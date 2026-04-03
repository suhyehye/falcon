"""
ablation_vectors.py - CoOp learnable context vector 수 ablation
================================================================
n_ctx = [2, 4] 로 각각 학습 → test MAE/RMSE 비교.

파이프라인:
  1. SAM3 → crop → 라벨링 (데이터셋은 한번만 생성)
  2. n_ctx 별로 CoOp 학습
  3. n_ctx 별로 SAM3 → crop → CoOp 분류 → count → MAE/RMSE
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import open_clip
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
import torchvision.ops as ops

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("SAM3 모듈을 찾을 수 없습니다."); exit()

# ==========================================
# 설정
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "hf-hub:timm/PE-Core-bigG-14-448"
CTX_INIT = "a photo of a"

BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
LR = 0.002
EPOCHS = 10
GRAD_CLIP = 1.0
MIN_CROP_SIZE = 16
BOX_PAD = 10
NMS_IOU_THRESH = 0.6

SAM3_CHECKPOINT = "checkpoints/sam3.pt"
SPLIT_JSON = "../dataset/ioc/anno/split_config.json"
ANNO_JSON = "../dataset/ioc/anno/annotations.json"
DATASET_ROOT = "../dataset/ioc"
SAVE_DIR = "checkpoints/ablation_vectors"

N_CTX_LIST = [2, 8]

os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================================
# CoOp 모델
# ==========================================
def _get_text_tower(clip_model):
    if hasattr(clip_model, 'text'): return clip_model.text
    return clip_model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        text_model = _get_text_tower(clip_model)
        self.transformer = text_model.transformer
        self.positional_embedding = text_model.positional_embedding
        self.ln_final = text_model.ln_final
        self.text_projection = text_model.text_projection
        self.dtype = next(text_model.parameters()).dtype
        self.attn_mask = getattr(text_model, 'attn_mask', None)

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        mask = self.attn_mask.to(x.device) if self.attn_mask is not None else None
        x = self.transformer(x, attn_mask=mask)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
        if self.text_projection is not None:
            x = x @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx):
        super().__init__()
        text_model = _get_text_tower(clip_model)
        dtype = next(text_model.parameters()).dtype
        tokenizer = open_clip.get_tokenizer(MODEL_ID)

        prompts = [f"{CTX_INIT} {name.replace('_', ' ')}" for name in classnames]
        self.tokenized_prompts = tokenizer(prompts).to(DEVICE)

        with torch.no_grad():
            base_emb = text_model.token_embedding(self.tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", base_emb[:, 0:1, :])
        self.register_buffer("token_suffix", base_emb[:, n_ctx+1:, :])
        self.ctx = nn.Parameter(base_emb[0, 1:n_ctx+1, :].clone())
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)

class CoOp(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model, n_ctx)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale

    def forward(self, image):
        with torch.no_grad():
            image_features = self.image_encoder(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * image_features @ text_features.t()
        return logits


# ==========================================
# Dataset (한번만 생성)
# ==========================================
def is_point_in_box(point, box, padding=5):
    px, py = point
    x1, y1, x2, y2 = box
    return (x1 - padding <= px <= x2 + padding) and (y1 - padding <= py <= y2 + padding)

class SAMCropDataset(Dataset):
    def __init__(self, split, preprocess, sam_processor, augment=False):
        self.preprocess = preprocess
        self.augment = augment
        self.samples = []

        with open(SPLIT_JSON, 'r') as f:
            file_list = json.load(f).get(split, [])
        with open(ANNO_JSON, 'r') as f:
            full_anno = json.load(f)

        all_ref_exps = set()
        for anno in full_anno.values():
            for ref_exp in anno.keys():
                all_ref_exps.add(ref_exp)

        self.class_names = sorted(list(all_ref_exps))
        self.cls_to_idx = {c: i for i, c in enumerate(self.class_names)}

        print(f"\n[Dataset] Collecting crops for '{split}' split...")
        for img_rel_path in tqdm(file_list, desc="Extracting Patches"):
            if img_rel_path not in full_anno: continue
            img_full_path = os.path.join(DATASET_ROOT, img_rel_path)
            if not os.path.exists(img_full_path): continue

            try:
                raw_image = Image.open(img_full_path).convert("RGB")
                inference_state = sam_processor.set_image(raw_image)
            except: continue

            anno = full_anno[img_rel_path]
            img_groups = defaultdict(list)
            for ref_exp, details in anno.items():
                base_cls = details.get('class', 'object')
                pts = details.get('points', [])
                if pts: img_groups[base_cls].append({'ref_exp': ref_exp, 'points': pts})

            matched_points_tracker = set()
            for base_cls, sub_groups in img_groups.items():
                output = sam_processor.set_text_prompt(state=inference_state, prompt=base_cls)
                boxes = output["boxes"]
                if boxes is None or len(boxes) == 0: continue

                scores_tensor = output.get("iou_predictions", output.get("scores"))
                if scores_tensor is not None: scores_tensor = scores_tensor.flatten()
                else: scores_tensor = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

                nms_indices = ops.nms(boxes.float(), scores_tensor.float(), iou_threshold=NMS_IOU_THRESH)
                filtered_boxes = boxes[nms_indices].cpu().numpy()
                if filtered_boxes.ndim == 1: filtered_boxes = filtered_boxes[None, :]

                for box in filtered_boxes:
                    x1, y1, x2, y2 = map(int, box.flatten().tolist())
                    matched_ref = None
                    for group in sub_groups:
                        for pt in group['points']:
                            pt_tuple = tuple(pt)
                            if pt_tuple in matched_points_tracker: continue
                            if is_point_in_box(pt, (x1, y1, x2, y2), padding=5):
                                matched_ref = group['ref_exp']
                                matched_points_tracker.add(pt_tuple)
                                break
                        if matched_ref: break
                    if not matched_ref: continue

                    cx1 = max(0, x1 - BOX_PAD)
                    cy1 = max(0, y1 - BOX_PAD)
                    cx2 = min(raw_image.width, x2 + BOX_PAD)
                    cy2 = min(raw_image.height, y2 + BOX_PAD)
                    if cx2 - cx1 < MIN_CROP_SIZE or cy2 - cy1 < MIN_CROP_SIZE: continue

                    crop = raw_image.crop((cx1, cy1, cx2, cy2))
                    self.samples.append((crop, self.cls_to_idx[matched_ref]))

        print(f"Total Patches: {len(self.samples)}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        pil_img, label = self.samples[idx]
        return self.preprocess(pil_img), label


# ==========================================
# SAM3 crop 유틸 (테스트용)
# ==========================================
def get_crops_from_image(raw_image, sam_proc, base_cls):
    inference_state = sam_proc.set_image(raw_image)
    output = sam_proc.set_text_prompt(state=inference_state, prompt=base_cls)
    boxes = output["boxes"]
    if boxes is None or len(boxes) == 0: return []

    scores = output.get("iou_predictions", output.get("scores"))
    if scores is not None: scores = scores.flatten()
    else: scores = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    nms_idx = ops.nms(boxes.float(), scores.float(), iou_threshold=NMS_IOU_THRESH)
    filtered = boxes[nms_idx].cpu().numpy()
    if filtered.ndim == 1: filtered = filtered[None, :]

    crops = []
    for box in filtered:
        x1, y1, x2, y2 = map(int, box.flatten().tolist())
        cx1 = max(0, x1 - BOX_PAD)
        cy1 = max(0, y1 - BOX_PAD)
        cx2 = min(raw_image.width, x2 + BOX_PAD)
        cy2 = min(raw_image.height, y2 + BOX_PAD)
        if cx2 - cx1 < MIN_CROP_SIZE or cy2 - cy1 < MIN_CROP_SIZE: continue
        crops.append(raw_image.crop((cx1, cy1, cx2, cy2)))
    return crops


# ==========================================
# 학습 함수
# ==========================================
def train_one_config(n_ctx, clip_model, train_loader, class_names):
    print(f"\n{'='*60}")
    print(f"  Training n_ctx={n_ctx}")
    print(f"{'='*60}")

    model = CoOp(class_names, clip_model, n_ctx).to(DEVICE)

    for name, param in model.named_parameters():
        if "prompt_learner.ctx" not in name:
            param.requires_grad = False
    model.prompt_learner.ctx.requires_grad = True

    trainable = model.prompt_learner.ctx.numel()
    hidden_dim = model.prompt_learner.ctx.shape[-1]
    print(f"  Trainable params: {trainable:,} (ctx: [{n_ctx} x {hidden_dim}])")

    optimizer = optim.SGD([model.prompt_learner.ctx], lr=LR, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    best_ctx = None

    for epoch in range(EPOCHS):
        model.train()
        model.image_encoder.eval()
        total_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{EPOCHS}")
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, labels) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([model.prompt_learner.ctx], GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item()*GRAD_ACCUM_STEPS:.4f}",
                           acc=f"{100.*correct/total:.1f}%")

        scheduler.step()
        train_acc = 100.0 * correct / total

        # validation on train set accuracy as proxy
        if train_acc > best_acc:
            best_acc = train_acc
            best_ctx = model.prompt_learner.ctx.data.cpu().clone()

        print(f"  Epoch {epoch+1} | loss={total_loss/len(train_loader):.4f} | acc={train_acc:.2f}%")

    # save
    ckpt_path = os.path.join(SAVE_DIR, f"coop_ctx_nctx{n_ctx}.pth")
    torch.save({
        'coop_state_dict': {'ctx': best_ctx},
        'n_ctx': n_ctx,
    }, ckpt_path)
    print(f"  Saved: {ckpt_path}")

    return ckpt_path


# ==========================================
# 평가 함수
# ==========================================
def evaluate(n_ctx, ckpt_path, clip_model, preprocess, sam_proc, class_names):
    cls_to_idx = {c: i for i, c in enumerate(class_names)}

    model = CoOp(class_names, clip_model, n_ctx).to(DEVICE)
    saved = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.prompt_learner.load_state_dict(saved['coop_state_dict'], strict=False)
    model.eval()

    with open(SPLIT_JSON, 'r') as f:
        test_files = json.load(f).get('test', [])
    with open(ANNO_JSON, 'r') as f:
        full_anno = json.load(f)

    errors = []

    for img_rel_path in tqdm(test_files, desc=f"  Eval n_ctx={n_ctx}", leave=False):
        if img_rel_path not in full_anno: continue
        img_full_path = os.path.join(DATASET_ROOT, img_rel_path)
        if not os.path.exists(img_full_path): continue

        try:
            raw_image = Image.open(img_full_path).convert("RGB")
        except: continue

        anno = full_anno[img_rel_path]
        cls_groups = defaultdict(list)
        for ref_exp, details in anno.items():
            base_cls = details.get('class', 'object')
            gt_count = len(details.get('points', []))
            cls_groups[base_cls].append((ref_exp, gt_count))

        for base_cls, ref_list in cls_groups.items():
            try:
                crops = get_crops_from_image(raw_image, sam_proc, base_cls)
            except: continue

            if len(crops) == 0:
                for ref_exp, gt_count in ref_list:
                    if ref_exp in cls_to_idx:
                        errors.append(abs(0 - gt_count))
                continue

            crop_tensors = torch.stack([preprocess(c) for c in crops]).to(DEVICE)
            with torch.no_grad(), torch.amp.autocast('cuda'):
                logits = model(crop_tensors)
                pred_idxs = logits.argmax(dim=-1)

            for ref_exp, gt_count in ref_list:
                if ref_exp not in cls_to_idx: continue
                target_idx = cls_to_idx[ref_exp]
                pred_count = (pred_idxs == target_idx).sum().item()
                errors.append(abs(pred_count - gt_count))

    errors = np.array(errors, dtype=np.float32)
    mae = errors.mean()
    rmse = np.sqrt((errors ** 2).mean())
    return mae, rmse, len(errors)


# ==========================================
# 메인
# ==========================================
def main():
    print(f"Ablation: n_ctx vectors on {DEVICE}")
    torch.cuda.empty_cache()

    # -- PE-Core 로드 --
    print("Loading PE-Core...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_ID, device=DEVICE, force_custom_text=True
    )

    # -- SAM3 로드 --
    print("Loading SAM3...")
    sam_model = build_sam3_image_model(load_from_HF=False, checkpoint_path=SAM3_CHECKPOINT)
    sam_proc = Sam3Processor(sam_model)

    # -- 데이터셋 (한번만 생성) --
    train_dataset = SAMCropDataset('train', preprocess, sam_proc)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    class_names = train_dataset.class_names

    meta_path = os.path.join(SAVE_DIR, "classes.json")
    with open(meta_path, 'w') as f:
        json.dump(class_names, f)

    print(f"\nClasses: {len(class_names)}")
    print(f"Train patches: {len(train_dataset)}")
    print(f"n_ctx configs: {N_CTX_LIST}")

    # -- n_ctx 별 학습 & 평가 --
    results = {}

    for n_ctx in N_CTX_LIST:
        start_t = time.time()
        ckpt_path = train_one_config(n_ctx, clip_model, train_loader, class_names)
        train_time = time.time() - start_t

        mae, rmse, total = evaluate(n_ctx, ckpt_path, clip_model, preprocess, sam_proc, class_names)
        results[n_ctx] = {'mae': mae, 'rmse': rmse, 'total': total, 'time': train_time}

        print(f"\n  n_ctx={n_ctx} | MAE={mae:.4f} | RMSE={rmse:.4f} | pairs={total} | time={train_time:.1f}s")

    # -- 결과 요약 --
    print("\n" + "=" * 60)
    print(f"  Ablation Results: Learnable Context Vectors")
    print(f"  Model: PE-Core bigG/14-448")
    print("=" * 60)
    print(f"  {'n_ctx':>5} | {'MAE':>8} | {'RMSE':>8} | {'Params':>10} | {'Time':>8}")
    print(f"  {'-'*5} | {'-'*8} | {'-'*8} | {'-'*10} | {'-'*8}")

    hidden_dim = clip_model.text.transformer.width if hasattr(clip_model.text, 'transformer') else 1280
    for n_ctx in N_CTX_LIST:
        r = results[n_ctx]
        params = n_ctx * hidden_dim
        print(f"  {n_ctx:>5} | {r['mae']:>8.4f} | {r['rmse']:>8.4f} | {params:>10,} | {r['time']:>7.1f}s")
    print("=" * 60)

    # -- 결과 저장 --
    result_path = os.path.join(SAVE_DIR, "ablation_results.txt")
    with open(result_path, 'w') as f:
        f.write("n_ctx,mae,rmse,total_pairs,train_time\n")
        for n_ctx in N_CTX_LIST:
            r = results[n_ctx]
            f.write(f"{n_ctx},{r['mae']:.4f},{r['rmse']:.4f},{r['total']},{r['time']:.1f}\n")
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
