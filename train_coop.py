import os
import json
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

# --- SAM3 모듈 ---
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("❌ SAM3 모듈을 찾을 수 없습니다."); exit()

# ==========================================
#  설정 (Configuration)
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "hf-hub:timm/PE-Core-bigG-14-448"

# 훈련 하이퍼파라미터
BATCH_SIZE = 16       
GRAD_ACCUM_STEPS = 2  
LR = 0.002           
EPOCHS = 10          
CTX_INIT = "a photo of a"
N_CTX = 4            

SAM3_CHECKPOINT = "checkpoints/sam3.pt"
SPLIT_JSON = "../dataset/ioc/anno/split_config.json"
ANNO_JSON = "../dataset/ioc/anno/annotations.json"
DATASET_ROOT = "../dataset/ioc"
SAVE_DIR = "checkpoints/coop_language_guided"

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 1. CoOp 모델 (Vision-Language Alignment)
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
        
        # 모델 본연의 마스크를 안전하게 가져옴
        self.attn_mask = getattr(text_model, 'attn_mask', None)

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        
        # 🔥 [수정됨] OpenCLIP은 batch_first=True 이므로 permute(1, 0, 2)를 삭제합니다!
        
        mask = self.attn_mask.to(x.device) if self.attn_mask is not None else None
        x = self.transformer(x, attn_mask=mask)
        
        # 🔥 [수정됨] 원래대로 되돌리는 permute도 삭제합니다!
        
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
        
        if self.text_projection is not None:
            x = x @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        text_model = _get_text_tower(clip_model)
        dtype = next(text_model.parameters()).dtype

        print(f"🧩 Initializing Bulletproof Learnable Context")
        tokenizer = open_clip.get_tokenizer(MODEL_ID)
        
        # 1. 완벽한 문장으로 토큰화
        prompts = [f"{CTX_INIT} {name.replace('_', ' ')}" for name in classnames]
        self.tokenized_prompts = tokenizer(prompts).to(DEVICE)
        
        with torch.no_grad():
            base_emb = text_model.token_embedding(self.tokenized_prompts).type(dtype)
        
        # =================================================================
        # index 0: [SOS]
        # index 1~4: "a photo of a" (학습할 부분)
        # index 5~: 클래스명 + [EOS] + PAD
        # =================================================================
        self.register_buffer("token_prefix", base_emb[:, 0:1, :])  # 앞부분 고정
        self.register_buffer("token_suffix", base_emb[:, 5:, :])   # 뒷부분 고정
        
        # 딱 4칸의 단어만 파라미터로 선언하여 학습 활성화
        self.ctx = nn.Parameter(base_emb[0, 1:5, :].clone())
        self.n_cls = len(classnames)
        
    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        return prompts

class CoOp(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        
    def forward(self, image):
        with torch.no_grad():
            image_features = self.image_encoder(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        prompts = self.prompt_learner()
        tokenized = self.tokenized_prompts
        
        # 텍스트 인코더 자체는 학습하지 않으므로 VRAM 절약을 위해 no_grad 처리
        # (단, prompts에 대한 그래디언트는 흘러가도록 모델 내부 파라미터만 requires_grad=False로 처리됨)
        text_features = self.text_encoder(prompts, tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        return logits

# ==========================================
# 2. Dataset 구성 (NMS 및 Point Tracker 적용)
# ==========================================
def is_point_in_box(point, box, padding=5):
    px, py = point
    x1, y1, x2, y2 = box
    return (x1 - padding <= px <= x2 + padding) and (y1 - padding <= py <= y2 + padding)

class SAMCoCoOpDataset(Dataset):
    def __init__(self, split, preprocess, sam_processor):
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
        
        print(f"\n📦 [Dataset] Collecting crops for '{split}' split...")
        print(f"   - Target Ref Exps: {len(self.class_names)} classes")
        
        for img_rel_path in tqdm(file_list, desc="Extracting & Caching Patches"):
            if img_rel_path not in full_anno: continue
            img_full_path = os.path.join(DATASET_ROOT, img_rel_path)
            if not os.path.exists(img_full_path): continue

            
            raw_image = Image.open(img_full_path).convert("RGB")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                inference_state = sam_processor.set_image(raw_image)

            anno = full_anno[img_rel_path]
            img_groups = defaultdict(list)
            for ref_exp, details in anno.items():
                base_cls = details.get('class', 'object')
                pts = details.get('points', [])
                if pts: img_groups[base_cls].append({'ref_exp': ref_exp, 'points': pts})

            matched_points_tracker = set()

            for base_cls, sub_groups in img_groups.items():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = sam_processor.set_text_prompt(state=inference_state, prompt=base_cls)
                boxes = output["boxes"]
                if boxes is None or len(boxes) == 0: continue
                
                scores_tensor = output.get("iou_predictions", output.get("scores"))
                if scores_tensor is not None: scores_tensor = scores_tensor.flatten()
                else: scores_tensor = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                
                nms_indices = ops.nms(boxes.float(), scores_tensor.float(), iou_threshold=0.6)
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

                    pad = 10
                    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                    cx2, cy2 = min(raw_image.width, x2 + pad), min(raw_image.height, y2 + pad)
                    if cx2 - cx1 < 10 or cy2 - cy1 < 10: continue
                    
                    crop = raw_image.crop((cx1, cy1, cx2, cy2))
                    tensor_img = preprocess(crop)
                    label_idx = self.cls_to_idx[matched_ref]
                    self.samples.append((tensor_img, label_idx))
                    
        print(f"✅ Caching Complete! Total Patches: {len(self.samples)}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# ==========================================
# 3. 메인 학습 실행
# ==========================================
def main():
    print(f"🚀 Training Language-Guided CoOp on {DEVICE}...")
    torch.cuda.empty_cache()

    print("Loading PE-Core & SAM3...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_ID, device=DEVICE, force_custom_text=True
    )
    
    sam_model = build_sam3_image_model(load_from_HF=False, checkpoint_path=SAM3_CHECKPOINT)
    sam_proc = Sam3Processor(sam_model)
        
    train_dataset = SAMCoCoOpDataset(split='train', preprocess=preprocess, sam_processor=sam_proc)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    print("\nBuilding CoOp Alignment Model...")
    model = CoOp(train_dataset.class_names, clip_model).to(DEVICE)
    
    for name, param in model.named_parameters():
        if "prompt_learner.ctx" not in name:
            param.requires_grad = False
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔥 Trainable Parameters: {trainable_params} (Only the Context Vectors!)")
            
    optimizer = optim.SGD(model.prompt_learner.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda') 
    
    print(f"\n🔥 Start Training for {EPOCHS} epochs...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss, correct, total = 0, 0, 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, labels) / GRAD_ACCUM_STEPS 
            
            scaler.scale(loss).backward()
            
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            current_loss = loss.item() * GRAD_ACCUM_STEPS
            total_loss += current_loss
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_description(f"Epoch {epoch+1} Loss: {current_loss:.4f} Acc: {100.*correct/total:.2f}%")
            
        scheduler.step()
        
    save_path = os.path.join(SAVE_DIR, "coop_language_guided_model.pth")
    torch.save(model.prompt_learner.state_dict(), save_path)
    
    meta_path = os.path.join(SAVE_DIR, "classes.json")
    with open(meta_path, 'w') as f:
        json.dump(train_dataset.class_names, f)
        
    print(f"\n✅ Training Complete! Model & Classes saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()