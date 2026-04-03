import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import open_clip
from tqdm import tqdm
from PIL import Image
import cv2
from collections import defaultdict
import torchvision.ops as ops
import math

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

CTX_INIT = "a photo of a"
N_CTX = 4            
CONF_THRESHOLD = 0.5   

SAM3_CHECKPOINT = "checkpoints/sam3.pt"
SPLIT_JSON = "../dataset/ioc/anno/split_config.json"
ANNO_JSON = "../dataset/ioc/anno/annotations.json"
DATASET_ROOT = "../dataset/ioc"
MODEL_WEIGHTS = "checkpoints/coop_language_guided/coop_language_guided_model.pth"
CLASSES_JSON = "checkpoints/coop_language_guided/classes.json"
RESULT_DIR = "coop_result"

os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================================
# 1. 모델 구조 (학습 코드와 100% 동일)
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
    def __init__(self, classnames, clip_model):
        super().__init__()
        text_model = _get_text_tower(clip_model)
        dtype = next(text_model.parameters()).dtype

        tokenizer = open_clip.get_tokenizer(MODEL_ID)
        prompts = [f"{CTX_INIT} {name.replace('_', ' ')}" for name in classnames]
        self.tokenized_prompts = tokenizer(prompts).to(DEVICE)
        
        with torch.no_grad():
            base_emb = text_model.token_embedding(self.tokenized_prompts).type(dtype)
        
        self.register_buffer("token_prefix", base_emb[:, 0:1, :])  
        self.register_buffer("token_suffix", base_emb[:, 5:, :])   
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
        
        text_features = self.text_encoder(prompts, tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        return logits

# ==========================================
# 2. 평가 루프 및 시각화 저장
# ==========================================
def main():
    print("🚀 Initializing Inference Pipeline...")
    
    with open(CLASSES_JSON, 'r') as f:
        class_names = json.load(f) # Referring Expressions 목록 (ex. 'crushed fuse')
    
    with open(SPLIT_JSON, 'r') as f:
        test_files = json.load(f).get('test', [])
        
    with open(ANNO_JSON, 'r') as f:
        full_anno = json.load(f)

    # 예측된 속성이 어떤 폴더(Base Class)에 들어가야 하는지 찾기 위함
    ref_to_base_map = {}
    for anno in full_anno.values():
        for ref_exp, details in anno.items():
            ref_to_base_map[ref_exp] = details.get('class', 'object')

    # 모델 준비
    clip_model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID, device=DEVICE, force_custom_text=True)
    clip_model.eval()
    
    sam_model = build_sam3_image_model(load_from_HF=False, checkpoint_path=SAM3_CHECKPOINT)
    sam_proc = Sam3Processor(sam_model)
    
    model = CoOp(class_names, clip_model).to(DEVICE)
    model.prompt_learner.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()

    # 평가 메트릭 변수
    absolute_errors = []
    squared_errors = []

    print(f"\n🏃‍♂️ Starting Inference on {len(test_files)} test images...")
    
    for img_rel_path in tqdm(test_files, desc="Evaluating"):
        if img_rel_path not in full_anno: continue
        img_full_path = os.path.join(DATASET_ROOT, img_rel_path)
        if not os.path.exists(img_full_path): continue
        
        raw_image = Image.open(img_full_path).convert("RGB")
        img_cv = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = sam_proc.set_image(raw_image)
        
        anno = full_anno[img_rel_path]
        
        # 정답(GT) 카운트 수집 및 SAM 텍스트 프롬프트용 클래스 추출
        gt_counts = defaultdict(int)
        base_classes = set()
        for ref_exp, details in anno.items():
            gt_counts[ref_exp] = len(details.get('points', []))
            base_classes.add(details.get('class', 'object'))
            
        # SAM으로 패치 추출
        all_crops = []
        all_boxes = []
        
        for base_cls in base_classes:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = sam_proc.set_text_prompt(state=inference_state, prompt=base_cls)
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
                pad = 10
                cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                cx2, cy2 = min(raw_image.width, x2 + pad), min(raw_image.height, y2 + pad)
                if cx2 - cx1 < 10 or cy2 - cy1 < 10: continue
                
                crop = raw_image.crop((cx1, cy1, cx2, cy2))
                all_crops.append(preprocess(crop))
                all_boxes.append((x1, y1, x2, y2))
                
        # CoOp 모델 추론
        pred_boxes_per_ref = defaultdict(list)
        
        if len(all_crops) > 0:
            crops_tensor = torch.stack(all_crops).to(DEVICE)
            with torch.no_grad(), torch.amp.autocast('cuda'):
                logits = model(crops_tensor)
                probs = logits.softmax(dim=1)
                max_probs, preds = probs.max(dim=1)
                
            for idx, (pred_idx, prob) in enumerate(zip(preds, max_probs)):
                if prob.item() >= CONF_THRESHOLD: 
                    pred_ref_exp = class_names[pred_idx.item()] # ex) 'crushed fuse'
                    pred_boxes_per_ref[pred_ref_exp].append(all_boxes[idx])
                    
        # 이미지별 시각화 및 오차 계산
        # eval_ref_exps = set(gt_counts.keys()).union(set(pred_boxes_per_ref.keys()))
        eval_ref_exps = set(gt_counts.keys())
        img_name_no_ext = os.path.splitext(os.path.basename(img_rel_path))[0]
        
        for ref_exp in eval_ref_exps:
            p_count = len(pred_boxes_per_ref[ref_exp])
            g_count = gt_counts.get(ref_exp, 0)
            
            err = abs(p_count - g_count)
            absolute_errors.append(err)
            squared_errors.append(err ** 2)
            
            draw_img = img_cv.copy()
            for (x1, y1, x2, y2) in pred_boxes_per_ref[ref_exp]:
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
            text = f"Exp: {ref_exp} | GT: {g_count} | Pred: {p_count}"
            cv2.putText(draw_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 🔥 [경로 생성 로직 수정] coop_result/클래스명(SAM)/이미지명/referring_expression.jpg
            base_class_name = ref_to_base_map.get(ref_exp, "unknown_class")
            
            safe_base_cls = base_class_name.replace(" ", "_").replace("/", "_")
            safe_ref_exp = ref_exp.replace(" ", "_").replace("/", "_")
            
            save_dir = os.path.join(RESULT_DIR, safe_base_cls, img_name_no_ext)
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, f"{safe_ref_exp}.jpg")
            cv2.imwrite(save_path, draw_img)

    # ==========================================
    # 3. 최종 메트릭 출력
    # ==========================================
    if len(absolute_errors) > 0:
        mae = np.mean(absolute_errors)
        rmse = math.sqrt(np.mean(squared_errors))
        print("\n" + "="*40)
        print("🎯 Evaluation Metrics on Test Set")
        print("="*40)
        print(f"Total Evaluated Pairs : {len(absolute_errors)}")
        print(f"MAE (Mean Abs Error)  : {mae:.4f}")
        print(f"RMSE (Root Mean Sq)   : {rmse:.4f}")
        print("="*40)
        print(f"✅ Visualization images saved in '{RESULT_DIR}/<class_name>/<image_name>/<ref_exp>.jpg'")
    else:
        print("\n⚠️ No classes evaluated. Please check if annotations match the test split.")

if __name__ == "__main__":
    main()