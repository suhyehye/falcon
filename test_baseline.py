import os
import json
import torch
import torch.nn as nn
import numpy as np
import open_clip
from PIL import Image
from tqdm import tqdm
import torchvision.ops as ops

# --- SAM3 모듈 ---
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("❌ SAM3 모듈을 찾을 수 없습니다."); exit()

# ==========================================
# ⚙️ 설정 (Configuration)
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "hf-hub:timm/PE-Core-bigG-14-448"
CONF_THRESHOLD = 0.5   

SAM3_CHECKPOINT = "checkpoints/sam3.pt"
SPLIT_JSON = "../dataset/ioc/anno/split_config.json"
ANNO_JSON = "../dataset/ioc/anno/annotations.json"
DATASET_ROOT = "../dataset/ioc"

# ==========================================
# 1. Zero-shot Evaluation 함수
# ==========================================
def run_zeroshot_evaluation():
    print(f"🚀 Initializing PE-Core Zero-shot Evaluation on {DEVICE}...")

    # 1. 모델 및 데이터 로드
    clip_model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID, device=DEVICE)
    tokenizer = open_clip.get_tokenizer(MODEL_ID)
    clip_model.eval()

    sam_model = build_sam3_image_model(load_from_HF=False, checkpoint_path=SAM3_CHECKPOINT)
    sam_proc = Sam3Processor(sam_model)

    with open(SPLIT_JSON, 'r') as f:
        test_files = json.load(f).get('test', [])
    with open(ANNO_JSON, 'r') as f:
        full_anno = json.load(f)

    # 2. 클래스 목록 및 텍스트 임베딩 생성 (Zero-shot)
    all_ref_exps = set()
    for anno in full_anno.values():
        for ref_exp in anno.keys():
            all_ref_exps.add(ref_exp)
    class_names = sorted(list(all_ref_exps))
    
    # "a photo of a [defect]" 형식으로 프롬프트 생성
    prompts = tokenizer([f"a photo of a {name}" for name in class_names]).to(DEVICE)
    
    with torch.no_grad():
        text_features = clip_model.encode_text(prompts)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # 3. 테스트 루프
    errors = []
    print(f"📊 Evaluating {len(test_files)} images...")

    for img_rel_path in tqdm(test_files):
        if img_rel_path not in full_anno: continue
        img_full_path = os.path.join(DATASET_ROOT, img_rel_path)
        
        raw_image = Image.open(img_full_path).convert("RGB")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = sam_proc.set_image(raw_image)

        img_anno = full_anno[img_rel_path]
        
        # 각 타겟 상태(예: broken fuse)별로 카운팅 수행
        for target_prompt, details in img_anno.items():
            base_class = details.get('class', 'object')
            gt_count = len(details.get('points', []))
            
            # SAM3 Region Proposal
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = sam_proc.set_text_prompt(state=inference_state, prompt=base_class)
            boxes = output["boxes"]
            if boxes is None or len(boxes) == 0:
                errors.append(gt_count) # 0개 예측한 셈
                continue

            # NMS
            scores = output.get("iou_predictions", output.get("scores")).flatten()
            nms_idx = ops.nms(boxes.float(), scores.float(), iou_threshold=0.6)
            filtered_boxes = boxes[nms_idx].cpu().numpy()

            # 패치 추출 및 분류
            pred_count = 0
            all_crops = []
            for box in filtered_boxes:
                x1, y1, x2, y2 = map(int, box.flatten().tolist())
                pad = 10
                cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                cx2, cy2 = min(raw_image.width, x2+pad), min(raw_image.height, y2+pad)
                if cx2-cx1 < 10 or cy2-cy1 < 10: continue
                all_crops.append(preprocess(raw_image.crop((cx1, cy1, cx2, cy2))))

            if all_crops:
                crops_tensor = torch.stack(all_crops).to(DEVICE)
                with torch.no_grad():
                    image_features = clip_model.encode_image(crops_tensor)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    logits = (100.0 * image_features @ text_features.t()).softmax(dim=-1)
                    max_probs, preds = logits.max(dim=1)
                    
                    target_idx = class_names.index(target_prompt)
                    # 예측 결과 중 target_prompt와 일치하고 임계값을 넘는 것만 카운트
                    for p_idx, prob in zip(preds, max_probs):
                        if p_idx == target_idx and prob >= CONF_THRESHOLD:
                            pred_count += 1
            
            errors.append(abs(gt_count - pred_count))

    # 4. 최종 메트릭 계산
    errors = np.array(errors)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))

    print(f"\n✨ [Final Zero-shot Results]")
    print(f"   - MAE: {mae:.4f}")
    print(f"   - RMSE: {rmse:.4f}")

if __name__ == "__main__":
    run_zeroshot_evaluation()