# CoOp Fine-tuning with PE-Core bigG/14 + SAM3

Context Optimization (CoOp) 기반의 referring-expression counting 파이프라인.
SAM3로 객체를 탐지하고, PE-Core (ViT-bigG/14-448) 의 텍스트 임베딩에 learnable context vector를 삽입하여 referring-expression 분류 성능을 향상시킵니다.

## Pipeline

```
Input Image
    |
    v
[SAM3] -- base_cls text prompt --> 후보 박스 탐지
    |
    v
NMS 필터링 --> 개별 크롭 추출
    |
    v
[PE-Core + CoOp] -- learnable ctx + class name --> 크롭 분류
    |
    v
ref_exp 별 카운팅 --> MAE / RMSE 평가
```

## Structure

```
coop-pe-core/
├── README.md
├── requirements.txt
├── train_coop.py          # CoOp 학습 (SAM3 crop + PE-Core 분류)
├── test_coop.py           # CoOp 학습 후 테스트 (MAE/RMSE + 시각화)
├── test_baseline.py       # CoOp 없이 zero-shot PE-Core baseline 테스트
├── ablation_vectors.py    # Learnable context vector 수 ablation (n_ctx=2,8)
└── checkpoints/
    └── sam3.pt            # SAM3 checkpoint (3.3GB)
```

## Environment Setup

```bash
conda create -n coop-pe-core python=3.12 -y
conda activate coop-pe-core
pip install -r requirements.txt
```

SAM3 모듈 설치 (프로젝트 폴더 밖에서 clone):

```bash
cd ..
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
cd ../coop-pe-core
```

**주의**: `coop-pe-core/` 폴더 안에 `sam3`라는 이름의 디렉토리가 있으면 Python namespace가 충돌하여 import가 실패합니다. SAM3 repo는 반드시 프로젝트 폴더 밖에 clone하세요.

### Tested Versions

- Python 3.12.12
- PyTorch 2.9.1+cu128
- open_clip_torch 3.2.0
- timm 1.0.24

## Data

- IoC dataset: `../dataset/ioc/`
  - `anno/annotations.json`
  - `anno/split_config.json`
- SAM3 checkpoint: `checkpoints/sam3.pt` 

## Usage

### 1. Baseline (Zero-shot PE-Core, CoOp 없음)

```bash
python test_baseline.py
```

### 2. CoOp 학습

```bash
python train_coop.py
```

- 학습된 ctx 벡터: `checkpoints/coop_language_guided/coop_language_guided_model.pth`
- 클래스 목록: `checkpoints/coop_language_guided/classes.json`

### 3. CoOp 테스트 (MAE/RMSE + 시각화)

```bash
python test_coop.py
```

- 시각화 결과: `coop_result/<class_name>/<image_name>/<ref_exp>.jpg`

### 4. Ablation Study (context vector 수)

```bash
python ablation_vectors.py
```

learnable context vector 수(n_ctx)에 따른 성능 비교:

| n_ctx | MAE  | RMSE |
|-------|------|------|
| 2     | 2.00 | 4.58 |
| 4     | 1.80 | 4.49 |
| 8     | 6.29 | 9.25 |

## Model

- **Vision encoder**: PE-Core ViT-bigG/14-448 (`hf-hub:timm/PE-Core-bigG-14-448`)
- **Object detector**: SAM3 (text-prompted segmentation)
- **CoOp**: 4 learnable context vectors (default), dim=1280
- **Optimizer**: SGD (lr=0.002, momentum=0.9)
- **Epochs**: 10
