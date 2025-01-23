# CAMTER_CATEGORY

## 프로젝트 개요

CAMTER_CATEGORY는 캠핑 용품 이미지를 10개의 주요 카테고리로 분류하고 딥러닝 모델을 활용하여 분류 작업을 수행하는 브랜치
전처리 과정에서 데이터를 정리하고, EfficientNet B0 모델을 활용하여 학습 및 평가를 진행

## 디렉토리 구조
```
CAMTER_CATEGORY
├── models            # 모델 관련 파일 저장
│   ├── camping_model.pth  # 학습된 모델 파일
│   ├── model_utils.py     # 모델 로드/저장 관련 유틸리티
├── scripts           # 스크립트 (기능별 분리)
│   ├── preprocess
│   │   ├── JsonTransfer.py        # 20개 카테고리로 분류
│   │   ├── CategoryTransfer.py    # 10개 카테고리로 변환
│   ├── train
│   │   ├── CategoryClassficationV1.py  # 모델 학습 코드
├── README.md         # 프로젝트 설명
├── requirements.txt  # Python 패키지 의존성
```

## 주요 파일 설명

### 1. `JsonTransfer.py`
- JSON 라벨링 데이터를 읽어들여 이미지를 20개의 세부 카테고리로 분류

### 2. `CategoryTransfer.py`
- 20개의 세부 카테고리를 10개의 주요 카테고리로 매핑

### 3. `CategoryClassficationV1.py`
- EfficientNet B0 모델을 활용하여 학습과 평가를 수행
- 학습된 모델은 `models/camping_model.pth`에 저장

## 설치 및 실행 방법

### 1. 환경 설정
- 필요한 Python 패키지를 설치합
pip install -r requirements.txt
```

### 2. 데이터 준비
- 전처리 과정을 통해 데이터를 `data/processed` 디렉토리로 정리

### 3. 전처리
- `JsonTransfer.py`와 `CategoryTransfer.py`를 순서대로 실행하여 데이터를 전처리

### 4. 학습 및 평가
- `CategoryClassficationV1.py`를 실행하여 모델을 학습
```bash
python scripts/train/CategoryClassficationV1.py
```

## 결과물
- 학습된 모델 파일: `models/camping_model.pth`

## 참고 사항
- 데이터는 프로젝트 디렉토리에 포함되지 않으며, 외부 스토리지를 통해 관리
- AIhub의 'Small object detection을 위한 이미지 데이터'중 캠핑용품 데이터 사용하였으며, 매트/침낭 항목의 데이터는 아직 구비를 못함
- "https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=476"
