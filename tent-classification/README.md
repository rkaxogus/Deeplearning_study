# CAMTER_CATEGORY

## 프로젝트 개요

tent-classification는 현재 텐트를 5개 브랜드 29개 텐트로 분류하고, 딥러닝 모델을 활용하여 분류작업을 수행하는 브랜치
전처리 과정에서 데이터를 정리하고, EfficientNet B0 모델을 활용하여 학습 및 평가를 진행


## 디렉토리 구조
```
tent-classification/
├── models                  # 모델 관련 파일 저장
│   ├── tent_model.h5       # 학습된 모델 가중치 파일
│   ├── tent_model_product.h5  # 프로덕션용 최종 모델 가중치 파일
│   ├── model_utils.py      # 모델 로드/저장 관련 유틸리티 스크립트
├── scripts                 # 데이터 처리 및 모델 학습 스크립트
│   ├── preprocess          # 데이터 전처리 관련 스크립트
│   │   ├── tent.py         # 데이터셋 폴더 구조 생성 스크립트
│   │   ├── tent2.py        # 폴더 생성 대안 스크립트
│   ├── train               # 모델 학습 관련 스크립트
│   │   ├── efficientnet.ipynb  # EfficientNet 기반 모델 학습 노트북
│   │   ├── efficientnet2.ipynb # EfficientNet 추가 실험 노트북
├── README.md               # 프로젝트 설명 및 설정 방법
```

## 주요 파일 설명


### 1. efficientnet.ipynb와 efficientnet2.ipynb,efficientnet3.ipynb
-efficientnet.ipynb: 1번째:for문을 통해 드롭아웃(0,0.2,0.3,0.5)별로 학습을 진행한 코드입니다
                     2번째:리콜값을 반환하는 코드입니다   

-efficientnet2.ipynb: 1번쨰:모델을 학습시키는 가장 기본 형태의 코드가 있습니다. 다른 모델 학습 코드는 전부 이 코드 기반으로 변형된 코드입니다
                      2번째:리콜을 통해 낮은 정확도를 보이는 클래스를 제외한 6개의 클래스로 모델을 학습시킨 코드이며, 최종목표인 80퍼센트를 달성했습니다(클래스가 6개 밖에 안되긴 하여 206개로 했을때는 모름) 
                      3,4번쨰:저장된 모델을 사용해 이미지를 넣어 잘 맞추는지 테스트 하는 코드입니다 

-efficientnet2.ipynb: 테스트 페이지 이며 신경쓰지 않으셔도 됩니다                      
### 2. `tent_model.h5와 tent_model_product.h5`
-efficientnet_tent_model.pth: 가장 기본 코드의 학습된 모델 가중치 파일.
-efficientnet_tent_model2.pth: 프로덕션 환경에서 사용할 최종 모델 가중치 파일, 리콜을 통해 계산한 상위 6개의 클래스를 사용하였으며, 최대 82%의 정확도가 나옴
-efficientnet_tent_model3.pth:가장 기본 코드의 최종 학습된 모델 가중치 파일, 11개의 클래스의 824개 데이터를 전부 사용하였으며, 최대 정확도가 70%이상 나옴
## 설치 및 실행 방법



## 결과물
- 학습된 모델 파일: -efficientnet_tent_model.pth,efficientnet_tent_model2(main).pth,efficientnet_tent_model3(test).pth

## 참고 사항
-데이터셋은 프로젝트 디렉토리에 포함되지 않으며, 별도로 준비해야 합니다.
-EfficientNet 모델을 활용하여 텐트 이미지의 브랜드와 모델을 정확히 분류합니다.
-현재 최고 정확도 82%