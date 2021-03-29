[site](https://megaproduct.lotte.net/competitionSummary/6)

# 주제
- 유통 진열 상품 약 1,000종 대상 이미지 분류 AI 모델 개발

# 배경
- 국내 매장에서 유통중인 다량의 상품에 대해 각 상품을 특정하는 이미지 분류
- Vision AI 기반으로 분류 정확도를 끌어 올릴 수 있는 인공지능 모델 개발

# 주최
- 롯데정보통신

# 일정
- 2021.03.15 - 2021.03.26

# 데이터
- 학습용 : 48,000장 
- 제출용 : 72,000장
- 각 이미지 shape : (256, 256, 3)
- class 수 : 1,000개
- 촬영정보
  - 360도 roll 촬영
  - 이미지에 한제품이 여러개 촬영된 것도 있음.
  - 훈련 데이터에는 0도, 30도 각도에서 찍은 것이 있지만 시험(제출) 데이터에는 0,30,60도 각도가 있음.

# 데이터 훝기 & 전략 세우기
- 데이터는 360도 roll 촬영된 것으로 영상을 frame 단위로 저장한 것일 수 있음 -> 훈련, 검증으로 나눌 시 overfitting 될 가능성이 높음.
- 상품 이미지의 text를 추출하여 feature로 사용하는 방법 -> 이미지의 해상도가 작기 때문에 정보를 추출하기 어려움
- baseline code를 만들어 제출해 보았을 때 75~79 성능을 냄 -> train/validation에 overfitting 된것을 확인
- overfitting 을 완화하기 위해 augmentation 전략 세우는 것을 최우선으로 둠.
- baseline 을 통해 앙상블시 1~3 점 점수 향상이 있었음.
- 

- Data Augmentation 방법
  - 데이터의 촬영 정보를 보면 훈련 데이터로 주어진 것은 60도각도가 없다. 60도 각도 뿐만 아니라 여러 각도를 보충 해줄 수 있는 방법을 고려
  - 예시 이미지를 보면 각도가 달라질 수록 object의 색감도 변할 수 있다.
  - 사용된 augmentation 방법 (albumentation library 사용)
    - SiftScaleRotate
    - GridDistortion
    - Blur
    - HorizontalFlip
    - Rotate
  - Some Tricks
    - mixup
    - cutmix
```
aug = A.Compose([
            A.ShiftScaleRotate(border_mode=1),
            A.GridDistortion(border_mode=1),
            A.Blur(blur_limit=1),
            A.HorizontalFlip(p=0.5),
            A.Rotate(border_mode=1),
            A.OneOf([
                    A.CLAHE(),
                    A.RandomBrightnessContrast(),]),
                ])
```


# Experiments Parameters
  ## Network
    - EfficientNet B0
    - EfficientNet B1
    - EfficientNet B2

  ## optimizer
    - [AdamP](https://github.com/clovaai/AdamP)
    - Adam
    - [SAM](https://github.com/davda54/sam)
  ## scheduler
    - Cosine Annealing 
    - [Polynomial decay](https://github.com/cmpark0126/pytorch-polynomial-lr-decay)

  ## Epochs & Batch size
    - Epochs : 50
    - Batch Size : 128

  ## Loss
    - CrossEntropy Loss

  ## Model Save
    - validation best accuracy

# Ensemble
  - 각 Network를 훈련시킬 때 5-fold로 분리를 한 후 prob soft voting ensemble.

# Experiments Results
  ## augmentation에 따른 성능 변화

  <center>  

  |제외 aug method|score|
  |---------------|-----|
  |None|91.457|
  |mixup & cutmix|81.510|
  |cutmix|87.225|
  |Rotate|90.393|
  |mixup|90.707|
  |Blur|91.011|
  |HorizontalFlip|91.024|
  |GridDistortion|90.935|
  |CLAHE|91.114|
  |ShiftScaleRotate|91.294|  

  </center>

# 
