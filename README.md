롯데정보통신 Vision AI 경진대회(채용연계) [site](https://megaproduct.lotte.net/competitionSummary/6)

model weight : https://drive.google.com/file/d/1VPAZU8OM1jFHF5E38WlMY8_ztOGYsInv/view?usp=sharing


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
- 데이터는 360도 roll 촬영된 것으로 영상을 frame 단위로 저장한 것일 수 있음 -> train으로 주어진 데이터로만 훈련, 검증으로 나눌 시 overfitting 될 가능성이 높음. 어쩔 수 없는 부분.
- 상품 이미지의 text를 추출하여 feature로 사용하는 방법 -> 이미지의 해상도가 작기 때문에 정보를 추출하기 어려움
- baseline code를 만들어 제출해 보았을 때 validation 99.xx가 75~79 성능을 냄 -> train/validation분포에만 overfitting 된것을 확인
- overfitting 을 완화하기 위해 augmentation 전략 세우는 것을 최우선으로 둠.
- baseline model 을 통해 soft voting ensemble 시도하여 1~3 점 점수 향상이 있었음. -> ensemble로 분류 성능이 

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
    - [cutmix](https://github.com/clovaai/CutMix-PyTorch)
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
  ## LearningRate
    - init LR : 1e-3
    - end LR : 1e-6
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
  ## augmentation에 따른 실험 결과
    - megaproduct 시스템에 제출된 점수 (testset의 validation, testset의 private은 공개되지 않음)
    - None은 전체 aug method를 사용한 결과.
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

  
  - cutmix와 mixup 방법을 제외 했을 때 아주 큰 성능 차이를 보였음.
  - 즉, cutmix와 mixup 방법으로 아주 큰 성능 향상을 보였음.
  - 나머지 방법들에 대해서도 약간의 성능 변화가 있었지만 각 1회 훈련한 결과이므로 차이가 있을 수 있음.
  - 이후 실험들에 대해서는 모두 사용하여 훈련을 하였음.



  ## model ensemble에 따른 실험 결과

  |model|score|
  |---------------|-----|
  |EfficientNet B0|94.233|
  |EfficientNet B1|94.404|
  |EfficientNet B2|94.603|
  
  - 해당 성능은 TTA를 적용시켜 제출한 결과.
  - TTA시 약 0.02 정도 점수가 향상됨.
  - 추가) 해당 결과는 Adam optimizer를 사용한 결과로 AdamP 혹은 SAM 을 사용하면 약 0.01~0.02 향상 함.
  - single fold 에서 91.xxx 성능을 보이던 것을 soft voting ensemble하면 2~3 정도 향상.
  - b0, b1, b2 모델이 더 깊어질수록 성능도 조금씩 향상되는 모습을 보였음.
  
  ## Others
  - Epoch과 Batch Size에 따른 실험 결과는 성능 변화가 크게 있지 않으므로 생략.
  - Batch Size 128 일때 Epoch 40~55 가 좋은 성능을 보임.
  - Learning Rate는 init으로 1e-3, end로 1e-6 scheduler를 통해 조절했을 때 가장 성능이 좋았음.

# 결론
- Train, Test 에는 pinch에 따른 데이터 분포 차이가 존재 -> 여러 augmentation 방법을 통해 일부 overfitting 되는 부분을 완화 시킴.
- 특히 mixup 과 cutmix를 통해 아주 큰 성능 향상을 보였음.
- 모델의 크기가 커질 수록 성능이 향상. (실험한 모델보다 더 큰 모델인 b3-b7 모델들을 통해 직접확인 해 보지 못했지만 큰 성능 향상은 없을것으로 보임.)
