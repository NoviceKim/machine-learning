### 주제: 로지스틱 회귀를 통해 잠수함 레이더 수치로 기뢰를 탐지하는 분류 모델 학습
- 이번에 사용한 데이터 세트의 총 데이터(행) 수는 207개
- 기뢰란, 선박을 파괴할 목적으로 수중에 설치하는 폭탄을 의미  
  
---
  
#### Features
- 총 60종의 float 타입 데이터
  
#### Targets
- 기뢰 / 바위 여부를 이진 분류
> - M: 기뢰
> - R: 바위
  
---
  
### 데이터 전처리
  
데이터 전처리에 앞서, 기존 데이터 세트 복사  
이후 결측치와 중복 여부를 검사했지만, 데이터 세트에서는 해당 사항들이 발견되지 않음
  
레이블 인코딩에 앞서 현재 타겟 데이터의 분포를 히스토그램으로 시각화
  
---
  
#### 레이블 인코딩
- target의 클래스가 2가지(R, M)이기 때문에 레이블 인코딩 실행

이 과정에서 Sklearn의 LabelEncoder를 사용하여 인코딩 했을 때,  
M(기뢰)가 0, R(바위)가 1로 변환되는 현상 확인  
  
보통 양성, 즉 문제 발생 상태를 1로 취급하는 것을 고려해서  
이번 프로젝트에서는 LabelEncoder 대신 별도로 선언한 함수로 인코딩  

```
# target 클래스를 레이블 인코딩하는 함수
# R → 0, M → 1로 변환
def target_encoder(target):
    if target == 'R':
        return 0
    if target == 'M':
        return 1
```

다만, 이 방식으로 인코딩 했을 때 데이터 타입이 수치형으로 바뀌지는 않기 때문에  
astype(np.int8) 메서드로 타겟 데이터의 타입을 수치형으로 변환,  
인코딩 후에는 아래와 같은 히스토그램을 시각화 할 수 있었음    
  
---
  
### 데이터 세트 분할
- 타겟 데이터의 클래스 별 데이터 수가 균일하지는 않지만, 그 차이가 크지 않기 때문에 SMOTE를 통한 오버 샘플링도 병행
  
sklearn의 train_test_split를 이용해 데이터를 분할한 뒤,  
imblearn의 SMOTE를 통한 오버 샘플링으로 훈련 데이터를 추가 생성  

```
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 데이터 세트 분할
features, targets = pre_m_df.iloc[:, :-1], pre_m_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(features, targets, stratify=targets, test_size=0.2, random_state=124)

# 오버 샘플링
smote = SMOTE(random_state=124)

over_X_train, over_y_train = smote.fit_resample(X_train, y_train)
```

이후 176개의 훈련 데이터와 42개의 테스트 데이터가 만들어짐  
  
---
  
### 차원 축소
- 현재 feature 수가 60개로 많기 때문에 feature들 간의 다중 공선성, 데이터 희소화 등의 이슈가 발생할 수 있음
- 따라서 표현력 손실을 감수하더라도 위 문제를 해결하고, 연산 속도를 보완하기 위해 차원 축소 실행
  
#### 과정
- 축소 이후의 차원 수는 보존률이 0.7 이상이면서, 0.7에 가장 근접할 때의 수치로 선정
- PCA 방식으로 차원 축소해보고, LDA 방식으로도 시도해 본 다음 두 가지 방식의 결과 비교
  
---

### PCA 방식으로 차원 축소

분할한 데이터를 차원 축소하기에 앞서,  
반복문을 통해 PCA 방식을 사용해서 1 ~ 15차원으로 축소했을 때 각 차원의 기존 데이터 보존률을 출력
  
```
from sklearn.decomposition import PCA

# 1 ~ 15차원까지 늘려보면서 각 차원의 데이터 보존률 출력
# fit하기 적합한 차원 수를 알아보기 위함
for i in range(15):
    pca = PCA(n_components=(i + 1))

    pca_train = pca.fit_transform(over_X_train)
    pca_test = pca.transform(X_test)
    
    # 차원 축소 후 데이터 보존률 확인
    print(f'{i + 1}차원')
    print(pca.explained_variance_ratio_.sum())
    print('=' * 30)
```
  
위 코드의 출력 결과로 5차원으로 축소했을 때 보존률이 0.7에 가까운, 이상적인 수치를 보이는 것을 확인  
하지만 그 전에, 차원 축소 이후 데이터의 분포를 산점도로 시각화하고 싶었기 때문에 우선 2차원으로 축소해서 결과를 확인함  



위의 시각화 결과를 통해, PCA 방식을 사용해서 2차원으로 축소하면 클래스 간 구분이 어렵다는 사실을 알 수 있었음  

이후, 원래 목표대로 기존 데이터를 5차원으로 축소 후  
차원 축소된 훈련용, 테스트용 데이터 세트를 각각 생성

```
from sklearn.decomposition import PCA

# PCA 방식을 통해 feature들을 5차원으로 축소
pca = PCA(n_components=5)
pca_train = pca.fit_transform(train_df.iloc[:, :-1])
pca_test = pca.transform(test_df.iloc[:, :-1])

# 차원 축소가 완료된 데이터 세트 생성
pca_columns = [f'pca{i + 1}' for i in range(pca_train.shape[1])]
pca_train_df = pd.DataFrame(pca_train, columns=pca_columns)
pca_train_df.loc[:, 'target'] = train_df.target

pca_columns = [f'pca{i + 1}' for i in range(pca_test.shape[1])]
pca_test_df = pd.DataFrame(pca_test, columns=pca_columns)
pca_test_df.loc[:, 'target'] = test_df.target
```

#### Pytorch를 이용한 로지스틱 회귀

Pytorch의 torch에 포함된 여러 메서드를 사용해서 차원 축소된 데이터 세트로 모델 학습 실시  
학습에 앞서, manual_seed()로 시드값을 고정해서 여러 번 돌리더라도 항상 같은 결과를 유지하게 하고,  
분할했던 데이터 세트들은 FloatTensor()를 통해 Pytorch에서 쓸 수 있는 tensor 타입 데이터로 변환  

```
import torch
from torch.nn import Sequential, Linear, Sigmoid
from torch.optim import SGD
from torch.nn.functional import binary_cross_entropy
from sklearn.model_selection import train_test_split
import numpy as np

# torch의 시드값 고정
torch.manual_seed(124)

# 데이터 세트 분할 후 Tensor 타입으로 변경
# y(정답) 데이터 세트들은 ndarray 타입이기 때문에 view를 사용해서 차원 변환
X_train = torch.FloatTensor(pca_train_df.iloc[:, :-1].values)
y_train = torch.FloatTensor(pca_train_df.target.values).view(-1, 1)
X_test = torch.FloatTensor(pca_test_df.iloc[:, :-1].values)
y_test = torch.FloatTensor(pca_test_df.target.values).view(-1, 1)
```
  
그 다음 Sequential로 Linear → Sigmoid 순서를 거치는 Pytorch 모델 생성,
학습을 200,000번 반복하며 10,000번마다 W(비중)와 b(편향), loss를 출력  

```
# Linear(5차원) → Sigmoid 순서의 층 선언
logistic_r = Sequential(
    Linear(5, 1),
    Sigmoid(),
)

optimizer = SGD(logistic_r.parameters(), lr=0.005)

# 반복 횟수
epochs = 200000

for epoch in range(1, epochs + 1):
    H = logistic_r(X_train)
    loss = binary_cross_entropy(H, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10000 == 0:
        print(f'Epoch: {epoch}/{epochs}')
        for i, w in enumerate(list(logistic_r.parameters())[0][0]):
            print(f'W{i + 1}: {np.round(w.item(), 4)}, ', end='')
        print(f'b: {np.round(list(logistic_r.parameters())[1].item(), 4)}')
        print(f'loss: {np.round(loss.item(), 4)}')
        print('=' * 60)
```

위 코드를 실행했을 때, 약 160,000번째 반복부터는 더 이상 W, b, loss가 변하지 않는 것을 확인  
  
학습이 완료된 모델은 별도의 함수로 분류 모델의 지표인  
정확도, 정밀도, 재현율, F1 Score와, 타겟이 이진 분류였기에 ROC-AUC까지 출력

```
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# 분류 모델의 평가 지표를 출력하고, 분류기와 문제(X_test)를 전달받으면 오차 행렬도 시각화 해주는 함수
def get_evaluation(y_test, prediction, classifier=None, X_test=None):
    confusion = confusion_matrix(y_test, prediction)
    accuracy = accuracy_score(y_test , prediction)
    precision = precision_score(y_test , prediction, average='macro')
    recall = recall_score(y_test , prediction, average='macro')
    f1 = f1_score(y_test, prediction, average='macro')
    auc = roc_auc_score(y_test, prediction, average='macro')
    
    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1: {3:.4f}, ROC-AUC: {4:.4f}'.format(accuracy, precision, recall, f1, auc))
    print("#" * 80)
    
    if classifier is not None and X_test is not None:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
        titles_options = [("Confusion matrix", None), ("Normalized confusion matrix", "true")]

        for (title, normalize), ax in zip(titles_options, axes.flatten()):
            disp = ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, ax=ax, cmap=plt.cm.Blues, normalize=normalize)
            disp.ax_.set_title(title)
        plt.show()
```

위 함수 선언 후 아래 코드로 평가 지표 출력

```
# 모델 평가
# X_test와 y_test가 현재 tensor 타입인 것을 감안해서 인자 할당
get_evaluation(y_test.detach(), logistic_r(X_test) >= 0.5)
```

출력 결과, 모든 평가 지표가 약 0.79 전후를 기록함  

---

#### Sklearn의 LogisticRegression을 이용한 로지스틱 회귀

이번에는 Sklearn의 LogisticRegression 메서드를 사용해서 로지스틱 회귀 실행

```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 데이터 세트 분할
X_train, y_train = pca_train_df.iloc[:, :-1], pca_train_df.target
X_test, y_test = pca_test_df.iloc[:, :-1], pca_test_df.target

# LogisticRegression 모델 객체 생성
logistic_r = LogisticRegression(solver='liblinear', penalty='l2', C=1, random_state=124)
logistic_r.fit(X_train, y_train)

# 훈련 후 테스트 데이터 예측
prediction = logistic_r.predict(X_test)
```

모델 훈련 뒤, 마찬가지로 평가 지표를 출력해 보았으며  
이번에는 오차 행렬 시각화도 병행

---

#### PCA 방식의 차원 축소 후 로지스틱 회귀 결과
- 전반적인 평가 지표가 약 0.78 정도로 측정
- 다만 0(바위)을 예측하는 경향이 강하기 때문에 LDA 차원 축소 결과 보고 임계치 조정

---

### LDA 방식으로 차원 축소
- 현재 target의 클래스는 2가지
- LDA 방식으로 차원 축소 할 경우, 축소 이후의 차원 수는 target 클래스 수 - 1이 최대이므로,  
  현재 데이터 세트는 1차원으로의 축소밖에 불가능함


LDA 방식을 통해 기존 feature들을 1차원으로 축소한 뒤,  
PCA 때와 마찬가지로 축소 이후의 데이터 세트를 따로 생성함

```
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA 방식으로 데이터 세트 차원 축소
lda = LinearDiscriminantAnalysis(n_components=1)

# LDA로 fit할 때는 target 데이터를 같이 넣어준다
lda_train = lda.fit_transform(train_df.iloc[:, :-1], train_df.iloc[:, -1])
lda_test = lda.transform(test_df.iloc[:, :-1])

# 차원 축소된 데이터 세트 생성
lda_columns = [f'lda{i + 1}' for i in range(lda_train.shape[1])]
lda_train_df = pd.DataFrame(lda_train, columns=lda_columns)
lda_train_df.loc[:, 'target'] = train_df['target']

lda_columns = [f'lda{i + 1}' for i in range(lda_test.shape[1])]
lda_test_df = pd.DataFrame(lda_test, columns=lda_columns)
lda_test_df.loc[:, 'target'] = test_df['target']
```

---

#### Pytorch를 이용한 로지스틱 회귀

PCA 때와 동일한 코드로 Pytorch를 통한 로지스틱 회귀 진행

LDA 방식으로 차원 축소했을 떄, 110,000번째 반복 이후로 W, b, loss가 더 이상 변하지 않는 것을 확인

그 후 get_evaluation() 함수로 평가 지표를 출력한 결과, 약 0.76 전후의 평가를 기록함

---

#### Sklearn의 LogisticRegression을 이용한 로지스틱 회귀

마찬가지로 PCA 때와 같은 코드로 로지스틱 회귀 진행  
다만, 이번에는 LogisticRegression의 C(Cost) 파라미터의 값을 0.0001로 기존에 비해 낮춤으로서  
모델에 가해지는 규제를 강화함

```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 데이터 세트 분할
X_train, y_train = lda_train_df.iloc[:, :-1], lda_train_df.target
X_test, y_test = lda_test_df.iloc[:, :-1], lda_test_df.target

# LogisticRegression 모델 객체 생성
logistic_r = LogisticRegression(solver='liblinear', penalty='l2', C=0.0001, random_state=124)
logistic_r.fit(X_train, y_train)

# 훈련 후 테스트 데이터 예측
prediction = logistic_r.predict(X_test)
```

그 후 get_evaluation() 함수로 평가 지표를 출력해본 결과 약 0.78 정도의 평가를 기록함

---

#### LDA 방식의 차원 축소 후 로지스틱 회귀 결과
- 전반적인 평가 지표가 0.78 정도로 PCA와 비슷한 양상을 보임
- 본 모델은 지뢰(1)을 예측하는 것에 초점을 두고 싶기 때문에 재현율을 0.83 이상으로 향상시키는 것이 이상적이라고 판단,  
  따라서 trade-off 시각화하고 임계치 조정해서 다시 학습

---

위 과정 이전에 파이프라인을 구축,  
매번 축소된 데이터 세트를 직접 만들 필요없이 학습까지 바로 진행하도록 했으며  
교차 검증을 통한 최적의 하이퍼 파라미터 탐색도 병행  

---

### LDA 파이프라인 구축
- LogisticRegression 파라미터
> - solver: liblinear, saga
>> l1 또는 l2 Penalty를 적용할 수 있는 알고리즘으로 선정
> - penalty: l1, l2
> - C: 0.5, 0.1, 0.05, 0.01, 0.005, 0.001

- 파이프라인 순서
> - LDA
> - LogisticRegression

- 모든 feature들에 MinMaxScaler가 적용되어 있기 때문에 추가적인 스케일링은 생략

학습에 앞서, 훈련 및 테스트 데이터를 재정의 하기 위해
데이터 분할과 오버 샘플링을 다시 실행

```
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 데이터 세트 분할
features, targets = pre_m_df.iloc[:, :-1], pre_m_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(features, targets, stratify=targets, test_size=0.2, random_state=124)

# 오버 샘플링
smote = SMOTE(random_state=124)

over_X_train, over_y_train = smote.fit_resample(X_train, y_train)
```

그 후 LDA를 통해 1차원으로 축소한 뒤, LogisticRegression 모델을 통한 학습을 순서대로 거치도록 하는 파이프라인 생성  
단, 이 과정에서 이슈가 발생했기 때문에 sklearn.pipeline의 FeatureUnion을 파이프라인 과정에 포함함



---


























