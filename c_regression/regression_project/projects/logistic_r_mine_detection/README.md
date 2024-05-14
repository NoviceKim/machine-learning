### 데이터 세트: 잠수함 레이더로 기뢰 탐지
- 총 데이터(행) 수는 207개
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

#### PCA 방식으로 차원 축소

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




---



