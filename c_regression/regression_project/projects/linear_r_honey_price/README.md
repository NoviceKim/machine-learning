### 주제: 꿀 가격 회귀 예측

---

#### Features (총 10종)
- CS: 색깔 (Color Score)
> - 점수가 낮을수록 밝은 색, 높을 수록 어두운 색을 띄는 꿀

- Density: 밀도
- WC: 수분 함유량 (Water Content)
- pH
- EC: 전기 전도도 (Electrical Conductivity)
- F: 과당 함유량 (Fructose level)
- G: 포도당 함유량 (Glucose level)
- Pollen_analysis: 밀원식물 종류
> - Clover, Wildflower, Orange Blossom 등을 포함한 총 19종의 고유 클래스

- Viscosity: 점도 (단위는 centipoise)
> - 특이사항: 2500 ~ 9500 사이의 값은 아래의 Purity에 최적인 것으로 간주됨

- Purity: 순도

#### Targets
- Price: 가격
