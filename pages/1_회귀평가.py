import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. 예시 데이터 생성 (1906년 ~ 2025년, 총 120년 데이터)
# * 실제 데이터를 사용할 경우 pd.read_csv() 등을 활용하세요.
# ---------------------------------------------------------
np.random.seed(42)
years = np.arange(1906, 2026)

# 지구 온난화 경향 반영: 최근으로 갈수록 기온 상승 폭이 커지는 가상 데이터
base_temp = 12.0
trend = (years - 1906) * 0.015 + ((years - 1906) / 100) ** 2 * 0.5
noise = np.random.normal(0, 0.4, len(years))
temperatures = base_temp + trend + noise

df = pd.DataFrame({'Year': years, 'Temp': temperatures})

# ---------------------------------------------------------
# 2. 데이터셋 분할
# ---------------------------------------------------------
# 공통 테스트 데이터 (최근 20년: 2006 ~ 2025)
test_df = df[(df['Year'] >= 2006) & (df['Year'] <= 2025)]
X_test = test_df[['Year']]
y_test = test_df['Temp']

# 학습 데이터 1: 전체 데이터 (1906 ~ 2025)
X_all = df[['Year']]
y_all = df['Temp']

# 학습 데이터 2: 최근 50년 (1956 ~ 2005)
train_50_df = df[(df['Year'] >= 1956) & (df['Year'] <= 2005)]
X_train_50 = train_50_df[['Year']]
y_train_50 = train_50_df['Temp']

# 학습 데이터 3: 최근 100년 (1906 ~ 2005)
train_100_df = df[(df['Year'] >= 1906) & (df['Year'] <= 2005)]
X_train_100 = train_100_df[['Year']]
y_train_100 = train_100_df['Temp']

# ---------------------------------------------------------
# 3. 모델 학습 및 예측 함수
# ---------------------------------------------------------
def train_and_evaluate(X_tr, y_tr, X_te, y_te, model_name):
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    
    # 예측 (공통 테스트 데이터 대상)
    y_pred = model.predict(X_te)
    
    # 평가지표 계산
    mae = mean_absolute_error(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    return {
        'Model': model_name,
        'Slope (기울기)': slope,
        'Intercept (절편)': intercept,
        'MAE': mae,
        'MSE': mse,
        'R²': r2
    }

# ---------------------------------------------------------
# 4. 모델 실행 및 결과 수집
# ---------------------------------------------------------
results = []

# (1) 전체 데이터 평가 (전체 데이터로 학습 후 테스트 데이터 평가)
results.append(train_and_evaluate(X_all, y_all, X_test, y_test, "전체 데이터 (1906~2025)"))

# (2) 최근 50년 학습 모델 평가
results.append(train_and_evaluate(X_train_50, y_train_50, X_test, y_test, "최근 50년 학습 (1956~2005)"))

# (3) 최근 100년 학습 모델 평가
results.append(train_and_evaluate(X_train_100, y_train_100, X_test, y_test, "최근 100년 학습 (1906~2005)"))

# 결과 출력
results_df = pd.DataFrame(results)
print("=== 모델별 회귀선 기울기 및 공통 테스트 데이터(2006~2025) 예측 성능 비교 ===")
print(results_df.to_string(index=False))
