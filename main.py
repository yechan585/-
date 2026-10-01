import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write(
    "서울의 과거 기온 데이터를 바탕으로 선형 회귀 분석을 진행하여 기온 변화율과 미래 기온을 예측합니다."
)


# 데이터 로드 및 전처리
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")

    # '날짜' 열을 datetime 타입으로 변환 후 '연도' 추출
    df["날짜"] = pd.to_datetime(df["날짜"].str.strip())
    df["연도"] = df["날짜"].dt.year

    # 연도별 관측일수 및 평균기온 계산
    yearly_stats = (
        df.groupby("연도")
        .agg(관측일수=("평균기온", "count"), 연평균기온=("평균기온", "mean"))
        .reset_index()
    )

    # 필터링 조건 적용: 2025년 이하 & 관측일수 300일 이상
    filtered_df = yearly_stats[
        (yearly_stats["연도"] <= 2025) & (yearly_stats["관측일수"] >= 300)
    ].copy()

    # 독립 변수 X: 1908년부터 지난 연수
    filtered_df["X_passed_years"] = filtered_df["연도"] - 1908

    return filtered_df


df_all = load_and_preprocess_data()

# ---------------------------------------------------------
# 1. 전체 기간 회귀 분석
# ---------------------------------------------------------
X_all = df_all["X_passed_years"].values
Y_all = df_all["연평균기온"].values
slope_all, intercept_all = np.polyfit(X_all, Y_all, 1)

corr_all = np.corrcoef(df_all["연도"], df_all["연평균기온"])[0, 1]
rate_100_all = slope_all * 100  # 100년당 온실가스/기온 상승폭 (°C)

start_year_all = int(df_all["연도"].min())
end_year_all = int(df_all["연도"].max())
count_years_all = len(df_all)


# ---------------------------------------------------------
# 2. 최근 20년 회귀 분석
# ---------------------------------------------------------
latest_20_start = end_year_all - 19
df_20 = df_all[df_all["연도"] >= latest_20_start].copy()

X_20 = df_20["X_passed_years"].values
Y_20 = df_20["연평균기온"].values
slope_20, intercept_20 = np.polyfit(X_20, Y_20, 1)

corr_20 = np.corrcoef(df_20["연도"], df_20["연평균기온"])[0, 1]
rate_100_20 = slope_20 * 100  # 최근 20년 기준 100년당 상승폭 (°C)

start_year_20 = int(df_20["연도"].min())
end_year_20 = int(df_20["연도"].max())
count_years_20 = len(df_20)


# ---------------------------------------------------------
# 100년당 기온 상승폭 크게 비교 표시
# ---------------------------------------------------------
st.subheader("🔥 100년당 기온 상승 속도 비교")

col_rate1, col_rate2 = st.columns(2)

with col_rate1:
    st.markdown("### 🌐 전체 기간 기준 (100년당)")
    st.markdown(f"# **+{rate_100_all:.2f} °C**")
    st.caption(
        f"분석 대상: {start_year_all}년 ~ {end_year_all}년 ({count_years_all}개 연도)"
    )
    st.text(f"상관계수 (r): {corr_all:.4f}")

with col_rate2:
    st.markdown("### ⚡ 최근 20년 기준 (100년당)")
    st.markdown(f"# **+{rate_100_20:.2f} °C**")
    st.caption(
        f"분석 대상: {start_year_20}년 ~ {end_year_20}년 ({count_years_20}개 연도)"
    )
    st.text(f"상관계수 (r): {corr_20:.4f}")

st.divider()

# ---------------------------------------------------------
# 예측 슬라이더 및 나란히 비교
# ---------------------------------------------------------
selected_year = st.slider("예측할 연도를 선택하세요", 1900, 2100, 2026)

x_pred = selected_year - 1908
pred_all = slope_all * x_pred + intercept_all
pred_20 = slope_20 * x_pred + intercept_20

st.subheader(f"🔮 {selected_year}년 예상 연평균 기온 비교")

col_p1, col_p2 = st.columns(2)

with col_p1:
    st.markdown("#### 전체 기간 회귀선 예측")
    st.markdown(f"## **{pred_all:.2f} °C**")

with col_p2:
    st.markdown("#### 최근 20년 회귀선 예측")
    st.markdown(f"## **{pred_20:.2f} °C**")

st.divider()

# ---------------------------------------------------------
# Plotly 시각화 (두 회귀선 함께 표시)
# ---------------------------------------------------------
fig = px.scatter(
    df_all,
    x="연도",
    y="연평균기온",
    title="서울 연도별 연평균기온 및 회귀 직선 비교",
    labels={"연도": "연도", "연평균기온": "연평균기온 (°C)"},
    hover_data={"연도": True, "연평균기온": ":.2f"},
)

# 회귀선 x 범위 설정 (1900~2100년)
years_range = np.arange(1900, 2101)
x_range_passed = years_range - 1908

# 1. 전체 기간 회귀선 (빨간색)
trend_y_all = slope_all * x_range_passed + intercept_all
fig.add_trace(
    go.Scatter(
        x=years_range,
        y=trend_y_all,
        mode="lines",
        name=f"전체 기간 회귀선 (100년당 +{rate_100_all:.2f}°C)",
        line=dict(color="red", width=2),
    )
)

# 2. 최근 20년 회귀선 (주황색 점선)
trend_y_20 = slope_20 * x_range_passed + intercept_20
fig.add_trace(
    go.Scatter(
        x=years_range,
        y=trend_y_20,
        mode="lines",
        name=f"최근 20년 회귀선 (100년당 +{rate_100_20:.2f}°C)",
        line=dict(color="orange", width=2, dash="dash"),
    )
)

# 3. 선택 연도 예측점 (전체 기간)
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[pred_all],
        mode="markers",
        name=f"전체 기준 예측({selected_year}년)",
        marker=dict(color="red", size=12, symbol="star"),
    )
)

# 4. 선택 연도 예측점 (최근 20년)
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[pred_20],
        mode="markers",
        name=f"최근 20년 기준 예측({selected_year}년)",
        marker=dict(color="orange", size=12, symbol="diamond"),
    )
)

# 레이아웃 설정
fig.update_layout(
    xaxis=dict(title="연도", tickmode="linear", tick0=1900, dtick=20),
    yaxis=dict(title="연평균기온 (°C)"),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)

st.plotly_chart(fig, use_container_width=True)
