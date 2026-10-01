import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write("서울의 과거 기온 데이터를 바탕으로 선형 회귀 분석을 통해 미래 기온을 예측합니다.")


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


df = load_and_preprocess_data()

# 선형 회귀 계산 (1908년 대비 경과 연수 기준)
X = df["X_passed_years"].values
Y = df["연평균기온"].values

# 1차 다항식(선형 회귀) 피팅: Y = slope * X + intercept
slope, intercept = np.polyfit(X, Y, 1)

# 상관계수 계산
correlation = np.corrcoef(df["연도"], df["연평균기온"])[0, 1]

# 정보 요약 표시
start_year = int(df["연도"].min())
end_year = int(df["연도"].max())
count_years = len(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("회귀선 분석 연도 개수", f"{count_years} 개")
col2.metric("시작 연도", f"{start_year} 년")
col3.metric("끝 연도", f"{end_year} 년")
col4.metric("상관계수 (r)", f"{correlation:.4f}")

st.divider()

# 예측 슬라이더 및 큰 숫자 표시
selected_year = st.slider("예측할 연도를 선택하세요", 1900, 2100, 2026)

# 선택된 연도의 경과 연수 계산 후 예측
x_pred = selected_year - 1908
predicted_temp = slope * x_pred + intercept

st.subheader(f"🔮 {selected_year}년 예상 연평균 기온")
st.markdown(f"# **{predicted_temp:.2f} °C**")

st.divider()

# Plotly 시각화
# 1. 관측값 산점도
fig = px.scatter(
    df,
    x="연도",
    y="연평균기온",
    title="서울 연도별 연평균기온 및 회귀 직선",
    labels={"연도": "연도", "연평균기온": "연평균기온 (°C)"},
    hover_data={"연도": True, "연평균기온": ":.2f"},
)

# 2. 회귀선 추세선 추가 (1900년부터 2100년까지 확장선 생성)
years_range = np.arange(1900, 2101)
x_range_passed = years_range - 1908
trend_y = slope * x_range_passed + intercept

fig.add_trace(
    go.Scatter(
        x=years_range,
        y=trend_y,
        mode="lines",
        name="회귀 직선",
        line=dict(color="red", width=2),
    )
)

# 3. 선택한 연도 예측점 표시
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"선택한 연도({selected_year}년)",
        marker=dict(color="green", size=14, symbol="star"),
    )
)

# 그래프 레이아웃 설정 (가로축 연도 설정)
fig.update_layout(
    xaxis=dict(title="연도", tickmode="linear", tick0=1900, dtick=20),
    yaxis=dict(title="연평균기온 (°C)"),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)

st.plotly_chart(fig, use_container_width=True)
