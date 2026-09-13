
import streamlit as st
import pandas as pd
from scanner import Settings, load_universe, run_scan

st.set_page_config(
    page_title="Swing Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1100px;}
h1 {font-size: 1.65rem !important; margin-bottom: .25rem !important;}
div[data-testid="stMetricValue"] {font-size: 1.35rem;}
.stButton > button {width:100%; height:3rem; font-size:1.05rem; font-weight:700;}
@media (max-width: 700px) {
  .block-container {padding-left: .8rem; padding-right: .8rem;}
  h1 {font-size: 1.45rem !important;}
}
</style>
""", unsafe_allow_html=True)

st.title("📈 미국주식 스윙 스캐너")
st.caption("RSI + MACD + 상승 다이버전스로 매수 후보를 자동 선별합니다.")

with st.expander("검색 조건", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        universe = st.selectbox(
            "종목군",
            ["S&P500 + Nasdaq100", "S&P500", "Nasdaq100"],
            index=0
        )
        rsi_max = st.slider("RSI 최대", 20, 50, 35)
    with c2:
        min_score = st.slider("최소 점수", 0, 100, 50)
        min_dvol_m = st.slider("최소 20일 평균 거래대금 ($M)", 1, 100, 5)

    only_new = st.toggle("최근 5거래일 내 MACD 골든크로스 우선", value=False)

map_universe = {
    "S&P500 + Nasdaq100": "both",
    "S&P500": "sp500",
    "Nasdaq100": "nasdaq100",
}

run = st.button("🔎 지금 스캔", type="primary")

st.markdown("""
**점수**
- RSI 과매도 25
- 상승 다이버전스 30
- MACD 골든크로스/임박 25
- 거래량 증가 10
- 20일선 회복 10
""")

if run:
    settings = Settings(
        rsi_max=float(rsi_max),
        min_score=float(min_score),
        min_avg_dollar_vol=float(min_dvol_m) * 1_000_000,
    )

    with st.spinner("종목 목록 불러오는 중..."):
        tickers = load_universe(map_universe[universe])

    with st.spinner(f"{len(tickers)}개 종목 분석 중..."):
        result = run_scan(tickers, settings)

    if only_new and not result.empty:
        result = result[result["MACD_Cross"].eq("YES")].copy()

    if result.empty:
        st.warning("현재 조건에 맞는 종목이 없습니다. RSI 최대값이나 최소 점수를 조금 완화해보세요.")
    else:
        best = result.iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("후보 수", len(result))
        m2.metric("TOP", best["Ticker"])
        m3.metric("TOP 점수", f'{best["Score"]:.0f}')

        st.subheader("오늘의 후보")
        show_cols = [
            "Ticker","Score","Price","RSI14","BullDiv",
            "MACD_Cross","MACD_Near","Vol_vs_20D","Above_20DMA"
        ]
        st.dataframe(
            result[show_cols],
            use_container_width=True,
            hide_index=True,
            height=min(700, 58 + 35 * min(len(result), 18)),
        )

        st.download_button(
            "CSV 저장",
            result.to_csv(index=False).encode("utf-8-sig"),
            file_name="swing_candidates.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.divider()
st.caption("후보 선별 도구입니다. 다이버전스·MACD 신호는 반등을 보장하지 않습니다.")
