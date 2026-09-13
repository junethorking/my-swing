
import streamlit as st
from scanner import Settings, load_universe, run_scan

st.set_page_config(
    page_title="US Swing Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

st.title("📈 미국주식 스윙 스캐너")
st.caption("RSI + MACD + 상승 다이버전스로 미국 상장주 매수 후보를 자동 선별합니다.")

with st.expander("검색 조건", expanded=True):
    universe_label = st.selectbox(
        "종목군",
        [
            "미국주식 전체 (NYSE + Nasdaq + American)",
            "S&P500 + Nasdaq100",
            "S&P500",
            "Nasdaq100",
        ],
        index=0,
    )

    c1, c2 = st.columns(2)

    with c1:
        rsi_max = st.slider("RSI 최대", 20, 50, 35)
        min_price = st.slider("최소 주가 ($)", 1, 30, 3)

    with c2:
        min_score = st.slider("최소 점수", 0, 100, 50)
        min_dvol_m = st.slider("최소 20일 평균 거래대금 ($M)", 1, 100, 5)

    only_new = st.toggle("최근 5거래일 내 MACD 골든크로스만 보기", value=False)

universe_map = {
    "미국주식 전체 (NYSE + Nasdaq + American)": "all_us",
    "S&P500 + Nasdaq100": "both",
    "S&P500": "sp500",
    "Nasdaq100": "nasdaq100",
}

run = st.button("🔎 지금 스캔", type="primary")

st.markdown(
    """
**기본 점수**
- RSI 과매도: 최대 25점
- 상승 다이버전스: 최대 30점
- MACD 골든크로스/임박: 최대 25점
- 거래량 증가: 최대 10점
- 20일선 회복: 10점

**미국주식 전체**는 ETF·테스트종목·워런트·권리·우선주 등을 최대한 제외하고,
주가와 거래대금 필터를 통과한 종목을 분석합니다.
"""
)

if run:
    settings = Settings(
        rsi_max=float(rsi_max),
        min_score=float(min_score),
        min_avg_dollar_vol=float(min_dvol_m) * 1_000_000,
        min_price=float(min_price),
    )

    with st.spinner("미국 상장 종목 목록 불러오는 중..."):
        tickers = load_universe(universe_map[universe_label])

    st.info(f"분석 대상: {len(tickers):,}개 종목")

    with st.spinner(
        "가격 데이터 분석 중... 미국주식 전체는 시간이 꽤 걸릴 수 있습니다."
    ):
        result = run_scan(tickers, settings)

    if only_new and not result.empty:
        result = result[result["MACD_Cross"].eq("YES")].copy()

    if result.empty:
        st.warning(
            "현재 조건에 맞는 종목이 없습니다. RSI 최대값이나 최소 점수를 조금 완화해보세요."
        )
    else:
        best = result.iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("후보 수", len(result))
        m2.metric("TOP", best["Ticker"])
        m3.metric("TOP 점수", f'{best["Score"]:.0f}')

        st.subheader("오늘의 후보")
        show_cols = [
            "Ticker",
            "Score",
            "Price",
            "RSI14",
            "BullDiv",
            "MACD_Cross",
            "MACD_Near",
            "Vol_vs_20D",
            "Above_20DMA",
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
st.caption(
    "후보 선별용 도구입니다. 다이버전스·MACD 신호는 반등을 보장하지 않습니다."
)
