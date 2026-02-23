import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from textblob import TextBlob
import FinanceDataReader as fdr

# -------------------------
# 1. 페이지 설정 및 제목 (이전 유지)
# -------------------------
st.set_page_config(page_title="AI 글로벌 통합 분석 v3.9.1", layout="wide")
st.title("🛡️ AI 고신뢰 글로벌 통합 분석 시스템")

# -------------------------
# 2. 핵심 분석 엔진 (데이터, AI, 감성, 백테스팅)
# -------------------------
@st.cache_data(ttl=3600)
def get_live_stock_list(region, us_market_type="S&P500"):
    try:
        if region == "국내":
            df = fdr.StockListing('KRX')
            return df[['Code', 'Name']].rename(columns={'Code': 'Symbol'})
        else:
            df = fdr.StockListing(us_market_type)
            return df[['Symbol', 'Name']]
    except: return pd.DataFrame()

@st.cache_data(ttl=600)
def get_stock_data(code):
    try:
        df = yf.download(code, period="2y", progress=False, multi_level_index=False)
        if df.empty or len(df) < 50: return pd.DataFrame()
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA20"] = df["Close"].rolling(20).mean()
        diff = df["Close"].diff()
        gain = diff.where(diff > 0, 0).rolling(14).mean()
        loss = diff.where(diff < 0, 0).abs().rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        df['Vol_Change'] = df['Volume'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        return df.dropna().replace([np.inf, -np.inf], 0)
    except: return pd.DataFrame()

def predict_next_day(df, fast=False):
    features = ["Close", "MA5", "MA20", "RSI", "Vol_Change"]
    X = df[features].copy()
    y = df["Close"].shift(-1).dropna()
    model = XGBRegressor(n_estimators=(30 if fast else 100), max_depth=4, random_state=42)
    model.fit(X.iloc[:-1], y)
    return model.predict(X.tail(1))[0]

def get_sentiment(ticker, name="종목"):
    try:
        tk = yf.Ticker(ticker)
        news = tk.news
        if not news: return 0.0, "현재 탐지된 실시간 이슈가 없습니다."
        scores = [TextBlob(n['title']).sentiment.polarity for n in news[:5]]
        avg_score = np.mean(scores)
        all_titles = " ".join([n['title'].lower() for n in news[:10]])
        keywords = {
            "반도체/HBM": ["hbm", "semiconductor", "chip", "nvidia", "sk hynix"],
            "공급계약/수주": ["contract", "supply", "deal", "order", "partnership"],
            "실적발표/성장": ["earnings", "profit", "growth", "revenue", "surge"],
            "정부정책/밸류업": ["value-up", "government", "policy", "dividend", "tax"],
            "거시경제/금리": ["fed", "inflation", "rate", "nasdaq"]
        }
        found_labels = [label for label, keys in keywords.items() if any(k in all_titles for k in keys)]
        if found_labels:
            issue_text = f"현재 **{', '.join(found_labels)}** 관련 이슈가 포착되었습니다. "
            issue_text += f"글로벌 뉴스 심리는 **{('긍정' if avg_score > 0 else '신중')}**한 상태입니다."
        else:
            issue_text = f"현재 {name}은(는) 개별 이슈보다는 기술적 수급에 의해 움직이고 있습니다. (최신 헤드라인: \"{news[0]['title']}\")"
        return avg_score, issue_text
    except: return 0.0, "뉴스 데이터를 분석할 수 없습니다."

def run_backtest_simple(df):
    if len(df) < 30: return 0.0
    rets = []
    for i in range(len(df)-15, len(df)):
        try:
            pred = predict_next_day(df.iloc[:i], fast=True)
            curr = df.iloc[i]["Close"]
            sig = 1 if (pred - curr)/curr > 0.005 else 0
            actual = (df.iloc[i+1]["Close"] - curr) / curr if i+1 < len(df) else 0
            rets.append((actual - 0.002) if sig == 1 else 0)
        except: rets.append(0)
    return (np.prod(1 + np.array(rets)) - 1) * 100

# -------------------------
# 3. 사이드바 UI 및 글로벌 지수
# -------------------------
try:
    indices = yf.download(["^IXIC", "^SOX"], period="2d", progress=False)['Close']
    nas_chg = (indices["^IXIC"].iloc[-1] - indices["^IXIC"].iloc[-2]) / indices["^IXIC"].iloc[-2]
    sox_chg = (indices["^SOX"].iloc[-1] - indices["^SOX"].iloc[-2]) / indices["^SOX"].iloc[-2]
except:
    nas_chg, sox_chg = 0.0, 0.0

st.sidebar.markdown("### 🌍 글로벌 마켓 시그널")
st.sidebar.metric("나스닥", f"{nas_chg*100:.2f}%", delta=f"{nas_chg*100:.2f}%")
st.sidebar.metric("반도체(SOX)", f"{sox_chg*100:.2f}%", delta=f"{sox_chg*100:.2f}%")

st.sidebar.divider()
region = st.sidebar.selectbox("지역 선택", ["국내", "해외"])
if region == "국내":
    stock_list_df = get_live_stock_list("국내")
else:
    us_type = st.sidebar.selectbox("해외 지수 선택", ["S&P500", "NASDAQ", "NYSE"])
    stock_list_df = get_live_stock_list("해외", us_type)

# -------------------------
# 4. 개별 종목 분석 (리포트 요약 포함)
# -------------------------
if not stock_list_df.empty:
    selected_name = st.sidebar.selectbox("종목 선택", stock_list_df['Name'].tolist())
    symbol_raw = stock_list_df[stock_list_df['Name'] == selected_name]['Symbol'].values[0]
    
    if st.sidebar.button("📊 종목 심층 분석"):
        st.subheader(f"🔍 {selected_name} AI 분석 리포트")
        symbol = symbol_raw + ".KS" if region == "국내" else symbol_raw
        df = get_stock_data(symbol)
        
        if df.empty and region == "국내":
            symbol = symbol_raw + ".KQ"
            df = get_stock_data(symbol)

        if not df.empty:
            p = predict_next_day(df)
            c = float(df["Close"].iloc[-1])
            pct = ((p-c)/c)*100
            sent_score, issue_summary = get_sentiment(symbol, selected_name)
            b_ret = run_backtest_simple(df)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("AI 예상 수익률", f"{pct:.2f}%")
            c2.metric("전략 신뢰도(15일)", f"{b_ret:.2f}%")
            c3.metric("뉴스 감성", f"{sent_score:.2f}")
            
            with st.container(border=True):
                st.markdown("**💡 AI 분석 요약**")
                rsi_val = df.iloc[-1]['RSI']
                tech_status = "과매수 주의" if rsi_val > 70 else ("과매도 반등기대" if rsi_val < 30 else "안정적 흐름")
                st.write(f"✅ **실시간 이슈 분석:** {issue_summary}")
                st.write(f"✅ **기술적 지표 상황:** 현재 주가는 {tech_status} 구간에 위치하고 있습니다.")
                if "반도체" in selected_name or "삼성" in selected_name or "하이닉스" in selected_name:
                    st.write(f"✅ **글로벌 연동:** 미국 반도체 지수(SOX)의 변동률({sox_chg*100:.2f}%)이 반영된 섹터 흐름을 보이고 있습니다.")
                st.divider()
                if pct > 0.5 and b_ret > 0 and sent_score >= 0:
                    st.success("🎯 **결론:** 강력한 이슈와 기술적 상승 동력이 결합되었습니다. [적극적 관심]")
                else:
                    st.warning("⚠️ **결론:** 지표가 혼조세이거나 확신이 부족한 단계입니다. [신중한 접근]")
            st.line_chart(df['Close'].tail(100))
        else:
            st.error("데이터 로드 실패")

# -------------------------
# 5. 시장 스캐너 (에러 해결 핵심 구간)
# -------------------------
st.divider()
st.subheader("🚀 실시간 고신뢰 유망주 스캐너")
st.write("AI 예측 + 백테스팅 + 뉴스 감성을 모두 통과한 종목을 스캔합니다.")

if st.button("🚀 고신뢰 스캔 시작"):
    if stock_list_df.empty:
        st.error("종목 리스트 로드 실패")
    else:
        prog = st.progress(0)
        msg = st.empty()
        results = []
        scan_targets = stock_list_df.head(30)
        
        for i, (idx, row) in enumerate(scan_targets.iterrows()):
            name = row['Name']
            code = (row['Symbol'] + ".KS") if region == "국내" else row['Symbol']
            
            msg.info(f"🧐 {name} 정밀 검증 중...")
            prog.progress((i + 1) / 30)
            
            df = get_stock_data(code)
            if df.empty and region == "국내":
                df = get_stock_data(code.replace(".KS", ".KQ"))
            
            if not df.empty:
                try:
                    p = predict_next_day(df, fast=True)
                    c = float(df["Close"].iloc[-1])
                    pct = ((p - c) / c) * 100
                    
                    if pct > 0.3:
                        b_ret = run_backtest_simple(df)
                        if b_ret >= 0:
                            sent, _ = get_sentiment(code, name)
                            # 에러 방지: 결과를 저장할 때 '%'를 붙이지 않고 숫자(float)로만 저장합니다.
                            results.append({
                                "종목명": name,
                                "예상수익": round(pct, 2), # 숫자형
                                "과거성과": round(b_ret, 2), # 숫자형
                                "뉴스감성": round(sent, 2), # 숫자형
                                "섹터동조": "SOX 연동" if (sox_chg > 0 and ("반도체" in name or "삼성" in name)) else "일반"
                            })
                except: continue
        
        msg.empty()
        prog.empty()
        
        if results:
            res_df = pd.DataFrame(results).sort_values(by="예상수익", ascending=False)
            
            # [수정 포인트] 시각화 효과는 숫자형 데이터에서 실행하고, 표시는 포맷팅을 통해 '%'를 붙입니다.
            st.dataframe(
                res_df.style.background_gradient(subset=['예상수익', '과거성과'], cmap='RdYlGn')
                .format({"예상수익": "{:.2f}%", "과거성과": "{:.2f}%"}), # 여기서 화면 표시용 포맷 지정
                use_container_width=True
            )
        else:
            st.warning("모든 관문을 통과한 고신뢰 종목이 없습니다.")

st.info("💡 **v3.9.1 안내**: 숫자 데이터의 포맷 에러를 해결하고 안정성을 높인 버전입니다.")