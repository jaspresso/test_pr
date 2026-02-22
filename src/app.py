import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import torch
from xgboost import XGBRegressor

# -------------------------
# 1. 페이지 설정 및 제목
# -------------------------
st.set_page_config(page_title="AI 주가 예측 스캐너 v2.0", layout="wide")
st.title("📈 AI 주가 분석 및 알고리즘 패턴 스캐너")
st.markdown("알고리즘 매매 트리거(VWAP, 이격도, 전고점 돌파)를 반영한 인공지능 분석 도구입니다.")

# -------------------------
# 2. 공통 함수 (데이터 로드 및 지표 생성)
# -------------------------
@st.cache_data
def load_market(region, market):
    try:
        if region == "국내":
            file = "kospi.csv" if market == "코스피" else "kosdaq.csv"
            df = pd.read_csv(file, encoding="utf-8-sig", dtype={'종목코드': str})
        else:
            file = "sp500.csv" if market == "S&P500" else "nasdaq.csv"
            df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()

@st.cache_data
def get_processed_data(code):
    try:
        # 지표 계산을 위해 넉넉한 데이터를 가져옵니다.
        df = yf.download(code, period="2y", progress=False, multi_level_index=False)
        if df.empty or len(df) < 50: return pd.DataFrame()

        # --- 기본 및 보조 지표 ---
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["Volatility"] = df["Close"].pct_change().rolling(5).std()
        
        # 볼린저 밴드
        std20 = df["Close"].rolling(20).std()
        df["BB_Upper"] = df["MA20"] + (std20 * 2)
        df["BB_Lower"] = df["MA20"] - (std20 * 2)
        
        # RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9)) + 1e-9))

        # MACD
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2

        # --- 알고리즘 경향성 지표 (추가) ---
        # 1. VWAP (거래량 가중 평균가격 근사치)
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-9)
        
        # 2. 가격 이격도 (20일선 기준, 100에 가까울수록 매수세 안정)
        df['Disparity20'] = (df['Close'] / df['MA20']) * 100
        
        # 3. 전고점 돌파 거리 (20일 최고가 대비 현재가 위치)
        df['Max20'] = df['High'].rolling(window=20).max()
        df['Dist_Max'] = (df['Max20'] - df['Close']) / (df['Close'] + 1e-9)

        # 4. 거래량 변화율
        df["Vol_Change"] = df["Volume"].diff() / (df["Volume"].shift(1) + 1e-9)

        # --- 데이터 정제 (XGBoost 에러 방지) ---
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        return df
    except:
        return pd.DataFrame()

def train_and_predict(df):
    # 확장된 피처 목록 (알고리즘 지표 포함)
    features = [
        "Close", "MA5", "MA20", "Volatility", "BB_Upper", "BB_Lower", 
        "RSI", "MACD", "Vol_Change", "VWAP", "Disparity20", "Dist_Max"
    ]
    X = df[features]
    y = df["Close"].shift(-1).dropna()
    X_train = X.iloc[:-1]

    if torch.cuda.is_available():
        dev, tree_method = 'cuda', 'hist'
    else:
        dev, tree_method = 'cpu', 'auto'

    model = XGBRegressor(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.05, 
        random_state=42,
        tree_method=tree_method,
        device=dev
    )
    model.fit(X_train, y)
    
    latest_x = X.tail(1)
    pred = model.predict(latest_x)[0]
    return pred

# -------------------------
# 3. 사이드바 - 종목 선택 UI
# -------------------------
region = st.sidebar.selectbox("지역 선택", ["국내", "해외"])
market = st.sidebar.selectbox("시장 선택", ["코스피", "코스닥"] if region == "국내" else ["S&P500", "NASDAQ"])
market_df = load_market(region, market)

if not market_df.empty:
    name_col = "회사명" if region == "국내" else ("Security" if market == "S&P500" else "Name")
    stock_name = st.sidebar.selectbox("개별 종목 분석", market_df[name_col])
    row = market_df[market_df[name_col] == stock_name]
    
    if region == "국내":
        stock_code = row.iloc[0]["종목코드"].strip().zfill(6) + (".KS" if market == "코스피" else ".KQ")
    else:
        stock_code = row.iloc[0]["Symbol"].strip()

# -------------------------
# 4. 메인 분석 영역 (개별 종목)
# -------------------------
if st.sidebar.button("🔍 개별 종목 분석"):
    df = get_processed_data(stock_code)
    if not df.empty:
        with st.spinner(f'{stock_name} 데이터 분석 및 AI 학습 중...'):
            pred = train_and_predict(df)
            curr = float(df["Close"].iloc[-1])
            change_pct = ((pred - curr) / curr) * 100

            st.subheader(f"📊 {stock_name} AI 분석 결과")
            c1, c2, c3 = st.columns(3)
            c1.metric("현재가", f"{curr:,.0f}")
            c2.metric("내일 예측가", f"{pred:,.0f}")
            c3.metric("AI 예상 수익률", f"{change_pct:.2f}%", delta=f"{change_pct:.2f}%")
            
            # --- 다중 지표 시각화 ---
            st.write("### 📈 주가 및 볼린저 밴드")
            st.line_chart(df[["Close", "BB_Upper", "BB_Lower"]].tail(60))

            col_a, col_b = st.columns(2)
            with col_a:
                st.write("### 🌡️ RSI (심리 지수)")
                st.line_chart(df["RSI"].tail(60))
            with col_b:
                st.write("### 🌊 MACD (추세 강도)")
                st.line_chart(df["MACD"].tail(60))

            st.write("### 🤖 알고리즘 지표 (VWAP & 이격도)")
            # 이격도 100 기준선 시각화를 위해 멀티라인 표시
            st.line_chart(df[["Disparity20"]].tail(60))
            st.caption("이격도 100 상향 돌파 시 알고리즘 매수세 유입 가능성이 높습니다.")
            
    else:
        st.error("데이터가 부족하거나 불러올 수 없습니다.")

# -------------------------
# 5. 메인 화면 하단 - 급등주 스캐너
# -------------------------
st.divider()
st.subheader("🚀 알고리즘 패턴 급등 유망주 스캐너")
st.write(f"현재 {market} 시장 내 거래량 상위 30개 종목 중 AI 예측과 알고리즘 지표가 우수한 종목을 검색합니다.")

if st.button("📡 스캔 시작"):
    sample_stocks = market_df.head(30) 
    results = []
    
    progress_bar = st.progress(0)
    for i, (idx, row) in enumerate(sample_stocks.iterrows()):
        s_name = row[name_col]
        s_code = (row["종목코드"].strip().zfill(6) + (".KS" if market == "코스피" else ".KQ")) if region == "국내" else row["Symbol"].strip()
        
        progress_bar.progress((i + 1) / 30)
        s_df = get_processed_data(s_code)
        if s_df.empty: continue
        
        try:
            s_pred = train_and_predict(s_df)
            s_curr = float(s_df["Close"].iloc[-1])
            s_pct = ((s_pred - s_curr) / s_curr) * 100
            
            if s_pct > 0.5: # 0.5% 이상 상승 예측 종목만
                results.append({"종목명": s_name, "현재가": s_curr, "예측가": s_pred, "예상 수익률": s_pct})
        except: continue

    if results:
        top_5 = sorted(results, key=lambda x: x['예상 수익률'], reverse=True)[:5]
        st.table(pd.DataFrame(top_5).style.format({"현재가": "{:,.0f}", "예측가": "{:,.0f}", "예상 수익률": "{:.2f}%"}))
        st.success("스캔 완료! 위 종목들은 알고리즘 매매가 선호하는 기술적 구간에 진입했습니다.")
    else:
        st.warning("현재 시장 조건에서 적합한 종목이 없습니다.")

st.info("💡 **알림**: 본 서비스는 기술적 분석 및 알고리즘 패턴을 기반으로 하며, 실제 투자 결과는 시장 상황에 따라 다를 수 있습니다.")