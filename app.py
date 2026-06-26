# app.py - Cloud Compatible Version
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import warnings
import time
import os
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')
os.environ['YFINANCE_NO_CACHE'] = '1'

# ── Page Config ──
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0a0e1a;
    --bg2: #0f1629;
    --card: #141b2d;
    --blue: #00d4ff;
    --purple: #7c3aed;
    --green: #00ff88;
    --red: #ff4757;
    --gold: #ffd700;
    --text: #e8eaf6;
    --muted: #8892b0;
    --border: rgba(0,212,255,0.15);
}

.stApp { background: var(--bg); font-family: 'Inter', sans-serif; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem; max-width: 1600px; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(#00d4ff, #7c3aed);
    border-radius: 3px;
}

.hero {
    background: linear-gradient(135deg, #0a0e1a, #0f1629, #0a0e1a);
    border-bottom: 1px solid var(--border);
    padding: 2rem 2rem 1.5rem;
    margin: -1rem -2rem 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse at 20% 50%, rgba(0,212,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 50%, rgba(124,58,237,0.07) 0%, transparent 60%);
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #00d4ff, #7c3aed, #00ff88);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0; position: relative; z-index: 1;
}
.hero-sub {
    color: var(--muted); font-size: 1rem;
    margin: 0.4rem 0 0; position: relative; z-index: 1;
}
.badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 50px; padding: 3px 12px;
    font-size: 0.75rem; font-weight: 600;
    color: #00d4ff; letter-spacing: 1px;
    text-transform: uppercase; margin-bottom: 0.8rem;
    position: relative; z-index: 1;
}
.dot {
    width: 7px; height: 7px;
    background: #00ff88; border-radius: 50%;
    animation: blink 1.5s ease infinite;
    display: inline-block;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px; padding: 1.4rem 1.6rem;
    position: relative; overflow: hidden;
    transition: all 0.3s ease;
}
.card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4ff, #7c3aed);
}
.card:hover { border-color: rgba(0,212,255,0.35); transform: translateY(-2px); }
.card.g::before { background: linear-gradient(90deg, #00ff88, #00cc6a); }
.card.r::before { background: linear-gradient(90deg, #ff4757, #cc2233); }
.card.gold::before { background: linear-gradient(90deg, #ffd700, #ff9500); }
.card.p::before { background: linear-gradient(90deg, #7c3aed, #a855f7); }
.clabel { font-size:0.7rem; font-weight:700; color:var(--muted); text-transform:uppercase; letter-spacing:1.2px; margin-bottom:0.25rem; }
.cval { font-family:'Space Grotesk',sans-serif; font-size:1.75rem; font-weight:700; color:var(--text); line-height:1; }
.cdelta { font-size:0.78rem; font-weight:600; margin-top:0.25rem; }
.pos { color: #00ff88; } .neg { color: #ff4757; }

.sec {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px; padding: 1.6rem;
    margin-bottom: 1.2rem;
}
.sec-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem; font-weight: 700; color: var(--text);
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: 8px;
}
.ico {
    width: 28px; height: 28px;
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(124,58,237,0.2));
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 7px; display: flex;
    align-items: center; justify-content: center;
    font-size: 0.85rem;
}

.pred-box {
    background: linear-gradient(135deg,
        rgba(0,212,255,0.08),
        rgba(124,58,237,0.08),
        rgba(0,255,136,0.08));
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 18px; padding: 2rem;
    text-align: center; margin: 1rem 0;
}
.pred-price {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(135deg, #00d4ff, #00ff88);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sig-buy  { background:rgba(0,255,136,0.12); border:2px solid rgba(0,255,136,0.4); color:#00ff88; padding:6px 22px; border-radius:50px; font-weight:700; display:inline-block; margin-top:0.5rem; }
.sig-sell { background:rgba(255,71,87,0.12);  border:2px solid rgba(255,71,87,0.4);  color:#ff4757; padding:6px 22px; border-radius:50px; font-weight:700; display:inline-block; margin-top:0.5rem; }
.sig-hold { background:rgba(255,215,0,0.12); border:2px solid rgba(255,215,0,0.4); color:#ffd700; padding:6px 22px; border-radius:50px; font-weight:700; display:inline-block; margin-top:0.5rem; }

[data-testid="stSidebar"] { background: #0f1629 !important; border-right: 1px solid var(--border) !important; }
.stButton>button {
    background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important; width: 100% !important;
    padding: 0.65rem 1.5rem !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.2) !important;
    transition: all 0.3s !important;
}
.stButton>button:hover { transform:translateY(-2px) !important; box-shadow:0 8px 30px rgba(0,212,255,0.35) !important; }

.stTabs [data-baseweb="tab-list"] {
    background: var(--card) !important;
    border-radius: 10px !important; padding: 3px !important;
    border: 1px solid var(--border) !important; gap: 3px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: var(--muted) !important;
    border-radius: 7px !important; font-size: 0.83rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,rgba(0,212,255,0.18),rgba(124,58,237,0.18)) !important;
    color: #00d4ff !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
}
.stProgress>div>div>div>div {
    background: linear-gradient(90deg, #00d4ff, #7c3aed, #00ff88) !important;
}
.stSelectbox>div>div, .stTextInput>div>div>input {
    background: rgba(20,27,45,0.9) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 9px !important; color: #e8eaf6 !important;
}
div[data-testid="stSidebar"] .stMarkdown p { color: #8892b0; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# DATA FETCHING  — robust multi-try
# ══════════════════════════════════════════
@st.cache_data(ttl=600, show_spinner=False)
def fetch_stock_data(ticker: str, period: str):
    """Fetch stock data with curl_cffi to bypass Yahoo Finance blocks."""
    ticker = ticker.strip().upper()

    try:
        from curl_cffi import requests as cffi_requests
        session = cffi_requests.Session(impersonate="chrome")
    except ImportError:
        session = None

    df = None

    # Method 1: yfinance with browser-impersonating session
    try:
        if session:
            stock = yf.Ticker(ticker, session=session)
        else:
            stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval="1d", auto_adjust=True)
        if df is not None and len(df) > 5:
            pass
        else:
            df = None
    except Exception as e:
        df = None

    # Method 2: Fallback to download
    if df is None or df.empty:
        try:
            df = yf.download(
                ticker,
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
                session=session
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        except Exception:
            df = None

    # Method 3: Last-resort with shorter period
    if df is None or df.empty:
        try:
            short_map = {"5y":"2y","2y":"1y","1y":"6mo","6mo":"3mo"}
            sp = short_map.get(period, "3mo")
            stock = yf.Ticker(ticker, session=session) if session else yf.Ticker(ticker)
            df = stock.history(period=sp, interval="1d", auto_adjust=True)
        except Exception:
            df = None

    if df is None or df.empty:
        return None, {}

    # Clean column names
    df.columns = [str(c).strip().capitalize() for c in df.columns]
    if "Adj close" in df.columns and "Close" not in df.columns:
        df["Close"] = df["Adj close"]

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = df.get("Close", pd.Series(dtype=float))

    df.dropna(subset=["Close"], inplace=True)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Build info
    info = {
        'name': ticker,
        'sector': 'N/A',
        'market_cap': 0,
        'pe_ratio': 0,
        'beta': 0,
        'week_52_high': float(df['High'].max()),
        'week_52_low': float(df['Low'].min()),
    }

    try:
        stock = yf.Ticker(ticker, session=session) if session else yf.Ticker(ticker)
        try:
            fi = stock.fast_info
            mc = getattr(fi, 'market_cap', 0)
            if mc: info['market_cap'] = mc
        except Exception:
            pass

        try:
            raw = stock.info
            if raw:
                info['name']     = raw.get('longName', ticker) or ticker
                info['sector']   = raw.get('sector', 'N/A') or 'N/A'
                info['pe_ratio'] = raw.get('trailingPE', 0) or 0
                info['beta']     = raw.get('beta', 0) or 0
                if not info['market_cap']:
                    info['market_cap'] = raw.get('marketCap', 0) or 0
        except Exception:
            pass
    except Exception:
        pass

    return df, info

# ══════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════
def add_indicators(df):
    df = df.copy()
    c = df['Close']
    df['MA7']  = c.rolling(7).mean()
    df['MA20'] = c.rolling(20).mean()
    df['MA50'] = c.rolling(50).mean()
    df['BB_mid']   = c.rolling(20).mean()
    df['BB_std']   = c.rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2*df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2*df['BB_std']
    delta = c.diff()
    gain  = delta.where(delta>0,0).rolling(14).mean()
    loss  = (-delta.where(delta<0,0)).rolling(14).mean()
    df['RSI'] = 100 - 100/(1+gain/(loss+1e-9))
    e1 = c.ewm(span=12,adjust=False).mean()
    e2 = c.ewm(span=26,adjust=False).mean()
    df['MACD']        = e1-e2
    df['MACD_signal'] = df['MACD'].ewm(span=9,adjust=False).mean()
    df['MACD_hist']   = df['MACD']-df['MACD_signal']
    df['Volume_MA']   = df['Volume'].rolling(20).mean()
    df['Return']      = c.pct_change()
    df['Volatility']  = df['Return'].rolling(20).std()
    return df

# ══════════════════════════════════════════
# ML FEATURES
# ══════════════════════════════════════════
def make_features(df, lookback=30):
    df  = df.copy().dropna()
    rows, targets = [], []
    for i in range(lookback, len(df)):
        row = list(df['Close'].iloc[i-lookback:i].values)
        row += [
            df['MA7'].iloc[i-1]       if not pd.isna(df['MA7'].iloc[i-1])       else df['Close'].iloc[i-1],
            df['MA20'].iloc[i-1]      if not pd.isna(df['MA20'].iloc[i-1])      else df['Close'].iloc[i-1],
            df['MA50'].iloc[i-1]      if not pd.isna(df['MA50'].iloc[i-1])      else df['Close'].iloc[i-1],
            df['RSI'].iloc[i-1]       if not pd.isna(df['RSI'].iloc[i-1])       else 50,
            df['MACD'].iloc[i-1]      if not pd.isna(df['MACD'].iloc[i-1])      else 0,
            df['BB_upper'].iloc[i-1]  if not pd.isna(df['BB_upper'].iloc[i-1])  else df['Close'].iloc[i-1],
            df['BB_lower'].iloc[i-1]  if not pd.isna(df['BB_lower'].iloc[i-1]) else df['Close'].iloc[i-1],
            float(df['Volume'].iloc[i-1]),
            df['Volatility'].iloc[i-1] if not pd.isna(df['Volatility'].iloc[i-1]) else 0,
        ]
        rows.append(row)
        targets.append(float(df['Close'].iloc[i]))
    X   = np.array(rows,  dtype=np.float32)
    y   = np.array(targets, dtype=np.float32)
    idx = df.index[lookback:]
    return X, y, idx

def fmt_cap(v):
    if v>=1e12: return f"${v/1e12:.2f}T"
    if v>=1e9:  return f"${v/1e9:.2f}B"
    if v>=1e6:  return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"

LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,14,26,0.85)',
    font=dict(family='Inter', color='#8892b0', size=11),
    xaxis=dict(gridcolor='rgba(0,212,255,0.05)', linecolor='rgba(0,212,255,0.12)', tickfont=dict(color='#8892b0')),
    yaxis=dict(gridcolor='rgba(0,212,255,0.05)', linecolor='rgba(0,212,255,0.12)', tickfont=dict(color='#8892b0')),
    legend=dict(bgcolor='rgba(14,21,39,0.9)', bordercolor='rgba(0,212,255,0.18)', borderwidth=1, font=dict(color='#e8eaf6', size=11)),
    hovermode='x unified',
    hoverlabel=dict(bgcolor='rgba(14,21,39,0.96)', bordercolor='rgba(0,212,255,0.25)', font=dict(family='Inter',color='#e8eaf6')),
    margin=dict(l=8, r=8, t=35, b=8)
)

# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 1.4rem;border-bottom:1px solid rgba(0,212,255,0.12);margin-bottom:1.2rem;">
        <div style="font-size:2.6rem;margin-bottom:0.4rem;">📈</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.15rem;font-weight:700;
                    background:linear-gradient(135deg,#00d4ff,#7c3aed);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
            AI Stock Predictor
        </div>
        <div style="font-size:0.7rem;color:#374151;margin-top:3px;">Powered by Machine Learning</div>
    </div>
    """, unsafe_allow_html=True)

    def sb_sec(title, content_fn):
        st.markdown(f'<div style="background:rgba(0,212,255,0.03);border:1px solid rgba(0,212,255,0.1);border-radius:10px;padding:0.9rem;margin-bottom:0.8rem;"><div style="font-size:0.7rem;font-weight:700;color:#00d4ff;text-transform:uppercase;letter-spacing:1.4px;margin-bottom:0.7rem;">{title}</div>', unsafe_allow_html=True)
        content_fn()
        st.markdown('</div>', unsafe_allow_html=True)

    def stock_section():
        global ticker
        stocks = {
            "Apple (AAPL)":"AAPL","Microsoft (MSFT)":"MSFT",
            "Google (GOOGL)":"GOOGL","Amazon (AMZN)":"AMZN",
            "Tesla (TSLA)":"TSLA","NVIDIA (NVDA)":"NVDA",
            "Meta (META)":"META","Netflix (NFLX)":"NFLX",
            "Custom Ticker":"CUSTOM"
        }
        choice = st.selectbox("Stock", list(stocks.keys()), label_visibility="collapsed")
        if choice == "Custom Ticker":
            ticker = st.text_input("Ticker Symbol", value="SPY", label_visibility="collapsed")
        else:
            ticker = stocks[choice]

    def time_section():
        global period, forecast_days, lookback
        pm = {"6 Months":"6mo","1 Year":"1y","2 Years":"2y","5 Years":"5y"}
        pl = st.selectbox("Period", list(pm.keys()), index=1, label_visibility="collapsed")
        period        = pm[pl]
        forecast_days = st.slider("Forecast Days", 1, 60, 30)
        lookback      = st.slider("Lookback Window", 10, 60, 30)

    def model_section():
        global model_type
        model_type = st.selectbox("Algorithm", [
            "Gradient Boosting","Random Forest","Ridge Regression","Ensemble (All 3)"
        ], label_visibility="collapsed")

    ticker = "AAPL"; period = "1y"; forecast_days = 30; lookback = 30; model_type = "Gradient Boosting"

    sb_sec("📊 Stock Selection", stock_section)
    sb_sec("⏱️ Time Settings",   time_section)
    sb_sec("🤖 Model",           model_section)

    run_btn = st.button("🚀 Run AI Prediction")
    st.markdown('<div style="text-align:center;margin-top:1rem;color:#374151;font-size:0.7rem;">⚠️ Educational only · Not financial advice</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════
st.markdown(f"""
<div class="hero">
    <div class="badge"><span class="dot"></span> AI-Powered Analysis</div>
    <h1 class="hero-title">Stock Price Trend Predictor</h1>
    <p class="hero-sub">Machine Learning · Real-Time Forecasting · Interactive Analytics &nbsp;|&nbsp;
        Analyzing: <strong style="color:#00d4ff;">{ticker}</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# FETCH DATA
# ══════════════════════════════════════════
if 'results' not in st.session_state:
    st.session_state.results = None

fetch_placeholder = st.empty()
with fetch_placeholder:
    with st.spinner(f"📡 Fetching {ticker} data from Yahoo Finance..."):
        df_raw, info = fetch_stock_data(ticker, period)

fetch_placeholder.empty()

if df_raw is None or df_raw.empty:
    st.markdown(f"""
    <div style="background:rgba(255,71,87,0.08);border:1px solid rgba(255,71,87,0.3);
                border-radius:14px;padding:2rem;text-align:center;margin:2rem 0;">
        <div style="font-size:2.5rem;margin-bottom:0.8rem;">😕</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.3rem;
                    font-weight:700;color:#ff4757;margin-bottom:0.5rem;">
            Could Not Fetch Data for "{ticker}"
        </div>
        <div style="color:#8892b0;font-size:0.9rem;max-width:500px;margin:0 auto;">
            Yahoo Finance may be temporarily unavailable or rate-limiting requests.
            <br><br>
            <strong style="color:#e8eaf6;">Try these fixes:</strong><br>
            1. Wait 30 seconds and refresh the page<br>
            2. Try a different stock (e.g., MSFT, GOOGL)<br>
            3. Try a different time period<br>
            4. The market may be closed — try again later
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_retry1, col_retry2, col_retry3 = st.columns(3)
    with col_retry1:
        if st.button("🔄 Retry AAPL"):
            st.cache_data.clear()
            st.rerun()
    with col_retry2:
        if st.button("🔄 Try MSFT"):
            st.cache_data.clear()
            st.rerun()
    with col_retry3:
        if st.button("🗑️ Clear Cache & Retry"):
            st.cache_data.clear()
            st.rerun()
    st.stop()

# ── Process data ──
df = add_indicators(df_raw.copy())
cur  = float(df['Close'].iloc[-1])
prev = float(df['Close'].iloc[-2]) if len(df) > 1 else cur
chg  = cur - prev
pct  = chg / prev * 100 if prev != 0 else 0

# ══════════════════════════════════════════
# METRIC CARDS
# ══════════════════════════════════════════
def mcard(icon, label, val, delta=None, sign=None, cls=""):
    d = ""
    if delta:
        c = "pos" if sign=="+" else "neg"
        a = "▲" if sign=="+" else "▼"
        d = f'<div class="cdelta {c}">{a} {delta}</div>'
    return f"""
    <div class="card {cls}">
        <div style="font-size:1.5rem;margin-bottom:3px">{icon}</div>
        <div class="clabel">{label}</div>
        <div class="cval">{val}</div>
        {d}
    </div>"""

c1,c2,c3,c4,c5,c6 = st.columns(6)
with c1: st.markdown(mcard("💰","Price",f"${cur:.2f}",f"{abs(pct):.2f}%","+" if chg>=0 else "-"),unsafe_allow_html=True)
with c2: st.markdown(mcard("📊","Change",f"{'+'if chg>=0 else''}{chg:.2f}",f"${abs(chg):.2f}","+" if chg>=0 else "-","g" if chg>=0 else "r"),unsafe_allow_html=True)
with c3:
    vol=df['Volume'].iloc[-1]
    st.markdown(mcard("📦","Volume",f"{vol/1e6:.1f}M" if vol>=1e6 else f"{vol/1e3:.0f}K",cls="p"),unsafe_allow_html=True)
with c4:
    rsi=df['RSI'].iloc[-1]
    rc="g" if rsi<30 else("r" if rsi>70 else "gold")
    st.markdown(mcard("⚡","RSI (14)",f"{rsi:.1f}" if not pd.isna(rsi) else "N/A",cls=rc),unsafe_allow_html=True)
with c5:
    h52=info.get('week_52_high',0)
    st.markdown(mcard("🎯","52W High",f"${h52:.2f}" if h52 else "N/A",cls="gold"),unsafe_allow_html=True)
with c6:
    mc=info.get('market_cap',0)
    st.markdown(mcard("🏦","Mkt Cap",fmt_cap(mc) if mc else "N/A"),unsafe_allow_html=True)

st.markdown("<div style='height:1.2rem'></div>",unsafe_allow_html=True)

# ══════════════════════════════════════════
# TABS
# ══════════════════════════════════════════
t1,t2,t3,t4,t5 = st.tabs(["📈 Price Chart","🔬 Technical","🤖 AI Prediction","📊 Performance","ℹ️ Info"])

# ── TAB 1 ──────────────────────────────────
with t1:
    ca, cb = st.columns([3,1])
    with ca:
        st.markdown('<div class="sec"><div class="sec-title"><div class="ico">📈</div>Price History</div>', unsafe_allow_html=True)
        ctype = st.radio("", ["Candlestick","Line","OHLC"], horizontal=True, label_visibility="collapsed")
        dp = df.dropna(subset=['Close'])

        fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.75,0.25],vertical_spacing=0.03)

        if ctype=="Candlestick":
            fig.add_trace(go.Candlestick(x=dp.index,open=dp['Open'],high=dp['High'],low=dp['Low'],close=dp['Close'],
                name="OHLC",increasing_line_color='#00ff88',decreasing_line_color='#ff4757',
                increasing_fillcolor='rgba(0,255,136,0.25)',decreasing_fillcolor='rgba(255,71,87,0.25)'),row=1,col=1)
        elif ctype=="Line":
            fig.add_trace(go.Scatter(x=dp.index,y=dp['Close'],name="Close",
                line=dict(color='#00d4ff',width=2),fill='tozeroy',fillcolor='rgba(0,212,255,0.04)'),row=1,col=1)
        else:
            fig.add_trace(go.Ohlc(x=dp.index,open=dp['Open'],high=dp['High'],low=dp['Low'],close=dp['Close'],
                name="OHLC",increasing_line_color='#00ff88',decreasing_line_color='#ff4757'),row=1,col=1)

        if not dp['MA20'].isna().all():
            fig.add_trace(go.Scatter(x=dp.index,y=dp['MA20'],name="MA20",line=dict(color='#ffd700',width=1.5,dash='dot')),row=1,col=1)
        if not dp['MA50'].isna().all():
            fig.add_trace(go.Scatter(x=dp.index,y=dp['MA50'],name="MA50",line=dict(color='#a78bfa',width=1.5,dash='dot')),row=1,col=1)
        fig.add_trace(go.Scatter(x=dp.index,y=dp['BB_upper'],name="BB+",line=dict(color='rgba(0,212,255,0.4)',width=1,dash='dash')),row=1,col=1)
        fig.add_trace(go.Scatter(x=dp.index,y=dp['BB_lower'],name="BB-",line=dict(color='rgba(0,212,255,0.4)',width=1,dash='dash'),
            fill='tonexty',fillcolor='rgba(0,212,255,0.03)'),row=1,col=1)

        vc=['#00ff88' if dp['Close'].iloc[i]>=dp['Open'].iloc[i] else '#ff4757' for i in range(len(dp))]
        fig.add_trace(go.Bar(x=dp.index,y=dp['Volume'],name="Vol",marker_color=vc,opacity=0.55),row=2,col=1)

        fig.update_layout(**LAYOUT,height=520)
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
        st.markdown('</div>',unsafe_allow_html=True)

    with cb:
        st.markdown('<div class="sec"><div class="sec-title"><div class="ico">📋</div>Stats</div>', unsafe_allow_html=True)
        def row(l,v):
            return f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(0,212,255,0.07);"><span style="color:#8892b0;font-size:0.78rem;">{l}</span><span style="color:#e8eaf6;font-weight:600;font-size:0.82rem;">{v}</span></div>'

        for k,v in {
            "Open":f"${df['Open'].iloc[-1]:.2f}","High":f"${df['High'].iloc[-1]:.2f}",
            "Low":f"${df['Low'].iloc[-1]:.2f}","Close":f"${df['Close'].iloc[-1]:.2f}",
            "MA7":f"${df['MA7'].iloc[-1]:.2f}" if not pd.isna(df['MA7'].iloc[-1]) else "N/A",
            "MA20":f"${df['MA20'].iloc[-1]:.2f}" if not pd.isna(df['MA20'].iloc[-1]) else "N/A",
            "MACD":f"{df['MACD'].iloc[-1]:.3f}" if not pd.isna(df['MACD'].iloc[-1]) else "N/A",
        }.items():
            st.markdown(row(k,v),unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

        st.markdown('<div class="sec"><div class="sec-title"><div class="ico">📉</div>Returns</div>', unsafe_allow_html=True)
        for lbl,days in [("1W",5),("1M",21),("3M",63),("6M",126),("1Y",252)]:
            if len(df)>days:
                r=(df['Close'].iloc[-1]/df['Close'].iloc[-days]-1)*100
                co="#00ff88" if r>=0 else "#ff4757"
                ar="▲" if r>=0 else "▼"
                st.markdown(f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(0,212,255,0.07);"><span style="color:#8892b0;font-size:0.78rem;">{lbl}</span><span style="color:{co};font-weight:600;font-size:0.82rem;">{ar} {abs(r):.2f}%</span></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

# ── TAB 2 ──────────────────────────────────
with t2:
    st.markdown('<div class="sec"><div class="sec-title"><div class="ico">🔬</div>Technical Indicators</div>', unsafe_allow_html=True)
    dta = df.dropna().copy()
    fig2 = make_subplots(rows=4,cols=1,shared_xaxes=True,row_heights=[0.40,0.20,0.20,0.20],
                         vertical_spacing=0.04,subplot_titles=("Price & BB","RSI (14)","MACD","Volume"))
    fig2.add_trace(go.Scatter(x=dta.index,y=dta['Close'],name="Close",line=dict(color='#00d4ff',width=2)),row=1,col=1)
    fig2.add_trace(go.Scatter(x=dta.index,y=dta['MA20'],name="MA20",line=dict(color='#ffd700',width=1.5,dash='dot')),row=1,col=1)
    fig2.add_trace(go.Scatter(x=dta.index,y=dta['BB_upper'],name="BB+",line=dict(color='rgba(0,212,255,0.45)',width=1,dash='dash')),row=1,col=1)
    fig2.add_trace(go.Scatter(x=dta.index,y=dta['BB_lower'],name="BB-",line=dict(color='rgba(0,212,255,0.45)',width=1,dash='dash'),
        fill='tonexty',fillcolor='rgba(0,212,255,0.04)'),row=1,col=1)
    fig2.add_trace(go.Scatter(x=dta.index,y=dta['RSI'],name="RSI",line=dict(color='#a78bfa',width=2)),row=2,col=1)
    fig2.add_hline(y=70,line_dash="dash",line_color="rgba(255,71,87,0.4)",row=2,col=1)
    fig2.add_hline(y=30,line_dash="dash",line_color="rgba(0,255,136,0.4)",row=2,col=1)
    mch=['rgba(0,255,136,0.65)' if v>=0 else 'rgba(255,71,87,0.65)' for v in dta['MACD_hist']]
    fig2.add_trace(go.Bar(x=dta.index,y=dta['MACD_hist'],name="Hist",marker_color=mch),row=3,col=1)
    fig2.add_trace(go.Scatter(x=dta.index,y=dta['MACD'],name="MACD",line=dict(color='#00d4ff',width=1.5)),row=3,col=1)
    fig2.add_trace(go.Scatter(x=dta.index,y=dta['MACD_signal'],name="Signal",line=dict(color='#ff6b35',width=1.5)),row=3,col=1)
    vc2=['rgba(0,255,136,0.55)' if dta['Close'].iloc[i]>=dta['Open'].iloc[i] else 'rgba(255,71,87,0.55)' for i in range(len(dta))]
    fig2.add_trace(go.Bar(x=dta.index,y=dta['Volume'],name="Vol",marker_color=vc2),row=4,col=1)
    fig2.update_layout(**LAYOUT,height=730,showlegend=False)
    for ann in fig2.layout.annotations: ann.font.update(color='#8892b0',size=11)
    st.plotly_chart(fig2,use_container_width=True,config={'displayModeBar':False})
    st.markdown('</div>',unsafe_allow_html=True)

# ── TAB 3 ──────────────────────────────────
with t3:
    if not run_btn and st.session_state.results is None:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;">
            <div style="font-size:3.5rem;margin-bottom:0.8rem;">🤖</div>
            <h2 style="font-family:'Space Grotesk',sans-serif;font-size:1.6rem;font-weight:700;
                       background:linear-gradient(135deg,#00d4ff,#7c3aed);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
                Ready to Predict
            </h2>
            <p style="color:#8892b0;font-size:0.95rem;max-width:450px;margin:0.5rem auto 0;">
                Configure settings in the sidebar then click
                <strong style="color:#00d4ff;">🚀 Run AI Prediction</strong>
            </p>
        </div>""", unsafe_allow_html=True)

    if run_btn:
        pb   = st.progress(0)
        stxt = st.empty()

        stxt.markdown('<p style="color:#00d4ff;font-weight:600;">⚙️ Building features...</p>',unsafe_allow_html=True)
        X, y, idx = make_features(df, lookback)
        if len(X) < 50:
            st.error("Not enough data. Use a longer period or smaller lookback.")
            st.stop()
        pb.progress(20)

        stxt.markdown('<p style="color:#00d4ff;font-weight:600;">📐 Scaling data...</p>',unsafe_allow_html=True)
        sx = MinMaxScaler(); sy = MinMaxScaler()
        Xs = sx.fit_transform(X)
        ys = sy.fit_transform(y.reshape(-1,1)).ravel()
        split = int(len(Xs)*0.85)
        Xtr,Xte = Xs[:split],Xs[split:]
        ytr,yte = ys[:split],ys[split:]
        ite = idx[split:]
        pb.progress(35)

        stxt.markdown(f'<p style="color:#00d4ff;font-weight:600;">🤖 Training {model_type}...</p>',unsafe_allow_html=True)
        if model_type=="Gradient Boosting":
            mdl=GradientBoostingRegressor(n_estimators=200,learning_rate=0.05,max_depth=4,random_state=42)
            mdl.fit(Xtr,ytr); mdls={'GB':mdl}
        elif model_type=="Random Forest":
            mdl=RandomForestRegressor(n_estimators=200,max_depth=10,random_state=42,n_jobs=-1)
            mdl.fit(Xtr,ytr); mdls={'RF':mdl}
        elif model_type=="Ridge Regression":
            mdl=Ridge(alpha=1.0); mdl.fit(Xtr,ytr); mdls={'Ridge':mdl}
        else:
            m1=GradientBoostingRegressor(n_estimators=150,learning_rate=0.05,max_depth=4,random_state=42)
            m2=RandomForestRegressor(n_estimators=150,max_depth=8,random_state=42,n_jobs=-1)
            m3=Ridge(alpha=1.0)
            m1.fit(Xtr,ytr); m2.fit(Xtr,ytr); m3.fit(Xtr,ytr)
            mdls={'GB':m1,'RF':m2,'Ridge':m3}
        pb.progress(70)

        stxt.markdown('<p style="color:#00d4ff;font-weight:600;">📊 Evaluating...</p>',unsafe_allow_html=True)
        ps = np.mean([m.predict(Xte) for m in mdls.values()],axis=0)
        ya = sy.inverse_transform(yte.reshape(-1,1)).ravel()
        yp = sy.inverse_transform(ps.reshape(-1,1)).ravel()
        mae  = mean_absolute_error(ya,yp)
        rmse = np.sqrt(mean_squared_error(ya,yp))
        mape = np.mean(np.abs((ya-yp)/(ya+1e-9)))*100
        r2   = r2_score(ya,yp)
        pb.progress(85)

        stxt.markdown('<p style="color:#00d4ff;font-weight:600;">🔮 Forecasting...</p>',unsafe_allow_html=True)
        last = X[-1].copy(); fps=[]
        for _ in range(forecast_days):
            inp_s = sx.transform(last.reshape(1,-1))
            p_s   = np.mean([m.predict(inp_s)[0] for m in mdls.values()])
            pp    = sy.inverse_transform([[p_s]])[0][0]
            fps.append(pp)
            nr=last.copy(); nr[:lookback-1]=last[1:lookback]; nr[lookback-1]=pp; last=nr

        fdates = pd.bdate_range(start=df.index[-1]+timedelta(days=1),periods=forecast_days)
        pb.progress(100)
        stxt.markdown('<p style="color:#00ff88;font-weight:700;">✅ Complete!</p>',unsafe_allow_html=True)

        st.session_state.results={
            'ya':ya,'yp':yp,'fps':np.array(fps),
            'fdates':fdates,'mae':mae,'rmse':rmse,'mape':mape,'r2':r2,'ite':ite
        }

    if st.session_state.results is not None:
        r=st.session_state.results
        fp=r['fps']; tgt=float(fp[-1])
        pp=(tgt-cur)/cur*100
        sig = "BUY" if pp>2 else ("SELL" if pp<-2 else "HOLD")
        sc  = "sig-buy" if pp>2 else ("sig-sell" if pp<-2 else "sig-hold")

        st.markdown(f"""
        <div class="pred-box">
            <div style="font-size:0.8rem;color:#8892b0;text-transform:uppercase;letter-spacing:2px;margin-bottom:0.4rem;">
                AI Price Target ({forecast_days} Days)
            </div>
            <div class="pred-price">${tgt:.2f}</div>
            <div style="color:#8892b0;font-size:0.88rem;margin-top:0.3rem;">
                Current <strong style="color:#00d4ff;">${cur:.2f}</strong> →
                <strong style="color:{'#00ff88' if pp>=0 else '#ff4757'}">
                    {'▲' if pp>=0 else '▼'} {abs(pp):.2f}%
                </strong>
            </div>
            <div><span class="{sc}">{sig}</span></div>
            <div style="margin-top:0.8rem;background:rgba(0,255,136,0.1);border:1px solid rgba(0,255,136,0.25);
                        border-radius:50px;padding:3px 14px;font-size:0.8rem;font-weight:600;color:#00ff88;display:inline-block;">
                R² = {r['r2']:.4f}
            </div>
        </div>""", unsafe_allow_html=True)

        mc1,mc2,mc3,mc4=st.columns(4)
        with mc1: st.markdown(mcard("📏","MAE",f"${r['mae']:.2f}"),unsafe_allow_html=True)
        with mc2: st.markdown(mcard("📐","RMSE",f"${r['rmse']:.2f}",cls="p"),unsafe_allow_html=True)
        with mc3: st.markdown(mcard("📊","MAPE",f"{r['mape']:.2f}%",cls="gold"),unsafe_allow_html=True)
        with mc4: st.markdown(mcard("🎯","R²",f"{r['r2']:.4f}",cls="g"),unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem'></div>",unsafe_allow_html=True)

        st.markdown('<div class="sec"><div class="sec-title"><div class="ico">🔮</div>Forecast Chart</div>',unsafe_allow_html=True)
        hp = df['Close'].iloc[-90:] if len(df)>=90 else df['Close']
        lo = fp*(1-r['mape']/100); hi=fp*(1+r['mape']/100)

        ff=go.Figure()
        ff.add_trace(go.Scatter(x=hp.index,y=hp.values,name="Historical",line=dict(color='#00d4ff',width=2),fill='tozeroy',fillcolor='rgba(0,212,255,0.04)'))
        n=min(len(r['ite']),len(r['yp']))
        ff.add_trace(go.Scatter(x=r['ite'][:n],y=r['yp'][:n],name="Model Fit",line=dict(color='#ffd700',width=2,dash='dot')))
        ff.add_trace(go.Scatter(x=list(fdates)+list(fdates)[::-1],y=list(hi)+list(lo)[::-1],
            fill='toself',fillcolor='rgba(0,255,136,0.07)',line=dict(color='rgba(0,0,0,0)'),name="Confidence",hoverinfo='skip'))
        ff.add_trace(go.Scatter(x=fdates,y=fp,name=f"Forecast",line=dict(color='#00ff88',width=2.5),
            mode='lines+markers',marker=dict(size=4,color='#00ff88')))
        ff.add_hline(y=cur,line_dash="dash",line_color="rgba(255,215,0,0.45)",
                     annotation_text=f"Now: ${cur:.2f}",annotation_font_color="#ffd700")
        ff.update_layout(**LAYOUT,height=420)
        st.plotly_chart(ff,use_container_width=True,config={'displayModeBar':False})
        st.markdown('</div>',unsafe_allow_html=True)

        st.markdown('<div class="sec"><div class="sec-title"><div class="ico">📋</div>Forecast Table</div>',unsafe_allow_html=True)
        fdf=pd.DataFrame({
            'Date':pd.DatetimeIndex(fdates).strftime('%Y-%m-%d'),
            'Predicted':[ f"${p:.2f}" for p in fp],
            'Change':   [f"{'+'if p>=cur else''}{(p-cur)/cur*100:.2f}%" for p in fp],
            'Low Est':  [f"${p*(1-r['mape']/100):.2f}" for p in fp],
            'High Est': [f"${p*(1+r['mape']/100):.2f}" for p in fp],
        })
        st.dataframe(fdf,use_container_width=True,hide_index=True)
        st.markdown('</div>',unsafe_allow_html=True)

# ── TAB 4 ──────────────────────────────────
with t4:
    if st.session_state.results is None:
        st.info("Run the AI Prediction first.")
    else:
        r=st.session_state.results; n=min(len(r['ya']),len(r['yp']))
        p1,p2=st.columns(2)
        with p1:
            st.markdown('<div class="sec"><div class="sec-title"><div class="ico">🎯</div>Actual vs Predicted</div>',unsafe_allow_html=True)
            fa=go.Figure()
            fa.add_trace(go.Scatter(x=r['ite'][:n],y=r['ya'][:n],name="Actual",line=dict(color='#00d4ff',width=2)))
            fa.add_trace(go.Scatter(x=r['ite'][:n],y=r['yp'][:n],name="Predicted",line=dict(color='#ff6b35',width=2,dash='dot')))
            fa.update_layout(**LAYOUT,height=300)
            st.plotly_chart(fa,use_container_width=True,config={'displayModeBar':False})
            st.markdown('</div>',unsafe_allow_html=True)
        with p2:
            st.markdown('<div class="sec"><div class="sec-title"><div class="ico">📊</div>Error Distribution</div>',unsafe_allow_html=True)
            err=r['yp'][:n]-r['ya'][:n]
            fe=go.Figure()
            fe.add_trace(go.Histogram(x=err,nbinsx=35,marker_color='rgba(0,212,255,0.6)',marker_line_color='rgba(0,212,255,0.2)',marker_line_width=0.5))
            fe.add_vline(x=0,line_dash="dash",line_color="#00ff88")
            fe.update_layout(**LAYOUT,height=300,xaxis_title="Error ($)",yaxis_title="Count")
            st.plotly_chart(fe,use_container_width=True,config={'displayModeBar':False})
            st.markdown('</div>',unsafe_allow_html=True)

        st.markdown('<div class="sec"><div class="sec-title"><div class="ico">📈</div>Regression Plot</div>',unsafe_allow_html=True)
        fr=go.Figure()
        fr.add_trace(go.Scatter(x=r['ya'][:n],y=r['yp'][:n],mode='markers',marker=dict(color='#7c3aed',size=5,opacity=0.65)))
        mv=min(r['ya'][:n].min(),r['yp'][:n].min()); xv=max(r['ya'][:n].max(),r['yp'][:n].max())
        fr.add_trace(go.Scatter(x=[mv,xv],y=[mv,xv],mode='lines',name="Perfect",line=dict(color='#00ff88',width=2,dash='dash')))
        fr.update_layout(**LAYOUT,height=320,xaxis_title="Actual",yaxis_title="Predicted")
        st.plotly_chart(fr,use_container_width=True,config={'displayModeBar':False})
        st.markdown('</div>',unsafe_allow_html=True)

# ── TAB 5 ──────────────────────────────────
with t5:
    nm=info.get('name',ticker)
    st.markdown(f"""
    <div class="sec">
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:1.2rem;">
            <div style="width:48px;height:48px;background:linear-gradient(135deg,rgba(0,212,255,0.18),rgba(124,58,237,0.18));
                        border:1px solid rgba(0,212,255,0.25);border-radius:12px;
                        display:flex;align-items:center;justify-content:center;font-size:1.5rem;">📊</div>
            <div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:1.3rem;font-weight:700;color:#e8eaf6;">{nm}</div>
                <div style="color:#8892b0;font-size:0.85rem;">{ticker} · {info.get('sector','N/A')}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    ia,ib,ic=st.columns(3)
    def ir(l,v):
        return f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(0,212,255,0.07);"><span style="color:#8892b0;font-size:0.8rem;">{l}</span><span style="color:#e8eaf6;font-weight:600;font-size:0.83rem;">{v}</span></div>'

    with ia:
        st.markdown("**📈 Valuation**")
        st.markdown(ir("Market Cap", fmt_cap(info.get('market_cap',0)) if info.get('market_cap') else "N/A"),unsafe_allow_html=True)
        st.markdown(ir("P/E Ratio",  f"{info.get('pe_ratio',0):.2f}" if info.get('pe_ratio') else "N/A"),unsafe_allow_html=True)
        st.markdown(ir("Beta",       f"{info.get('beta',0):.2f}" if info.get('beta') else "N/A"),unsafe_allow_html=True)
    with ib:
        st.markdown("**🎯 Price Range**")
        st.markdown(ir("52W High",  f"${info.get('week_52_high',0):.2f}" if info.get('week_52_high') else "N/A"),unsafe_allow_html=True)
        st.markdown(ir("52W Low",   f"${info.get('week_52_low',0):.2f}"  if info.get('week_52_low')  else "N/A"),unsafe_allow_html=True)
        st.markdown(ir("Current",   f"${cur:.2f}"),unsafe_allow_html=True)
    with ic:
        st.markdown("**📊 Statistics**")
        st.markdown(ir("Data Points",   f"{len(df):,}"),unsafe_allow_html=True)
        st.markdown(ir("Period Return", f"{(df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100:.2f}%"),unsafe_allow_html=True)
        st.markdown(ir("Avg Volume",    f"{df['Volume'].mean()/1e6:.2f}M"),unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="sec"><div class="sec-title"><div class="ico">🔥</div>Correlation Matrix</div>',unsafe_allow_html=True)
    cc=['Close','Volume','MA20','MA50','RSI','MACD','BB_upper','BB_lower']
    cm=df[cc].dropna().corr()
    fc=go.Figure(go.Heatmap(z=cm.values,x=cm.columns.tolist(),y=cm.columns.tolist(),
        colorscale=[[0,'#ff4757'],[0.5,'#141b2d'],[1,'#00d4ff']],zmid=0,
        text=np.round(cm.values,2),texttemplate="%{text}",textfont=dict(size=10)))
    fc.update_layout(**LAYOUT,height=360)
    st.plotly_chart(fc,use_container_width=True,config={'displayModeBar':False})
    st.markdown('</div>',unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("""
<div style="border-top:1px solid rgba(0,212,255,0.08);padding:1.2rem 0;margin-top:1rem;
            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
    <span style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1rem;
                 background:linear-gradient(135deg,#00d4ff,#7c3aed);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        📈 AI Stock Predictor
    </span>
    <span style="color:#374151;font-size:0.72rem;">
        ⚠️ Educational only · Not financial advice · Past performance ≠ future results
    </span>
</div>
""", unsafe_allow_html=True)