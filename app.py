# app.py - Streamlit Cloud Compatible Version
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import warnings
import time
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #0f1629;
        --bg-card: #141b2d;
        --accent-blue: #00d4ff;
        --accent-purple: #7c3aed;
        --accent-green: #00ff88;
        --accent-red: #ff4757;
        --accent-gold: #ffd700;
        --text-primary: #e8eaf6;
        --text-secondary: #8892b0;
        --border: rgba(0, 212, 255, 0.15);
    }

    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', sans-serif;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        padding: 0rem 2rem 2rem 2rem;
        max-width: 1600px;
    }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00d4ff, #7c3aed);
        border-radius: 3px;
    }

    .hero-header {
        background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #0a0e1a 100%);
        border-bottom: 1px solid var(--border);
        padding: 2rem 2rem 1.5rem;
        margin: -1rem -2rem 2rem -2rem;
        position: relative;
        overflow: hidden;
    }

    .hero-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(ellipse at 20% 50%, rgba(0,212,255,0.08) 0%, transparent 60%),
            radial-gradient(ellipse at 80% 50%, rgba(124,58,237,0.08) 0%, transparent 60%);
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.1;
    }

    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0,212,255,0.1);
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 50px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #00d4ff;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .live-dot {
        width: 8px; height: 8px;
        background: #00ff88;
        border-radius: 50%;
        animation: pulse 1.5s ease-in-out infinite;
        display: inline-block;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    .metric-card {
        background: linear-gradient(135deg, #141b2d 0%, rgba(20,27,45,0.8) 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
    }

    .metric-card:hover {
        border-color: rgba(0,212,255,0.4);
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(0,212,255,0.2);
    }

    .metric-card.green::before { background: linear-gradient(90deg, #00ff88, #00cc6a); }
    .metric-card.red::before { background: linear-gradient(90deg, #ff4757, #cc1a2a); }
    .metric-card.gold::before { background: linear-gradient(90deg, #ffd700, #ff9500); }
    .metric-card.purple::before { background: linear-gradient(90deg, #7c3aed, #a855f7); }

    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
    }

    .metric-delta {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    .metric-delta.positive { color: #00ff88; }
    .metric-delta.negative { color: #ff4757; }

    .section-card {
        background: #141b2d;
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .section-title-icon {
        width: 30px; height: 30px;
        background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(124,58,237,0.2));
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.7rem 2rem !important;
        width: 100% !important;
        box-shadow: 0 4px 20px rgba(0,212,255,0.2) !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0,212,255,0.4) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: #141b2d !important;
        border-radius: 12px !important;
        padding: 4px !important;
        border: 1px solid var(--border) !important;
        gap: 4px !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(124,58,237,0.2)) !important;
        color: #00d4ff !important;
        border: 1px solid rgba(0,212,255,0.3) !important;
    }

    .prediction-banner {
        background: linear-gradient(135deg,
            rgba(0,212,255,0.1),
            rgba(124,58,237,0.1),
            rgba(0,255,136,0.1));
        border: 1px solid rgba(0,212,255,0.25);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .prediction-price {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .signal-buy {
        background: rgba(0,255,136,0.15);
        border: 2px solid rgba(0,255,136,0.5);
        color: #00ff88;
        padding: 8px 24px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
        margin-top: 0.5rem;
    }

    .signal-sell {
        background: rgba(255,71,87,0.15);
        border: 2px solid rgba(255,71,87,0.5);
        color: #ff4757;
        padding: 8px 24px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
        margin-top: 0.5rem;
    }

    .signal-hold {
        background: rgba(255,215,0,0.15);
        border: 2px solid rgba(255,215,0,0.5);
        color: #ffd700;
        padding: 8px 24px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
        margin-top: 0.5rem;
    }

    [data-testid="stSidebar"] {
        background: #0f1629 !important;
        border-right: 1px solid var(--border) !important;
    }

    .sidebar-logo {
        text-align: center;
        padding: 1rem 0 1.5rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.5rem;
    }

    .sidebar-section {
        background: rgba(0,212,255,0.04);
        border: 1px solid rgba(0,212,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .sidebar-section-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: #00d4ff;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.8rem;
    }

    .stSelectbox > div > div,
    .stTextInput > div > div > input {
        background: rgba(20,27,45,0.8) !important;
        border: 1px solid rgba(0,212,255,0.2) !important;
        border-radius: 10px !important;
        color: #e8eaf6 !important;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #7c3aed, #00ff88) !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================

@st.cache_data(ttl=300)
def fetch_stock_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        df    = stock.history(period=period, interval="1d")
        try:
            raw  = stock.info
            info = {
                'name':         raw.get('longName', ticker),
                'sector':       raw.get('sector', 'N/A'),
                'market_cap':   raw.get('marketCap', 0),
                'pe_ratio':     raw.get('trailingPE', 0),
                'week_52_high': raw.get('fiftyTwoWeekHigh', 0),
                'week_52_low':  raw.get('fiftyTwoWeekLow', 0),
                'beta':         raw.get('beta', 0),
            }
        except:
            info = {'name': ticker, 'sector': 'N/A', 'market_cap': 0,
                    'pe_ratio': 0, 'week_52_high': 0, 'week_52_low': 0, 'beta': 0}
        return df, info
    except Exception as e:
        return None, {}

def add_indicators(df):
    df = df.copy()
    df['MA7']  = df['Close'].rolling(7).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['BB_mid']   = df['Close'].rolling(20).mean()
    df['BB_std']   = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2*df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2*df['BB_std']
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/(loss+1e-10)))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']
    df['Volume_MA']   = df['Volume'].rolling(20).mean()
    df['Daily_Return']= df['Close'].pct_change()
    df['Volatility']  = df['Daily_Return'].rolling(20).std()
    return df

def make_features(df, lookback=30):
    """Create ML features from price data."""
    df = df.copy().dropna()
    rows = []
    targets = []
    for i in range(lookback, len(df)):
        row = []
        # Price lags
        for lag in range(1, lookback+1):
            row.append(df['Close'].iloc[i-lag])
        # Technical indicators
        row += [
            df['MA7'].iloc[i-1]   if not pd.isna(df['MA7'].iloc[i-1])   else df['Close'].iloc[i-1],
            df['MA20'].iloc[i-1]  if not pd.isna(df['MA20'].iloc[i-1])  else df['Close'].iloc[i-1],
            df['MA50'].iloc[i-1]  if not pd.isna(df['MA50'].iloc[i-1])  else df['Close'].iloc[i-1],
            df['RSI'].iloc[i-1]   if not pd.isna(df['RSI'].iloc[i-1])   else 50,
            df['MACD'].iloc[i-1]  if not pd.isna(df['MACD'].iloc[i-1]) else 0,
            df['BB_upper'].iloc[i-1] if not pd.isna(df['BB_upper'].iloc[i-1]) else df['Close'].iloc[i-1],
            df['BB_lower'].iloc[i-1] if not pd.isna(df['BB_lower'].iloc[i-1]) else df['Close'].iloc[i-1],
            df['Volume'].iloc[i-1],
            df['Volatility'].iloc[i-1] if not pd.isna(df['Volatility'].iloc[i-1]) else 0,
        ]
        rows.append(row)
        targets.append(df['Close'].iloc[i])
    return np.array(rows), np.array(targets), df.index[lookback:]

def format_mcap(cap):
    if cap >= 1e12: return f"${cap/1e12:.2f}T"
    if cap >= 1e9:  return f"${cap/1e9:.2f}B"
    if cap >= 1e6:  return f"${cap/1e6:.2f}M"
    return f"${cap:,.0f}"

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,14,26,0.8)',
    font=dict(family='Inter', color='#8892b0', size=11),
    xaxis=dict(gridcolor='rgba(0,212,255,0.06)',
               linecolor='rgba(0,212,255,0.15)',
               tickfont=dict(color='#8892b0')),
    yaxis=dict(gridcolor='rgba(0,212,255,0.06)',
               linecolor='rgba(0,212,255,0.15)',
               tickfont=dict(color='#8892b0')),
    legend=dict(bgcolor='rgba(14,21,39,0.9)',
                bordercolor='rgba(0,212,255,0.2)',
                borderwidth=1,
                font=dict(color='#e8eaf6', size=11)),
    hovermode='x unified',
    hoverlabel=dict(bgcolor='rgba(14,21,39,0.95)',
                    bordercolor='rgba(0,212,255,0.3)',
                    font=dict(family='Inter', color='#e8eaf6')),
    margin=dict(l=10, r=10, t=40, b=10)
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:3rem;margin-bottom:0.5rem;">📈</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;font-weight:700;
                    background:linear-gradient(135deg,#00d4ff,#7c3aed);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;">AI Stock Predictor</div>
        <div style="font-size:0.72rem;color:#4a5568;margin-top:4px;">Powered by Machine Learning</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">📊 Stock Selection</div>', unsafe_allow_html=True)
    stocks = {
        "Apple (AAPL)": "AAPL", "Microsoft (MSFT)": "MSFT",
        "Google (GOOGL)": "GOOGL", "Amazon (AMZN)": "AMZN",
        "Tesla (TSLA)": "TSLA", "NVIDIA (NVDA)": "NVDA",
        "Meta (META)": "META", "Netflix (NFLX)": "NFLX",
        "Custom Ticker": "CUSTOM"
    }
    choice = st.selectbox("Select Stock", list(stocks.keys()))
    if choice == "Custom Ticker":
        ticker = st.text_input("Enter Ticker", value="SPY")
    else:
        ticker = stocks[choice]
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">⏱️ Time Settings</div>', unsafe_allow_html=True)
    period_map   = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
    period_label = st.selectbox("Historical Period", list(period_map.keys()), index=1)
    period       = period_map[period_label]
    forecast_days = st.slider("Forecast Days", 1, 60, 30)
    lookback     = st.slider("Lookback Window", 10, 60, 30)
    st.markdown('</div>', unsafe_allow_html=True)

    # Model Settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">🤖 Model Settings</div>', unsafe_allow_html=True)
    model_type = st.selectbox("Algorithm", [
        "Gradient Boosting", "Random Forest", "Ridge Regression", "Ensemble (All)"
    ])
    st.markdown('</div>', unsafe_allow_html=True)

    run_btn = st.button("🚀 Run Prediction")

    st.markdown("""
    <div style="text-align:center;margin-top:1.5rem;color:#2d3748;font-size:0.72rem;">
        ⚠️ Educational purposes only<br>Not financial advice
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="hero-header">
    <div class="hero-badge">
        <span class="live-dot"></span>
        AI-Powered Analysis
    </div>
    <h1 class="hero-title">Stock Price Trend Predictor</h1>
    <p class="hero-subtitle">
        Machine Learning · Real-Time Forecasting · Interactive Analytics
        &nbsp;|&nbsp; Analyzing: <strong style="color:#00d4ff;">{ticker}</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# FETCH DATA
# ============================================================
if 'results' not in st.session_state:
    st.session_state.results = None

with st.spinner("Fetching live data..."):
    df_raw, info = fetch_stock_data(ticker, period)

if df_raw is None or df_raw.empty:
    st.error("❌ Could not fetch data. Check ticker symbol.")
    st.stop()

df = add_indicators(df_raw.copy())
cur  = float(df['Close'].iloc[-1])
prev = float(df['Close'].iloc[-2])
chg  = cur - prev
pct  = chg / prev * 100

# ============================================================
# METRIC CARDS
# ============================================================
def mcard(icon, label, val, delta=None, sign=None, cls="blue"):
    d = ""
    if delta:
        c = "positive" if sign=="+" else "negative"
        a = "▲" if sign=="+" else "▼"
        d = f'<div class="metric-delta {c}">{a} {delta}</div>'
    return f"""
    <div class="metric-card {cls}">
        <div style="font-size:1.6rem;margin-bottom:4px">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{val}</div>
        {d}
    </div>
    """

c1,c2,c3,c4,c5,c6 = st.columns(6)
with c1: st.markdown(mcard("💰","Current Price",f"${cur:.2f}",f"{abs(pct):.2f}%","+" if chg>=0 else "-","blue"),unsafe_allow_html=True)
with c2: st.markdown(mcard("📊","Day Change",f"{'+'if chg>=0 else''}{chg:.2f}",f"${abs(chg):.2f}","+" if chg>=0 else "-","green" if chg>=0 else "red"),unsafe_allow_html=True)
with c3:
    vol = df['Volume'].iloc[-1]
    st.markdown(mcard("📦","Volume",f"{vol/1e6:.1f}M" if vol>=1e6 else f"{vol/1e3:.0f}K",cls="purple"),unsafe_allow_html=True)
with c4:
    rsi = df['RSI'].iloc[-1]
    rc  = "green" if rsi<30 else ("red" if rsi>70 else "gold")
    st.markdown(mcard("⚡","RSI (14)",f"{rsi:.1f}",cls=rc),unsafe_allow_html=True)
with c5:
    h52 = info.get('week_52_high',0)
    st.markdown(mcard("🎯","52W High",f"${h52:.2f}" if h52 else "N/A",cls="gold"),unsafe_allow_html=True)
with c6:
    mc = info.get('market_cap',0)
    st.markdown(mcard("🏦","Market Cap",format_mcap(mc) if mc else "N/A",cls="blue"),unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Price Chart","🔬 Technical","🤖 AI Prediction","📊 Performance","ℹ️ Stock Info"
])

# ── TAB 1: PRICE CHART ──
with tab1:
    col_m, col_s = st.columns([3,1])
    with col_m:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title"><div class="section-title-icon">📈</div>{info.get("name",ticker)} — Price History</div>', unsafe_allow_html=True)

        ctype = st.radio("Chart Type", ["Candlestick","Line","OHLC"], horizontal=True, label_visibility="collapsed")
        dp = df.dropna(subset=['Close'])

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75,0.25], vertical_spacing=0.03)

        if ctype == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=dp.index, open=dp['Open'], high=dp['High'],
                low=dp['Low'], close=dp['Close'], name="OHLC",
                increasing_line_color='#00ff88', decreasing_line_color='#ff4757',
                increasing_fillcolor='rgba(0,255,136,0.3)',
                decreasing_fillcolor='rgba(255,71,87,0.3)'
            ), row=1, col=1)
        elif ctype == "Line":
            fig.add_trace(go.Scatter(
                x=dp.index, y=dp['Close'], name="Close",
                line=dict(color='#00d4ff', width=2),
                fill='tozeroy', fillcolor='rgba(0,212,255,0.05)'
            ), row=1, col=1)
        else:
            fig.add_trace(go.Ohlc(
                x=dp.index, open=dp['Open'], high=dp['High'],
                low=dp['Low'], close=dp['Close'], name="OHLC",
                increasing_line_color='#00ff88', decreasing_line_color='#ff4757'
            ), row=1, col=1)

        if not dp['MA20'].isna().all():
            fig.add_trace(go.Scatter(x=dp.index, y=dp['MA20'], name="MA20",
                                     line=dict(color='#ffd700', width=1.5, dash='dot')), row=1, col=1)
        if not dp['MA50'].isna().all():
            fig.add_trace(go.Scatter(x=dp.index, y=dp['MA50'], name="MA50",
                                     line=dict(color='#a78bfa', width=1.5, dash='dot')), row=1, col=1)

        fig.add_trace(go.Scatter(x=dp.index, y=dp['BB_upper'], name="BB Upper",
                                  line=dict(color='rgba(0,212,255,0.4)', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=dp.index, y=dp['BB_lower'], name="BB Lower",
                                  line=dict(color='rgba(0,212,255,0.4)', width=1, dash='dash'),
                                  fill='tonexty', fillcolor='rgba(0,212,255,0.04)'), row=1, col=1)

        vc = ['#00ff88' if dp['Close'].iloc[i]>=dp['Open'].iloc[i] else '#ff4757' for i in range(len(dp))]
        fig.add_trace(go.Bar(x=dp.index, y=dp['Volume'], name="Volume",
                              marker_color=vc, opacity=0.6), row=2, col=1)

        fig.update_layout(**PLOTLY_LAYOUT, height=550)
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_s:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><div class="section-title-icon">📋</div>Quick Stats</div>', unsafe_allow_html=True)

        stats = {
            "Open":   f"${df['Open'].iloc[-1]:.2f}",
            "High":   f"${df['High'].iloc[-1]:.2f}",
            "Low":    f"${df['Low'].iloc[-1]:.2f}",
            "Close":  f"${df['Close'].iloc[-1]:.2f}",
            "MA7":    f"${df['MA7'].iloc[-1]:.2f}"  if not pd.isna(df['MA7'].iloc[-1])  else "N/A",
            "MA20":   f"${df['MA20'].iloc[-1]:.2f}" if not pd.isna(df['MA20'].iloc[-1]) else "N/A",
            "MA50":   f"${df['MA50'].iloc[-1]:.2f}" if not pd.isna(df['MA50'].iloc[-1]) else "N/A",
            "MACD":   f"{df['MACD'].iloc[-1]:.3f}"  if not pd.isna(df['MACD'].iloc[-1]) else "N/A",
        }
        for k,v in stats.items():
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                        padding:6px 0;border-bottom:1px solid rgba(0,212,255,0.08);">
                <span style="color:#8892b0;font-size:0.8rem;">{k}</span>
                <span style="color:#e8eaf6;font-weight:600;font-size:0.85rem;">{v}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Returns card
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><div class="section-title-icon">📉</div>Returns</div>', unsafe_allow_html=True)
        for label, days in [("1W",5),("1M",21),("3M",63),("6M",126),("1Y",252)]:
            if len(df) > days:
                ret = (df['Close'].iloc[-1]/df['Close'].iloc[-days]-1)*100
                color = "#00ff88" if ret>=0 else "#ff4757"
                arrow = "▲" if ret>=0 else "▼"
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            padding:6px 0;border-bottom:1px solid rgba(0,212,255,0.08);">
                    <span style="color:#8892b0;font-size:0.8rem;">{label}</span>
                    <span style="color:{color};font-weight:600;font-size:0.85rem;">{arrow} {abs(ret):.2f}%</span>
                </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 2: TECHNICAL ──
with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><div class="section-title-icon">🔬</div>Technical Indicators</div>', unsafe_allow_html=True)

    dta = df.dropna().copy()
    fig_ta = make_subplots(rows=4, cols=1, shared_xaxes=True,
                           row_heights=[0.40,0.20,0.20,0.20],
                           vertical_spacing=0.04,
                           subplot_titles=("Price & Bollinger Bands","RSI (14)","MACD","Volume"))

    fig_ta.add_trace(go.Scatter(x=dta.index, y=dta['Close'], name="Close",
                                line=dict(color='#00d4ff', width=2)), row=1, col=1)
    fig_ta.add_trace(go.Scatter(x=dta.index, y=dta['MA20'], name="MA20",
                                line=dict(color='#ffd700', width=1.5, dash='dot')), row=1, col=1)
    fig_ta.add_trace(go.Scatter(x=dta.index, y=dta['BB_upper'], name="BB Upper",
                                line=dict(color='rgba(0,212,255,0.5)', width=1, dash='dash')), row=1, col=1)
    fig_ta.add_trace(go.Scatter(x=dta.index, y=dta['BB_lower'], name="BB Lower",
                                line=dict(color='rgba(0,212,255,0.5)', width=1, dash='dash'),
                                fill='tonexty', fillcolor='rgba(0,212,255,0.05)'), row=1, col=1)

    fig_ta.add_trace(go.Scatter(x=dta.index, y=dta['RSI'], name="RSI",
                                line=dict(color='#a78bfa', width=2)), row=2, col=1)
    fig_ta.add_hline(y=70, line_dash="dash", line_color="rgba(255,71,87,0.5)", row=2, col=1)
    fig_ta.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,136,0.5)", row=2, col=1)

    mch = ['rgba(0,255,136,0.7)' if v>=0 else 'rgba(255,71,87,0.7)' for v in dta['MACD_hist']]
    fig_ta.add_trace(go.Bar(x=dta.index, y=dta['MACD_hist'], name="Histogram",
                             marker_color=mch), row=3, col=1)
    fig_ta.add_trace(go.Scatter(x=dta.index, y=dta['MACD'], name="MACD",
                                line=dict(color='#00d4ff', width=1.5)), row=3, col=1)
    fig_ta.add_trace(go.Scatter(x=dta.index, y=dta['MACD_signal'], name="Signal",
                                line=dict(color='#ff6b35', width=1.5)), row=3, col=1)

    vc2 = ['rgba(0,255,136,0.6)' if dta['Close'].iloc[i]>=dta['Open'].iloc[i]
           else 'rgba(255,71,87,0.6)' for i in range(len(dta))]
    fig_ta.add_trace(go.Bar(x=dta.index, y=dta['Volume'], name="Volume",
                             marker_color=vc2), row=4, col=1)

    fig_ta.update_layout(**PLOTLY_LAYOUT, height=750, showlegend=False)
    for ann in fig_ta.layout.annotations:
        ann.font.color = '#8892b0'
        ann.font.size  = 11
    st.plotly_chart(fig_ta, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

    # Signal cards
    sc = st.columns(4)
    rn  = dta['RSI'].iloc[-1]
    mn  = dta['MACD'].iloc[-1]
    ms  = dta['MACD_signal'].iloc[-1]
    cn  = dta['Close'].iloc[-1]
    m20 = dta['MA20'].iloc[-1]
    m50 = dta['MA50'].iloc[-1]

    def sig_card(title, val, bull, bear):
        if bull:   c,s = "#00ff88","Bullish"
        elif bear: c,s = "#ff4757","Bearish"
        else:      c,s = "#ffd700","Neutral"
        return f"""
        <div class="metric-card" style="text-align:center;">
            <div class="metric-label">{title}</div>
            <div class="metric-value" style="font-size:1.3rem;color:{c};">{val}</div>
            <div style="color:{c};font-size:0.78rem;font-weight:700;margin-top:4px;">{s}</div>
        </div>"""

    with sc[0]: st.markdown(sig_card("RSI",f"{rn:.1f}",rn<30,rn>70),unsafe_allow_html=True)
    with sc[1]: st.markdown(sig_card("MACD",f"{mn:.3f}",mn>ms,mn<ms),unsafe_allow_html=True)
    with sc[2]: st.markdown(sig_card("MA Trend",f"${cn:.2f}",cn>m20 and cn>m50,cn<m20 and cn<m50),unsafe_allow_html=True)
    with sc[3]:
        bbu = dta['BB_upper'].iloc[-1]
        bbl = dta['BB_lower'].iloc[-1]
        pos = (cn-bbl)/(bbu-bbl)*100 if (bbu-bbl)!=0 else 50
        st.markdown(sig_card("Bollinger",f"{pos:.0f}%",pos<20,pos>80),unsafe_allow_html=True)

# ── TAB 3: AI PREDICTION ──
with tab3:
    if not run_btn and st.session_state.results is None:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;">
            <div style="font-size:4rem;margin-bottom:1rem;">🤖</div>
            <h2 style="font-family:'Space Grotesk',sans-serif;font-size:1.8rem;
                       background:linear-gradient(135deg,#00d4ff,#7c3aed);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                       background-clip:text;">Ready to Predict</h2>
            <p style="color:#8892b0;font-size:1rem;max-width:500px;margin:0 auto;">
                Configure settings in the sidebar and click
                <strong style="color:#00d4ff;">🚀 Run Prediction</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

    if run_btn:
        pb = st.progress(0)
        st_txt = st.empty()

        st_txt.markdown('<p style="color:#00d4ff;font-weight:600;">⚙️ Building feature matrix...</p>', unsafe_allow_html=True)
        X, y, idx = make_features(df, lookback)
        pb.progress(20)

        st_txt.markdown('<p style="color:#00d4ff;font-weight:600;">📐 Scaling data...</p>', unsafe_allow_html=True)
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        Xs = scaler_X.fit_transform(X)
        ys = scaler_y.fit_transform(y.reshape(-1,1)).ravel()
        pb.progress(35)

        split   = int(len(Xs)*0.85)
        X_train, X_test = Xs[:split], Xs[split:]
        y_train, y_test = ys[:split], ys[split:]
        idx_test = idx[split:]

        st_txt.markdown('<p style="color:#00d4ff;font-weight:600;">🤖 Training model...</p>', unsafe_allow_html=True)

        if model_type == "Gradient Boosting":
            model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                               max_depth=4, random_state=42)
            model.fit(X_train, y_train)
            models = {'Gradient Boosting': model}

        elif model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=200, max_depth=10,
                                           random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            models = {'Random Forest': model}

        elif model_type == "Ridge Regression":
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            models = {'Ridge': model}

        else:  # Ensemble
            m1 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                            max_depth=4, random_state=42)
            m2 = RandomForestRegressor(n_estimators=200, max_depth=10,
                                        random_state=42, n_jobs=-1)
            m3 = Ridge(alpha=1.0)
            m1.fit(X_train, y_train)
            m2.fit(X_train, y_train)
            m3.fit(X_train, y_train)
            models = {'Gradient Boosting': m1, 'Random Forest': m2, 'Ridge': m3}

        pb.progress(70)

        st_txt.markdown('<p style="color:#00d4ff;font-weight:600;">📊 Evaluating...</p>', unsafe_allow_html=True)

        if model_type == "Ensemble (All)":
            preds_s = np.mean([m.predict(X_test) for m in models.values()], axis=0)
        else:
            preds_s = model.predict(X_test)

        y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1,1)).ravel()
        y_pred_actual = scaler_y.inverse_transform(preds_s.reshape(-1,1)).ravel()

        mae  = mean_absolute_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        mape = np.mean(np.abs((y_test_actual - y_pred_actual)/(y_test_actual+1e-10)))*100
        r2   = r2_score(y_test_actual, y_pred_actual)
        pb.progress(85)

        st_txt.markdown('<p style="color:#00d4ff;font-weight:600;">🔮 Forecasting future prices...</p>', unsafe_allow_html=True)

        # Iterative forecast
        last_known = X[-1].copy()
        future_prices = []

        for step in range(forecast_days):
            inp_s    = scaler_X.transform(last_known.reshape(1,-1))
            if model_type == "Ensemble (All)":
                pred_s = np.mean([m.predict(inp_s)[0] for m in models.values()])
            else:
                pred_s = model.predict(inp_s)[0]
            pred_price = scaler_y.inverse_transform([[pred_s]])[0][0]
            future_prices.append(pred_price)
            # Shift lags
            new_row = last_known.copy()
            new_row[:lookback-1] = last_known[1:lookback]
            new_row[lookback-1]  = pred_price
            last_known = new_row

        future_dates = pd.bdate_range(
            start=df.index[-1] + timedelta(days=1), periods=forecast_days
        )
        pb.progress(100)
        st_txt.markdown('<p style="color:#00ff88;font-weight:700;">✅ Done!</p>', unsafe_allow_html=True)

        # Store results
        st.session_state.results = {
            'y_test_actual': y_test_actual,
            'y_pred_actual': y_pred_actual,
            'future_dates':  future_dates,
            'future_prices': future_prices,
            'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2,
            'idx_test': idx_test,
            'model_type': model_type,
        }

    # Show results if available
    if st.session_state.results is not None:
        res = st.session_state.results
        fp  = np.array(res['future_prices'])
        target = float(fp[-1])
        pred_pct = (target - cur) / cur * 100

        if pred_pct > 2:   sig, scls = "BUY",  "signal-buy"
        elif pred_pct < -2: sig, scls = "SELL", "signal-sell"
        else:               sig, scls = "HOLD", "signal-hold"

        st.markdown(f"""
        <div class="prediction-banner">
            <div style="font-size:0.82rem;color:#8892b0;text-transform:uppercase;
                        letter-spacing:2px;margin-bottom:0.5rem;">
                AI Price Target ({forecast_days} Days)
            </div>
            <div class="prediction-price">${target:.2f}</div>
            <div style="color:#8892b0;font-size:0.9rem;margin-top:0.3rem;">
                from current <strong style="color:#00d4ff;">${cur:.2f}</strong>
                &nbsp;→&nbsp;
                <strong style="color:{'#00ff88' if pred_pct>=0 else '#ff4757'}">
                    {'▲' if pred_pct>=0 else '▼'} {abs(pred_pct):.2f}%
                </strong>
            </div>
            <div style="margin-top:1rem;">
                <span class="{scls}">{sig}</span>
            </div>
            <div style="margin-top:0.8rem;background:rgba(0,255,136,0.15);
                        border:1px solid rgba(0,255,136,0.3);border-radius:50px;
                        padding:4px 16px;font-size:0.82rem;font-weight:600;
                        color:#00ff88;display:inline-block;">
                R² Score: {res['r2']:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        m1,m2,m3,m4 = st.columns(4)
        with m1: st.markdown(mcard("📏","MAE",f"${res['mae']:.2f}",cls="blue"),unsafe_allow_html=True)
        with m2: st.markdown(mcard("📐","RMSE",f"${res['rmse']:.2f}",cls="purple"),unsafe_allow_html=True)
        with m3: st.markdown(mcard("📊","MAPE",f"{res['mape']:.2f}%",cls="gold"),unsafe_allow_html=True)
        with m4: st.markdown(mcard("🎯","R² Score",f"{res['r2']:.4f}",cls="green"),unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Forecast chart
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><div class="section-title-icon">🔮</div>Price Forecast</div>', unsafe_allow_html=True)

        hist_plot = df['Close'].iloc[-120:] if len(df)>=120 else df['Close']

        lower = fp * (1 - res['mape']/100)
        upper = fp * (1 + res['mape']/100)

        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(
            x=hist_plot.index, y=hist_plot.values,
            name="Historical", line=dict(color='#00d4ff', width=2),
            fill='tozeroy', fillcolor='rgba(0,212,255,0.05)'
        ))

        n = min(len(res['idx_test']), len(res['y_pred_actual']))
        fig_f.add_trace(go.Scatter(
            x=res['idx_test'][:n], y=res['y_pred_actual'][:n],
            name="Model Fit", line=dict(color='#ffd700', width=2, dash='dot')
        ))

        fig_f.add_trace(go.Scatter(
            x=list(res['future_dates'])+list(res['future_dates'])[::-1],
            y=list(upper)+list(lower)[::-1],
            fill='toself', fillcolor='rgba(0,255,136,0.08)',
            line=dict(color='rgba(0,0,0,0)'),
            name="Confidence Band", hoverinfo='skip'
        ))

        fig_f.add_trace(go.Scatter(
            x=res['future_dates'], y=fp,
            name=f"Forecast ({forecast_days}D)",
            line=dict(color='#00ff88', width=2.5),
            mode='lines+markers',
            marker=dict(size=4, color='#00ff88')
        ))

        fig_f.add_hline(y=cur, line_dash="dash",
                         line_color="rgba(255,215,0,0.5)",
                         annotation_text=f"Current: ${cur:.2f}",
                         annotation_font_color="#ffd700")

        fig_f.update_layout(**PLOTLY_LAYOUT, height=450)
        st.plotly_chart(fig_f, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

        # Actual vs Predicted
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><div class="section-title-icon">📊</div>Actual vs Predicted</div>', unsafe_allow_html=True)

        fig_avp = go.Figure()
        test_idx = res['test_index']
        n_pts    = min(len(test_idx), len(res['y_test_actual']), len(res['y_pred_actual']))

        fig_avp.add_trace(go.Scatter(
            x=res['idx_test'][:n], y=res['y_test_actual'][:n],
            name="Actual", line=dict(color='#00d4ff', width=2)
        ))
        fig_avp.add_trace(go.Scatter(
            x=res['idx_test'][:n], y=res['y_pred_actual'][:n],
            name="Predicted", line=dict(color='#ff6b35', width=2, dash='dot')
        ))
        fig_avp.update_layout(**PLOTLY_LAYOUT, height=350)
        st.plotly_chart(fig_avp, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

        # Forecast table
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><div class="section-title-icon">📋</div>Forecast Table</div>', unsafe_allow_html=True)
        fdf = pd.DataFrame({
            'Date':            pd.DatetimeIndex(res['future_dates']).strftime('%Y-%m-%d'),
            'Predicted Price': [f"${p:.2f}" for p in fp],
            'Change from Today':[f"{'+'if p>=cur else''}{(p-cur)/cur*100:.2f}%" for p in fp],
            'Lower Bound':     [f"${p*(1-res['mape']/100):.2f}" for p in fp],
            'Upper Bound':     [f"${p*(1+res['mape']/100):.2f}" for p in fp],
        })
        st.dataframe(fdf, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 4: PERFORMANCE ──
with tab4:
    if st.session_state.results is None:
        st.info("Run the AI Prediction first to see performance metrics.")
    else:
        res = st.session_state.results
        cl, cr = st.columns(2)

        with cl:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title"><div class="section-title-icon">🎯</div>Regression Plot</div>', unsafe_allow_html=True)
            n = min(len(res['y_test_actual']),len(res['y_pred_actual']))
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(
                x=res['y_test_actual'][:n], y=res['y_pred_actual'][:n],
                mode='markers',
                marker=dict(color='#7c3aed', size=5, opacity=0.7)
            ))
            mn_v = min(res['y_test_actual'][:n].min(), res['y_pred_actual'][:n].min())
            mx_v = max(res['y_test_actual'][:n].max(), res['y_pred_actual'][:n].max())
            fig_s.add_trace(go.Scatter(
                x=[mn_v, mx_v], y=[mn_v, mx_v],
                mode='lines', name="Perfect",
                line=dict(color='#00ff88', width=2, dash='dash')
            ))
            fig_s.update_layout(**PLOTLY_LAYOUT, height=300,
                                xaxis_title="Actual", yaxis_title="Predicted")
            st.plotly_chart(fig_s, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        with cr:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title"><div class="section-title-icon">📈</div>Error Distribution</div>', unsafe_allow_html=True)
            errors = res['y_pred_actual'][:n] - res['y_test_actual'][:n]
            fig_e = go.Figure()
            fig_e.add_trace(go.Histogram(
                x=errors, nbinsx=40,
                marker_color='rgba(0,212,255,0.6)',
                marker_line_color='rgba(0,212,255,0.2)',
                marker_line_width=0.5
            ))
            fig_e.add_vline(x=0, line_dash="dash", line_color="#00ff88")
            fig_e.update_layout(**PLOTLY_LAYOUT, height=300,
                                xaxis_title="Error ($)", yaxis_title="Frequency")
            st.plotly_chart(fig_e, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        # Metrics summary
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><div class="section-title-icon">📊</div>Model Summary</div>', unsafe_allow_html=True)
        sm1,sm2,sm3,sm4 = st.columns(4)
        with sm1: st.markdown(mcard("📏","MAE",f"${res['mae']:.2f}",cls="blue"),unsafe_allow_html=True)
        with sm2: st.markdown(mcard("📐","RMSE",f"${res['rmse']:.2f}",cls="purple"),unsafe_allow_html=True)
        with sm3: st.markdown(mcard("📊","MAPE",f"{res['mape']:.2f}%",cls="gold"),unsafe_allow_html=True)
        with sm4: st.markdown(mcard("🎯","R²",f"{res['r2']:.4f}",cls="green"),unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 5: STOCK INFO ──
with tab5:
    info = st.session_state.info
    name = info.get('name', ticker)

    st.markdown(f"""
    <div class="section-card">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:1.5rem;">
            <div style="width:52px;height:52px;
                        background:linear-gradient(135deg,rgba(0,212,255,0.2),rgba(124,58,237,0.2));
                        border:1px solid rgba(0,212,255,0.3);border-radius:14px;
                        display:flex;align-items:center;justify-content:center;font-size:1.6rem;">📊</div>
            <div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;
                            font-weight:700;color:#e8eaf6;">{name}</div>
                <div style="color:#8892b0;font-size:0.9rem;">{ticker} · {info.get('sector','N/A')}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    ic1, ic2, ic3 = st.columns(3)

    def irow(l,v):
        return f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:8px 0;border-bottom:1px solid rgba(0,212,255,0.08);">
            <span style="color:#8892b0;font-size:0.82rem;">{l}</span>
            <span style="color:#e8eaf6;font-weight:600;font-size:0.85rem;">{v}</span>
        </div>"""

    with ic1:
        st.markdown("**📈 Valuation**")
        st.markdown(irow("Market Cap", format_mcap(info.get('market_cap',0)) if info.get('market_cap') else "N/A"), unsafe_allow_html=True)
        st.markdown(irow("P/E Ratio",  f"{info.get('pe_ratio',0):.2f}" if info.get('pe_ratio') else "N/A"), unsafe_allow_html=True)
        st.markdown(irow("Beta",       f"{info.get('beta',0):.2f}" if info.get('beta') else "N/A"), unsafe_allow_html=True)
    with ic2:
        st.markdown("**🎯 Price Levels**")
        st.markdown(irow("52W High",   f"${info.get('week_52_high',0):.2f}" if info.get('week_52_high') else "N/A"), unsafe_allow_html=True)
        st.markdown(irow("52W Low",    f"${info.get('week_52_low',0):.2f}"  if info.get('week_52_low')  else "N/A"), unsafe_allow_html=True)
        st.markdown(irow("Current",    f"${cur:.2f}"), unsafe_allow_html=True)
    with ic3:
        st.markdown("**📊 Data Stats**")
        st.markdown(irow("Data Points", f"{len(df):,}"), unsafe_allow_html=True)
        st.markdown(irow("Period Return", f"{(df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100:.2f}%"), unsafe_allow_html=True)
        st.markdown(irow("Avg Volume", f"{df['Volume'].mean()/1e6:.2f}M"), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><div class="section-title-icon">🔥</div>Correlation Matrix</div>', unsafe_allow_html=True)
    cc = ['Close','Volume','MA20','MA50','RSI','MACD','BB_upper','BB_lower']
    cm = df[cc].dropna().corr()
    fig_c = go.Figure(go.Heatmap(
        z=cm.values, x=cm.columns.tolist(), y=cm.columns.tolist(),
        colorscale=[[0,'#ff4757'],[0.5,'#141b2d'],[1,'#00d4ff']],
        zmid=0, text=np.round(cm.values,2),
        texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>'
    ))
    fig_c.update_layout(**PLOTLY_LAYOUT, height=380)
    st.plotly_chart(fig_c, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("""
<div style="border-top:1px solid rgba(0,212,255,0.1);padding:1.5rem 0;margin-top:1rem;">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
        <div>
            <span style="font-family:'Space Grotesk',sans-serif;font-weight:700;
                         background:linear-gradient(135deg,#00d4ff,#7c3aed);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                         background-clip:text;font-size:1.1rem;">
                📈 AI Stock Predictor
            </span>
            <span style="color:#2d3748;font-size:0.78rem;margin-left:12px;">
                Built with Streamlit · Plotly · Scikit-learn
            </span>
        </div>
        <div style="color:#2d3748;font-size:0.75rem;">
            ⚠️ Educational purposes only · Not financial advice
        </div>
    </div>
</div>
""", unsafe_allow_html=True)