# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
import time
from datetime import datetime, timedelta
import io
import base64

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS - STUNNING DARK THEME
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #0f1629;
        --bg-card: #141b2d;
        --bg-card-hover: #1a2238;
        --accent-blue: #00d4ff;
        --accent-purple: #7c3aed;
        --accent-green: #00ff88;
        --accent-red: #ff4757;
        --accent-gold: #ffd700;
        --accent-orange: #ff6b35;
        --text-primary: #e8eaf6;
        --text-secondary: #8892b0;
        --text-muted: #4a5568;
        --border: rgba(0, 212, 255, 0.15);
        --glow-blue: 0 0 30px rgba(0, 212, 255, 0.3);
        --glow-purple: 0 0 30px rgba(124, 58, 237, 0.3);
        --glow-green: 0 0 30px rgba(0, 255, 136, 0.3);
    }

    /* ── Global Reset ── */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', sans-serif;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    /* ── Hide Streamlit Defaults ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding: 0rem 2rem 2rem 2rem;
        max-width: 1600px;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent-blue), var(--accent-purple));
        border-radius: 3px;
    }

    /* ── Animated Header ── */
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
        background: radial-gradient(ellipse at 20% 50%, rgba(0,212,255,0.08) 0%, transparent 60%),
                    radial-gradient(ellipse at 80% 50%, rgba(124,58,237,0.08) 0%, transparent 60%);
        pointer-events: none;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.1;
        animation: titleGlow 3s ease-in-out infinite alternate;
    }

    @keyframes titleGlow {
        from { filter: brightness(1); }
        to { filter: brightness(1.2); }
    }

    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1.1rem;
        font-weight: 400;
        margin: 0.5rem 0 0 0;
        letter-spacing: 0.5px;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 50px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--accent-blue);
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .live-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-green);
        border-radius: 50%;
        animation: pulse 1.5s ease-in-out infinite;
        box-shadow: 0 0 8px var(--accent-green);
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.8); }
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(135deg, var(--bg-card) 0%, rgba(20,27,45,0.8) 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
    }

    .metric-card:hover {
        border-color: rgba(0, 212, 255, 0.4);
        transform: translateY(-2px);
        box-shadow: var(--glow-blue);
    }

    .metric-icon {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
        display: block;
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
    }

    .metric-delta {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    .metric-delta.positive { color: var(--accent-green); }
    .metric-delta.negative { color: var(--accent-red); }

    .metric-card.blue::before { background: linear-gradient(90deg, #00d4ff, #0099cc); }
    .metric-card.purple::before { background: linear-gradient(90deg, #7c3aed, #a855f7); }
    .metric-card.green::before { background: linear-gradient(90deg, #00ff88, #00cc6a); }
    .metric-card.gold::before { background: linear-gradient(90deg, #ffd700, #ff9500); }
    .metric-card.red::before { background: linear-gradient(90deg, #ff4757, #cc1a2a); }

    /* ── Section Cards ── */
    .section-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .section-card::after {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(124,58,237,0.05) 0%, transparent 70%);
        pointer-events: none;
    }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .section-title-icon {
        width: 32px; height: 32px;
        background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(124,58,237,0.2));
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }

    /* ── Sidebar ── */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding: 1.5rem 1rem;
    }

    .sidebar-logo {
        text-align: center;
        padding: 1rem 0 1.5rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.5rem;
    }

    .sidebar-logo-icon {
        font-size: 3rem;
        display: block;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 0 15px rgba(0,212,255,0.5));
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }

    .sidebar-logo-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
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
        color: var(--accent-blue);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.8rem;
    }

    /* ── Streamlit Widget Overrides ── */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: rgba(20, 27, 45, 0.8) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 2px rgba(0,212,255,0.1) !important;
    }

    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple)) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.7rem 2rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        box-shadow: 0 4px 20px rgba(0,212,255,0.2) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0,212,255,0.4) !important;
        filter: brightness(1.1) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ── Progress Bar ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple), var(--accent-green)) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        border: 1px solid var(--border) !important;
        gap: 4px !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(124,58,237,0.2)) !important;
        color: var(--accent-blue) !important;
        border: 1px solid rgba(0,212,255,0.3) !important;
    }

    .stTabs [data-baseweb="tab-panel"] {
        background: transparent !important;
        padding: 1rem 0 0 !important;
    }

    /* ── Alerts / Info Boxes ── */
    .stAlert {
        background: rgba(0, 212, 255, 0.08) !important;
        border: 1px solid rgba(0, 212, 255, 0.25) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }

    /* ── DataFrame ── */
    .stDataFrame {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }

    /* ── Status Cards ── */
    .status-card {
        display: flex;
        align-items: center;
        gap: 12px;
        background: rgba(0,255,136,0.06);
        border: 1px solid rgba(0,255,136,0.2);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
    }

    .status-card.error {
        background: rgba(255,71,87,0.06);
        border-color: rgba(255,71,87,0.2);
    }

    .status-card.warning {
        background: rgba(255,215,0,0.06);
        border-color: rgba(255,215,0,0.2);
    }

    .status-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .status-dot.green { background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); }
    .status-dot.red { background: var(--accent-red); box-shadow: 0 0 8px var(--accent-red); }
    .status-dot.gold { background: var(--accent-gold); box-shadow: 0 0 8px var(--accent-gold); }

    /* ── Prediction Banner ── */
    .prediction-banner {
        background: linear-gradient(135deg, rgba(0,212,255,0.1) 0%, rgba(124,58,237,0.1) 50%, rgba(0,255,136,0.1) 100%);
        border: 1px solid rgba(0,212,255,0.25);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }

    .prediction-banner::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(0,212,255,0.05), transparent, rgba(124,58,237,0.05), transparent);
        animation: rotate 8s linear infinite;
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .prediction-price {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        z-index: 1;
    }

    .prediction-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
        position: relative;
        z-index: 1;
        margin-bottom: 0.5rem;
    }

    .prediction-confidence {
        background: rgba(0,255,136,0.15);
        border: 1px solid rgba(0,255,136,0.3);
        border-radius: 50px;
        padding: 4px 16px;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--accent-green);
        display: inline-block;
        position: relative;
        z-index: 1;
        margin-top: 0.5rem;
    }

    /* ── Stock Tags ── */
    .stock-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(124,58,237,0.15);
        border: 1px solid rgba(124,58,237,0.3);
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #a78bfa;
        margin: 2px;
        cursor: pointer;
        transition: all 0.2s;
    }

    .stock-tag:hover {
        background: rgba(124,58,237,0.3);
        transform: translateY(-1px);
    }

    /* ── Divider ── */
    .neon-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-blue), var(--accent-purple), transparent);
        margin: 1.5rem 0;
        border: none;
        opacity: 0.4;
    }

    /* ── Chart Container ── */
    .chart-container {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* ── Tooltip ── */
    .tooltip-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: var(--text-secondary);
        margin: 0.5rem 0;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }

    /* ── Staggered animation ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in { animation: fadeInUp 0.6s ease forwards; }
    .fade-in-1 { animation: fadeInUp 0.6s 0.1s ease forwards; opacity: 0; }
    .fade-in-2 { animation: fadeInUp 0.6s 0.2s ease forwards; opacity: 0; }
    .fade-in-3 { animation: fadeInUp 0.6s 0.3s ease forwards; opacity: 0; }

    /* ── Signal indicator ── */
    .signal-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .signal-buy {
        background: rgba(0,255,136,0.15);
        border: 2px solid rgba(0,255,136,0.5);
        color: var(--accent-green);
        box-shadow: 0 0 20px rgba(0,255,136,0.2);
    }

    .signal-sell {
        background: rgba(255,71,87,0.15);
        border: 2px solid rgba(255,71,87,0.5);
        color: var(--accent-red);
        box-shadow: 0 0 20px rgba(255,71,87,0.2);
    }

    .signal-hold {
        background: rgba(255,215,0,0.15);
        border: 2px solid rgba(255,215,0,0.5);
        color: var(--accent-gold);
        box-shadow: 0 0 20px rgba(255,215,0,0.2);
    }

    /* ── Loading animation ── */
    .loading-text {
        color: var(--accent-blue);
        font-weight: 600;
        font-size: 0.9rem;
        animation: blink 1s ease-in-out infinite;
    }

    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* ── Model Architecture Card ── */
    .arch-card {
        background: linear-gradient(135deg, rgba(0,212,255,0.05), rgba(124,58,237,0.05));
        border: 1px solid rgba(124,58,237,0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.4rem 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .arch-icon {
        width: 36px; height: 36px;
        background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(124,58,237,0.2));
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }

    .arch-name { font-weight: 600; font-size: 0.9rem; color: var(--text-primary); }
    .arch-detail { font-size: 0.78rem; color: var(--text-secondary); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data(ttl=300)
def fetch_stock_data(ticker, period, interval="1d"):
    """Fetch stock data from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        info = {}
        try:
            raw_info = stock.info
            info = {
                'name': raw_info.get('longName', ticker),
                'sector': raw_info.get('sector', 'N/A'),
                'market_cap': raw_info.get('marketCap', 0),
                'pe_ratio': raw_info.get('trailingPE', 0),
                'volume_avg': raw_info.get('averageVolume', 0),
                'week_52_high': raw_info.get('fiftyTwoWeekHigh', 0),
                'week_52_low': raw_info.get('fiftyTwoWeekLow', 0),
                'dividend_yield': raw_info.get('dividendYield', 0),
                'beta': raw_info.get('beta', 0),
            }
        except:
            info = {'name': ticker, 'sector': 'N/A', 'market_cap': 0,
                    'pe_ratio': 0, 'volume_avg': 0, 'week_52_high': 0,
                    'week_52_low': 0, 'dividend_yield': 0, 'beta': 0}
        return df, info
    except Exception as e:
        return None, {}

def add_technical_indicators(df):
    """Add technical indicators to the DataFrame."""
    df = df.copy()
    # Moving Averages
    df['MA7']  = df['Close'].rolling(window=7).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    # Bollinger Bands
    df['BB_mid']   = df['Close'].rolling(window=20).mean()
    df['BB_std']   = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
    # RSI
    delta = df['Close'].diff()
    gain  = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']
    # Volume MA
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    # Daily Return & Volatility
    df['Daily_Return']   = df['Close'].pct_change()
    df['Volatility']     = df['Daily_Return'].rolling(window=20).std()
    # ATR
    df['TR']  = np.maximum(df['High'] - df['Low'],
                np.maximum(abs(df['High'] - df['Close'].shift()),
                           abs(df['Low']  - df['Close'].shift())))
    df['ATR'] = df['TR'].rolling(window=14).mean()
    return df

def prepare_sequences(data, seq_length):
    """Prepare sequences for LSTM."""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_model(model_type, seq_length, n_features, units, dropout_rate, learning_rate):
    """Build deep learning model."""
    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(units, return_sequences=True,
                       input_shape=(seq_length, n_features),
                       kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units // 2, return_sequences=True,
                       kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units // 4, return_sequences=False))
        model.add(Dropout(dropout_rate))
    elif model_type == "Bidirectional LSTM":
        model.add(Bidirectional(LSTM(units, return_sequences=True),
                                input_shape=(seq_length, n_features)))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(units // 2, return_sequences=True)))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(units // 4, return_sequences=False)))
        model.add(Dropout(dropout_rate))
    elif model_type == "GRU":
        model.add(GRU(units, return_sequences=True,
                      input_shape=(seq_length, n_features)))
        model.add(Dropout(dropout_rate))
        model.add(GRU(units // 2, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(GRU(units // 4, return_sequences=False))
        model.add(Dropout(dropout_rate))
    elif model_type == "Hybrid LSTM+GRU":
        model.add(LSTM(units, return_sequences=True,
                       input_shape=(seq_length, n_features)))
        model.add(Dropout(dropout_rate))
        model.add(GRU(units // 2, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units // 4, return_sequences=False))
        model.add(Dropout(dropout_rate))

    model.add(Dense(32, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='huber',
                  metrics=['mae'])
    return model

def format_market_cap(cap):
    if cap >= 1e12: return f"${cap/1e12:.2f}T"
    if cap >= 1e9:  return f"${cap/1e9:.2f}B"
    if cap >= 1e6:  return f"${cap/1e6:.2f}M"
    return f"${cap:,.0f}"

def get_trend_signal(predictions, current_price):
    avg_pred  = np.mean(predictions)
    change_pct = (avg_pred - current_price) / current_price * 100
    if change_pct > 2:   return "BUY",  change_pct, "signal-buy"
    if change_pct < -2:  return "SELL", change_pct, "signal-sell"
    return "HOLD", change_pct, "signal-hold"

# ============================================================
# PLOTLY THEME
# ============================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,14,26,0.8)',
    font=dict(family='Inter', color='#8892b0', size=11),
    xaxis=dict(
        gridcolor='rgba(0,212,255,0.06)',
        zerolinecolor='rgba(0,212,255,0.1)',
        linecolor='rgba(0,212,255,0.15)',
        tickfont=dict(color='#8892b0'),
        showgrid=True
    ),
    yaxis=dict(
        gridcolor='rgba(0,212,255,0.06)',
        zerolinecolor='rgba(0,212,255,0.1)',
        linecolor='rgba(0,212,255,0.15)',
        tickfont=dict(color='#8892b0'),
        showgrid=True
    ),
    legend=dict(
        bgcolor='rgba(14,21,39,0.9)',
        bordercolor='rgba(0,212,255,0.2)',
        borderwidth=1,
        font=dict(color='#e8eaf6', size=11)
    ),
    hovermode='x unified',
    hoverlabel=dict(
        bgcolor='rgba(14,21,39,0.95)',
        bordercolor='rgba(0,212,255,0.3)',
        font=dict(family='Inter', color='#e8eaf6')
    ),
    margin=dict(l=10, r=10, t=40, b=10)
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="sidebar-logo-icon">📈</span>
        <div class="sidebar-logo-text">AI Stock Predictor</div>
        <div style="font-size:0.72rem;color:#4a5568;margin-top:4px;">Powered by Deep Learning</div>
    </div>
    """, unsafe_allow_html=True)

    # Stock Selection
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">📊 Stock Selection</div>', unsafe_allow_html=True)

    popular_stocks = {
        "Apple (AAPL)": "AAPL", "Microsoft (MSFT)": "MSFT",
        "Google (GOOGL)": "GOOGL", "Amazon (AMZN)": "AMZN",
        "Tesla (TSLA)": "TSLA", "NVIDIA (NVDA)": "NVDA",
        "Meta (META)": "META", "Netflix (NFLX)": "NFLX",
        "Custom Ticker": "CUSTOM"
    }

    stock_choice = st.selectbox("Select Stock", list(popular_stocks.keys()), index=0)
    if stock_choice == "Custom Ticker":
        ticker = st.text_input("Enter Ticker Symbol", value="SPY",
                               placeholder="e.g., AAPL, BTC-USD")
    else:
        ticker = popular_stocks[stock_choice]
    st.markdown('</div>', unsafe_allow_html=True)

    # Time Settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">⏱️ Time Settings</div>', unsafe_allow_html=True)

    period_map = {
        "6 Months": "6mo", "1 Year": "1y",
        "2 Years": "2y", "5 Years": "5y"
    }
    period_label = st.selectbox("Historical Period", list(period_map.keys()), index=1)
    period       = period_map[period_label]

    forecast_days = st.slider("Forecast Days", min_value=1, max_value=60, value=30, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    # Model Settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">🤖 Model Settings</div>', unsafe_allow_html=True)

    model_type  = st.selectbox("Architecture",
                               ["LSTM", "Bidirectional LSTM", "GRU", "Hybrid LSTM+GRU"], index=0)
    seq_length  = st.slider("Sequence Length", min_value=10, max_value=90, value=60, step=5)
    units       = st.select_slider("LSTM Units", options=[32, 64, 128, 256], value=128)
    dropout_rate = st.slider("Dropout Rate", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    epochs      = st.slider("Max Epochs", min_value=20, max_value=200, value=60, step=10)
    batch_size  = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
    learning_rate = st.select_slider("Learning Rate",
                                     options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
    st.markdown('</div>', unsafe_allow_html=True)

    # Feature Selection
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">🔬 Features</div>', unsafe_allow_html=True)

    use_volume   = st.checkbox("Volume", value=True)
    use_ma       = st.checkbox("Moving Averages", value=True)
    use_rsi      = st.checkbox("RSI", value=True)
    use_macd     = st.checkbox("MACD", value=True)
    use_bb       = st.checkbox("Bollinger Bands", value=True)
    use_atr      = st.checkbox("ATR", value=False)
    st.markdown('</div>', unsafe_allow_html=True)

    run_button = st.button("🚀 Run Prediction", use_container_width=True)

    st.markdown("""
    <div style="text-align:center;margin-top:1.5rem;color:#2d3748;font-size:0.72rem;">
        ⚠️ For educational purposes only<br>Not financial advice
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
        Deep Learning · Real-Time Forecasting · Interactive Analytics
        &nbsp;|&nbsp; Currently analyzing: <strong style="color:#00d4ff;">{ticker}</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if 'results' not in st.session_state:
    st.session_state.results = None
if 'df_full' not in st.session_state:
    st.session_state.df_full = None
if 'info' not in st.session_state:
    st.session_state.info = {}

# ============================================================
# FETCH LIVE DATA (always)
# ============================================================
with st.spinner(""):
    df_raw, stock_info = fetch_stock_data(ticker, period)

if df_raw is None or df_raw.empty:
    st.markdown("""
    <div class="status-card error">
        <div class="status-dot red"></div>
        <div><strong>Data Fetch Failed</strong> — Please check the ticker symbol and try again.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df_full = add_technical_indicators(df_raw.copy())
st.session_state.df_full = df_full
st.session_state.info    = stock_info

current_price = float(df_full['Close'].iloc[-1])
prev_price    = float(df_full['Close'].iloc[-2])
price_change  = current_price - prev_price
price_pct     = price_change / prev_price * 100

# ============================================================
# METRIC CARDS ROW
# ============================================================
col1, col2, col3, col4, col5, col6 = st.columns(6)

def metric_card(icon, label, value, delta=None, delta_sign=None, card_class="blue"):
    delta_html = ""
    if delta is not None:
        cls = "positive" if delta_sign == "+" else "negative"
        arrow = "▲" if delta_sign == "+" else "▼"
        delta_html = f'<div class="metric-delta {cls}">{arrow} {delta}</div>'
    return f"""
    <div class="metric-card {card_class}">
        <span class="metric-icon">{icon}</span>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

with col1:
    st.markdown(metric_card(
        "💰", "Current Price",
        f"${current_price:.2f}",
        f"{abs(price_pct):.2f}%",
        "+" if price_change >= 0 else "-",
        "blue"
    ), unsafe_allow_html=True)

with col2:
    st.markdown(metric_card(
        "📊", "Day Change",
        f"{'+'if price_change>=0 else ''}{price_change:.2f}",
        f"${abs(price_change):.2f}",
        "+" if price_change >= 0 else "-",
        "green" if price_change >= 0 else "red"
    ), unsafe_allow_html=True)

with col3:
    vol = df_full['Volume'].iloc[-1]
    st.markdown(metric_card(
        "📦", "Volume",
        f"{vol/1e6:.2f}M" if vol >= 1e6 else f"{vol/1e3:.1f}K",
        card_class="purple"
    ), unsafe_allow_html=True)

with col4:
    rsi_val = df_full['RSI'].iloc[-1]
    rsi_class = "green" if rsi_val < 30 else ("red" if rsi_val > 70 else "gold")
    st.markdown(metric_card("⚡", "RSI (14)", f"{rsi_val:.1f}", card_class=rsi_class), unsafe_allow_html=True)

with col5:
    hi52 = stock_info.get('week_52_high', 0)
    lo52 = stock_info.get('week_52_low', 0)
    st.markdown(metric_card(
        "🎯", "52W High",
        f"${hi52:.2f}" if hi52 else "N/A",
        card_class="gold"
    ), unsafe_allow_html=True)

with col6:
    mc = stock_info.get('market_cap', 0)
    st.markdown(metric_card(
        "🏦", "Market Cap",
        format_market_cap(mc) if mc else "N/A",
        card_class="blue"
    ), unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Price Chart", "🔬 Technical Analysis",
    "🤖 AI Prediction", "📊 Performance", "ℹ️ Stock Info"
])

# ──────────────────────────────────────────────────────────────
# TAB 1 — PRICE CHART
# ──────────────────────────────────────────────────────────────
with tab1:
    col_main, col_side = st.columns([3, 1])

    with col_main:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="section-title">
            <div class="section-title-icon">📈</div>
            {stock_info.get('name', ticker)} — Price History & Candlestick
        </div>
        """, unsafe_allow_html=True)

        chart_type = st.radio("Chart Type", ["Candlestick", "Line", "OHLC"],
                              horizontal=True, label_visibility="collapsed")

        df_plot = df_full.dropna(subset=['Close'])

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25], vertical_spacing=0.03)

        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=df_plot.index, open=df_plot['Open'],
                high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'],
                name="OHLC",
                increasing_line_color='#00ff88', decreasing_line_color='#ff4757',
                increasing_fillcolor='rgba(0,255,136,0.3)',
                decreasing_fillcolor='rgba(255,71,87,0.3)'
            ), row=1, col=1)
        elif chart_type == "Line":
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot['Close'], name="Close",
                line=dict(color='#00d4ff', width=2),
                fill='tonexty' if False else 'tozeroy',
                fillcolor='rgba(0,212,255,0.05)'
            ), row=1, col=1)
        else:
            fig.add_trace(go.Ohlc(
                x=df_plot.index, open=df_plot['Open'],
                high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'],
                name="OHLC",
                increasing_line_color='#00ff88', decreasing_line_color='#ff4757'
            ), row=1, col=1)

        # MA overlays
        if not df_plot['MA20'].isna().all():
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'],
                                     name="MA20", line=dict(color='#ffd700', width=1.5, dash='dot')), row=1, col=1)
        if not df_plot['MA50'].isna().all():
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'],
                                     name="MA50", line=dict(color='#a78bfa', width=1.5, dash='dot')), row=1, col=1)

        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_upper'],
                                  name="BB Upper", line=dict(color='rgba(0,212,255,0.4)', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_lower'],
                                  name="BB Lower", line=dict(color='rgba(0,212,255,0.4)', width=1, dash='dash'),
                                  fill='tonexty', fillcolor='rgba(0,212,255,0.04)'), row=1, col=1)

        # Volume
        colors = ['#00ff88' if df_plot['Close'].iloc[i] >= df_plot['Open'].iloc[i]
                  else '#ff4757' for i in range(len(df_plot))]
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'],
                              name="Volume", marker_color=colors, opacity=0.6), row=2, col=1)

        fig.update_layout(**PLOTLY_LAYOUT, height=550,
                          title=dict(text=f"", font=dict(size=12)))
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_side:
        # Price stats
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><div class="section-title-icon">📋</div>Quick Stats</div>', unsafe_allow_html=True)

        stats = {
            "Open":     f"${df_full['Open'].iloc[-1]:.2f}",
            "High":     f"${df_full['High'].iloc[-1]:.2f}",
            "Low":      f"${df_full['Low'].iloc[-1]:.2f}",
            "Close":    f"${df_full['Close'].iloc[-1]:.2f}",
            "MA7":      f"${df_full['MA7'].iloc[-1]:.2f}" if not pd.isna(df_full['MA7'].iloc[-1]) else "N/A",
            "MA20":     f"${df_full['MA20'].iloc[-1]:.2f}" if not pd.isna(df_full['MA20'].iloc[-1]) else "N/A",
            "MA50":     f"${df_full['MA50'].iloc[-1]:.2f}" if not pd.isna(df_full['MA50'].iloc[-1]) else "N/A",
            "MACD":     f"{df_full['MACD'].iloc[-1]:.3f}" if not pd.isna(df_full['MACD'].iloc[-1]) else "N/A",
            "ATR":      f"{df_full['ATR'].iloc[-1]:.2f}" if not pd.isna(df_full['ATR'].iloc[-1]) else "N/A",
            "Volatility": f"{df_full['Volatility'].iloc[-1]*100:.2f}%" if not pd.isna(df_full['Volatility'].iloc[-1]) else "N/A",
        }

        for k, v in stats.items():
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:6px 0;border-bottom:1px solid rgba(0,212,255,0.08);">
                <span style="color:#8892b0;font-size:0.8rem;">{k}</span>
                <span style="color:#e8eaf6;font-weight:600;font-size:0.85rem;">{v}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Returns card
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><div class="section-title-icon">📉</div>Returns</div>', unsafe_allow_html=True)

        for label, days in [("1W", 5), ("1M", 21), ("3M", 63), ("6M", 126), ("1Y", 252)]:
            if len(df_full) > days:
                ret = (df_full['Close'].iloc[-1] / df_full['Close'].iloc[-days] - 1) * 100
                color = "#00ff88" if ret >= 0 else "#ff4757"
                arrow = "▲" if ret >= 0 else "▼"
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:6px 0;border-bottom:1px solid rgba(0,212,255,0.08);">
                    <span style="color:#8892b0;font-size:0.8rem;">{label}</span>
                    <span style="color:{color};font-weight:600;font-size:0.85rem;">{arrow} {abs(ret):.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# TAB 2 — TECHNICAL ANALYSIS
# ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><div class="section-title-icon">🔬</div>Technical Indicators Dashboard</div>', unsafe_allow_html=True)

    df_ta = df_full.dropna().copy()

    fig_ta = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.40, 0.20, 0.20, 0.20],
        vertical_spacing=0.04,
        subplot_titles=("Price & Bollinger Bands", "RSI (14)", "MACD", "Volume")
    )

    # Price + Bollinger
    fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta['Close'], name="Close",
                                line=dict(color='#00d4ff', width=2)), row=1, col=1)
    fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta['MA20'], name="MA20",
                                line=dict(color='#ffd700', width=1.5, dash='dot')), row=1, col=1)
    fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_upper'], name="BB Upper",
                                line=dict(color='rgba(0,212,255,0.5)', width=1, dash='dash')), row=1, col=1)
    fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_lower'], name="BB Lower",
                                line=dict(color='rgba(0,212,255,0.5)', width=1, dash='dash'),
                                fill='tonexty', fillcolor='rgba(0,212,255,0.05)'), row=1, col=1)

    # RSI
    fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta['RSI'], name="RSI",
                                line=dict(color='#a78bfa', width=2)), row=2, col=1)
    fig_ta.add_hline(y=70, line_dash="dash", line_color="rgba(255,71,87,0.5)", row=2, col=1)
    fig_ta.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,136,0.5)", row=2, col=1)
    fig_ta.add_hrect(y0=70, y1=100, fillcolor="rgba(255,71,87,0.05)", row=2, col=1)
    fig_ta.add_hrect(y0=0, y1=30, fillcolor="rgba(0,255,136,0.05)", row=2, col=1)

    # MACD
    colors_macd = ['rgba(0,255,136,0.7)' if v >= 0 else 'rgba(255,71,87,0.7)'
                   for v in df_ta['MACD_hist']]
    fig_ta.add_trace(go.Bar(x=df_ta.index, y=df_ta['MACD_hist'],
                             name="Histogram", marker_color=colors_macd), row=3, col=1)
    fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta['MACD'], name="MACD",
                                line=dict(color='#00d4ff', width=1.5)), row=3, col=1)
    fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta['MACD_signal'], name="Signal",
                                line=dict(color='#ff6b35', width=1.5)), row=3, col=1)

    # Volume
    vol_colors = ['rgba(0,255,136,0.6)' if df_ta['Close'].iloc[i] >= df_ta['Open'].iloc[i]
                  else 'rgba(255,71,87,0.6)' for i in range(len(df_ta))]
    fig_ta.add_trace(go.Bar(x=df_ta.index, y=df_ta['Volume'],
                             name="Volume", marker_color=vol_colors), row=4, col=1)

    fig_ta.update_layout(**PLOTLY_LAYOUT, height=750, showlegend=False)
    for ann in fig_ta.layout.annotations:
        ann.font.color = '#8892b0'
        ann.font.size  = 11
    st.plotly_chart(fig_ta, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

    # Signal Summary
    st.markdown('<div class="section-title" style="margin-top:1rem"><div class="section-title-icon">🎯</div>Signal Summary</div>', unsafe_allow_html=True)
    sig_cols = st.columns(5)

    def signal_card(title, value, cond_green, cond_red, fmt=""):
        if cond_green:  color, signal_text = "#00ff88", "Bullish"
        elif cond_red:  color, signal_text = "#ff4757", "Bearish"
        else:           color, signal_text = "#ffd700", "Neutral"
        return f"""
        <div class="metric-card" style="text-align:center;">
            <div class="metric-label">{title}</div>
            <div class="metric-value" style="font-size:1.4rem;color:{color};">{value}{fmt}</div>
            <div style="color:{color};font-size:0.78rem;font-weight:700;margin-top:4px;">{signal_text}</div>
        </div>
        """

    rsi_now  = df_ta['RSI'].iloc[-1]
    macd_now = df_ta['MACD'].iloc[-1]
    macd_sig = df_ta['MACD_signal'].iloc[-1]
    close_now = df_ta['Close'].iloc[-1]
    ma20_now  = df_ta['MA20'].iloc[-1]
    ma50_now  = df_ta['MA50'].iloc[-1]
    bb_upper  = df_ta['BB_upper'].iloc[-1]
    bb_lower  = df_ta['BB_lower'].iloc[-1]
    vol_now   = df_ta['Volume'].iloc[-1]
    vol_ma    = df_ta['Volume_MA'].iloc[-1]

    with sig_cols[0]:
        st.markdown(signal_card("RSI", f"{rsi_now:.1f}",
                                rsi_now < 30, rsi_now > 70), unsafe_allow_html=True)
    with sig_cols[1]:
        st.markdown(signal_card("MACD", f"{macd_now:.3f}",
                                macd_now > macd_sig, macd_now < macd_sig), unsafe_allow_html=True)
    with sig_cols[2]:
        above_ma = close_now > ma20_now and close_now > ma50_now
        below_ma = close_now < ma20_now and close_now < ma50_now
        st.markdown(signal_card("MA Trend", f"${close_now:.2f}",
                                above_ma, below_ma), unsafe_allow_html=True)
    with sig_cols[3]:
        near_upper = close_now > bb_upper * 0.98
        near_lower = close_now < bb_lower * 1.02
        st.markdown(signal_card("Bollinger", f"{((close_now - bb_lower)/(bb_upper - bb_lower)*100):.0f}%",
                                near_lower, near_upper), unsafe_allow_html=True)
    with sig_cols[4]:
        high_vol = vol_now > vol_ma * 1.2
        low_vol  = vol_now < vol_ma * 0.8
        st.markdown(signal_card("Volume", f"{vol_now/1e6:.1f}M",
                                high_vol, low_vol), unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# TAB 3 — AI PREDICTION
# ──────────────────────────────────────────────────────────────
with tab3:
    if not run_button and st.session_state.results is None:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;">
            <div style="font-size:4rem;margin-bottom:1rem;filter:drop-shadow(0 0 20px rgba(0,212,255,0.5))">🤖</div>
            <h2 style="font-family:'Space Grotesk',sans-serif;font-size:1.8rem;
                       background:linear-gradient(135deg,#00d4ff,#7c3aed);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                       background-clip:text;margin-bottom:0.5rem;">
                Ready to Predict
            </h2>
            <p style="color:#8892b0;font-size:1rem;max-width:500px;margin:0 auto 2rem;">
                Configure your model settings in the sidebar and click
                <strong style="color:#00d4ff;">Run Prediction</strong> to start the AI analysis.
            </p>
            <div style="display:flex;justify-content:center;gap:1rem;flex-wrap:wrap;">
        """, unsafe_allow_html=True)

        for arch in ["LSTM", "Bidirectional LSTM", "GRU", "Hybrid LSTM+GRU"]:
            st.markdown(f'<span class="stock-tag">🧠 {arch}</span>', unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    if run_button:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">
                <div class="section-title-icon">⚙️</div>
                Training Pipeline
            </div>
        </div>
        """, unsafe_allow_html=True)

        progress_bar = st.progress(0)
        status_text  = st.empty()

        # Step 1: Prepare features
        status_text.markdown('<p class="loading-text">⚙️ Preparing feature matrix...</p>', unsafe_allow_html=True)
        time.sleep(0.3)

        feature_cols = ['Close', 'High', 'Low', 'Open']
        if use_volume: feature_cols.append('Volume')
        if use_ma:     feature_cols += ['MA7', 'MA20', 'MA50']
        if use_rsi:    feature_cols.append('RSI')
        if use_macd:   feature_cols += ['MACD', 'MACD_signal']
        if use_bb:     feature_cols += ['BB_upper', 'BB_lower']
        if use_atr:    feature_cols.append('ATR')

        df_model = df_full[feature_cols].dropna()
        if len(df_model) < seq_length + 50:
            st.error("Not enough data. Try a longer period or shorter sequence length.")
            st.stop()

        progress_bar.progress(10)

        # Step 2: Scale
        status_text.markdown('<p class="loading-text">📐 Scaling data (MinMax)...</p>', unsafe_allow_html=True)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(df_model.values)
        progress_bar.progress(20)

        # Step 3: Sequences
        status_text.markdown('<p class="loading-text">🔀 Creating training sequences...</p>', unsafe_allow_html=True)
        X, y = prepare_sequences(scaled, seq_length)
        split = int(len(X) * 0.85)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        progress_bar.progress(30)

        # Step 4: Build model
        status_text.markdown(f'<p class="loading-text">🏗️ Building {model_type} architecture...</p>', unsafe_allow_html=True)
        model = build_model(model_type, seq_length, X.shape[2], units, dropout_rate, learning_rate)
        progress_bar.progress(40)

        # Step 5: Train
        status_text.markdown('<p class="loading-text">🚀 Training model...</p>', unsafe_allow_html=True)
        callbacks = [
            EarlyStopping(patience=12, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6)
        ]
        history = model.fit(
            X_train, y_train,
            epochs=epochs, batch_size=batch_size,
            validation_split=0.15, callbacks=callbacks,
            verbose=0
        )
        progress_bar.progress(75)

        # Step 6: Evaluate
        status_text.markdown('<p class="loading-text">📊 Evaluating model performance...</p>', unsafe_allow_html=True)
        y_pred_scaled = model.predict(X_test, verbose=0)

        # Inverse transform (only close price col = index 0)
        def inverse_close(arr_scaled):
            dummy = np.zeros((len(arr_scaled), scaled.shape[1]))
            dummy[:, 0] = arr_scaled.flatten()
            return scaler.inverse_transform(dummy)[:, 0]

        y_test_actual = inverse_close(y_test)
        y_pred_actual = inverse_close(y_pred_scaled)

        mae   = mean_absolute_error(y_test_actual, y_pred_actual)
        rmse  = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        mape  = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
        r2    = r2_score(y_test_actual, y_pred_actual)
        progress_bar.progress(85)

        # Step 7: Forecast
        status_text.markdown(f'<p class="loading-text">🔮 Forecasting {forecast_days} days ahead...</p>', unsafe_allow_html=True)
        last_seq     = scaled[-seq_length:].copy()
        future_preds = []

        for _ in range(forecast_days):
            inp      = last_seq.reshape(1, seq_length, scaled.shape[1])
            pred_s   = model.predict(inp, verbose=0)[0, 0]
            new_row  = last_seq[-1].copy()
            new_row[0] = pred_s
            last_seq = np.vstack([last_seq[1:], new_row])
            future_preds.append(pred_s)

        future_prices = inverse_close(np.array(future_preds))
        last_date      = df_model.index[-1]
        future_dates   = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)
        progress_bar.progress(100)
        status_text.markdown('<p style="color:#00ff88;font-weight:700;">✅ Training complete!</p>', unsafe_allow_html=True)

        # Store results
        st.session_state.results = {
            'history': history, 'model': model,
            'y_test_actual': y_test_actual, 'y_pred_actual': y_pred_actual,
            'future_dates': future_dates, 'future_prices': future_prices,
            'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2,
            'df_model': df_model, 'scaled': scaled,
            'test_index': df_model.index[split + seq_length:],
            'feature_cols': feature_cols
        }

    # Show results if available
    if st.session_state.results is not None:
        res = st.session_state.results

        # Prediction Banner
        future_avg  = float(np.mean(res['future_prices']))
        cur_pr      = float(df_full['Close'].iloc[-1])
        pred_change = (future_avg - cur_pr) / cur_pr * 100
        signal, spct, signal_cls = get_trend_signal(res['future_prices'], cur_pr)
        target_price = float(res['future_prices'][-1])

        st.markdown(f"""
        <div class="prediction-banner">
            <div class="prediction-label">AI Price Target ({forecast_days}D)</div>
            <div class="prediction-price">${target_price:.2f}</div>
            <div style="color:#8892b0;font-size:0.9rem;margin-top:0.3rem;position:relative;z-index:1;">
                from current <strong style="color:#00d4ff;">${cur_pr:.2f}</strong>
                &nbsp;→&nbsp;
                <strong style="color:{'#00ff88' if pred_change>=0 else '#ff4757'}">
                    {'▲' if pred_change>=0 else '▼'} {abs(pred_change):.2f}%
                </strong>
            </div>
            <div style="margin-top:1rem;position:relative;z-index:1;">
                <span class="signal-badge {signal_cls}">{signal}</span>
            </div>
            <div class="prediction-confidence">
                Model Accuracy: R² = {res['r2']:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(metric_card("📏", "MAE", f"${res['mae']:.2f}", card_class="blue"), unsafe_allow_html=True)
        with m2:
            st.markdown(metric_card("📐", "RMSE", f"${res['rmse']:.2f}", card_class="purple"), unsafe_allow_html=True)
        with m3:
            st.markdown(metric_card("📊", "MAPE", f"{res['mape']:.2f}%", card_class="gold"), unsafe_allow_html=True)
        with m4:
            st.markdown(metric_card("🎯", "R² Score", f"{res['r2']:.4f}", card_class="green"), unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Forecast Chart
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><div class="section-title-icon">🔮</div>Price Forecast</div>', unsafe_allow_html=True)

        df_hist_plot = df_full['Close'].iloc[-120:] if len(df_full) >= 120 else df_full['Close']

        fig_pred = go.Figure()

        # Historical
        fig_pred.add_trace(go.Scatter(
            x=df_hist_plot.index, y=df_hist_plot.values,
            name="Historical", line=dict(color='#00d4ff', width=2),
            fill='tozeroy', fillcolor='rgba(0,212,255,0.05)'
        ))

        # Test predictions
        test_idx = res['test_index']
        if len(test_idx) == len(res['y_pred_actual']):
            fig_pred.add_trace(go.Scatter(
                x=test_idx, y=res['y_pred_actual'],
                name="Model Fit", line=dict(color='#ffd700', width=2, dash='dot')
            ))

        # Confidence band for forecast
        lower = res['future_prices'] * (1 - res['mape']/100)
        upper = res['future_prices'] * (1 + res['mape']/100)

        fig_pred.add_trace(go.Scatter(
            x=list(res['future_dates']) + list(res['future_dates'])[::-1],
            y=list(upper) + list(lower)[::-1],
            name="Confidence Band", fill='toself',
            fillcolor='rgba(0,255,136,0.08)',
            line=dict(color='rgba(0,255,136,0)', width=0),
            hoverinfo='skip'
        ))

        fig_pred.add_trace(go.Scatter(
            x=res['future_dates'], y=res['future_prices'],
            name=f"Forecast ({forecast_days}D)",
            line=dict(color='#00ff88', width=2.5),
            mode='lines+markers',
            marker=dict(size=4, color='#00ff88',
                        line=dict(color='#0a0e1a', width=1))
        ))

        # Current price line
        fig_pred.add_hline(
            y=cur_pr, line_dash="dash",
            line_color="rgba(255,215,0,0.5)",
            annotation_text=f"Current: ${cur_pr:.2f}",
            annotation_font_color="#ffd700"
        )

        fig_pred.update_layout(**PLOTLY_LAYOUT, height=480,
                               title=dict(text="", font=dict(size=0)))
        st.plotly_chart(fig_pred, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

        # Actual vs Predicted
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><div class="section-title-icon">📊</div>Actual vs Predicted (Test Set)</div>', unsafe_allow_html=True)

        fig_avp = go.Figure()
        test_idx = res['test_index']
        n_pts    = min(len(test_idx), len(res['y_test_actual']), len(res['y_pred_actual']))

        fig_avp.add_trace(go.Scatter(
            x=test_idx[:n_pts], y=res['y_test_actual'][:n_pts],
            name="Actual", line=dict(color='#00d4ff', width=2)
        ))
        fig_avp.add_trace(go.Scatter(
            x=test_idx[:n_pts], y=res['y_pred_actual'][:n_pts],
            name="Predicted", line=dict(color='#ff6b35', width=2, dash='dot')
        ))

        error_arr = res['y_pred_actual'][:n_pts] - res['y_test_actual'][:n_pts]
        fig_avp.add_trace(go.Bar(
            x=test_idx[:n_pts], y=error_arr, name="Error",
            marker_color=['rgba(0,255,136,0.4)' if e >= 0 else 'rgba(255,71,87,0.4)' for e in error_arr],
            yaxis='y2', opacity=0.5
        ))

        fig_avp.update_layout(
            **PLOTLY_LAYOUT, height=380,
            yaxis2=dict(overlaying='y', side='right', showgrid=False,
                        tickfont=dict(color='#8892b0'))
        )
        st.plotly_chart(fig_avp, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

        # Forecast table
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><div class="section-title-icon">📋</div>Forecast Table</div>', unsafe_allow_html=True)

        forecast_df = pd.DataFrame({
            'Date':          pd.DatetimeIndex(res['future_dates']).strftime('%Y-%m-%d'),
            'Predicted Price': [f"${p:.2f}" for p in res['future_prices']],
            'Change from Today': [f"{'+'if p>=cur_pr else ''}{(p-cur_pr)/cur_pr*100:.2f}%" for p in res['future_prices']],
            'Lower Bound':   [f"${p*(1-res['mape']/100):.2f}" for p in res['future_prices']],
            'Upper Bound':   [f"${p*(1+res['mape']/100):.2f}" for p in res['future_prices']],
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# TAB 4 — PERFORMANCE
# ──────────────────────────────────────────────────────────────
with tab4:
    if st.session_state.results is None:
        st.info("Run the AI Prediction first to see performance metrics.")
    else:
        res = st.session_state.results

        col_l, col_r = st.columns(2)

        with col_l:
            # Training history
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title"><div class="section-title-icon">📉</div>Training Loss</div>', unsafe_allow_html=True)

            hist = res['history'].history
            ep_range = list(range(1, len(hist['loss']) + 1))

            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=ep_range, y=hist['loss'],
                                          name="Train Loss", line=dict(color='#00d4ff', width=2)))
            fig_loss.add_trace(go.Scatter(x=ep_range, y=hist['val_loss'],
                                          name="Val Loss", line=dict(color='#ff6b35', width=2, dash='dot')))
            fig_loss.update_layout(**PLOTLY_LAYOUT, height=300,
                                   xaxis_title="Epoch", yaxis_title="Loss (Huber)")
            st.plotly_chart(fig_loss, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

            # Scatter: actual vs predicted
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title"><div class="section-title-icon">🎯</div>Regression Plot</div>', unsafe_allow_html=True)

            n_pts = min(len(res['y_test_actual']), len(res['y_pred_actual']))
            fig_scat = go.Figure()
            fig_scat.add_trace(go.Scatter(
                x=res['y_test_actual'][:n_pts], y=res['y_pred_actual'][:n_pts],
                mode='markers',
                marker=dict(color='#7c3aed', size=5, opacity=0.7,
                            line=dict(color='rgba(124,58,237,0.2)', width=0.5)),
                name="Predictions"
            ))
            min_v = min(res['y_test_actual'][:n_pts].min(), res['y_pred_actual'][:n_pts].min())
            max_v = max(res['y_test_actual'][:n_pts].max(), res['y_pred_actual'][:n_pts].max())
            fig_scat.add_trace(go.Scatter(
                x=[min_v, max_v], y=[min_v, max_v],
                mode='lines', name="Perfect Fit",
                line=dict(color='#00ff88', width=2, dash='dash')
            ))
            fig_scat.update_layout(**PLOTLY_LAYOUT, height=300,
                                   xaxis_title="Actual Price",
                                   yaxis_title="Predicted Price")
            st.plotly_chart(fig_scat, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        with col_r:
            # MAE history
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title"><div class="section-title-icon">📊</div>MAE History</div>', unsafe_allow_html=True)

            fig_mae = go.Figure()
            if 'mae' in hist:
                fig_mae.add_trace(go.Scatter(x=ep_range, y=hist['mae'],
                                              name="Train MAE", line=dict(color='#a78bfa', width=2)))
            if 'val_mae' in hist:
                fig_mae.add_trace(go.Scatter(x=ep_range, y=hist['val_mae'],
                                              name="Val MAE", line=dict(color='#ffd700', width=2, dash='dot')))
            fig_mae.update_layout(**PLOTLY_LAYOUT, height=300,
                                   xaxis_title="Epoch", yaxis_title="MAE")
            st.plotly_chart(fig_mae, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

            # Error distribution
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title"><div class="section-title-icon">📈</div>Error Distribution</div>', unsafe_allow_html=True)

            errors = res['y_pred_actual'][:n_pts] - res['y_test_actual'][:n_pts]
            fig_err = go.Figure()
            fig_err.add_trace(go.Histogram(
                x=errors, nbinsx=40,
                marker_color='rgba(0,212,255,0.6)',
                marker_line_color='rgba(0,212,255,0.2)',
                marker_line_width=0.5, name="Errors"
            ))
            fig_err.add_vline(x=0, line_dash="dash", line_color="#00ff88",
                               annotation_text="Zero Error", annotation_font_color="#00ff88")
            fig_err.update_layout(**PLOTLY_LAYOUT, height=300,
                                   xaxis_title="Prediction Error ($)",
                                   yaxis_title="Frequency")
            st.plotly_chart(fig_err, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        # Model architecture summary
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><div class="section-title-icon">🏗️</div>Model Architecture</div>', unsafe_allow_html=True)

        arch_cols = st.columns(3)
        with arch_cols[0]:
            st.markdown(f"""
            <div class="arch-card">
                <div class="arch-icon">🧠</div>
                <div>
                    <div class="arch-name">{model_type}</div>
                    <div class="arch-detail">Architecture Type</div>
                </div>
            </div>
            <div class="arch-card">
                <div class="arch-icon">📐</div>
                <div>
                    <div class="arch-name">{units} → {units//2} → {units//4}</div>
                    <div class="arch-detail">Layer Units</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with arch_cols[1]:
            total_params = res['model'].count_params()
            st.markdown(f"""
            <div class="arch-card">
                <div class="arch-icon">🔢</div>
                <div>
                    <div class="arch-name">{total_params:,}</div>
                    <div class="arch-detail">Total Parameters</div>
                </div>
            </div>
            <div class="arch-card">
                <div class="arch-icon">⏱️</div>
                <div>
                    <div class="arch-name">{len(hist['loss'])} / {epochs}</div>
                    <div class="arch-detail">Epochs Trained</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with arch_cols[2]:
            st.markdown(f"""
            <div class="arch-card">
                <div class="arch-icon">📦</div>
                <div>
                    <div class="arch-name">{batch_size}</div>
                    <div class="arch-detail">Batch Size</div>
                </div>
            </div>
            <div class="arch-card">
                <div class="arch-icon">🎛️</div>
                <div>
                    <div class="arch-name">{len(res['feature_cols'])} features</div>
                    <div class="arch-detail">Input Dimensions</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# TAB 5 — STOCK INFO
# ──────────────────────────────────────────────────────────────
with tab5:
    info = st.session_state.info
    name = info.get('name', ticker)

    st.markdown(f"""
    <div class="section-card">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:1.5rem;">
            <div style="width:56px;height:56px;background:linear-gradient(135deg,rgba(0,212,255,0.2),rgba(124,58,237,0.2));
                        border:1px solid rgba(0,212,255,0.3);border-radius:14px;
                        display:flex;align-items:center;justify-content:center;font-size:1.8rem;">📊</div>
            <div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:1.5rem;font-weight:700;color:#e8eaf6;">{name}</div>
                <div style="color:#8892b0;font-size:0.9rem;">{ticker} &nbsp;·&nbsp; {info.get('sector','N/A')}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    info_col1, info_col2, info_col3 = st.columns(3)

    def info_row(label, value):
        return f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:8px 0;border-bottom:1px solid rgba(0,212,255,0.08);">
            <span style="color:#8892b0;font-size:0.82rem;">{label}</span>
            <span style="color:#e8eaf6;font-weight:600;font-size:0.85rem;">{value}</span>
        </div>
        """

    with info_col1:
        st.markdown("**📈 Valuation**")
        pe   = info.get('pe_ratio', 0)
        mc   = info.get('market_cap', 0)
        div  = info.get('dividend_yield', 0)
        beta = info.get('beta', 0)
        st.markdown(info_row("Market Cap",      format_market_cap(mc) if mc else "N/A"), unsafe_allow_html=True)
        st.markdown(info_row("P/E Ratio",       f"{pe:.2f}" if pe else "N/A"), unsafe_allow_html=True)
        st.markdown(info_row("Dividend Yield",  f"{div*100:.2f}%" if div else "N/A"), unsafe_allow_html=True)
        st.markdown(info_row("Beta",            f"{beta:.2f}" if beta else "N/A"), unsafe_allow_html=True)

    with info_col2:
        st.markdown("**🎯 Price Levels**")
        hi52 = info.get('week_52_high', 0)
        lo52 = info.get('week_52_low', 0)
        st.markdown(info_row("52W High",      f"${hi52:.2f}" if hi52 else "N/A"), unsafe_allow_html=True)
        st.markdown(info_row("52W Low",       f"${lo52:.2f}" if lo52 else "N/A"), unsafe_allow_html=True)
        st.markdown(info_row("Current",       f"${current_price:.2f}"), unsafe_allow_html=True)
        if hi52 and lo52:
            pos = (current_price - lo52) / (hi52 - lo52) * 100
            st.markdown(info_row("52W Position", f"{pos:.1f}%"), unsafe_allow_html=True)

    with info_col3:
        st.markdown("**📊 Statistics**")
        vol_avg = info.get('volume_avg', 0)
        st.markdown(info_row("Avg Volume",    f"{vol_avg/1e6:.2f}M" if vol_avg >= 1e6 else f"{vol_avg:,}"), unsafe_allow_html=True)
        st.markdown(info_row("Last Volume",   f"{df_full['Volume'].iloc[-1]/1e6:.2f}M"), unsafe_allow_html=True)
        st.markdown(info_row("Period Return", f"{(df_full['Close'].iloc[-1]/df_full['Close'].iloc[0]-1)*100:.2f}%"), unsafe_allow_html=True)
        st.markdown(info_row("Data Points",   f"{len(df_full):,}"), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><div class="section-title-icon">🔥</div>Feature Correlation Matrix</div>', unsafe_allow_html=True)

    corr_cols  = ['Close', 'Volume', 'MA20', 'MA50', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'ATR']
    corr_data  = df_full[corr_cols].dropna()
    corr_matrix = corr_data.corr()

    fig_corr = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.columns.tolist(),
        colorscale=[[0,'#ff4757'], [0.5,'#141b2d'], [1,'#00d4ff']],
        zmid=0, text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>'
    ))
    fig_corr.update_layout(**PLOTLY_LAYOUT, height=400)
    st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

    # Returns distribution
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><div class="section-title-icon">📊</div>Daily Returns Distribution</div>', unsafe_allow_html=True)

    returns = df_full['Daily_Return'].dropna() * 100
    fig_ret = go.Figure()
    fig_ret.add_trace(go.Histogram(
        x=returns, nbinsx=60, name="Daily Returns",
        marker_color='rgba(124,58,237,0.6)',
        marker_line_color='rgba(124,58,237,0.2)',
        marker_line_width=0.5
    ))
    fig_ret.add_vline(x=float(returns.mean()), line_dash="dash",
                       line_color="#00d4ff",
                       annotation_text=f"Mean: {returns.mean():.3f}%",
                       annotation_font_color="#00d4ff")
    fig_ret.update_layout(**PLOTLY_LAYOUT, height=300,
                           xaxis_title="Daily Return (%)", yaxis_title="Frequency")
    st.plotly_chart(fig_ret, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
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
                Built with Keras · Streamlit · Plotly
            </span>
        </div>
        <div style="color:#2d3748;font-size:0.75rem;text-align:right;">
            ⚠️ For educational & research purposes only &nbsp;·&nbsp;
            Not financial advice &nbsp;·&nbsp;
            Past performance ≠ future results
        </div>
    </div>
</div>
""", unsafe_allow_html=True)