import streamlit as st
import os
import pickle
import warnings
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from datetime import datetime, timedelta
from GoogleNews import GoogleNews
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import difflib
import pandas_ta as ta
import traceback
import sys
import time

warnings.filterwarnings("ignore")

# ---------------------------
# Configuration and Constants
# ---------------------------

st.set_page_config(page_title="Indian Stock Movement Prediction", page_icon="📈", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 30

XGB_PATH = "xgboost_models"
os.makedirs(XGB_PATH, exist_ok=True)

# Feature columns must exactly match model training (37 features for 30 days = 1110 features)
# FEATURE_COLS = [
#     # Basic OHLCV (5)
#     "Open", "High", "Low", "Close", "Volume",
    
#     # Sentiment features (5) 
#     "avg_positive", "avg_neutral", "avg_negative", "news_count", "sentiment_std",
    
#     # Core technical indicators (27)
#     "SMA_5", "SMA_10", "SMA_20", "EMA_12", "EMA_26", "RSI",
#     "MACD", "MACD_signal", "MACD_hist",
#     "BB_upper", "BB_middle", "BB_lower", "OBV", "VWAP",
#     "Returns", "Log_Returns", "High_Low_Ratio","BBB_20_2.0","BBP_20_2.0",
#     "Price_Change_1d", "Price_Change_3d", "Price_Change_5d", 
#     "Volume_SMA_10", "Volume_Ratio",
#     "Sentiment_Trend_3d", "Sentiment_Trend_7d",
#     "CCI", "Williams_R", "ATR"
# ]
FEATURE_COLS = [
    'Close', 'High', 'Low', 'Open', 'Volume', 
    'avg_positive', 'avg_neutral', 'avg_negative', 'news_count', 'sentiment_std', 
    'SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26', 'RSI', 
    'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 
    'BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 
    'OBV', 'VWAP', 'Returns', 'Log_Returns', 'High_Low_Ratio', 
    'Sentiment_Trend_3d', 'Sentiment_Trend_7d', 'Sentiment_Volatility', 
    'News_Volume_MA', 'News_Volume_Spike', 'Extreme_Positive', 'Extreme_Negative'
    # Add these 3 missing features that were in training:
    # (You'll need to identify what these are from your training output)
]



AVAILABLE_FINBERT_MODELS = {
    "Indian Fine-tuned": "./finbert-indian-sentiment",
    "ProsusAI FinBERT": "ProsusAI/finbert",
    "YiYang FinBERT-Tone": "yiyanghkust/finbert-tone"
}

# Example ticker-to-company mapping (extend as needed)
TICKER_TO_COMPANY = {
    'ADANIENT.NS': 'Adani Enterprises',
    'ADANIPORTS.NS': 'Adani Ports and Special Economic Zone',
    'APOLLOHOSP.NS': 'Apollo Hospitals',
    'ASIANPAINT.NS': 'Asian Paints',
    'AXISBANK.NS': 'Axis Bank',
    'BAJAJ-AUTO.NS': 'Bajaj Auto',
    'BAJAJFINSV.NS': 'Bajaj Finserv',
    'BAJFINANCE.NS': 'Bajaj Finance',
    'BEL.NS': 'Bharat Electronics',
    'BHARTIARTL.NS': 'Bharti Airtel',
    'BPCL.NS': 'Bharat Petroleum Corporation',
    'BRITANNIA.NS': 'Britannia Industries',
    'CIPLA.NS': 'Cipla',
    'COALINDIA.NS': 'Coal India',
    'DIVISLAB.NS': "Divi's Laboratories",
    'DRREDDY.NS': "Dr. Reddy's Laboratories",
    'EICHERMOT.NS': 'Eicher Motors',
    'GRASIM.NS': 'Grasim Industries',
    'HCLTECH.NS': 'HCL Technologies',
    'HDFCBANK.NS': 'HDFC Bank',
    'HDFCLIFE.NS': 'HDFC Life Insurance',
    'HEROMOTOCO.NS': 'Hero MotoCorp',
    'HINDALCO.NS': 'Hindalco Industries',
    'HINDUNILVR.NS': 'Hindustan Unilever',
    'ICICIBANK.NS': 'ICICI Bank',
    'INDUSINDBK.NS': 'IndusInd Bank',
    'INFY.NS': 'Infosys',
    'ITC.NS': 'ITC',
    'JIOFIN.NS': 'Jio Financial Services',
    'JSWSTEEL.NS': 'JSW Steel',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank',
    'LT.NS': 'Larsen & Toubro',
    'M&M.NS': 'Mahindra & Mahindra',
    'MARUTI.NS': 'Maruti Suzuki India',
    'NESTLEIND.NS': 'Nestle India',
    'NTPC.NS': 'NTPC',
    'ONGC.NS': 'Oil & Natural Gas Corporation',
    'POWERGRID.NS': 'Power Grid Corporation',
    'RELIANCE.NS': 'Reliance Industries',
    'SBILIFE.NS': 'SBI Life Insurance',
    'SBIN.NS': 'State Bank of India',
    'SHRIRAMFIN.NS': 'Shriram Finance',
    'SUNPHARMA.NS': 'Sun Pharmaceutical Industries',
    'TATACONSUM.NS': 'Tata Consumer Products',
    'TATAMOTORS.NS': 'Tata Motors',
    'TATASTEEL.NS': 'Tata Steel',
    'TCS.NS': 'Tata Consultancy Services',
    'TECHM.NS': 'Tech Mahindra',
    'TITAN.NS': 'Titan Company',
    'TRENT.NS': 'Trent',
    'ULTRACEMCO.NS': 'UltraTech Cement',
    'WIPRO.NS': 'Wipro'
}

# ---------------------------
# Utility and Helper Functions
# ---------------------------

def last_trading_day(date=None):
    """Returns the last trading day that is not a weekend."""
    if date is None:
        date = datetime.today().date()
    while date.weekday() >= 5:
        date -= timedelta(days=1)
    return date

def load_finbert_model(model_path):
    """Load FinBERT tokenizer and model with error handling."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, use_safetensors=True, trust_remote_code=False)
        except:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, trust_remote_code=False)
        model = model.to(DEVICE).eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading FinBERT model from {model_path}: {e}")
        return None, None

def load_xgb(ticker):
    """Load XGBoost model for the given ticker, with fallback path tries."""
    model_name = f"{ticker.replace('.NS','')}_xgboost_model.pkl"
    model_path = os.path.join(XGB_PATH, model_name)
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            st.info(f"Loaded XGBoost model: {model_name}")
            print("Features",model.get_booster().feature_names)
            return model
        except Exception as e:
            st.error(f"Error loading model from {model_path}: {e}")
    else:
        st.warning(f"Model not found for ticker {ticker} at path {model_path}")
    return None

def get_last_30_days_comprehensive_data(ticker):
    """Fetch stock data for ~60 days (to warm up indicators), add sentiment defaults and features."""
    end_date = last_trading_day()
    start_date = end_date - timedelta(days=60)
    st.write(f"Fetching data for {ticker} from {start_date} to {end_date}")
    try:
        data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'),
                           end=end_date.strftime('%Y-%m-%d'), interval='1d',
                           auto_adjust=True, progress=False)
        if data.empty:
            st.warning("No price data available.")
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.tail(60).reset_index()
        data['Ticker'] = ticker

        # Add default neutral sentiment columns
        for col in ["avg_positive", "avg_neutral", "avg_negative", "news_count", "sentiment_std"]:
            data[col] = 0.0

        data = create_comprehensive_features_exact(data)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def get_latest_headlines(company, days=1):
    """Fetch latest news headline titles for the company using GoogleNews."""
    try:
        gnews = GoogleNews(lang="en", period=f"{days}d")
        gnews.search(company + " Company News")
        time.sleep(1)  # to avoid rate limiting
        return [r["title"] for r in gnews.results() if "title" in r]
    except Exception as e:
        st.warning(f"Error fetching news for {company}: {e}")
        return []

def finbert_scores(titles, tokenizer, model):
    """Tokenize and get sentiment scores as a numpy array."""
    if not titles:
        return np.zeros((0, 3))
    batch_size = 32
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(titles), batch_size):
            batch = titles[i:i+batch_size]
            toks = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            toks = {k: v.to(DEVICE) for k, v in toks.items()}
            model = model.to(DEVICE)
            logits = model(**toks).logits
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)

def get_sentiment_features_for_today(ticker, model, tokenizer):
    """Fetch headlines & compute FinBERT sentiment features."""
    company = TICKER_TO_COMPANY.get(ticker, ticker)
    headlines = get_latest_headlines(company)
    if not headlines:
        # Return neutral defaults if no news
        return {'avg_positive': 33.33, 'avg_neutral': 33.33, 'avg_negative': 33.33,
                'news_count': 0, 'sentiment_std': 0}

    probs = finbert_scores(headlines, tokenizer, model)
    if len(probs) == 0:
        return {'avg_positive': 33.33, 'avg_neutral': 33.33, 'avg_negative': 33.33,
                'news_count': 0, 'sentiment_std': 0}

    return {
        'avg_positive': probs[:, 0].mean() * 100,
        'avg_negative': probs[:, 1].mean() * 100,
        'avg_neutral': probs[:, 2].mean() * 100,
        'news_count': len(probs),
        'sentiment_std': np.std(probs[:, 0] - probs[:, 1])
    }

# def create_comprehensive_features_exact(df):
#     """Add technical indicators matching model training."""
#     try:
#         df = df.copy()
#         df['Date'] = pd.to_datetime(df['Date'])
#         df = df.sort_values('Date').set_index('Date')
#         for col in ['High', 'Low', 'Close', 'Volume']:
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#         df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

#         # Calculate indicators
#         df['SMA_5'] = ta.sma(df['Close'], 5)
#         df['SMA_10'] = ta.sma(df['Close'], 10)
#         df['SMA_20'] = ta.sma(df['Close'], 20)
#         df['EMA_12'] = ta.ema(df['Close'], 12)
#         df['EMA_26'] = ta.ema(df['Close'], 26)
#         df['RSI'] = ta.rsi(df['Close'], 14)

#         macd = ta.macd(df['Close'])
#         df['MACD'] = macd['MACD_12_26_9'].fillna(0)
#         df['MACD_signal'] = macd['MACDs_12_26_9'].fillna(0)
#         df['MACD_hist'] = macd['MACDh_12_26_9'].fillna(0)

#         bb = ta.bbands(df['Close'], 20)
#         df['BB_upper'] = bb['BBU_20_2.0']
#         df['BB_middle'] = bb['BBM_20_2.0']
#         df['BB_lower'] = bb['BBL_20_2.0']

#         df['OBV'] = ta.obv(df['Close'], df['Volume'])
#         df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)

#         tp = (df['High'] + df['Low'] + df['Close']) / 3
#         df['VWAP'] = (tp * df['Volume']).cumsum() / df['Volume'].cumsum()

#         df['Returns'] = df['Close'].pct_change()
#         df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
#         df['High_Low_Ratio'] = df['High'] / df['Low']
#         df['Price_Change_1d'] = df['Close'].pct_change(1)
#         df['Price_Change_3d'] = df['Close'].pct_change(3)
#         df['Price_Change_5d'] = df['Close'].pct_change(5)
#         df['Volume_SMA_10'] = ta.sma(df['Volume'], 10)
#         df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10']

#         if 'avg_positive' in df.columns and 'avg_negative' in df.columns:
#             df['Sentiment_Trend_3d'] = (df['avg_positive'].rolling(3).mean() -
#                                        df['avg_negative'].rolling(3).mean())
#             df['Sentiment_Trend_7d'] = (df['avg_positive'].rolling(7).mean() -
#                                        df['avg_negative'].rolling(7).mean())

#         df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], 20)
#         df['Williams_R'] = ta.willr(df['High'], df['Low'], df['Close'], 14)

#         df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
#         return df.reset_index()
#     except Exception:
#         st.error("Error in feature creation")
#         traceback.print_exc(file=sys.stdout)
#         return pd.DataFrame()
def create_comprehensive_features_exact(df):
    """Add technical indicators EXACTLY matching model training."""
    try:
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')
        
        # Clean data
        for col in ['High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Basic indicators - EXACTLY as in training
        df['SMA_5'] = ta.sma(df['Close'], length=5)
        df['SMA_10'] = ta.sma(df['Close'], length=10) 
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['EMA_12'] = ta.ema(df['Close'], length=12)
        df['EMA_26'] = ta.ema(df['Close'], length=26)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # MACD - use the SAME method as training
        df.ta.macd(close='Close', append=True)
        df.ta.bbands(close='Close', length=20, append=True)
        
        # Other indicators
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['High_Low_Ratio'] = df['High'] / df['Low']
        
        # Fill NaN values for technical indicators
        technical_cols = ['SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26', 'RSI', 
                         'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 'BBU_20_2.0', 'BBM_20_2.0', 
                         'BBL_20_2.0', 'OBV', 'VWAP', 'Returns', 'Log_Returns']
        
        for col in technical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(df[col].mean())
        
        # Sentiment features - EXACTLY as in training
        if 'avg_positive' in df.columns:
            df['Sentiment_Trend_3d'] = (df['avg_positive'].rolling(3).mean() - df['avg_negative'].rolling(3).mean())
            df['Sentiment_Trend_7d'] = (df['avg_positive'].rolling(7).mean() - df['avg_negative'].rolling(7).mean())
            df['Sentiment_Volatility'] = ((df['avg_positive'] - df['avg_negative']).rolling(5).std())
            df['News_Volume_MA'] = df['news_count'].rolling(5).mean()
            df['News_Volume_Spike'] = (df['news_count'] > df['News_Volume_MA'] * 1.5).astype(int)
            df['Extreme_Positive'] = (df['avg_positive'] > 70).astype(int)
            df['Extreme_Negative'] = (df['avg_negative'] > 70).astype(int)
            
            sentiment_derived_cols = ['Sentiment_Trend_3d', 'Sentiment_Trend_7d', 'Sentiment_Volatility', 'News_Volume_MA']
            for col in sentiment_derived_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
        
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        return df.reset_index()
        
    except Exception as e:
        st.error(f"Error in feature creation: {e}")
        traceback.print_exc()
        return pd.DataFrame()

# ---------------------------
# Prediction Functions
# ---------------------------

# def predict_with_fresh_data(ticker, finbert_model, tokenizer):
#     """Fetch recent data, append sentiment, compute features, and predict."""
#     recent_data = get_last_30_days_comprehensive_data(ticker)
#     if recent_data.empty or len(recent_data) < SEQUENCE_LENGTH:
#         raise ValueError(f"Insufficient data: {len(recent_data)} rows, need at least {SEQUENCE_LENGTH}")

#     sentiment_features = get_sentiment_features_for_today(ticker, finbert_model, tokenizer)

#     last_idx = recent_data.index[-1]
#     for key, value in sentiment_features.items():
#         recent_data.at[last_idx, key] = value

#     recent_data = create_comprehensive_features_exact(recent_data)

#     model = load_xgb(ticker)
#     if model is None:
#         raise ValueError(f"No trained model found for {ticker}")

#     missing_cols = [col for col in FEATURE_COLS if col not in recent_data.columns]
#     if missing_cols:
#         raise ValueError(f"Missing required features: {missing_cols}")

#     sequence = recent_data.tail(SEQUENCE_LENGTH)[FEATURE_COLS].values.flatten().reshape(1, -1)

#     prediction = model.predict(sequence)[0]
#     probability = model.predict_proba(sequence)[0]
#     return prediction, probability, sentiment_features, recent_data
def predict_with_fresh_data(ticker, finbert_model, tokenizer):
    """Fetch recent data, append sentiment, compute features, and predict."""
    recent_data = get_last_30_days_comprehensive_data(ticker)
    if recent_data.empty or len(recent_data) < SEQUENCE_LENGTH:
        raise ValueError(f"Insufficient data: {len(recent_data)} rows, need at least {SEQUENCE_LENGTH}")

    sentiment_features = get_sentiment_features_for_today(ticker, finbert_model, tokenizer)
    last_idx = recent_data.index[-1]
    for key, value in sentiment_features.items():
        recent_data.at[last_idx, key] = value

    recent_data = create_comprehensive_features_exact(recent_data)
    
    # DEBUG: Check available features
    available_features = [col for col in FEATURE_COLS if col in recent_data.columns]
    missing_features = [col for col in FEATURE_COLS if col not in recent_data.columns]
    
    st.write(f"Available features ({len(available_features)}): {available_features}")
    st.write(f"Missing features ({len(missing_features)}): {missing_features}")
    st.write(f"Total columns in data: {recent_data.columns.tolist()}")
    
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    sequence = recent_data.tail(SEQUENCE_LENGTH)[FEATURE_COLS].values.flatten().reshape(1, -1)
    st.write(f"Sequence shape: {sequence.shape} (expected: (1, 1170))")
    
    model = load_xgb(ticker)
    if model is None:
        raise ValueError(f"No trained model found for {ticker}")
        
    prediction = model.predict(sequence)[0]
    probability = model.predict_proba(sequence)[0]
    
    return prediction, probability, sentiment_features, recent_data


def predict_next_day_streamlit(ticker, tokenizer, finbert):
    """Run prediction with Streamlit UI updates."""
    st.subheader(f"Processing {ticker}...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("📊 Fetching data and sentiment...")
        progress_bar.progress(30)

        pred, proba, sent, recent_data = predict_with_fresh_data(ticker, finbert, tokenizer)

        progress_bar.progress(80)

        # Display latest OHLCV
        latest = recent_data.iloc[-1]
        st.markdown("#### Latest Stock Data")
        st.dataframe(pd.DataFrame({
            'Metric': ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
            'Value': [latest['Date'].strftime('%Y-%m-%d'), f"₹{latest['Open']:.2f}",
                      f"₹{latest['High']:.2f}", f"₹{latest['Low']:.2f}", f"₹{latest['Close']:.2f}",
                      f"{int(latest['Volume']):,}"]
        }), use_container_width=True)

        # Get news headlines
        headlines = get_latest_headlines(TICKER_TO_COMPANY.get(ticker, ticker))

        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        return pred, proba, sent, headlines

    except Exception as e:
        status_text.error(f"Error: {e}")
        progress_bar.empty()
        return None, None, None, None

# ---------------------------
# Streamlit App UI
# ---------------------------

def main():
    st.title("📈 Indian Stock Movement Prediction")
    st.markdown("Predict next-day stock movements using news sentiment and technical indicators.")

    # Sidebar: Model selection
    st.sidebar.header("⚙️ Model Configuration")
    model_name = st.sidebar.selectbox("Select FinBERT Model:", list(AVAILABLE_FINBERT_MODELS.keys()))
    model_path = AVAILABLE_FINBERT_MODELS[model_name]

    with st.spinner(f"Loading {model_name}..."):
        tokenizer, finbert = load_finbert_model(model_path)

    if tokenizer is None or finbert is None:
        st.error(f"Failed to load FinBERT model {model_name}")
        st.stop()
    else:
        st.sidebar.success(f"{model_name} loaded")

    # Sidebar: XGBoost models info
    st.sidebar.header("Trained XGBoost Models")
    if os.path.exists(XGB_PATH):
        models = [f for f in os.listdir(XGB_PATH) if f.endswith('_xgboost_model.pkl')]
        st.sidebar.info(f"{len(models)} pre-trained models found.")
        if st.sidebar.button("Show Model List"):
            with st.sidebar.expander("Model List", expanded=True):
                for m in sorted(models):
                    m_ticker = m.replace('_xgboost_model.pkl', '') + '.NS'
                    name = TICKER_TO_COMPANY.get(m_ticker, m_ticker)
                    st.write(f"• {name} ({m_ticker})")
    else:
        st.sidebar.warning("No XGBoost model directory found.")

    # Search & predict
    search_query = st.text_input("Search Company or Ticker", placeholder="e.g. Reliance, INFY")
    if search_query:
        matches = search_tickers(search_query)
        if matches:
            options = [f"{company} ({ticker})" for ticker, company, _ in matches]
            selected = st.radio("Select a company:", options)
            selected_ticker = selected.split("(")[-1].strip(")")

            model_available = load_xgb(selected_ticker) is not None
            if model_available:
                st.success(f"Model found for {selected_ticker}")
            else:
                st.warning(f"No model available for {selected_ticker}")

            if st.button("Predict Next-Day Movement", disabled=not model_available):
                pred, proba, sent, headlines = predict_next_day_streamlit(selected_ticker, tokenizer, finbert)
                if pred is not None:
                    st.markdown("---")
                    st.header("Prediction Results")
                    direction = "📈 UP" if pred == 1 else "📉 DOWN"
                    confidence = max(proba) * 100
                    st.success(f"Direction: {direction} with confidence {confidence:.1f}%")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("UP Probability", f"{proba[1]*100:.1f}%")
                    col2.metric("DOWN Probability", f"{proba[0]*100:.1f}%")

                    st.markdown("### Sentiment Analysis")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Positive", f"{sent['avg_positive']:.1f}%")
                    col2.metric("Neutral", f"{sent['avg_neutral']:.1f}%")
                    col3.metric("Negative", f"{sent['avg_negative']:.1f}%")
                    col4.metric("News Count", int(sent['news_count']))

                    if headlines:
                        with st.expander("Latest News Headlines", expanded=False):
                            for i, headline in enumerate(headlines[:10], 1):
                                st.write(f"{i}. {headline}")
                else:
                    st.error("Prediction failed.")
        else:
            st.warning("No matches found.")

    st.markdown("---")
    st.caption("You can search by company name (e.g., 'Reliance') or ticker symbol (e.g., 'INFY.NS').")

def search_tickers(query):
    """Search tickers and companies, including fuzzy matches."""
    if not query:
        return []
    query = query.lower()
    results = []
    for ticker, company in TICKER_TO_COMPANY.items():
        if query in ticker.lower():
            results.append((ticker, company, "ticker"))
        elif query in company.lower():
            results.append((ticker, company, "company"))

    if not results:
        # Fuzzy matches
        companies = list(TICKER_TO_COMPANY.values())
        tickers = list(TICKER_TO_COMPANY.keys())
        fuzzy_companies = difflib.get_close_matches(query, companies, n=5, cutoff=0.3)
        for fc in fuzzy_companies:
            ticker = [k for k, v in TICKER_TO_COMPANY.items() if v == fc][0]
            results.append((ticker, fc, "fuzzy_company"))
        fuzzy_tickers = difflib.get_close_matches(query, tickers, n=5, cutoff=0.3)
        for ft in fuzzy_tickers:
            results.append((ft, TICKER_TO_COMPANY[ft], "fuzzy_ticker"))

    return results[:10]

if __name__ == "__main__":
    main()
