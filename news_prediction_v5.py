# --------------------------------------------------------------------------------
# Complete Streamlit News Prediction App - FIXED VERSION
# --------------------------------------------------------------------------------
import streamlit as st
import os, pickle, warnings, yfinance as yf, numpy as np, pandas as pd, torch, xgboost as xgb
from datetime import datetime, timedelta
from GoogleNews import GoogleNews
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import difflib
import pandas_ta as ta
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="Indian Stock Movement Prediction", page_icon="📈", layout="wide")

# --------------------------------------------------------------------------------
# Global Config - FIXED
# --------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_ORDER = ["positive", "negative", "neutral"]
SEQUENCE_LENGTH = 30

# Exact feature list to match your trained models (47 features = 1410 / 30)
# Update FEATURE_COLS to match training (37 features)
# CORRECTED: Use exactly 37 features to match your trained models (37 × 30 = 1110)
FEATURE_COLS_CORRECT = [
    # Basic OHLCV (5)
    "Open", "High", "Low", "Close", "Volume",
    
    # Sentiment features (5) 
    "avg_positive", "avg_neutral", "avg_negative", "news_count", "sentiment_std",
    
    # Core technical indicators (27)
    "SMA_5", "SMA_10", "SMA_20", "EMA_12", "EMA_26", "RSI",
    "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_middle", "BB_lower", "OBV", "VWAP",
    "Returns", "Log_Returns", "High_Low_Ratio",
    "Price_Change_1d", "Price_Change_3d", "Price_Change_5d", 
    "Volume_SMA_10", "Volume_Ratio",
    "Sentiment_Trend_3d", "Sentiment_Trend_7d",
    "CCI", "Williams_R", "ATR"
]

# Replace your current FEATURE_COLS with this
FEATURE_COLS = FEATURE_COLS_CORRECT

XGB_PATH = "xgboost_models"
os.makedirs(XGB_PATH, exist_ok=True)

# Available FinBERT Models - FIXED SYNTAX
AVAILABLE_FINBERT_MODELS = {
    "Indian Fine-tuned": "./finbert-indian-sentiment",
    "ProsusAI FinBERT": "ProsusAI/finbert", 
    "YiYang FinBERT-Tone": "yiyanghkust/finbert-tone"
}

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

# --------------------------------------------------------------------------------
# Model Loading Functions - FIXED
# --------------------------------------------------------------------------------
@st.cache_resource
def load_finbert_model(model_path):
    """Load FinBERT model with proper device handling"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Try SafeTensors first, fallback to regular loading
        try:
            finbert = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                use_safetensors=True,
                trust_remote_code=False
            )
        except:
            finbert = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                trust_remote_code=False
            )
        
        # Ensure model is properly moved to device
        finbert = finbert.to(DEVICE).eval()
        
        return tokenizer, finbert
        
    except Exception as e:
        st.error(f"Error loading FinBERT model from {model_path}: {e}")
        return None, None

# def load_xgb(ticker: str):
#     """Load XGBoost model with correct path structure"""
#     model_filename = f"{ticker.replace('.NS', '')}_xgboost_model.pkl"
#     model_path = os.path.join(XGB_PATH, model_filename)
    
#     if os.path.exists(model_path):
#         with open(model_path, 'rb') as f:
#             return pickle.load(f)
#     else:
#         # Fallback: try the old naming convention
#         old_path = os.path.join(XGB_PATH, f"xgboost_model_{ticker.split('.')[0]}.pkl")
#         if os.path.exists(old_path):
#             with open(old_path, 'rb') as f:
#                 return pickle.load(f)
#     return None
def load_xgb(ticker: str):
    """Load XGBoost model with correct path structure and print model info"""
    model_filename = f"{ticker.replace('.NS', '')}_xgboost_model.pkl"
    model_path = os.path.join(XGB_PATH, model_filename)
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Print model loading info
        print(f"✅ Loaded model: {model_filename}")
        print(f"   Path: {model_path}")
        
        # Try to determine feature count
        try:
            # Test with different feature counts to determine expectations
            try:
                model.predict(np.zeros((1, 1410)))
                feature_expectation = "1410 features (47 per timestep)"
            except:
                try:
                    model.predict(np.zeros((1, 1110)))
                    feature_expectation = "1110 features (37 per timestep)"
                except:
                    feature_expectation = "Unknown feature count"
            
            print(f"   Model expects: {feature_expectation}")
            
            # Check if model has feature count attribute
            if hasattr(model, 'n_features_in_'):
                print(f"   n_features_in_: {model.n_features_in_}")
            
        except Exception as e:
            print(f"   ⚠️ Could not determine feature count: {e}")
        
        return model
        
    else:
        # Fallback: try the old naming convention
        old_path = os.path.join(XGB_PATH, f"xgboost_model_{ticker.split('.')[0]}.pkl")
        if os.path.exists(old_path):
            with open(old_path, 'rb') as f:
                model = pickle.load(f)
            
            print(f"✅ Loaded model (old format): xgboost_model_{ticker.split('.')[0]}.pkl")
            print(f"   Path: {old_path}")
            
            # Try to determine feature count for old format too
            try:
                try:
                    model.predict(np.zeros((1, 1410)))
                    feature_expectation = "1410 features (47 per timestep)"
                except:
                    try:
                        model.predict(np.zeros((1, 1110)))
                        feature_expectation = "1110 features (37 per timestep)"
                    except:
                        feature_expectation = "Unknown feature count"
                
                print(f"   Model expects: {feature_expectation}")
                
                if hasattr(model, 'n_features_in_'):
                    print(f"   n_features_in_: {model.n_features_in_}")
                
            except Exception as e:
                print(f"   ⚠️ Could not determine feature count: {e}")
            
            return model
    
    print(f"❌ No model found for {ticker}")
    print(f"   Tried: {model_path}")
    fallback_path = os.path.join(XGB_PATH, f"xgboost_model_{ticker.split('.')[0]}.pkl")
    print(f"   Also tried: {fallback_path}")
    return None


# --------------------------------------------------------------------------------
# Data Fetching Functions - FIXED  
# --------------------------------------------------------------------------------
def last_trading_day(date=None):
    if date is None:
        date = datetime.today().date()
    while date.weekday() >= 5:  # 5=Sat,6=Sun
        date -= timedelta(days=1)
    return date

def get_last_30_days_comprehensive_data(ticker):
    """Fetch last 30 days of stock data with features"""
    end_date = last_trading_day()
    start_date = end_date - timedelta(days=60)
    print(f"Fetching data for {ticker} from {start_date} to {end_date}")
    print(f"yfinance version: {yf.__version__}")

    
    try:
        stock_data = yf.download(
            ticker, 
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1d',
            auto_adjust=True,
            progress=False
        )
        print(stock_data.head())
        
        if stock_data.empty:
            print("Data is empty")
            return pd.DataFrame()
        
        # Handle MultiIndex columns
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)
        
        # Get last 30 trading days
        stock_data = stock_data.tail(60).reset_index()
        stock_data['Ticker'] = ticker
        print("Data for ticker")
        # Add default sentiment values
        stock_data['avg_positive'] = 33.33
        stock_data['avg_neutral'] = 33.33
        stock_data['avg_negative'] = 33.33
        stock_data['news_count'] = 0
        stock_data['sentiment_std'] = 0
        
        # Apply technical indicators
        stock_data = create_comprehensive_features_exact(stock_data)
        print("Features created")
        return stock_data
        
    except Exception as e:
        print("Error:",e)
        st.error(f"Error fetching 30-day data for {ticker}: {e}")
        return pd.DataFrame()

def get_latest_headlines(company: str, days: int = 1) -> list[str]:
    """Fetch latest news headlines"""
    try:
        gnews = GoogleNews(lang="en", period=f"{days}d")
        gnews.search(company + " Company News")
        return [r["title"] for r in gnews.results() if "title" in r]
    except Exception as e:
        st.warning(f"Error fetching news for {company}: {e}")
        return []

# --------------------------------------------------------------------------------
# Sentiment Analysis Functions - FIXED
# --------------------------------------------------------------------------------
def finbert_scores(titles: list[str], tokenizer, finbert_model) -> np.ndarray:
    """Return N×3 array of softmax probs with proper device handling"""
    if not titles:
        return np.zeros((0, 3))
    
    batch, out = 32, []
    
    with torch.no_grad():
        for i in range(0, len(titles), batch):
            batch_titles = titles[i:i+batch]
            
            # Tokenize and ensure ALL tensors go to the same device as model
            toks = tokenizer(
                batch_titles, 
                padding=True, 
                truncation=True,
                max_length=128, 
                return_tensors="pt"
            )
            
            # Move ALL tokenized tensors to the same device as the model
            toks = {k: v.to(DEVICE) for k, v in toks.items()}
            
            # Ensure model is also on DEVICE
            finbert_model = finbert_model.to(DEVICE)
            
            # Forward pass
            logits = finbert_model(**toks).logits
            probs = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
            out.append(probs)
    
    return np.vstack(out)

def get_sentiment_features_for_today(ticker: str, finbert_model, tokenizer, device, label_order) -> dict:
    """Get sentiment features with proper device handling"""
    company_name = TICKER_TO_COMPANY.get(ticker, ticker)
    titles = get_latest_headlines(company_name)
    
    if not titles:
        return dict(
            avg_positive=33.33, avg_neutral=33.33, avg_negative=33.33,
            news_count=0, sentiment_std=0
        )
    
    # Ensure model is on correct device before inference
    finbert_model = finbert_model.to(device)
    
    probs = finbert_scores(titles, tokenizer, finbert_model)
    
    if len(probs) == 0:
        return dict(
            avg_positive=33.33, avg_neutral=33.33, avg_negative=33.33,
            news_count=0, sentiment_std=0
        )
    
    return dict(
        avg_positive=probs[:, 0].mean() * 100,
        avg_neutral=probs[:, 2].mean() * 100,
        avg_negative=probs[:, 1].mean() * 100,
        news_count=len(probs),
        sentiment_std=np.std(probs[:, 0] - probs[:, 1])
    )

# --------------------------------------------------------------------------------
# Feature Engineering Functions - FIXED TO MATCH TRAINING
# --------------------------------------------------------------------------------
# def create_comprehensive_features_exact(df):
#     df = df.copy()
#     df['Date'] = pd.to_datetime(df['Date'])
#     df = df.sort_values('Date').set_index('Date')
    
#     # Core technical indicators (exactly matching training)
#     df['SMA_5'] = ta.sma(df['Close'], 5)
#     df['SMA_10'] = ta.sma(df['Close'], 10)
#     df['SMA_20'] = ta.sma(df['Close'], 20)
#     df['EMA_12'] = ta.ema(df['Close'], 12)
#     df['EMA_26'] = ta.ema(df['Close'], 26)
#     df['RSI'] = ta.rsi(df['Close'], 14)
    
#     # MACD
#     macd = ta.macd(df['Close'])
#     df['MACD'] = macd['MACD_12_26_9']
#     df['MACD_signal'] = macd['MACDs_12_26_9']
#     df['MACD_hist'] = macd['MACDh_12_26_9']
    
#     # Bollinger Bands
#     bb = ta.bbands(df['Close'], 20)
#     df['BB_upper'] = bb['BBU_20_2.0']
#     df['BB_middle'] = bb['BBM_20_2.0']
#     df['BB_lower'] = bb['BBL_20_2.0']
    
#     # Volume and other indicators
#     df['OBV'] = ta.obv(df['Close'], df['Volume'])
#     df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
    
#     # VWAP
#     tp = (df['High'] + df['Low'] + df['Close']) / 3
#     df['VWAP'] = (tp * df['Volume']).cumsum() / df['Volume'].cumsum()
    
#     # Price features
#     df['Returns'] = df['Close'].pct_change()
#     df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
#     df['High_Low_Ratio'] = df['High'] / df['Low']
#     df['Price_Change_1d'] = df['Close'].pct_change(1)
#     df['Price_Change_3d'] = df['Close'].pct_change(3)
#     df['Price_Change_5d'] = df['Close'].pct_change(5)
#     df['Volume_SMA_10'] = ta.sma(df['Volume'], 10)
#     df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10']
    
#     # Basic sentiment features (only the core ones used in training)
#     if 'avg_positive' in df.columns:
#         df['Sentiment_Trend_3d'] = (df['avg_positive'].rolling(3).mean() - 
#                                    df['avg_negative'].rolling(3).mean())
#         df['Sentiment_Trend_7d'] = (df['avg_positive'].rolling(7).mean() - 
#                                    df['avg_negative'].rolling(7).mean())
    
#     # Additional technical indicators
#     df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], 20)
#     df['Williams_R'] = ta.willr(df['High'], df['Low'], df['Close'], 14)
    
#     # Fill NaN values
#     for col in df.columns:
#         df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
#     return df.reset_index()
def create_comprehensive_features_exact(df):
    import numpy as np
    import pandas_ta as ta
    import sys
    import traceback

    try:
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')

        # Convert columns to numeric and fill missing values before TA calculations
        for col in ['High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Calculate technical indicators
        df['SMA_5'] = ta.sma(df['Close'], 5)
        df['SMA_10'] = ta.sma(df['Close'], 10)
        df['SMA_20'] = ta.sma(df['Close'], 20)
        df['EMA_12'] = ta.ema(df['Close'], 12)
        df['EMA_26'] = ta.ema(df['Close'], 26)
        df['RSI'] = ta.rsi(df['Close'], 14)

        print(df['Close'])
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9'].fillna(0)
        df['MACD_signal'] = macd['MACDs_12_26_9'].fillna(0)
        df['MACD_hist'] = macd['MACDh_12_26_9'].fillna(0)

        bb = ta.bbands(df['Close'], 20)
        df['BB_upper'] = bb['BBU_20_2.0']
        df['BB_middle'] = bb['BBM_20_2.0']
        df['BB_lower'] = bb['BBL_20_2.0']

        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)

        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (tp * df['Volume']).cumsum() / df['Volume'].cumsum()

        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Price_Change_1d'] = df['Close'].pct_change(1)
        df['Price_Change_3d'] = df['Close'].pct_change(3)
        df['Price_Change_5d'] = df['Close'].pct_change(5)
        df['Volume_SMA_10'] = ta.sma(df['Volume'], 10)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10']

        if 'avg_positive' in df.columns and 'avg_negative' in df.columns:
            df['Sentiment_Trend_3d'] = (df['avg_positive'].rolling(3).mean() -
                                       df['avg_negative'].rolling(3).mean())
            df['Sentiment_Trend_7d'] = (df['avg_positive'].rolling(7).mean() -
                                       df['avg_negative'].rolling(7).mean())

        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], 20)
        df['Williams_R'] = ta.willr(df['High'], df['Low'], df['Close'], 14)

        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        return df.reset_index()

    except Exception:
        print("Error in create_comprehensive_features_exact:")
        traceback.print_exc(file=sys.stdout)
        # Optionally return empty df on error or re-raise
        return pd.DataFrame()



# --------------------------------------------------------------------------------
# Ticker Search Functions
# --------------------------------------------------------------------------------
def search_tickers(query):
    """Search for tickers based on company name or ticker symbol"""
    if not query:
        return []
    
    query = query.lower()
    matches = []
    
    # Search in both ticker and company name
    for ticker, company in TICKER_TO_COMPANY.items():
        if query in ticker.lower():
            matches.append((ticker, company, "ticker"))
        elif query in company.lower():
            matches.append((ticker, company, "company"))
    
    # Use difflib for fuzzy matching if no direct matches
    if not matches:
        company_names = list(TICKER_TO_COMPANY.values())
        ticker_symbols = list(TICKER_TO_COMPANY.keys())
        
        company_matches = difflib.get_close_matches(query, company_names, n=5, cutoff=0.3)
        for match in company_matches:
            ticker = [k for k, v in TICKER_TO_COMPANY.items() if v == match][0]
            matches.append((ticker, match, "fuzzy_company"))
        
        ticker_matches = difflib.get_close_matches(query, ticker_symbols, n=5, cutoff=0.3)
        for match in ticker_matches:
            matches.append((match, TICKER_TO_COMPANY[match], "fuzzy_ticker"))
    
    return matches[:10]

# --------------------------------------------------------------------------------
# Prediction Functions - FIXED
# --------------------------------------------------------------------------------
def predict_with_fresh_data_fixed(ticker, finbert_model, tokenizer, device, label_order, 
                                 feature_cols, sequence_length):
    """Safe prediction with exact feature matching - FIXED INDEXING"""
    
    # Get last 30 days of data
    recent_data = get_last_30_days_comprehensive_data(ticker)
    
    if len(recent_data) < sequence_length:
        raise ValueError(f"Insufficient recent data: {len(recent_data)} < {sequence_length}")
    
    # Get today's sentiment
    sentiment_features = get_sentiment_features_for_today(ticker, finbert_model, tokenizer, device, label_order)
    
    # FIXED: Update last row with current sentiment using .at
    last_idx = recent_data.index[-1]
    recent_data.at[last_idx, 'avg_positive'] = sentiment_features['avg_positive']
    recent_data.at[last_idx, 'avg_neutral'] = sentiment_features['avg_neutral']
    recent_data.at[last_idx, 'avg_negative'] = sentiment_features['avg_negative']
    recent_data.at[last_idx, 'news_count'] = sentiment_features['news_count']
    recent_data.at[last_idx, 'sentiment_std'] = sentiment_features['sentiment_std']
    
    # Load model
    model = load_xgb(ticker)
    if model is None:
        raise ValueError(f"No trained model found for {ticker}")
    
    # Filter features to exactly match training (47 features)
    available_features = [col for col in feature_cols if col in recent_data.columns]
    
    if len(available_features) != 47:
        st.warning(f"Feature count mismatch: expected 47, got {len(available_features)}")
        # Take only the first 47 features
        available_features = available_features[:47]
    
    # Create sequence from recent data
    sequence = recent_data.tail(sequence_length)[available_features].values.flatten().reshape(1, -1)
    
    # Verify final shape
    expected_shape = 47 * sequence_length  # Should be 1410
    if sequence.shape[1] != expected_shape:
        raise ValueError(f"Final shape mismatch: expected {expected_shape}, got {sequence.shape[1]}")
    
    prediction = model.predict(sequence)[0]
    probability = model.predict_proba(sequence)[0]
    
    return prediction, probability, sentiment_features, recent_data

def predict_with_fresh_data_fixed_debug_v1(ticker, finbert_model, tokenizer, device, label_order, 
                                       feature_cols, sequence_length):
    """Debug version to track exactly where the error occurs"""
    
    print(f"🔍 DEBUG: Starting prediction for {ticker}")
    
    try:
        # Step 1: Get data
        print("🔍 DEBUG: Step 1 - Getting last 30 days data...")
        recent_data = get_last_30_days_comprehensive_data(ticker)
        print(f"🔍 DEBUG: Data shape: {recent_data.shape}")
        print(f"🔍 DEBUG: Data columns: {list(recent_data.columns)}")
        
        if len(recent_data) < sequence_length:
            raise ValueError(f"Insufficient recent data: {len(recent_data)} < {sequence_length}")
        
        # Step 2: Get sentiment
        print("🔍 DEBUG: Step 2 - Getting sentiment features...")
        sentiment_features = get_sentiment_features_for_today(ticker, finbert_model, tokenizer, device, label_order)
        print(f"🔍 DEBUG: Sentiment features: {sentiment_features}")
        
        # Step 3: Update sentiment values (THIS IS LIKELY WHERE ERROR OCCURS)
        print("🔍 DEBUG: Step 3 - Updating sentiment values...")
        print(f"🔍 DEBUG: recent_data.index type: {type(recent_data.index)}")
        print(f"🔍 DEBUG: recent_data.index: {recent_data.index}")
        
        try:
            last_idx = recent_data.index[-1]
            print(f"🔍 DEBUG: Last index: {last_idx}")
            
            print("🔍 DEBUG: Updating avg_positive...")
            recent_data.at[last_idx, 'avg_positive'] = sentiment_features['avg_positive']
            
            print("🔍 DEBUG: Updating avg_neutral...")
            recent_data.at[last_idx, 'avg_neutral'] = sentiment_features['avg_neutral']
            
            print("🔍 DEBUG: Updating avg_negative...")
            recent_data.at[last_idx, 'avg_negative'] = sentiment_features['avg_negative']
            
            print("🔍 DEBUG: Updating news_count...")
            recent_data.at[last_idx, 'news_count'] = sentiment_features['news_count']
            
            print("🔍 DEBUG: Updating sentiment_std...")
            recent_data.at[last_idx, 'sentiment_std'] = sentiment_features['sentiment_std']
            
            print("🔍 DEBUG: ✅ All sentiment updates completed")
            
        except Exception as e:
            print(f"❌ DEBUG ERROR in sentiment update: {e}")
            print(f"❌ DEBUG ERROR type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Step 4: Load model
        print("🔍 DEBUG: Step 4 - Loading XGBoost model...")
        model = load_xgb(ticker)
        if model is None:
            raise ValueError(f"No trained model found for {ticker}")
        print("🔍 DEBUG: ✅ Model loaded successfully")
        
        # Step 5: Prepare features
        print("🔍 DEBUG: Step 5 - Preparing features...")
        available_features = [col for col in feature_cols if col in recent_data.columns]
        print(f"🔍 DEBUG: Available features count: {len(available_features)}")
        
        if len(available_features) != 47:
            print(f"🔍 DEBUG: ⚠️ Feature count mismatch: expected 47, got {len(available_features)}")
            available_features = available_features[:47]
        
        # Step 6: Create sequence
        print("🔍 DEBUG: Step 6 - Creating sequence...")
        try:
            sequence_data = recent_data.tail(sequence_length)[available_features]
            print(f"🔍 DEBUG: Sequence data shape: {sequence_data.shape}")
            
            sequence = sequence_data.values.flatten().reshape(1, -1)
            print(f"🔍 DEBUG: Final sequence shape: {sequence.shape}")
            
        except Exception as e:
            print(f"❌ DEBUG ERROR in sequence creation: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Step 7: Make prediction
        print("🔍 DEBUG: Step 7 - Making prediction...")
        expected_shape = 47 * sequence_length
        if sequence.shape[1] != expected_shape:
            raise ValueError(f"Final shape mismatch: expected {expected_shape}, got {sequence.shape[1]}")
        
        prediction = model.predict(sequence)[0]
        probability = model.predict_proba(sequence)[0]
        
        print("🔍 DEBUG: ✅ Prediction completed successfully")
        return prediction, probability, sentiment_features, recent_data
        
    except Exception as e:
        print(f"❌ DEBUG: MAIN ERROR in predict_with_fresh_data_fixed: {e}")
        print(f"❌ DEBUG: ERROR type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise
def predict_with_fresh_data_fixed_debug(ticker, finbert_model, tokenizer, device, label_order, 
                                       feature_cols, sequence_length):
    """CORRECTED Debug version to track exactly where the error occurs"""
    
    print(f"🔍 DEBUG: Starting prediction for {ticker}")
    
    try:
        # Step 1: Get data
        print("🔍 DEBUG: Step 1 - Getting last 30 days data...")
        recent_data = get_last_30_days_comprehensive_data(ticker)
        print(f"🔍 DEBUG: Data shape: {recent_data.shape}")
        print(f"🔍 DEBUG: Data columns: {list(recent_data.columns)}")
        
        if len(recent_data) < sequence_length:
            raise ValueError(f"Insufficient recent data: {len(recent_data)} < {sequence_length}")
        
        # Step 2: Get sentiment
        print("🔍 DEBUG: Step 2 - Getting sentiment features...")
        sentiment_features = get_sentiment_features_for_today(ticker, finbert_model, tokenizer, device, label_order)
        print(f"🔍 DEBUG: Sentiment features: {sentiment_features}")
        
        # Step 3: Update sentiment values
        print("🔍 DEBUG: Step 3 - Updating sentiment values...")
        print(f"🔍 DEBUG: recent_data.index type: {type(recent_data.index)}")
        print(f"🔍 DEBUG: recent_data.index: {recent_data.index}")
        
        try:
            last_idx = recent_data.index[-1]
            print(f"🔍 DEBUG: Last index: {last_idx}")
            
            print("🔍 DEBUG: Updating avg_positive...")
            recent_data.at[last_idx, 'avg_positive'] = sentiment_features['avg_positive']
            
            print("🔍 DEBUG: Updating avg_neutral...")
            recent_data.at[last_idx, 'avg_neutral'] = sentiment_features['avg_neutral']
            
            print("🔍 DEBUG: Updating avg_negative...")
            recent_data.at[last_idx, 'avg_negative'] = sentiment_features['avg_negative']
            
            print("🔍 DEBUG: Updating news_count...")
            recent_data.at[last_idx, 'news_count'] = sentiment_features['news_count']
            
            print("🔍 DEBUG: Updating sentiment_std...")
            recent_data.at[last_idx, 'sentiment_std'] = sentiment_features['sentiment_std']
            
            print("🔍 DEBUG: ✅ All sentiment updates completed")
            
        except Exception as e:
            print(f"❌ DEBUG ERROR in sentiment update: {e}")
            print(f"❌ DEBUG ERROR type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Step 4: Load model and check expected features
        print("🔍 DEBUG: Step 4 - Loading XGBoost model...")
        model = load_xgb(ticker)
        if model is None:
            raise ValueError(f"No trained model found for {ticker}")
        
        # 🔧 NEW: Check model expected features
        print("🔍 DEBUG: Step 4.1 - Checking model feature expectations...")
        try:
            # Test with 1110 features (your actual trained models)
            test_seq_1110 = np.zeros((1, 1110))
            model.predict(test_seq_1110)
            expected_features = 1110
            expected_per_timestep = 37
            print(f"🔍 DEBUG: ✅ Model expects 1110 features (37 per timestep)")
        except:
            try:
                # Test with 1410 features (if you have newer models)
                test_seq_1410 = np.zeros((1, 1410))
                model.predict(test_seq_1410)
                expected_features = 1410
                expected_per_timestep = 47
                print(f"🔍 DEBUG: ✅ Model expects 1410 features (47 per timestep)")
            except Exception as e2:
                print(f"🔍 DEBUG: ❌ Could not determine model feature requirements: {e2}")
                raise ValueError(f"Cannot determine feature requirements for model {ticker}")
        
        print(f"🔍 DEBUG: ✅ Model loaded successfully - expects {expected_features} total features")
        
        # Step 5: Prepare features based on model expectations
        print("🔍 DEBUG: Step 5 - Preparing features...")
        
        # Define feature sets for different model types
        if expected_per_timestep == 37:
            # Legacy model feature set (1110 total)
            required_features = [
                "Open", "High", "Low", "Close", "Volume",
                "avg_positive", "avg_neutral", "avg_negative", "news_count", "sentiment_std",
                "SMA_5", "SMA_10", "SMA_20", "EMA_12", "EMA_26", "RSI",
                "MACD", "MACD_signal", "MACD_hist",
                "BB_upper", "BB_middle", "BB_lower", "OBV", "VWAP",
                "Returns", "Log_Returns", "High_Low_Ratio",
                "Price_Change_1d", "Price_Change_3d", "Price_Change_5d",
                "Volume_SMA_10", "Volume_Ratio",
                "Sentiment_Trend_3d", "Sentiment_Trend_7d",
                "CCI", "Williams_R", "ATR"
            ]
        else:
            # Newer model feature set (1410 total)
            required_features = feature_cols[:47]
        
        available_features = [col for col in required_features if col in recent_data.columns]
        print(f"🔍 DEBUG: Available features count: {len(available_features)}")
        print(f"🔍 DEBUG: Required features count: {expected_per_timestep}")
        
        if len(available_features) != expected_per_timestep:
            print(f"🔍 DEBUG: ⚠️ Feature count mismatch: expected {expected_per_timestep}, got {len(available_features)}")
            print(f"🔍 DEBUG: Missing features: {set(required_features) - set(available_features)}")
            # Take only the available features up to the required count
            available_features = available_features[:expected_per_timestep]
        
        # Step 6: Create sequence
        print("🔍 DEBUG: Step 6 - Creating sequence...")
        try:
            sequence_data = recent_data.tail(sequence_length)[available_features]
            print(f"🔍 DEBUG: Sequence data shape: {sequence_data.shape}")
            
            sequence = sequence_data.values.flatten().reshape(1, -1)
            print(f"🔍 DEBUG: Final sequence shape: {sequence.shape}")
            
        except Exception as e:
            print(f"❌ DEBUG ERROR in sequence creation: {e}")
            print(f"❌ DEBUG Available columns: {list(recent_data.columns)}")
            print(f"❌ DEBUG Required features: {available_features}")
            import traceback
            traceback.print_exc()
            raise
        
        # Step 7: Make prediction
        print("🔍 DEBUG: Step 7 - Making prediction...")
        
        if sequence.shape[1] != expected_features:
            raise ValueError(f"Final shape mismatch: expected {expected_features}, got {sequence.shape[1]}")
        
        prediction = model.predict(sequence)[0]
        probability = model.predict_proba(sequence)[0]
        
        print(f"🔍 DEBUG: ✅ Prediction completed successfully")
        print(f"🔍 DEBUG: Used {len(available_features)} features per timestep, {sequence.shape[1]} total")
        return prediction, probability, sentiment_features, recent_data
        
    except Exception as e:
        print(f"❌ DEBUG: MAIN ERROR in predict_with_fresh_data_fixed: {e}")
        print(f"❌ DEBUG: ERROR type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise




def predict_next_day_streamlit(ticker: str, tokenizer, finbert):
    """Streamlit version of prediction with UI updates"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Processing {ticker}")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("📊 Fetching latest data and analyzing...")
            progress_bar.progress(50)
            
            pred, proba, sent, recent_data = predict_with_fresh_data_fixed_debug(
                ticker, finbert, tokenizer, DEVICE, LABEL_ORDER, 
                FEATURE_COLS, SEQUENCE_LENGTH
            )
            
            progress_bar.progress(100)
            
            # Display OHLCV data
            st.markdown("#### 📊 Latest OHLCV Data")
            
            latest_price = recent_data.iloc[-1]
            
            ohlcv_display = pd.DataFrame({
                'Metric': ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
                'Value': [
                    latest_price['Date'].strftime('%Y-%m-%d') if hasattr(latest_price['Date'], 'strftime') else str(latest_price['Date']),
                    f"₹{float(latest_price['Open']):.2f}",
                    f"₹{float(latest_price['High']):.2f}",
                    f"₹{float(latest_price['Low']):.2f}",
                    f"₹{float(latest_price['Close']):.2f}",
                    f"{int(latest_price['Volume']):,}"
                ]
            })

            st.dataframe(ohlcv_display, use_container_width=True, hide_index=True)

            
            # Get headlines for display
            company_name = TICKER_TO_COMPANY.get(ticker, ticker)
            headlines = get_latest_headlines(company_name)
            
            progress_bar.empty()
            status_text.empty()
            
            return pred, proba, None, sent, headlines
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return None, None, None, None, None
    
    with col2:
        st.markdown("#### 🤖 Model Status")
        model_exists = load_xgb(ticker) is not None
        if model_exists:
            st.success("✅ Pre-trained model available")
        else:
            st.warning("⚠️ No existing model found")

# --------------------------------------------------------------------------------
# Main Streamlit App
# --------------------------------------------------------------------------------
def main():
    st.title("📈 Indian Stock Movement Prediction")
    st.markdown("Predict next-day stock movements using news sentiment analysis")
    st.markdown("---")
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Model Configuration")
    
    # FinBERT Model Selection
    st.sidebar.subheader("🤖 FinBERT Model Selection")
    selected_model_name = st.sidebar.selectbox(
        "Choose FinBERT model for sentiment analysis:",
        options=list(AVAILABLE_FINBERT_MODELS.keys()),
        index=0,
        help="Different models may give different sentiment analysis results"
    )
    
    selected_model_path = AVAILABLE_FINBERT_MODELS[selected_model_name]
    
    # Display model info
    with st.sidebar.expander("ℹ️ Model Information"):
        if selected_model_name == "Indian Fine-tuned":
            st.write("**Fine-tuned on Indian financial news data**")
            st.write("- Best for Indian market sentiment")
            st.write("- Trained on local financial terminology")
        elif "ProsusAI" in selected_model_name:
            st.write("**General financial sentiment model**")
            st.write("- Broad financial domain coverage")
            st.write("- Good for international context")
        elif "YiYang" in selected_model_name:
            st.write("**Academic research model**")
            st.write("- High-quality pre-training")
            st.write("- Good baseline performance")
    
    # Load selected model
    with st.spinner(f"🔄 Loading {selected_model_name}..."):
        tokenizer, finbert = load_finbert_model(selected_model_path)
    
    if tokenizer is None or finbert is None:
        st.error(f"❌ Failed to load {selected_model_name}. Please check the model path.")
        st.stop()
    else:
        st.sidebar.success(f"✅ {selected_model_name} loaded successfully")
    
    # XGBoost Model Status
    st.sidebar.subheader("🎯 XGBoost Models")
    
    if os.path.exists(XGB_PATH):
        model_files = [f for f in os.listdir(XGB_PATH) if f.endswith('_xgboost_model.pkl')]
        st.sidebar.info(f"📊 Found {len(model_files)} trained XGBoost models")
        
        if st.sidebar.button("🔍 View Available Models"):
            with st.sidebar.expander("Available XGBoost Models", expanded=True):
                for model_file in sorted(model_files):
                    ticker = model_file.replace('_xgboost_model.pkl', '') + '.NS'
                    company = TICKER_TO_COMPANY.get(ticker, ticker)
                    st.write(f"• {company} ({ticker})")
    else:
        st.sidebar.warning("⚠️ No XGBoost models directory found")
    
    # Main content
    st.header("🔍 Stock Search & Prediction")
    
    # Search box
    search_query = st.text_input(
        "🔍 Search for a company or ticker symbol",
        placeholder="e.g., 'Reliance', 'INFY', 'Tata', 'HDFC'...",
        help="Type company name or ticker symbol to find matches"
    )
    
    # Show search results
    if search_query:
        matches = search_tickers(search_query)
        
        if matches:
            st.markdown("#### 📋 Search Results")
            
            options = []
            for ticker, company, match_type in matches:
                if match_type == "ticker":
                    label = f"🎯 {ticker} - {company} (Ticker Match)"
                elif match_type == "company":
                    label = f"🏢 {ticker} - {company} (Company Match)"
                else:
                    label = f"🔍 {ticker} - {company} (Similar)"
                options.append((label, ticker))
            
            if options:
                selected_option = st.radio(
                    "Select a company:",
                    options=[opt[0] for opt in options],
                    key="ticker_selection"
                )
                
                selected_ticker = next(opt[1] for opt in options if opt[0] == selected_option)
                company_name = TICKER_TO_COMPANY[selected_ticker]
                
                # Show model status for this ticker
                model_available = load_xgb(selected_ticker) is not None
                if model_available:
                    st.success(f"**Selected:** {company_name} ({selected_ticker}) ✅ **Model Available**")
                else:
                    st.warning(f"**Selected:** {company_name} ({selected_ticker}) ⚠️ **No Trained Model**")
                
                st.info(f"**Sentiment Analysis:** Using {selected_model_name}")
                
                # Prediction button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔮 Predict Next-Day Movement", type="primary", use_container_width=True):
                        
                        result = predict_next_day_streamlit(selected_ticker, tokenizer, finbert)
                        
                        if result[0] is not None:
                            pred, proba, df_updated, sent, headlines = result
                            
                            # Display prediction results
                            st.markdown("---")
                            st.header("🎯 Prediction Results")
                            
                            st.caption(f"🤖 **Models Used:** {selected_model_name} (Sentiment) + XGBoost (Prediction)")
                            
                            # Main prediction
                            direction = "📈 UP" if pred == 1 else "📉 DOWN"
                            confidence = max(proba) * 100
                            
                            if pred == 1:
                                st.success(f"## {direction}")
                                st.success(f"**Confidence:** {confidence:.1f}%")
                            else:
                                st.error(f"## {direction}")
                                st.error(f"**Confidence:** {confidence:.1f}%")
                            
                            # Probability distribution
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("📈 UP Probability", f"{proba[1]*100:.1f}%")
                            with col2:
                                st.metric("📉 DOWN Probability", f"{proba[0]*100:.1f}%")
                            
                            # Sentiment analysis
                            st.markdown("#### 🎯 Sentiment Analysis")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("😊 Positive", f"{sent['avg_positive']:.1f}%")
                            with col2:
                                st.metric("😐 Neutral", f"{sent['avg_neutral']:.1f}%")
                            with col3:
                                st.metric("😟 Negative", f"{sent['avg_negative']:.1f}%")
                            with col4:
                                st.metric("📰 News Count", int(sent['news_count']))
                            
                            # News headlines
                            if headlines:
                                with st.expander("📰 Latest News Headlines", expanded=False):
                                    for i, headline in enumerate(headlines[:10], 1):
                                        st.write(f"{i}. {headline}")
                                    if len(headlines) > 10:
                                        st.write(f"*... and {len(headlines) - 10} more headlines*")
                            else:
                                st.info("📰 No recent news headlines found")
        else:
            st.warning(f"❓ No matches found for '{search_query}'. Try a different search term.")
    
    # Footer
    st.markdown("---")
    st.markdown("**💡 Tip:** You can search by company name (e.g., 'Reliance', 'Infosys') or ticker symbol (e.g., 'INFY.NS', 'TCS.NS')")

if __name__ == "__main__":
    main()
