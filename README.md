# News Sentiment for Indian Stock Movement Prediction

**M.Tech (AI & ML) Major Project, BITS Pilani (WILP)** · Abhishek Chatterjee

Does financial news sentiment help predict the next-day direction of Indian large-cap stocks? This project builds an end-to-end pipeline to test that question on NIFTY 50 companies. It fine-tunes FinBERT on Indian financial news, turns headline sentiment into daily time-series features, combines them with technical indicators, and trains a per-stock XGBoost classifier to predict whether the next day's close will be up or down. A Streamlit app runs the trained models on live prices and same-day Google News headlines.

---

## Pipeline

```
Kaggle India news headlines ──► filter business news ──► map to NIFTY 50 tickers
                                                                │
          FinBERT (fine-tuned on Indian financial news) ◄───────┘
                                │  positive / neutral / negative probabilities per headline
                                ▼
            daily sentiment aggregates  +  Yahoo Finance OHLCV  +  technical indicators
                                │
                                ▼
        30-day sliding windows (36 features × 30 days)  ──►  per-ticker XGBoost  ──►  P(up tomorrow)
```

### 1. News data
- Source: [India News Headlines dataset](https://www.kaggle.com/datasets/therohk/india-headlines-news-dataset) (~3.88M headlines).
- Filtered to business categories (~166K headlines), then matched to NIFTY 50 companies by name: **7,361 headlines, May 2001 – June 2023**.
- `additional_nifty50_news.csv` has more recent headlines for supplementing and testing.

### 2. Sentiment model
- Base models compared: `yiyanghkust/finbert-tone`, `ProsusAI/finbert`, `yiyanghkust/finbert-esg`.
- Fine-tuned on [`kdave/Indian_Financial_News`](https://huggingface.co/datasets/kdave/Indian_Financial_News) (3-class: negative / neutral / positive) with Hugging Face `Trainer` (lr 2e-5, 3 epochs, batch 16, max length 512). The goal is to adapt FinBERT, which was trained mainly on US and global financial text, to Indian market language.

### 3. Feature engineering (36 features per trading day)
| Group | Features |
|---|---|
| Price | Open, High, Low, Close, Volume |
| Daily sentiment | mean positive / neutral / negative probability, news count, sentiment std |
| Sentiment dynamics | 3-day and 7-day sentiment trend, 5-day sentiment volatility, news-volume moving average, news-volume spike flag, extreme positive / negative flags |
| Technical | SMA 5/10/20, EMA 12/26, RSI 14, MACD (line, signal, histogram), Bollinger Bands (upper, middle, lower, bandwidth, %B), OBV, VWAP, returns, log returns, high/low ratio |

Days without news are forward-filled, or set to neutral (33.3% each class) when there is no earlier value.

### 4. Model
- **Target:** `Price_Direction_1d`, which is 1 if the next day's close is above today's.
- **Input:** a flattened 30-day window (30 × 36 = 1,080 features).
- **Split:** chronological per ticker (70% train / 10% validation / 20% test) so that no future data leaks into training.
- **Classifier:** XGBoost (200 trees, depth 6, lr 0.1, subsample 0.8, early stopping on validation log-loss), one model per ticker.

### 5. App
`news_prediction_final.py` is a Streamlit app. You search for a company, pick a FinBERT variant, and it fetches the last ~60 days of prices and today's headlines. It then rebuilds the same features and shows the predicted direction, the class probabilities and the sentiment breakdown.

---

## Results

Test-set results across **47 NIFTY 50 tickers**:

| Metric | Value |
|---|---|
| Mean accuracy | **50.2%** |
| Median accuracy | 50.0% |
| Range | 45.8% – 54.7% |
| Mean weighted F1 | 0.478 |
| Tickers above 50% | 23 / 47 |

**Interpretation.** Next-day direction prediction comes out at roughly chance level. This matches much of the literature on daily-horizon prediction for liquid large-cap stocks, which behave close to the weak-form efficient-market hypothesis: by the next close, public headlines are largely priced in. The feature-importance analysis (`feature_importance_analysis.csv`) spreads importance thinly across many features. Technical features (EMA, Bollinger Bands, VWAP) and sentiment features (extreme-positive flags, sentiment volatility) both show up near the top, but no single signal dominates.

## Limitations and next steps
- **Sparse news coverage.** About 7.4K headlines across ~50 companies and 22 years leaves most trading days with no company-specific news, so the sentiment features are mostly imputed. Denser, timestamped news (intraday, with publication time relative to market close) is the most important improvement.
- **Ablation.** Train an identical price-only baseline to isolate how much sentiment adds, and report results against a majority-class baseline.
- **Evaluation.** Use walk-forward validation instead of a single split, and add trading-oriented metrics (directional returns, Sharpe) next to accuracy.
- **Horizon and target.** Try longer horizons (the pipeline already builds 5-day targets) and filter out near-zero moves, where up/down labels are mostly noise.
- **Pooled models.** A single cross-sectional model across tickers would see far more data than 47 separate small models.

---

## Repository structure
| File | Purpose |
|---|---|
| `data_processing.ipynb` | Main pipeline with outputs: data loading, FinBERT fine-tuning, features, XGBoost training, feature importance |
| `data_processing_no_output.ipynb` | Same pipeline without cell outputs |
| `Clean Code.ipynb` | Annotated end-to-end version of the pipeline |
| `Final_Code.ipnyb` | Earlier pipeline version (sentiment and features, before XGBoost) |
| `news_prediction_final.py` | Streamlit inference app |
| `news_prediction_v5.py` | Earlier version of the app |
| `feature_importance_analysis.csv` | Average XGBoost feature importance across tickers |
| `additional_nifty50_news.csv` | Supplementary recent NIFTY 50 headlines |
| `requirements.txt` | Python dependencies |

## Running it
```bash
pip install -r requirements.txt

# 1. Run data_processing.ipynb to download the data, fine-tune FinBERT
#    (saved to ./finbert-indian-sentiment) and train models (saved to ./xgboost_models)
# 2. Launch the app
streamlit run news_prediction_final.py
```
Trained model weights are not committed. The notebook regenerates them. The Kaggle dataset requires `kagglehub` with Kaggle credentials configured.

## Tech stack
Python · PyTorch · Hugging Face Transformers & Datasets · FinBERT · XGBoost · scikit-learn · pandas / pandas-ta · yfinance · GoogleNews · Streamlit

> This is an academic project and not investment advice.
