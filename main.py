import pandas as pd
import numpy as np
import datetime
import yfinance as yf

# Configuration
ETF_SYMBOL = '159941.SZ'
NDX_SYMBOL = '^NDX'
NQ_SYMBOL = 'NQ=F'
VIX_SYMBOL = '^VIX'
FX_SYMBOL = 'USDCNY=X'
LOOKBACK_DAYS = 60

def fetch_data(symbol, start, end, interval='1d'):
    return yf.download(symbol, start=start, end=end, interval=interval, progress=False)

def prepare_features(df_etf, df_ndx, df_nq, df_vix, df_fx):
    df = pd.DataFrame(index=df_etf.index)
    df['ETF_close'] = df_etf['Close']
    df['NDX'] = df_ndx['Close']
    df['NQ'] = df_nq['Close']
    df['VIX'] = df_vix['Close']
    df['USDCNY'] = df_fx['Close']
    df['ETF_prev'] = df['ETF_close'].shift(1)
    df['FX_ret'] = df['USDCNY'].pct_change()
    df['volume_change'] = df_etf['Volume'].pct_change()
    df.dropna(inplace=True)
    return df

def rolling_predict_np(df, lookback=LOOKBACK_DAYS):
    features = ['NDX', 'NQ', 'ETF_prev', 'FX_ret', 'volume_change']
    preds = []

    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i]
        X = window[features].values
        # add intercept
        X_design = np.hstack([np.ones((X.shape[0], 1)), X])
        y = window['ETF_close'].values
        
        # numpy least squares
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        
        # today's features with intercept
        x_today = np.hstack([1, df.iloc[i][features].values])
        pred = x_today.dot(beta)
        preds.append(pred)
    
    df_pred = df.iloc[lookback:].copy()
    df_pred['pred_open'] = preds
    return df_pred

# Main
today = datetime.date.today()
start = today - datetime.timedelta(days=LOOKBACK_DAYS*2)

df_etf = fetch_data(ETF_SYMBOL, start, today)
df_ndx = fetch_data(NDX_SYMBOL, start, today)
df_nq  = fetch_data(NQ_SYMBOL, start, today)
df_vix = fetch_data(VIX_SYMBOL, start, today)
df_fx  = fetch_data(FX_SYMBOL, start, today)

df = prepare_features(df_etf, df_ndx, df_nq, df_vix, df_fx)
df_pred = rolling_predict_np(df)

latest = df_pred.iloc[-1]
print(f"Date: {latest.name.date()}")
print(f"Predicted ETF Open (滚动OLS via numpy): {latest['pred_open']:.4f}")
