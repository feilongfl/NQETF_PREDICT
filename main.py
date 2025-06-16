import argparse
import pandas as pd
import numpy as np
import datetime

def fetch_data_yfinance(symbol, start, end, interval='1d'):
    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, interval=interval,
                     auto_adjust=False, progress=False)
    return df

def fetch_data_akshare(symbol, start, end):
    import akshare as ak
    # 支持中国A股/ETF数据，symbol 格式如 '159941'
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start.strftime("%Y%m%d"),
                             end_date=end.strftime("%Y%m%d"), adjust="")
    if df.empty:
        return df
    # 重命名并设置索引
    df = df.rename(columns={
        "日期": "Date",
        "开盘": "Open",
        "收盘": "Close",
        "最高": "High",
        "最低": "Low",
        "成交量": "Volume"
    }).set_index("Date")
    # 转换为 datetime index
    df.index = pd.to_datetime(df.index)
    return df

def fetch_data(symbol, start, end, source, interval='1d'):
    """根据 source 选择数据源"""
    # 如果使用 akshare 并且是中国A股/ETF 代码
    if source == "akshare" and symbol.isdigit():
        df = fetch_data_akshare(symbol, start, end)
        if not df.empty:
            return df
        else:
            print(f"Warning: akshare 无数据，回退到 yfinance 获取 {symbol}")
    # 默认使用 yfinance
    return fetch_data_yfinance(symbol, start, end, interval)

def prepare_features(df_etf, df_ndx, df_nq, df_vix, df_fx):
    df = pd.DataFrame(index=df_etf.index)
    df['ETF_close'] = df_etf['Close']
    df['NDX']       = df_ndx['Close'].reindex(df.index).fillna(method='ffill')
    df['NQ']        = df_nq['Close'].reindex(df.index).fillna(method='ffill')
    df['VIX']       = df_vix['Close'].reindex(df.index).fillna(method='ffill')
    df['USDCNY']    = df_fx['Close'].reindex(df.index).fillna(method='ffill')
    df['ETF_prev']  = df['ETF_close'].shift(1)
    df['FX_ret']        = df['USDCNY'].pct_change(fill_method=None)
    df['volume_change'] = df_etf['Volume'].pct_change(fill_method=None)
    df.dropna(inplace=True)
    return df

def rolling_predict_np(df, lookback):
    features = ['NDX', 'NQ', 'ETF_prev', 'FX_ret', 'volume_change']
    preds = []
    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i]
        X = window[features].values
        X_design = np.hstack([np.ones((X.shape[0], 1)), X])
        y = window['ETF_close'].values
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        x_today = np.hstack([1, df.iloc[i][features].values])
        preds.append(x_today.dot(beta))
    df_pred = df.iloc[lookback:].copy()
    df_pred['pred_open'] = preds
    return df_pred

def main():
    parser = argparse.ArgumentParser(description="ETF 开盘价滚动OLS预测")
    parser.add_argument("--source", choices=["yfinance", "akshare"], default="akshare",
                        help="选择数据源：yfinance 或 akshare")
    args = parser.parse_args()

    # 配置
    ETF_SYMBOL = '159941'  # 对于 akshare，去掉交易所后缀
    NDX_SYMBOL = '^NDX'
    NQ_SYMBOL = 'NQ=F'
    VIX_SYMBOL = '^VIX'
    FX_SYMBOL = 'USDCNY=X'
    LOOKBACK_DAYS = 60

    today = datetime.date.today()
    start = today - datetime.timedelta(days=LOOKBACK_DAYS*2)

    # 下载数据
    df_etf = fetch_data(ETF_SYMBOL, start, today, args.source)
    df_ndx = fetch_data(NDX_SYMBOL, start, today, args.source)
    df_nq  = fetch_data(NQ_SYMBOL, start, today, args.source)
    df_vix = fetch_data(VIX_SYMBOL, start, today, args.source)
    df_fx  = fetch_data(FX_SYMBOL, start, today, args.source)

    # 检查
    if df_etf.empty or df_ndx.empty:
        print("Error: ETF 或 NDX 数据缺失，无法继续")
        return

    df = prepare_features(df_etf, df_ndx, df_nq, df_vix, df_fx)
    df_pred = rolling_predict_np(df, LOOKBACK_DAYS)
    if df_pred.empty:
        print("Error: 预处理后数据不足，无法预测")
        return

    latest = df_pred.iloc[-1]
    print(f"Date: {latest.name.date()}")
    print(f"Predicted ETF Open (源={args.source}): {latest['pred_open']:.4f}")

if __name__ == "__main__":
    main()
