# requirements: pip install yfinance akshare fire loguru pandas numpy

from dataclasses import dataclass
from typing import Optional
import datetime

import pandas as pd
import numpy as np
import fire
from loguru import logger


@dataclass
class PredictorConfig:
    etf_symbol: str = "159941"
    ndx_symbol: str = "^NDX"
    nq_symbol: str = "NQ=F"
    vix_symbol: str = "^VIX"
    fx_symbol: str = "USDCNY=X"
    lookback_days: int = 60
    source: str = "yfinance"  # or "akshare"
    interval: str = "1d"


class ETFOpenPredictor:
    def __init__(self, config: Optional[PredictorConfig] = None):
        self.cfg = config or PredictorConfig()
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="INFO")

    def fetch_yfinance(self, symbol: str, start: datetime.date, end: datetime.date):
        import yfinance as yf
        logger.info(f"[fetch_yfinance] symbol={symbol}, start={start}, end={end}, interval={self.cfg.interval}")
        df = yf.download(symbol,
                         start=start,
                         end=end,
                         interval=self.cfg.interval,
                         auto_adjust=False,
                         progress=False)
        logger.info(f"[fetch_yfinance] {symbol} rows={len(df)}")
        return df

    def fetch_akshare(self, symbol: str, start: datetime.date, end: datetime.date):
        import akshare as ak
        logger.info(f"[fetch_akshare] symbol={symbol}, start={start}, end={end}")
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust=""
        )
        if df.empty:
            logger.warning(f"[fetch_akshare] no data for {symbol}")
            return df
        df = (
            df.rename(columns={
                "日期": "Date",
                "开盘": "Open",
                "收盘": "Close",
                "最高": "High",
                "最低": "Low",
                "成交量": "Volume"
            })
              .set_index("Date")
        )
        df.index = pd.to_datetime(df.index)
        logger.info(f"[fetch_akshare] {symbol} rows={len(df)}")
        return df

    def fetch_data(self, symbol, start: datetime.date, end: datetime.date):
        sym_str = str(symbol)
        logger.info(f"[fetch_data] source={self.cfg.source}, symbol={sym_str}")
        if self.cfg.source == "akshare" and sym_str.isdigit():
            df = self.fetch_akshare(sym_str, start, end)
            if not df.empty:
                return df
            logger.info(f"[fetch_data] fallback to yfinance for {sym_str}")
        return self.fetch_yfinance(sym_str, start, end)

    def prepare_features(self, df_etf, df_ndx, df_nq, df_vix, df_fx):
        logger.info(f"[prepare_features] lookback_days={self.cfg.lookback_days}")
        df = pd.DataFrame(index=df_etf.index)
        df["ETF_close"] = df_etf["Close"]
        df["NDX"] = df_ndx["Close"].reindex(df.index).ffill()
        df["NQ"] = df_nq["Close"].reindex(df.index).ffill()
        df["VIX"] = df_vix["Close"].reindex(df.index).ffill()
        df["USDCNY"] = df_fx["Close"].reindex(df.index).ffill()
        df["ETF_prev"] = df["ETF_close"].shift(1)
        df["FX_ret"] = df["USDCNY"].pct_change(fill_method=None)
        df["volume_change"] = df_etf["Volume"].pct_change(fill_method=None)
        df.dropna(inplace=True)
        logger.info(f"[prepare_features] final rows={len(df)}, columns={list(df.columns)}")
        return df

    def rolling_predict(self, df: pd.DataFrame):
        logger.info(f"[rolling_predict] start predicting, rows={len(df)}")
        features = ["NDX", "NQ", "ETF_prev", "FX_ret", "volume_change"]
        preds = []
        for i in range(self.cfg.lookback_days, len(df)):
            window = df.iloc[i - self.cfg.lookback_days : i]
            X = window[features].values
            X_design = np.hstack([np.ones((X.shape[0], 1)), X])
            y = window["ETF_close"].values
            beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
            x_today = np.hstack([1, df.iloc[i][features].values])
            preds.append(float(x_today.dot(beta)))
        df_pred = df.iloc[self.cfg.lookback_days :].copy()
        df_pred["pred_open"] = preds
        logger.info(f"[rolling_predict] predictions count={len(preds)}")
        return df_pred

    def run(self, 
            source: Optional[str] = None,
            lookback_days: Optional[int] = None,
            etf_symbol: Optional[str] = None):
        """Predict ETF open price.

        Args:
          source: 'yfinance' or 'akshare'
          lookback_days: rolling window size
          etf_symbol: ETF code without suffix for akshare
        """
        if source:
            self.cfg.source = source
        if lookback_days:
            self.cfg.lookback_days = lookback_days
        if etf_symbol:
            self.cfg.etf_symbol = str(etf_symbol)
        logger.info(f"[run] config: {self.cfg}")

        today = datetime.date.today()
        start = today - datetime.timedelta(days=self.cfg.lookback_days * 2)

        df_etf = self.fetch_data(self.cfg.etf_symbol, start, today)
        df_ndx = self.fetch_data(self.cfg.ndx_symbol, start, today)
        if df_etf.empty or df_ndx.empty:
            logger.error("[run] ETF or NDX data missing, aborting")
            return

        df_nq = self.fetch_data(self.cfg.nq_symbol, start, today)
        df_vix = self.fetch_data(self.cfg.vix_symbol, start, today)
        df_fx = self.fetch_data(self.cfg.fx_symbol, start, today)

        df = self.prepare_features(df_etf, df_ndx, df_nq, df_vix, df_fx)
        if df.empty:
            logger.error("[run] No sufficient data after preprocessing")
            return

        df_pred = self.rolling_predict(df)
        latest = df_pred.iloc[-1]
        logger.info(f"[run] Date: {latest.name.date()}")
        logger.info(f"[run] Predicted ETF Open: {latest['pred_open']:.4f}")


if __name__ == "__main__":
    fire.Fire(ETFOpenPredictor)
