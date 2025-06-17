from dataclasses import dataclass, asdict
from typing import Optional
import datetime
import pandas as pd
import numpy as np
import fire
from loguru import logger

# Configure loguru to output JSON
logger.remove()
logger.add("log.json", rotation="1 MB", serialize=True, level="INFO")

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
        logger.info("config", **asdict(self.cfg))

    def fetch_yfinance(self, symbol: str, start: datetime.date, end: datetime.date):
        import yfinance as yf
        logger.info("fetch_yfinance.start", symbol=symbol, start=str(start), end=str(end), interval=self.cfg.interval)
        df = yf.download(symbol, start=start, end=end, interval=self.cfg.interval,
                         auto_adjust=False, progress=False)
        logger.info("fetch_yfinance.end", symbol=symbol, rows=len(df))
        return df

    def fetch_akshare(self, symbol: str, start: datetime.date, end: datetime.date):
        import akshare as ak
        logger.info("fetch_akshare.start", symbol=symbol, start=str(start), end=str(end))
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust=""
        )
        if df.empty:
            logger.warning("fetch_akshare.empty", symbol=symbol)
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
        logger.info("fetch_akshare.end", symbol=symbol, rows=len(df))
        return df

    def fetch_data(self, symbol, start: datetime.date, end: datetime.date):
        sym_str = str(symbol)
        logger.info("fetch_data", source=self.cfg.source, symbol=sym_str)
        if self.cfg.source == "akshare" and sym_str.isdigit():
            df = self.fetch_akshare(sym_str, start, end)
            if not df.empty:
                return df
            logger.info("fetch_data.fallback", symbol=sym_str)
        return self.fetch_yfinance(sym_str, start, end)

    def prepare_features(self, df_etf, df_ndx, df_nq, df_vix, df_fx):
        logger.info("prepare_features.start", lookback_days=self.cfg.lookback_days)

        idx = df_etf.index
        fx_shift = df_fx["Close"].reindex(idx).shift(1)

        def ensure_series(df_col):
            if isinstance(df_col, pd.Series):
                return df_col
            elif isinstance(df_col, pd.DataFrame):
                return df_col.iloc[:, 0]
            else:
                raise ValueError("Unexpected type for df_col")

        ndx_close = ensure_series(df_ndx["Close"])
        nq_close = ensure_series(df_nq["Close"])

        ndx_cny = (ndx_close * fx_shift).reindex(idx)
        nq_cny  = (nq_close * fx_shift).reindex(idx)

        df = pd.DataFrame(index=idx)
        df["NDX_ret"] = np.log(ndx_cny).diff()
        df["NQ_ret"]  = np.log(nq_cny).diff()
        df["VIX_ret"] = np.log(ensure_series(df_vix["Close"]).reindex(idx)).diff()
        df["ETF_close_log"] = np.log(df_etf["Close"].shift(1))
        df["ETF_open_log"] = np.log(df_etf["Open"])

        df.dropna(inplace=True)
        logger.info("prepare_features.end", rows=len(df), columns=list(df.columns))
        return df

    def rolling_predict(self, df: pd.DataFrame):
        logger.info("rolling_predict.start", rows=len(df), lookback_days=self.cfg.lookback_days)
        features = ["NDX_ret", "NQ_ret", "VIX_ret", "ETF_close_log"]
        preds = []

        for i in range(self.cfg.lookback_days, len(df)):
            window = df.iloc[i - self.cfg.lookback_days : i]
            X = window[features].values
            X_design = np.hstack([np.ones((X.shape[0], 1)), X])
            y = window["ETF_open_log"].values

            beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
            x_today = np.hstack([1, df.iloc[i][features].values])
            pred_log = x_today.dot(beta)
            preds.append(float(np.exp(pred_log)))

        df_pred = df.iloc[self.cfg.lookback_days :].copy()
        df_pred["pred_open"] = preds
        logger.info("rolling_predict.end", predictions=len(preds))
        return df_pred

    def run(self, source: Optional[str] = None,
            lookback_days: Optional[int] = None,
            etf_symbol: Optional[str] = None):

        if source:
            self.cfg.source = source
        if lookback_days:
            self.cfg.lookback_days = lookback_days
        if etf_symbol:
            self.cfg.etf_symbol = str(etf_symbol)

        logger.info("run.start", **asdict(self.cfg))

        today = datetime.date.today()
        start = today - datetime.timedelta(days=self.cfg.lookback_days * 2)

        df_etf = self.fetch_data(self.cfg.etf_symbol, start, today)
        df_ndx = self.fetch_data(self.cfg.ndx_symbol, start, today)
        df_nq = self.fetch_data(self.cfg.nq_symbol, start, today)
        df_vix = self.fetch_data(self.cfg.vix_symbol, start, today)
        df_fx = self.fetch_data(self.cfg.fx_symbol, start, today)

        if df_etf.empty or df_ndx.empty:
            logger.error("run.error", message="ETF or NDX data missing")
            return

        df = self.prepare_features(df_etf, df_ndx, df_nq, df_vix, df_fx)
        if df.empty:
            logger.error("run.error", message="Insufficient data after preprocessing")
            return

        df_pred = self.rolling_predict(df)
        latest = df_pred.iloc[-1]
        result = {"date": str(latest.name.date()), "pred_open": latest['pred_open']}
        logger.info("run.end", **result)

        with open("result.json", "w") as f:
            import json
            json.dump({**asdict(self.cfg), **result}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    fire.Fire(ETFOpenPredictor)