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
        # Log initial config
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
        """
        生成对齐后的特征。
        所有美股数据先换算成人民币，再取对数收益率；目标是次日 ETF 开盘价。
        """
        logger.info("prepare_features.start", lookback_days=self.cfg.lookback_days)

        # 对齐交易日索引（用 ETF 交易日作为主轴）
        idx = df_etf.index
        fx_shift = df_fx["Close"].reindex(idx).shift(1)      # 前一自然日汇率

        # 价格 → 人民币，再取 log-return（先对齐再 shift）
        ndx_cny = (df_ndx["Close"] * fx_shift).reindex(idx)
        nq_cny  = (df_nq["Close"]  * fx_shift).reindex(idx)

        df = pd.DataFrame(index=idx)

        # 特征：前一天到今天的 log-return
        df["NDX_ret"] = np.log(ndx_cny).diff()
        df["NQ_ret"]  = np.log(nq_cny).diff()
        df["VIX_ret"] = np.log(df_vix["Close"].reindex(idx)).diff()

        # 前一日 ETF 收盘（取 log）
        df["ETF_close_log"] = np.log(df_etf["Close"].shift(1))

        # 目标：今天 ETF 开盘（取 log）
        df["ETF_open_log"] = np.log(df_etf["Open"])

        df.dropna(inplace=True)
        logger.info("prepare_features.end", rows=len(df), columns=list(df.columns))
        return df


    def rolling_predict(self, df: pd.DataFrame):
        """
        用滚动 OLS 预测 log(ETF_open)；预测值再取 exp 还原为价格。
        """
        logger.info("rolling_predict.start", rows=len(df), lookback_days=self.cfg.lookback_days)
        features = ["NDX_ret", "NQ_ret", "VIX_ret", "ETF_close_log"]
        preds = []
        betas = []                       # 可选：记录回归系数（含 γ）

        for i in range(self.cfg.lookback_days, len(df)):
            window = df.iloc[i - self.cfg.lookback_days : i]
            X = window[features].values
            X_design = np.hstack([np.ones((X.shape[0], 1)), X])      # 加截距
            y = window["ETF_open_log"].values

            beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
            betas.append(beta)            # 若需查看 γ，可取 beta[2] (对应 NQ_ret)

            x_today = np.hstack([1, df.iloc[i][features].values])
            pred_log = x_today.dot(beta)
            preds.append(float(np.exp(pred_log)))     # 还原为价格

        df_pred = df.iloc[self.cfg.lookback_days :].copy()
        df_pred["pred_open"] = preds
        logger.info("rolling_predict.end", predictions=len(preds))
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
        logger.info("run.start", **asdict(self.cfg))

        today = datetime.date.today()
        start = today - datetime.timedelta(days=self.cfg.lookback_days * 2)

        df_etf = self.fetch_data(self.cfg.etf_symbol, start, today)
        df_ndx = self.fetch_data(self.cfg.ndx_symbol, start, today)
        if df_etf.empty or df_ndx.empty:
            logger.error("run.error", message="ETF or NDX data missing")
            return

        df_nq = self.fetch_data(self.cfg.nq_symbol, start, today)
        df_vix = self.fetch_data(self.cfg.vix_symbol, start, today)
        df_fx = self.fetch_data(self.cfg.fx_symbol, start, today)

        df = self.prepare_features(df_etf, df_ndx, df_nq, df_vix, df_fx)
        if df.empty:
            logger.error("run.error", message="Insufficient data after preprocessing")
            return

        df_pred = self.rolling_predict(df)
        latest = df_pred.iloc[-1]
        result = {"date": str(latest.name.date()), "pred_open": latest['pred_open']}
        logger.info("run.end", **result)
        # Also write result JSON separately
        with open("result.json", "w") as f:
            import json
            json.dump({**asdict(self.cfg), **result}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    fire.Fire(ETFOpenPredictor)
