import os
import ccxt
import pandas as pd
from ..base_loader import BaseLoader


class BybitOHLCVLoader(BaseLoader):
    def __init__(self, symbol, timeframe, start, end, storage_path="storage"):
        self.exchange = ccxt.bybit()
        self.symbol = symbol
        self.timeframe = timeframe
        self.start = start
        self.end = end
        self.storage_path = storage_path

    def load(self) -> pd.DataFrame:
        os.makedirs(f"{self.storage_path}/{self.symbol}", exist_ok=True)

        start_ms = int(pd.Timestamp(self.start, tz="UTC").timestamp() * 1000)
        end_ms = int(pd.Timestamp(self.end, tz="UTC").timestamp() * 1000)

        data = []
        since = start_ms

        while since < end_ms:
            batch = self.exchange.fetch_ohlcv(
                self.symbol,
                self.timeframe,
                since=since,
                limit=1000
            )

            if not batch:
                break

            data.extend(batch)
            since = batch[-1][0] + 1

        df = pd.DataFrame(
            data,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df.loc[self.start:self.end]

        safe_symbol = self.symbol.replace("/", "_")
        os.makedirs(f"{self.storage_path}/{safe_symbol}", exist_ok=True)

        df.to_csv(f"{self.storage_path}/{safe_symbol}/{self.start}-{self.end}.csv")

        return df