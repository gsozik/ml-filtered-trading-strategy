import os
import requests
import pandas as pd
from ..base_loader import BaseLoader


class MoexOHLCVLoader(BaseLoader):
    def __init__(self, ticker, timeframe, start, end, board="TQBR", market="shares", storage_path="storage"):
        self.ticker = ticker
        self.timeframe = timeframe
        self.start = start
        self.end = end
        self.board = board
        self.market = market
        self.storage_path = storage_path

    def load(self) -> pd.DataFrame:
        os.makedirs(f"{self.storage_path}/{self.ticker}", exist_ok=True)

        tf = {
            "1m": 1,
            "10m": 10,
            "1h": 60,
            "1d": 24,
            "1w": 7,
            "1M": 31,
            "1Q": 4
        }

        interval = tf[self.timeframe]

        url = (
            f"https://iss.moex.com/iss/engines/stock/markets/{self.market}/"
            f"boards/{self.board}/securities/{self.ticker}/candles.json"
        )

        rows = []
        off = 0

        while True:
            response = requests.get(
                url,
                params={
                    "from": self.start,
                    "till": self.end,
                    "interval": interval,
                    "start": off
                }
            ).json()["candles"]

            chunk = response["data"]

            if not chunk:
                break

            rows += chunk
            off += len(chunk)

        if not rows:
            raise ValueError(
                f"MOEX вернул пустые данные для {self.ticker}. "
                f"Проверь market={self.market}, board={self.board}, ticker={self.ticker}"
            )

        df = pd.DataFrame(rows, columns=response["columns"])

        df["timestamp"] = (
            pd.to_datetime(df["begin"])
            .dt.tz_localize("Europe/Moscow")
            .dt.tz_convert("UTC")
        )

        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]

        start_ts = pd.Timestamp(self.start, tz="UTC")
        end_ts = pd.Timestamp(self.end, tz="UTC")

        df = df.loc[start_ts:end_ts]

        df.to_csv(f"{self.storage_path}/{self.ticker}/{self.start}-{self.end}.csv")

        return df