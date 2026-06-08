import pandas as pd
from ..base_loader import BaseLoader


class CsvOHLCVLoader(BaseLoader):
    def __init__(
        self,
        path,
        start=None,
        end=None,
        timestamp_col="timestamp",
        index_col=None
    ):
        self.path = path
        self.start = start
        self.end = end
        self.timestamp_col = timestamp_col
        self.index_col = index_col

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)

        if self.timestamp_col in df.columns:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], utc=True)
            df = df.set_index(self.timestamp_col)

        elif self.index_col is not None:
            df[df.columns[self.index_col]] = pd.to_datetime(df[df.columns[self.index_col]], utc=True)
            df = df.set_index(df.columns[self.index_col])

        else:
            df.index = pd.to_datetime(df.index, utc=True)

        df = df[["open", "high", "low", "close", "volume"]]
        df = df.sort_index()

        if self.start is not None:
            df = df.loc[pd.Timestamp(self.start, tz="UTC"):]

        if self.end is not None:
            df = df.loc[:pd.Timestamp(self.end, tz="UTC")]

        return df