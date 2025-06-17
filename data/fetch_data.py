import pandas as pd
import os
import shutil
import datetime
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Literal
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import logging
import baostock as bs
from baostock.data.resultset import ResultData
from qlib_dump_bin import DumpDataAll


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _read_all_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _write_all_text(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


IndexType = Literal["csi300", "csi500", "csi1000", "all"]


class DataManager:
    _stocks: List[str]
    _basic_info: pd.DataFrame
    _adjust_factors: pd.DataFrame
    _index_type: IndexType

    _adjust_columns: List[str] = [
        "foreAdjustFactor",
        "backAdjustFactor",
        "adjustFactor"
    ]
    _fields: List[str] = [
        "date", "open", "high", "low",
        "close", "preclose", "volume", "amount",
        "turn", "tradestatus", "pctChg", "peTTM",
        "psTTM", "pcfNcfTTM", "pbMRQ", "isST"
    ]
    _price_fields: List[str] = [
        "open", "high", "low", "close", "preclose"
    ]

    def __init__(
            self,
            save_path: str,
            qlib_export_path: str,
            qlib_base_data_path: str,
            index_type: IndexType = "csi300",
            forward_adjust_date: str = "2023-01-15",
            max_workers: int = 10,
    ):
        self._save_path = os.path.expanduser(save_path)
        self._export_path = f"{self._save_path}/export"
        os.makedirs(self._save_path, exist_ok=True)
        os.makedirs(self._export_path, exist_ok=True)
        self._qlib_export_path = os.path.expanduser(qlib_export_path)
        self._qlib_path = os.path.expanduser(qlib_base_data_path)
        self._index_type = index_type
        self._forward_adjust_date = forward_adjust_date
        self._max_workers = max_workers

    @classmethod
    def _login_baostock(cls) -> None:
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull):
                bs.login()

    @property
    def _stock_list_path(self) -> str:
        return f"{self._save_path}/{self._index_type}_list.txt"

    def _get_index_code(self) -> str:
        """Get the corresponding index code for baostock"""
        index_codes = {
            "csi300": "sh.000300",
            "csi500": "sh.000905",
            "csi1000": "sh.000852",
            "all": None  # No specific index for 'all'
        }
        return index_codes.get(self._index_type)

    def _is_stock_active(self, end_date_str: str) -> bool:
        """Check if a stock is still active based on its end date"""
        if not end_date_str or end_date_str.strip() == "":
            return True  # No end date means still active

        try:
            end_date = pd.Timestamp(end_date_str)
            current_date = pd.Timestamp.now()
            # Consider a stock active if it's been delisted less than 30 days ago
            # This provides a small buffer for recently delisted stocks
            return abs((current_date - end_date).days) < 10
        except (ValueError, TypeError):
            logger.warning(f"Invalid end date {end_date_str}, kept for safety.")
            return True

    def _load_stocks_base(self) -> None:
        if os.path.exists(self._stock_list_path):
            lines = _read_all_text(self._stock_list_path).split('\n')
            self._stocks = [line for line in lines if line != ""]
        else:
            # Read from qlib instruments file
            instruments_path = f"{self._qlib_path}/instruments/{self._index_type}.txt"
            if not os.path.exists(instruments_path):
                raise FileNotFoundError(f"Instruments file not found: {instruments_path}")

            lines = _read_all_text(instruments_path).split('\n')
            active_stocks = []
            total_stocks = 0
            filtered_stocks = 0

            for line in lines:
                if line.strip() == "":
                    continue

                parts = line.split('\t')
                if len(parts) < 2:
                    continue

                stock_id = parts[0]
                end_date = parts[-1]

                total_stocks += 1

                # Check if stock is still active
                if self._is_stock_active(end_date):
                    baostock_code = f"{stock_id[:2].lower()}.{stock_id[-6:]}"
                    active_stocks.append(baostock_code)
                else:
                    filtered_stocks += 1
                    print(f"Filtered out inactive stock: {stock_id} (delisted on {end_date})")

            self._stocks = active_stocks
            print(f"Total stocks in instruments file: {total_stocks}")
            print(f"Active stocks: {len(active_stocks)}")
            print(f"Filtered out inactive stocks: {filtered_stocks}")

    def _load_stocks(self):
        print(f"Loading {self._index_type.upper()} stock list")
        self._login_baostock()
        self._load_stocks_base()

        # Add corresponding index if not 'all'
        index_code = self._get_index_code()
        if index_code:
            self._stocks.append(index_code)

        self._stocks = list(set(self._stocks))
        print(f"Loaded {len(self._stocks)} active stocks/indices")

        _write_all_text(self._stock_list_path,
                        '\n'.join(str(s) for s in self._stocks))

    def _parallel_foreach(
            self,
            callable,
            input: List[dict],
            max_workers: Optional[int] = None
    ) -> list:
        if max_workers is None:
            max_workers = self._max_workers
        with tqdm(total=len(input)) as pbar:
            results = []
            with ProcessPoolExecutor(max_workers) as executor:
                futures = [executor.submit(callable, **elem) for elem in input]
                for f in as_completed(futures):
                    results.append(f.result())
                    pbar.update(n=1)
            return results

    def _fetch_basic_info_job(self, code: str) -> pd.DataFrame:
        self._login_baostock()
        return self._result_to_data_frame(bs.query_stock_basic(code))

    def _fetch_basic_info(self) -> pd.DataFrame:
        print("Fetching basic info")
        dfs = self._parallel_foreach(
            self._fetch_basic_info_job,
            [dict(code=code) for code in self._stocks]
        )
        # Filter out empty dataframes that might result from invalid/delisted stocks
        valid_dfs = [df for df in dfs if not df.empty]
        if not valid_dfs:
            raise ValueError("No valid stock data found")

        df = pd.concat(valid_dfs)
        df = df.sort_values(by="code").drop_duplicates(subset="code").set_index("code")
        df.to_csv(f"{self._save_path}/basic_info.csv")
        print(f"Successfully fetched basic info for {len(df)} stocks")
        return df

    def _fetch_adjust_factors_job(self, code: str, start: str) -> pd.DataFrame:
        self._login_baostock()
        return self._result_to_data_frame(bs.query_adjust_factor(code, start))

    def _fetch_adjust_factors(self) -> pd.DataFrame:
        def one_year_before_ipo(ipo: str) -> str:
            earliest_time = pd.Timestamp("1990-12-19")
            ts = pd.Timestamp(ipo) - pd.DateOffset(years=1)
            ts = earliest_time if earliest_time > ts else ts
            return ts.strftime("%Y-%m-%d")

        print("Fetch adjust factors")
        dfs: List[pd.DataFrame] = self._parallel_foreach(
            self._fetch_adjust_factors_job,
            [dict(code=code, start=one_year_before_ipo(data["ipoDate"]))
             for code, data in self._basic_info.iterrows()]
        )
        df = pd.concat([df for df in dfs if not df.empty])
        df = df.set_index(["code", "dividOperateDate"])
        df.to_csv(f"{self._save_path}/adjust_factors.csv")
        return df

    def _adjust_factors_for(self, code: str) -> pd.DataFrame:
        adj_factor_idx: pd.Index = self._adjust_factors.index.levels[0]  # type: ignore
        if code not in adj_factor_idx:
            start: str = self._basic_info.loc[code, "ipoDate"]  # type: ignore
            return pd.DataFrame(
                [[1., 1., 1.]],
                index=pd.Index([start]),
                columns=self._adjust_columns
            )
        return self._adjust_factors.xs(code, level="code").astype(float)  # type: ignore

    def _download_stock_data_job(self, code: str, data: pd.Series) -> None:
        fields_str = ",".join(self._fields)
        numeric_fields = self._fields.copy()
        numeric_fields.pop(0)

        self._login_baostock()
        query = bs.query_history_k_data_plus(
            code, fields_str,
            start_date=data["ipoDate"], adjustflag="2"
        )
        adj = self._adjust_factors_for(code)
        df = self._result_to_data_frame(query).join(adj, on="date", how="left")

        # Skip if no data was returned (might happen for invalid/delisted stocks)
        if df.empty:
            print(f"Warning: No data returned for {code}")
            return

        df[self._adjust_columns] = df[self._adjust_columns].ffill().fillna(1.)
        df[numeric_fields] = df[numeric_fields].replace("", "0.").astype(float)

        def as_of_date(df: pd.DataFrame, date: str) -> pd.Series:
            index: int = df.index.searchsorted(date, side="right") - 1  # type: ignore
            return df.iloc[index]

        ref_factor = as_of_date(df, self._forward_adjust_date)["foreAdjustFactor"]
        readjust_fields = self._price_fields + ["foreAdjustFactor"]
        df[readjust_fields] /= ref_factor
        df["volume"] /= df["foreAdjustFactor"]
        df["vwap"] = df["amount"] / df["volume"]
        df = df.set_index("date")
        df.to_pickle(f"{self._save_path}/k_data/{code}.pkl")

    def _download_stock_data(self) -> None:
        print("Download stock data")
        os.makedirs(f"{self._save_path}/k_data", exist_ok=True)
        self._parallel_foreach(
            self._download_stock_data_job,
            [dict(code=code, data=data)
             for code, data in self._basic_info.iterrows()]
        )

    def _save_csv_job(self, path: Path) -> None:
        code = path.stem
        code = f"{code[:2].upper()}{code[-6:]}"
        df: pd.DataFrame = pd.read_pickle(path)
        df.rename(columns={"foreAdjustFactor": "factor"}, inplace=True)
        df["code"] = code
        out = Path(self._export_path) / f"{code}.csv"
        df.to_csv(out)

    def _save_csv(self) -> None:
        print("Export to csv")
        children = list(Path(f"{self._save_path}/k_data").iterdir())
        self._parallel_foreach(
            self._save_csv_job,
            [dict(path=path) for path in children]
        )

    @classmethod
    def _result_to_data_frame(cls, res: ResultData) -> pd.DataFrame:
        lst = []
        while res.error_code == "0" and res.next():
            lst.append(res.get_row_data())
        return pd.DataFrame(lst, columns=res.fields)

    def _dump_qlib_data(self) -> None:
        DumpDataAll(
            csv_path=self._export_path,
            qlib_dir=self._qlib_export_path,
            max_workers=self._max_workers,
            exclude_fields="date,code",
            symbol_field_name="code"
        ).dump()
        shutil.copy(f"{self._qlib_export_path}/calendars/day.txt",
                    f"{self._qlib_export_path}/calendars/day_future.txt")
        self._fix_constituents()

    def _fix_constituents(self) -> None:
        today = str(datetime.date.today())
        path = f"{self._qlib_export_path}/instruments"

        for p in Path(path).iterdir():
            if p.stem == "all":
                continue
            df = pd.read_csv(p, sep='\t', header=None)
            df.sort_values([2, 1, 0], ascending=[False, False, True], inplace=True)  # type: ignore
            latest_data = df[2].max()
            df[2] = df[2].replace(latest_data, today)
            df.to_csv(p, header=False, index=False, sep='\t')

    def fetch_and_save_data(
            self,
            use_cached_basic_info: bool = False,
            use_cached_adjust_factor: bool = False
    ):
        self._load_stocks()
        if use_cached_basic_info:
            self._basic_info = pd.read_csv(f"{self._save_path}/basic_info.csv", index_col=0)
        else:
            self._basic_info = self._fetch_basic_info()
        if use_cached_adjust_factor:
            self._adjust_factors = pd.read_csv(f"{self._save_path}/adjust_factors.csv", index_col=[0, 1])
        else:
            self._adjust_factors = self._fetch_adjust_factors()
        self._download_stock_data()
        self._save_csv()
        self._dump_qlib_data()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download Chinese stock data for different indices")
    parser.add_argument(
        "--index",
        type=str,
        choices=["csi300", "csi500", "csi1000", "all"],
        default="csi300",
        help="Index type to download (default: csi300)"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="~/.qlib/tmp",
        help="Path to save temporary data (default: ~/.qlib/tmp)"
    )
    parser.add_argument(
        "--qlib-export-path",
        type=str,
        default="~/.qlib/qlib_data/cn_data_rolling",
        help="Path to export qlib data (default: ~/.qlib/qlib_data/cn_data_rolling)"
    )
    parser.add_argument(
        "--qlib-base-path",
        type=str,
        default="~/.qlib/qlib_data/cn_data",
        help="Path to base qlib data containing instruments (default: ~/.qlib/qlib_data/cn_data)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of parallel workers (default: 10)"
    )
    parser.add_argument(
        "--use-cached-basic-info",
        action="store_true",
        help="Use cached basic info if available"
    )
    parser.add_argument(
        "--use-cached-adjust-factor",
        action="store_true",
        help="Use cached adjust factors if available"
    )

    args = parser.parse_args()

    today = str(datetime.date.today())
    print(f"Downloading {args.index.upper()} data")
    print(f"Forward adjust date: {today}")

    # Update export path to include index type
    export_path = f"{args.qlib_export_path}_{args.index}"

    dm = DataManager(
        save_path=f"{args.save_path}_{args.index}",
        qlib_export_path=export_path,
        qlib_base_data_path=args.qlib_base_path,
        index_type=args.index,
        forward_adjust_date=today,
        max_workers=args.max_workers
    )

    dm.fetch_and_save_data(
        use_cached_basic_info=args.use_cached_basic_info,
        use_cached_adjust_factor=args.use_cached_adjust_factor
    )

    print(f"Data download completed! Data saved to: {export_path}")


if __name__ == "__main__":
    main()