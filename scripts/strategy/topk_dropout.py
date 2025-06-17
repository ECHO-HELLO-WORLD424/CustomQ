import polars as pl
from datetime import datetime
from typing import Dict, List, Set
import argparse
import json
from pathlib import Path


class TopKDropoutTrader:
    """
    Implementation of Top-K Dropout trading strategy.

    Parameters:
    - topk: The number of stocks to hold in portfolio
    - drop: The number of stocks to sell/buy on each trading day
    """

    def __init__(self, topk: int, drop: int, decisions_dir: str = "./decisions"):
        self.topk = topk
        self.drop = drop
        self.current_holdings: Set[str] = set()
        self.trading_history: List[Dict] = []
        self.decisions_dir = Path(decisions_dir)

        # Create decisions directory if it doesn't exist
        self.decisions_dir.mkdir(exist_ok=True)

    def _serialize_sets(self, obj):
        """Convert sets to lists for JSON serialization."""
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: self._serialize_sets(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_sets(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    def _deserialize_sets(self, obj):
        """Convert lists back to sets where appropriate for loaded data."""
        if isinstance(obj, dict):
            # Convert holdings_after back to set
            if 'holdings_after' in obj and isinstance(obj['holdings_after'], list):
                obj['holdings_after'] = set(obj['holdings_after'])
            # Convert date strings back to datetime
            if 'date' in obj and isinstance(obj['date'], str):
                obj['date'] = datetime.fromisoformat(obj['date'])
            return {k: self._deserialize_sets(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deserialize_sets(item) for item in obj]
        return obj

    def save_trading_history(self, filename: str = None) -> str:
        """
        Save trading history to JSON file.

        Args:
            filename: Optional filename. If None, generates filename based on strategy params and timestamp.

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_history_k{self.topk}_d{self.drop}_{timestamp}.json"

        filepath = self.decisions_dir / filename

        # Prepare data for serialization
        data = {
            'strategy_params': {
                'topk': self.topk,
                'drop': self.drop,
                'turnover_rate': self.calculate_turnover_rate()
            },
            'current_holdings': list(self.current_holdings),
            'trading_history': self._serialize_sets(self.trading_history),
            'metadata': {
                'saved_at': datetime.now().isoformat(),
                'total_trading_days': len(self.trading_history)
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Trading history saved to: {filepath}")
        return str(filepath)

    def load_trading_history(self, filename: str) -> Dict:
        """
        Load trading history from JSON file.

        Args:
            filename: Name of the JSON file to load

        Returns:
            Dictionary containing loaded data
        """
        filepath = self.decisions_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Trading history file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Deserialize the data
        data = self._deserialize_sets(data)

        # Restore trader state
        if 'strategy_params' in data:
            loaded_topk = data['strategy_params']['topk']
            loaded_drop = data['strategy_params']['drop']
            if loaded_topk != self.topk or loaded_drop != self.drop:
                print(f"Warning: Loaded strategy params (k={loaded_topk}, d={loaded_drop}) "
                      f"differ from current params (k={self.topk}, d={self.drop})")

        if 'current_holdings' in data:
            self.current_holdings = set(data['current_holdings'])

        if 'trading_history' in data:
            self.trading_history = data['trading_history']

        print(f"Trading history loaded from: {filepath}")
        print(f"Loaded {len(self.trading_history)} trading days")
        print(f"Current portfolio size: {len(self.current_holdings)}")

        return data

    def list_saved_histories(self) -> List[str]:
        """
        List all saved trading history files in the decisions directory.

        Returns:
            List of filenames
        """
        json_files = list(self.decisions_dir.glob("*.json"))
        return [f.name for f in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True)]

    def get_history_summary(self, filename: str) -> Dict:
        """
        Get summary information about a saved trading history file without fully loading it.

        Args:
            filename: Name of the JSON file

        Returns:
            Dictionary with summary information
        """
        filepath = self.decisions_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Trading history file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        summary = {
            'filename': filename,
            'strategy_params': data.get('strategy_params', {}),
            'metadata': data.get('metadata', {}),
            'current_portfolio_size': len(data.get('current_holdings', [])),
            'file_size_kb': round(filepath.stat().st_size / 1024, 2)
        }

        return summary

    def load_predictions(self, parquet_file: str, target_date: str = None) -> pl.DataFrame:
        """
        Load prediction scores from parquet file.

        Args:
            parquet_file: Path to parquet file containing predictions
            target_date: Target date for predictions (YYYY-MM-DD format)
                        If None, uses the latest date in the file

        Returns:
            DataFrame with filtered predictions for the target date
        """
        df = pl.read_parquet(parquet_file)

        # Convert datetime column if it's not already datetime type
        if df['datetime'].dtype != pl.Datetime:
            df = df.with_columns(pl.col('datetime').str.to_datetime())

        if target_date:
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            df = df.filter(pl.col('datetime').dt.date() == target_dt.date())
        else:
            # Use the latest date available
            latest_date = df.select(pl.col('datetime').max()).item()
            df = df.filter(pl.col('datetime') == latest_date)

        if df.height == 0:
            raise ValueError(f"No predictions found for date: {target_date or 'latest'}")

        return df.sort('score', descending=True)

    def initialize_portfolio(self, predictions_df: pl.DataFrame) -> Dict[str, List[str]]:
        """
        Initialize portfolio with top-K stocks on the first trading day.

        Args:
            predictions_df: DataFrame with columns ['score', 'datetime', 'instrument']
                           sorted by score in descending order

        Returns:
            Dictionary with 'buy' and 'sell' lists (sell will be empty for initialization)
        """
        if len(self.current_holdings) > 0:
            raise ValueError("Portfolio already initialized. Use generate_trading_decisions() instead.")

        # Get top-K instruments
        ranked_instruments = predictions_df['instrument'].to_list()
        top_k_stocks = ranked_instruments[:self.topk]

        # Initialize portfolio with top-K stocks
        self.current_holdings = set(top_k_stocks)

        # Store initial portfolio setup in history
        trading_date = predictions_df['datetime'][0]
        self.trading_history.append({
            'date': trading_date,
            'sells': [],  # No sells on initialization
            'buys': top_k_stocks.copy(),
            'holdings_after': self.current_holdings.copy(),
            'portfolio_size': len(self.current_holdings),
            'is_initialization': True
        })

        print(f"Portfolio initialized with top-{self.topk} stocks")
        return {
            'sell': [],
            'buy': top_k_stocks
        }

    def generate_trading_decisions(self, predictions_df: pl.DataFrame) -> Dict[str, List[str]]:
        """
        Generate trading decisions based on top-k dropout strategy.

        If portfolio is empty (first trading day), automatically initializes with top-K stocks.

        Args:
            predictions_df: DataFrame with columns ['score', 'datetime', 'instrument']
                           sorted by score in descending order

        Returns:
            Dictionary with 'buy' and 'sell' lists containing instrument names
        """
        # Check if this is the first trading day (empty portfolio)
        if len(self.current_holdings) == 0:
            print("Empty portfolio detected. Initializing with top-K stocks...")
            return self.initialize_portfolio(predictions_df)

        # Get all instruments ranked by prediction score (high to low)
        ranked_instruments = predictions_df['instrument'].to_list()

        # Calculate number of stocks to drop
        # d = number of currently held stocks with rank > K
        currently_held_low_rank = []
        for instrument in self.current_holdings:
            if instrument in ranked_instruments:
                rank = ranked_instruments.index(instrument)
                if rank >= self.topk:  # rank > K (0-indexed, so >= topk)
                    currently_held_low_rank.append((instrument, rank))

        # Sort by rank (worst first) and take up to 'drop' number
        currently_held_low_rank.sort(key=lambda x: x[1], reverse=True)
        d = min(len(currently_held_low_rank), self.drop)

        # Determine sells: worst performing currently held stocks
        stocks_to_sell = [instrument for instrument, _ in currently_held_low_rank[:d]]

        # Determine buys: best performing unheld stocks
        stocks_to_buy = []
        for instrument in ranked_instruments:
            if len(stocks_to_buy) >= d:
                break
            if instrument not in self.current_holdings:
                stocks_to_buy.append(instrument)

        # Update current holdings
        for stock in stocks_to_sell:
            self.current_holdings.discard(stock)
        for stock in stocks_to_buy:
            self.current_holdings.add(stock)

        # Store trading decision in history
        trading_date = predictions_df['datetime'][0]
        self.trading_history.append({
            'date': trading_date,
            'sells': stocks_to_sell.copy(),
            'buys': stocks_to_buy.copy(),
            'holdings_after': self.current_holdings.copy(),
            'portfolio_size': len(self.current_holdings)
        })

        return {
            'sell': stocks_to_sell,
            'buy': stocks_to_buy
        }

    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status."""
        return {
            'current_holdings': list(self.current_holdings),
            'portfolio_size': len(self.current_holdings),
            'target_size': self.topk
        }

    def get_trading_history(self) -> List[Dict]:
        """Get complete trading history."""
        return self.trading_history

    def calculate_turnover_rate(self) -> float:
        """Calculate theoretical turnover rate: 2 * Drop / K"""
        return (2 * self.drop) / self.topk if self.topk > 0 else 0


def main():
    """Main function to demonstrate the trading strategy."""
    parser = argparse.ArgumentParser(description='Top-K Dropout Trading Strategy')
    parser.add_argument('--parquet_file', type=str,
                        help='Path to parquet file containing predictions')
    parser.add_argument('--topk', type=int, default=50,
                        help='Number of stocks to hold (default: 50)')
    parser.add_argument('--drop', type=int, default=5,
                        help='Number of stocks to sell/buy each day (default: 5)')
    parser.add_argument('--date', type=str, default=None,
                        help='Target date for predictions (YYYY-MM-DD). Uses latest if not specified')
    parser.add_argument('--decisions_dir', type=str, default='./decisions',
                        help='Directory to save/load trading decisions (default: ./decisions)')

    # JSON file operations
    parser.add_argument('--save_history', action='store_true',
                        help='Save trading history to JSON file after processing')
    parser.add_argument('--load_history', type=str,
                        help='Load trading history from JSON file')
    parser.add_argument('--list_histories', action='store_true',
                        help='List all saved trading history files')
    parser.add_argument('--summary', type=str,
                        help='Show summary of a specific trading history file')

    args = parser.parse_args()

    # Initialize trader
    trader = TopKDropoutTrader(topk=args.topk, drop=args.drop, decisions_dir=args.decisions_dir)

    # Handle list histories command
    if args.list_histories:
        print("=== Saved Trading Histories ===")
        histories = trader.list_saved_histories()
        if not histories:
            print("No saved trading histories found.")
        else:
            for i, filename in enumerate(histories, 1):
                print(f"{i}. {filename}")
        return 0

    # Handle summary command
    if args.summary:
        try:
            summary = trader.get_history_summary(args.summary)
            print(f"=== Summary: {args.summary} ===")
            print(f"Strategy: Top-{summary['strategy_params'].get('topk', 'N/A')} "
                  f"Drop-{summary['strategy_params'].get('drop', 'N/A')}")
            print(f"Trading days: {summary['metadata'].get('total_trading_days', 'N/A')}")
            print(f"Current portfolio size: {summary['current_portfolio_size']}")
            print(f"Turnover rate: {summary['strategy_params'].get('turnover_rate', 'N/A'):.2%}")
            print(f"File size: {summary['file_size_kb']} KB")
            print(f"Saved at: {summary['metadata'].get('saved_at', 'N/A')}")
        except Exception as e:
            print(f"Error reading summary: {e}")
        return 0

    # Handle load history command
    if args.load_history:
        try:
            trader.load_trading_history(args.load_history)

            # Display current status after loading
            status = trader.get_portfolio_status()
            print(f"\n=== Current Portfolio Status ===")
            print(f"Holdings: {status['portfolio_size']}/{status['target_size']}")
            print(f"Current holdings: {status['current_holdings']}")

            if not args.parquet_file:
                return 0

        except Exception as e:
            print(f"Error loading trading history: {e}")
            return 1

    # If no parquet file provided and not doing file operations, show help
    if not args.parquet_file and not any([args.list_histories, args.summary, args.load_history]):
        parser.print_help()
        print("\nNote: Either provide --parquet_file for trading decisions or use file operations")
        return 1

    # Process trading decisions if parquet file is provided
    if args.parquet_file:
        try:
            # Load predictions
            print(f"Loading predictions from {args.parquet_file}...")
            predictions = trader.load_predictions(args.parquet_file, args.date)

            print(f"Found {predictions.height} predictions for date: {predictions['datetime'][0]}")
            print(f"Score range: {predictions['score'].min():.6f} to {predictions['score'].max():.6f}")

            # Generate trading decisions
            decisions = trader.generate_trading_decisions(predictions)

            # Display results
            is_initialization = len(trader.trading_history) == 1 and trader.trading_history[0].get('is_initialization',
                                                                                                   False)

            if is_initialization:
                print(f"\n=== Portfolio Initialization (Top-{args.topk}) ===")
                print(f"Initial stocks to BUY ({len(decisions['buy'])}): {decisions['buy']}")
            else:
                print(f"\n=== Trading Decisions (Top-{args.topk} Drop-{args.drop}) ===")
                print(f"Stocks to SELL ({len(decisions['sell'])}): {decisions['sell']}")
                print(f"Stocks to BUY ({len(decisions['buy'])}): {decisions['buy']}")

            # Portfolio status
            status = trader.get_portfolio_status()
            print(f"\n=== Portfolio Status ===")
            print(f"Current holdings: {status['portfolio_size']}/{status['target_size']}")
            print(f"Holdings: {status['current_holdings']}")

            # Strategy metrics
            turnover_rate = trader.calculate_turnover_rate()
            print(f"\n=== Strategy Metrics ===")
            print(f"Theoretical turnover rate: {turnover_rate:.2%}")

            # Show top and bottom performers
            print(f"\n=== Top 10 Predictions ===")
            top_10 = predictions.head(10)
            for row in top_10.iter_rows():
                score, dt, instrument = row
                print(f"{instrument}: {score:.6f}")

            # Save history if requested
            if args.save_history:
                trader.save_trading_history()

        except Exception as e:
            print(f"Error: {e}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())