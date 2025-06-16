from qlib.contrib.evaluate import backtest_daily
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from datetime import datetime
from pprint import pprint
import pandas as pd
import yaml
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config


def init(config):
    return (init_instance_by_config(config['task']['model']),
            init_instance_by_config(config['task']['dataset']))


def backtest(pred_score):
    # Validate and get date range from predictions
    if pred_score.empty:
        raise ValueError("Prediction data is empty")

    dates = pred_score.index.get_level_values('datetime')
    data_start = dates.min()
    data_end = dates.max()

    print(f"Prediction data range: {data_start.date()} to {data_end.date()}")

    STRATEGY_CONFIG = {
        "topk": 50,
        "n_drop": 5,
        "signal": pred_score,
    }

    strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)

    # Use the actual data range for backtesting
    backtest_start = max(data_start, pd.Timestamp("2024-06-13"))
    backtest_end = min(data_end, pd.Timestamp("2025-06-12"))

    print(f"Backtesting period: {backtest_start.date()} to {backtest_end.date()}")

    report_normal, positions_normal = backtest_daily(
        start_time=backtest_start.strftime('%Y-%m-%d'),
        end_time=backtest_end.strftime('%Y-%m-%d'),
        strategy=strategy_obj
    )

    analysis = dict()
    analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
    analysis["excess_return_with_cost"] = risk_analysis(
        report_normal["return"] - report_normal["bench"] - report_normal["cost"])

    analysis_df = pd.concat(analysis)
    pprint(analysis_df)


def main(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Init & train the model
    qlib.init(provider_uri=config['qlib_init']['provider_uri'], region=REG_CN)
    model, dataset = init(config)

    model.fit(dataset)
    pred = model.predict(dataset)

    # backtest
    backtest(pred)

    if isinstance(pred, pd.Series):
        pred = pred.to_frame('score')
    pprint(pred.head(500))

    save_name = f"./scores/{config['task']['model']['class']}" + datetime.now().strftime('_pred_%d_%m_%y-%H_%M.parquet')
    pred.to_parquet(save_name, engine='pyarrow')


if __name__ == '__main__':
    main('/home/patrick/application/PyCharm/project/CustomQ/configs/workflow_config_lightgbm_configurable_dataset.yaml')



