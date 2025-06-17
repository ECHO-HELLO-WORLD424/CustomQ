import os
from datetime import datetime
import logging
import yaml

import qlib
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config
from models.KRNN import KRNN # This is adapted from qlib implementation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _get_default_config():
    """Default configuration for KRNN model and data"""
    return {
        'qlib_config': {
            'provider_uri': '~/.qlib/qlib_data/cn_data_rolling_csi300',  # Path to Qlib data
            'region': REG_CN,
        },
        'market': {
            'market': 'csi300',
            'benchmark': 'SH000300',
        },
        'data': {
            'class': 'DatasetH',
            'module_path': 'qlib.data.dataset',
            'kwargs': {
                'handler': {
                    'class': 'Alpha158',
                    'module_path': 'qlib.contrib.data.handler',
                    'kwargs': {
                        'fit_start_time': '2008-01-01',
                        'fit_end_time': '2016-12-31',
                        'instruments': 'csi300'
                    }
                },
                'segments': {
                    'train': ('2008-01-01', '2014-12-31'),
                    'valid': ('2015-01-01', '2016-12-31'),
                    'test': ('2017-01-01', '2020-08-01'),
                },
            }
        }
    }


def _load_config(config_path):
    """Load configuration from file or use default"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        return _get_default_config()


class KRNNStockPredictor:
    def __init__(self, config_path=None):
        """
        Initialize KRNN Stock Predictor

        Args:
            config_path (str): Path to configuration file (optional)
        """
        self.config = _load_config(config_path)
        self.model = None
        self.dataset = None

    def initialize_qlib(self):
        """Initialize Qlib with configuration"""
        try:
            qlib.init(**self.config['qlib_config'])
            logger.info("Qlib initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Qlib: {e}")
            raise

    def prepare_dataset(self):
        """Prepare dataset for training"""
        try:
            logger.info("Preparing dataset...")
            dataset_config = self.config['data']
            self.dataset = init_instance_by_config(dataset_config)

            # Log dataset information
            logger.info(f"Dataset prepared")
        except Exception as e:
            logger.error(f"Failed to prepare dataset: {e}")
            raise

    def create_model(self):
        """Create KRNN model instance"""
        try:
            logger.info("Creating KRNN model...")
            self.model = KRNN()
            logger.info(f"KRNN model created")
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise

    def train_model(self):
        """Train the KRNN model"""
        if self.model is None or self.dataset is None:
            raise ValueError("Model and dataset must be initialized before training")

        try:
            logger.info("Starting model training...")

            # Train the model
            self.model.fit(self.dataset)

            logger.info("Model training completed successfully")
            return self.model

        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise

    def evaluate_model(self):
        """Evaluate the trained model"""
        if self.model is None or self.dataset is None:
            raise ValueError("Model and dataset must be initialized before evaluation")

        try:
            logger.info("Evaluating model...")

            # Make predictions on test set
            predictions = self.model.predict(self.dataset, segment='test')

            # Get actual labels for comparison
            test_df = self.dataset.prepare(segments='test', col_set='label')

            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            from scipy.stats import pearsonr

            mse = mean_squared_error(test_df.values, predictions.values)
            mae = mean_absolute_error(test_df.values, predictions.values)
            ic, _ = pearsonr(test_df.values.flatten(), predictions.values.flatten())

            metric = {
                'MSE': mse,
                'MAE': mae,
                'IC': ic,
            }

            logger.info(f"Evaluation metrics: {metric}")
            return metric, predictions

        except Exception as e:
            logger.error(f"Failed to evaluate model: {e}")
            raise

    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save model
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)

            logger.info(f"Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, filepath):
        """Load a trained model"""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)

            logger.info(f"Model loaded from {filepath}")
            return self.model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


def main():
    """Main execution function"""
    try:
        # Initialize predictor
        predict = KRNNStockPredictor()

        # Initialize Qlib
        predict.initialize_qlib()

        # Train model
        predict.prepare_dataset()
        predict.create_model()
        _ = predict.train_model()

        # Evaluate model
        metric, predictions = predict.evaluate_model()

        # Save model
        model_path = f"./models/krnn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        predict.save_model(model_path)

        # Save predictions
        predictions_path = f"./results/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        predictions.to_csv(predictions_path)
        logger.info(f"Predictions saved to {predictions_path}")

        return predict, metric

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    # Run main pipeline
    predictor, metrics = main()

    print("\n" + "=" * 50)
    print("KRNN Stock Score Prediction Training Complete!")
    print("=" * 50)
    print(f"Final Metrics: {metrics}")
    print("=" * 50)