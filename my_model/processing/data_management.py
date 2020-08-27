import pandas as pd
import joblib

import logging
import typing as t

from my_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

from sklearn.pipeline import Pipeline


_logger = logging.getLogger(__name__)


def load_dataset(file_name: str) -> pd.DataFrame:
    _logger.info('Loading data from {DATASET_DIR}/{file_name}')
    dataframe = pd.read_csv(f'{DATASET_DIR}/{file_name}')
    dataframe = dataframe.rename(config.data_config.rename_columns, axis=1)
    return dataframe


def save_pipeline(pipeline_to_persist: Pipeline) -> None:
    save_file_name = f'{config.app_config.pipeline_save_file}.pkl'
    save_path = TRAINED_MODEL_DIR / save_file_name
    remove_old_pipelines()
    joblib.dump(pipeline_to_persist)
    _logger.info(f"saved pipeline: {save_file_name}")


def load_pipeline(file_name: str) -> Pipeline:
    file_path = TRAINED_MODEL_DIR / file_name
    _logger.info('Loading pipeline object from {file_path}')
    trained_model = joblib.load(file_path)
    return trained_model


def remove_old_pipelines(files_to_keep: t.List[str]) -> None:
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name != '__init__.py':
            model_file.unlink()
            _logger.info(f"removed {model_file}")

