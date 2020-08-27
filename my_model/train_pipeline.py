from my_model.pipeline import pipe
from my_model.config.core import config
from my_model.processing.data_management import load_dataset, save_pipeline

from my_model import __version__ as _version

import logging

_logger = logging.getLogger(__name__)


def run_training() -> None:
    train_data = load_dataset(config.app_config.train_data_file)
    pipe.fit(train_data[config.data_config.features], train_data[config.data_config.target])

    _logger.info(f'Saving model version {_version} to {config.app_config.pipeline_save_file}')

    save_pipeline(pipe)


if __name__ == '__main__':
    run_training()

