import pytest
from my_model.processing.data_management import load_dataset
from my_model.config.core import config


@pytest.fixture()
def raw_train_data():
    return load_dataset(config.app_config.train_data_file)


@pytest.fixture()
def raw_test_data():
    return load_dataset(config.app_config.test_data_file)


@pytest.fixture()
def data_record():
    return {
        'passenger_id': 1,
        'Survived': 1,
        'p_class': 2,
        'name': "Sergey",
        'sex': "male",
        'age': 23,
        'sib_sp': 0,
        'parch': 0,
        'ticket': "ac 32443c",
        'fare': 300.0,
        'cabin': None,
        'embarked': 'S',
    }

