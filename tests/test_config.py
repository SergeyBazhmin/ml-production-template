from pathlib import Path
import pytest
from my_model.config.core import (
    create_and_validate_config,
    fetch_config_from_yaml
)
from pydantic import ValidationError


TEST_CONFIG_TEXT = """
app_config:
  package_name: my_model
  pipeline_name: my_model_pipeline
  pipeline_save_file: my_model_pipeline_draft
  train_data_file: train.csv
  test_data_file: test.csv


transformer_config:
  features_to_skip:
    - Name
    - PassengerId
    - Cabin
    - Age
  features_to_impute:
    - Embarked
    - Sex
    - SibSp
    - Parch
  features_to_encode:
    - Sex
    - Embarked

model_config:
  n_estimators: 150
  max_depth: 5
  random_state: 42

data_config:
  rename_columns:
    PassengerId: passenger_id
    Pclass: p_class
    Age: age
    SibSp: sib_sp
    Parch: parch
    Fare: fare
    Name: name
    Sex: sex
    Cabin: cabin
    Embarked: embarked
  features:
    - passenger_id
    - p_class
    - age
    - sib_sp
    - parch
    - fare
    - name
    - sex
    - ticket
    - cabin
    - embarked
  target: Survived
"""

INVALID_TEST_CONFIG = """
app_config:
  package_name: my_model
  pipeline_name: my_model_pipeline
  pipeline_save_file: my_model_pipeline_draft
  test_data_file: test.csv


transformer_config:
  features_to_skip:
    - Name
    - PassengerId
    - Cabin
    - Age
  features_to_impute:
    - Embarked
    - Sex
    - SibSp
    - Parch
  features_to_encode:
    - Sex
    - Embarked

model_config:
  n_estimators: 150
  max_depth: 5
  random_state: 42


features:
  - PassengerId
  - Pclass
  - Age
  - SibSp
  - Parch
  - Fare
  - Name
  - Sex
  - Ticket
  - Cabin
  - Embarked
"""


def test_fetch_correct_config(tmpdir):
    configdir = Path(tmpdir)
    config_1 = configdir / "sample_config.yml"
    config_1.write_text(TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(config_1)
    config = create_and_validate_config(parsed_config)

    assert config.app_config
    assert config.data_config
    assert config.model_config
    assert config.transformer_config


def test_fetch_invalid_config(tmpdir):
    configdir = Path(tmpdir)
    config_1 = configdir / "sample_config.yml"
    config_1.write_text(INVALID_TEST_CONFIG)
    parsed_config = fetch_config_from_yaml(config_1)

    with pytest.raises(ValidationError) as excinfo:
        create_and_validate_config(parsed_config=parsed_config)

    assert "field required" in str(excinfo.value)
