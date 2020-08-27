from my_model.pipeline import pipe
from my_model.config.core import config
from my_model.processing.validation import validate_inputs
import pandas as pd


def just_transform(pipe, data):
    for name, step in pipe.steps[:-1]:
        data = step.fit_transform(data)
    return data


def test_pipeline_drops_features(raw_train_data: pd.DataFrame):
    assert all(x in raw_train_data.columns for x in config.transformer_config.features_to_skip)

    X_transformed = just_transform(pipe, raw_train_data)

    assert all(x in raw_train_data for x in config.transformer_config.features_to_skip)
    assert all(x not in X_transformed.columns for x in config.transformer_config.features_to_skip)


def test_pipeline_imputes_features(raw_train_data: pd.DataFrame):
    assert raw_train_data[config.transformer_config.features_to_impute].isna().any().any()

    X_transformed = just_transform(pipe, raw_train_data)

    assert raw_train_data[config.transformer_config.features_to_impute].isna().any().any()
    assert not X_transformed[config.transformer_config.features_to_impute].isna().any().any()


def test_pipeline_pipeline_encodes_features(raw_train_data: pd.DataFrame):
    assert all(x in raw_train_data.select_dtypes('object').columns for x in config.transformer_config.features_to_encode)

    X_transformed = just_transform(pipe, raw_train_data)

    assert all(x in raw_train_data.select_dtypes('object').columns for x in config.transformer_config.features_to_encode)
    assert all(x in X_transformed.select_dtypes('int64').columns for x in config.transformer_config.features_to_encode)


def test_pipeline_takes_validated_input(raw_train_data: pd.DataFrame, raw_test_data: pd.DataFrame):
    raw_test_data[config.data_config.target] = 0
    ret_value_train = validate_inputs(raw_train_data)
    ret_value_test = validate_inputs(raw_test_data)

    assert ret_value_train['errors'] is None
    assert ret_value_train['validated_inputs'] is not None

    assert ret_value_test['errors'] is None
    assert ret_value_test['validated_inputs'] is not None

    validated_inputs = ret_value_train['validated_inputs']
    pipe.fit(validated_inputs[config.data_config.features], validated_inputs[config.data_config.target])

    validated_inputs = ret_value_test['validated_inputs'].dropna()
    pipe.predict(validated_inputs[config.data_config.features])

