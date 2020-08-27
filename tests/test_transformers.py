from my_model.processing.transformers import FeatureSkipper, ImputerWrapper, EncoderWrapper
from my_model.config.core import config
import pandas as pd


def test_transformer_drops_features(raw_train_data: pd.DataFrame):
    assert all(x in raw_train_data.columns for x in config.transformer_config.features_to_skip)

    X_transformed = FeatureSkipper(config.transformer_config.features_to_skip).fit_transform(raw_train_data)

    assert all(x in raw_train_data.columns for x in config.transformer_config.features_to_skip)
    assert all(x not in X_transformed.columns for x in config.transformer_config.features_to_skip)


def test_transformer_imputes_features(raw_train_data: pd.DataFrame):
    assert raw_train_data[config.transformer_config.features_to_impute].isna().any().any()

    X_transformed = ImputerWrapper(config.transformer_config.features_to_impute, 'most_frequent').fit_transform(raw_train_data)

    assert raw_train_data[config.transformer_config.features_to_impute].isna().any().any()
    assert not X_transformed[config.transformer_config.features_to_impute].isna().any().any()


def test_transformer_encodes_features(raw_train_data: pd.DataFrame):
    assert all(x in raw_train_data.select_dtypes('object').columns for x in config.transformer_config.features_to_encode)

    raw_train_data = raw_train_data.dropna()

    X_transformed = EncoderWrapper(config.transformer_config.features_to_encode).fit_transform(raw_train_data)

    assert all(x in raw_train_data.select_dtypes('object').columns for x in config.transformer_config.features_to_encode)
    assert all(x in X_transformed.select_dtypes('int64').columns for x in config.transformer_config.features_to_encode)

