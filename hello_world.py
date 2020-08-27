from my_model.config.core import fetch_config_from_yaml, create_and_validate_config
from my_model.processing.data_management import load_dataset
from my_model.config.core import config
from my_model.processing.validation import validate_inputs
from pathlib import Path
from my_model.processing.transformers import FeatureSkipper
import my_model

if __name__ == '__main__':
    package_root = Path(my_model.__file__).resolve().parent
    config_file = package_root / "config.yml"
    parsed_conf = fetch_config_from_yaml(config_file)
    data = create_and_validate_config(parsed_conf)
    df = load_dataset(config.app_config.train_data_file)
    print(load_dataset(config.app_config.train_data_file))
    print(validate_inputs(df)['validated_inputs'])

    X_transformed = FeatureSkipper(config.transformer_config.features_to_skip).fit_transform(df)
    print(X_transformed)
    assert all(x in df.columns for x in config.transformer_config.features_to_skip)
    assert all(x not in X_transformed.columns for x in config.transformer_config.features_to_skip)

