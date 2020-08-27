from pathlib import Path
from pydantic import BaseModel
import typing as t
from strictyaml import load, YAML
import my_model


PACKAGE_ROOT = Path(my_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "dataset"


class AppConfig(BaseModel):
    package_name: str
    pipeline_name: str
    pipeline_save_file: str
    train_data_file: str
    test_data_file: str


class TransformerConfig(BaseModel):
    features_to_skip: t.Sequence[str]
    features_to_impute: t.Sequence[str]
    features_to_encode: t.Sequence[str]


class DataConfig(BaseModel):
    rename_columns: t.Dict[str, str]
    features: t.Sequence[str]
    target: str


class ModelConfig(BaseModel):
    n_estimators: int
    max_depth: int
    random_state: int


class Config(BaseModel):
    app_config: AppConfig
    transformer_config: TransformerConfig
    model_config: ModelConfig
    data_config: DataConfig


def find_config() -> Path:
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f'Config file not found at {CONFIG_FILE_PATH!r}')


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    if not cfg_path:
        cfg_path = find_config()

    if cfg_path:
        with open(cfg_path, 'r') as config_file:
            parsed_config = load(config_file.read())
            return parsed_config
    raise Exception(f'Config file not found at {cfg_path}')


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    _config = Config(
        app_config = AppConfig(**parsed_config.data['app_config']),
        transformer_config = TransformerConfig(**parsed_config.data['transformer_config']),
        model_config = ModelConfig(**parsed_config.data['model_config']),
        data_config = DataConfig(**parsed_config.data['data_config'])
    )
    return _config


config = create_and_validate_config()

