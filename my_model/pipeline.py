from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from my_model.processing.transformers import FeatureSkipper, ImputerWrapper, EncoderWrapper
from sklearn.ensemble import RandomForestClassifier
from my_model.config.core import config

clf = RandomForestClassifier(
    n_estimators = config.model_config.n_estimators,
    random_state = config.model_config.random_state,
    max_depth = config.model_config.max_depth
)

pipe = Pipeline([
    ('feature_skipper', FeatureSkipper(config.transformer_config.features_to_skip)),
    ('embarked_imputer', ImputerWrapper(config.transformer_config.features_to_impute, strategy='most_frequent')),
    ('cat_encoder', EncoderWrapper(config.transformer_config.features_to_encode)),
    ('classifier', clf)
])

