from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import typing as t


class FeatureSkipper(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_skip: t.List[str]) -> None:
        self.features_to_skip = features_to_skip

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        return X.drop(self.features_to_skip, axis=1)


class EncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_encode: t.List[str]):
        self.features_to_encode = features_to_encode
        self.encoders = None

    def fit(self, X, y = None):
        self.encoders = [LabelEncoder() for _ in range(len(self.features_to_encode))]
        for idx, feature in enumerate(self.features_to_encode):
            self.encoders[idx].fit(X[feature])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for idx, feature in enumerate(self.features_to_encode):
            X[feature] = self.encoders[idx].transform(X[feature])
        return X


class ImputerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_impute: list, strategy: str):
        self.features_to_impute = features_to_impute
        self.imputer = SimpleImputer(strategy=strategy)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.features_to_impute:
            X[feature] = self.imputer.fit_transform(X[feature].values.reshape(-1, 1))
        return X

