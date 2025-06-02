from __future__ import annotations

import pickle
from abc import ABC, abstractmethod


class AbstractBaseModel(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs) -> AbstractBaseModel:
        """Fit the model. Should return self."""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Make predictions using the model."""
        pass

    def score(self, X, y) -> float:
        """
        Optional score method.
        Override in subclass if scoring is supported,
        otherwise NotImplementedError is raised.
        """
        raise NotImplementedError("Score method not implemented.")

    def save(self, filepath: str) -> None:
        """Persist the model to a file using pickle."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> AbstractBaseModel:
        """Load a persisted model from a file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


class GenericModelAdapter(AbstractBaseModel):
    """
    A wrapper for any estimator supporting fit and predict.
    Uses pickle for model persistence.
    """

    def __init__(self, estimator: object):
        self.estimator = estimator

    def fit(self, *args, **kwargs) -> GenericModelAdapter:
        self.estimator.fit(*args, **kwargs)
        return self

    def predict(self, *args, **kwargs):
        return self.estimator.predict(*args, **kwargs)

    def score(self, X, y) -> float:
        if hasattr(self.estimator, "score"):
            return self.estimator.score(X, y)
        raise NotImplementedError(f"{type(self.estimator)} does not support scoring.")
