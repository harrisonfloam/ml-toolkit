"""
This module defines the abstract base model class and a generic wrapper for any estimator.

Usage Example:

    from model.base import GenericWrapper
    from sklearn.linear_model import LogisticRegression

    # Initialize a scikit-learn estimator
    estimator = LogisticRegression()

    # Wrap the estimator using the GenericWrapper
    model = GenericWrapper(estimator)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate if the estimator supports scoring
    try:
        score = model.score(X_test, y_test)
    except NotImplementedError:
        score = None

    # Persist the model to disk
    model.save("model.pkl")

    # Load the model later
    model = GenericWrapper.load("model.pkl")
"""

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


class GenericModel(AbstractBaseModel):
    """
    A wrapper for any estimator supporting fit and predict.
    Uses pickle for model persistence.
    """

    def __init__(self, estimator: object):
        self.estimator = estimator

    def fit(self, *args, **kwargs) -> GenericModel:
        self.estimator.fit(*args, **kwargs)
        return self

    def predict(self, *args, **kwargs):
        return self.estimator.predict(*args, **kwargs)

    def score(self, X, y) -> float:
        if hasattr(self.estimator, "score"):
            return self.estimator.score(X, y)
        raise NotImplementedError("Underlying estimator does not support scoring.")
