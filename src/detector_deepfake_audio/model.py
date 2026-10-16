"""Treinamento e inferência do classificador binário real vs sintético."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TrainResult:
    """Resultado do treinamento do classificador.

    Attributes:
        accuracy: acurácia de treino do modelo ajustado.
        n_samples: número de amostras usadas no treinamento.
        n_features: número de características por amostra.
    """

    accuracy: float
    n_samples: int
    n_features: int


class DeepfakeAudioDetector:
    """Detector de áudio sintético (deepfake) baseado em MFCC + regressão logística.

    O pipeline internamente aplica :class:`~sklearn.preprocessing.StandardScaler`
    seguido de :class:`~sklearn.linear_model.LogisticRegression`.

    Attributes:
        n_mfcc: número de coeficientes MFCC usados nas características.
        random_state: semente para reprodutibilidade do treinamento.
    """

    def __init__(self, n_mfcc: int = 13, random_state: int = 42) -> None:
        self.n_mfcc = n_mfcc
        self.random_state = random_state
        self._pipeline: Pipeline | None = None
        self._is_fitted = False

    def _build_pipeline(self) -> Pipeline:
        """Constrói o pipeline de pré-processamento + classificador."""
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainResult:
        """Treina o classificador com as características e rótulos.

        Args:
            X: matriz de características com shape (n_samples, n_features).
            y: rótulos binários (0 = real, 1 = sintético) com shape (n_samples,).

        Returns:
            :class:`TrainResult` com acurácia de treino e metadados.

        Raises:
            ValueError: se ``X`` não for 2D ou se ``y`` não tiver uma linha por
                amostra de ``X``.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2:
            raise ValueError("X deve ser uma matriz 2D (n_samples, n_features).")
        if y.shape[0] != X.shape[0]:
            raise ValueError("y deve ter uma linha por amostra de X.")

        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X, y)
        self._is_fitted = True

        accuracy = float(self._pipeline.score(X, y))
        return TrainResult(
            accuracy=accuracy,
            n_samples=int(X.shape[0]),
            n_features=int(X.shape[1]),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz rótulos para as características.

        Args:
            X: matriz de características com shape (n_samples, n_features).

        Returns:
            Vetor de rótulos preditos (0 = real, 1 = sintético).

        Raises:
            RuntimeError: se o modelo ainda não foi treinado.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return self._pipeline.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna as probabilidades de cada classe para as características.

        Args:
            X: matriz de características com shape (n_samples, n_features).

        Returns:
            Array com shape (n_samples, 2): [P(real), P(sintético)].

        Raises:
            RuntimeError: se o modelo ainda não foi treinado.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return self._pipeline.predict_proba(X)  # type: ignore[union-attr]

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._pipeline is None:
            raise RuntimeError("O modelo ainda não foi treinado. Chame fit().")

    @property
    def is_fitted(self) -> bool:
        """True se o modelo já foi treinado."""
        return self._is_fitted
