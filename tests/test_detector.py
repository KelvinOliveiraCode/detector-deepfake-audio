"""Suíte de testes do detector-deepfake-audio.

Gera áudio sintético determinístico (senóides com/sem ruído) usando NumPy com
semente fixa, de modo que a suíte não dependa de arquivos de áudio externos.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

from detector_deepfake_audio import DeepfakeAudioDetector, TrainResult, extract_mfcc

SAMPLE_RATE = 22050
DURATION = 1.0
N_MFCC = 13
SEED = 42


def _sine_wave(frequency: float, duration: float = DURATION, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Gera uma senóide pura (áudio "real", sem ruído)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2.0 * np.pi * frequency * t).astype(np.float32)


def _sine_with_noise(
    frequency: float,
    noise_level: float = 0.15,
    duration: float = DURATION,
    sr: int = SAMPLE_RATE,
    seed: int = SEED,
) -> np.ndarray:
    """Gera uma senóide com ruído gaussiano determinístico (áudio "sintético")."""
    signal = _sine_wave(frequency, duration, sr)
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, noise_level, size=signal.shape).astype(np.float32)
    return (signal + noise).astype(np.float32)


def _make_dataset(
    n_per_class: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """Constrói dataset balanceado real vs sintético a partir de senóides.

    Returns:
        (X, y) com X = características MFCC e y = rótulos (0=real, 1=sintético).
    """
    X_list = []
    y_list = []
    for i in range(n_per_class):
        # Real: senóide pura em frequências variadas, sem ruído.
        freq_real = 200.0 + 50.0 * i
        X_list.append(extract_mfcc(y=_sine_wave(freq_real), sr=SAMPLE_RATE, n_mfcc=N_MFCC))
        y_list.append(0)
        # Sintético: mesma senóide + ruído gaussiano determinístico.
        freq_syn = 200.0 + 50.0 * i
        X_list.append(
            extract_mfcc(
                y=_sine_with_noise(freq_syn, noise_level=0.2, seed=SEED + i),
                sr=SAMPLE_RATE,
                n_mfcc=N_MFCC,
            )
        )
        y_list.append(1)
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    return X, y


def _split_train_test(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.33, seed: int = SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide X/y em treino e teste de forma determinística e estratificada."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx_real = np.where(y == 0)[0]
    idx_syn = np.where(y == 1)[0]
    rng.shuffle(idx_real)
    rng.shuffle(idx_syn)
    n_test_real = max(1, int(len(idx_real) * test_size))
    n_test_syn = max(1, int(len(idx_syn) * test_size))
    test_idx = np.sort(np.concatenate([idx_real[:n_test_real], idx_syn[:n_test_syn]]))
    train_idx = np.sort(np.concatenate([idx_real[n_test_real:], idx_syn[n_test_syn:]]))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class TestExtractMfcc:
    """Testes da função extract_mfcc."""

    def test_returns_correct_shape(self) -> None:
        """O vetor MFCC deve ter 2 * n_mfcc características."""
        y = _sine_wave(440.0)
        feat = extract_mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        assert feat.shape == (2 * N_MFCC,)

    def test_is_deterministic(self) -> None:
        """O mesmo sinal deve produzir o mesmo vetor de características."""
        y = _sine_wave(440.0)
        f1 = extract_mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        f2 = extract_mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        np.testing.assert_array_equal(f1, f2)

    def test_different_signals_produce_different_features(self) -> None:
        """Sinais diferentes devem produzir vetores diferentes."""
        f_low = extract_mfcc(y=_sine_wave(200.0), sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        f_high = extract_mfcc(y=_sine_wave(800.0), sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        assert not np.allclose(f_low, f_high)

    def test_raises_when_no_input(self) -> None:
        """Deve levantar ValueError se nem audio_path nem y forem informados."""
        with pytest.raises(ValueError, match="audio_path|sinal y"):
            extract_mfcc()  # type: ignore[call-arg]

    def test_raises_when_y_without_sr(self) -> None:
        """Deve levantar ValueError se y for passado sem sr."""
        with pytest.raises(ValueError, match="sr"):
            extract_mfcc(y=_sine_wave(440.0))  # type: ignore[call-arg]


class TestDeepfakeAudioDetector:
    """Testes do classificador DeepfakeAudioDetector."""

    def test_fit_returns_train_result(self) -> None:
        """fit() deve retornar um TrainResult com metadados coerentes."""
        X, y = _make_dataset(n_per_class=20)
        detector = DeepfakeAudioDetector(n_mfcc=N_MFCC, random_state=SEED)
        result = detector.fit(X, y)
        assert isinstance(result, TrainResult)
        assert result.n_samples == X.shape[0]
        assert result.n_features == X.shape[1]
        assert 0.0 <= result.accuracy <= 1.0

    def test_predict_before_fit_raises(self) -> None:
        """predict() antes de fit() deve levantar RuntimeError."""
        detector = DeepfakeAudioDetector()
        with pytest.raises(RuntimeError, match="não foi treinado"):
            detector.predict(np.zeros((1, 2 * N_MFCC)))

    def test_predict_proba_before_fit_raises(self) -> None:
        """predict_proba() antes de fit() deve levantar RuntimeError."""
        detector = DeepfakeAudioDetector()
        with pytest.raises(RuntimeError, match="não foi treinado"):
            detector.predict_proba(np.zeros((1, 2 * N_MFCC)))

    def test_predict_output_shape_and_values(self) -> None:
        """predict() deve retornar rótulos 0 ou 1 com shape (n_samples,)."""
        X, y = _make_dataset(n_per_class=20)
        detector = DeepfakeAudioDetector(random_state=SEED)
        detector.fit(X, y)
        preds = detector.predict(X)
        assert preds.shape == (X.shape[0],)
        assert set(np.unique(preds).tolist()) <= {0, 1}

    def test_predict_proba_output_shape(self) -> None:
        """predict_proba() deve retornar shape (n_samples, 2) somando 1."""
        X, y = _make_dataset(n_per_class=20)
        detector = DeepfakeAudioDetector(random_state=SEED)
        detector.fit(X, y)
        proba = detector.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_accuracy_on_deterministic_audio(self) -> None:
        """Acurácia em dataset de teste (senóides com/sem ruído) deve exceder 0.85."""
        X, y = _make_dataset(n_per_class=60)
        X_train, X_test, y_train, y_test = _split_train_test(X, y, test_size=0.33, seed=SEED)
        detector = DeepfakeAudioDetector(n_mfcc=N_MFCC, random_state=SEED)
        detector.fit(X_train, y_train)
        preds = detector.predict(X_test)
        accuracy = float(np.mean(preds == y_test))
        assert accuracy > 0.85, f"Acurácia {accuracy:.3f} não excede 0.85"

    def test_detector_is_fitted_after_fit(self) -> None:
        """A propriedade is_fitted deve ser True após fit()."""
        X, y = _make_dataset(n_per_class=10)
        detector = DeepfakeAudioDetector()
        assert detector.is_fitted is False
        detector.fit(X, y)
        assert detector.is_fitted is True

    def test_fit_raises_on_wrong_shapes(self) -> None:
        """fit() deve rejeitar X não 2D e y com tamanho incompatível."""
        detector = DeepfakeAudioDetector()
        with pytest.raises(ValueError, match="matriz 2D"):
            detector.fit(np.zeros((5,)), np.zeros(5, dtype=int))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="uma linha por amostra"):
            detector.fit(np.zeros((5, 4)), np.zeros(3, dtype=int))  # type: ignore[arg-type]
