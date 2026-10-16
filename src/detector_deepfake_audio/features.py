"""Extração de características de áudio (MFCC) para o detector de deepfake."""

from __future__ import annotations

from typing import Optional

import librosa
import numpy as np


def extract_mfcc(
    audio_path: Optional[str] = None,
    y: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Extrai um vetor de características MFCC de um áudio.

    O vetor de saída concatena a média e o desvio padrão dos coeficientes MFCC
    ao longo do tempo, produzindo uma representação determinística e de
    tamanho fixo (2 * n_mfcc) independente da duração do áudio.

    Args:
        audio_path: caminho do arquivo de áudio. Obrigatório quando ``y`` não
            for fornecido.
        y: sinal de áudio já carregado (amostras). Obrigatório quando
            ``audio_path`` não for fornecido.
        sr: taxa de amostragem do sinal ``y``. Obrigatório quando ``y`` for
            informado.
        n_mfcc: número de coeficientes MFCC a extrair.
        n_fft: tamanho da janela FFT (em amostras).
        hop_length: tamanho do salto entre quadros consecutivos (em amostras).

    Returns:
        Array NumPy 1D com 2 * n_mfcc características (média + desvio padrão
        dos MFCCs por coeficiente).

    Raises:
        ValueError: se nem ``audio_path`` nem ``y`` forem fornecidos, ou se
            ``y`` for informado sem ``sr``.
    """
    if y is None:
        if audio_path is None:
            raise ValueError("Informe audio_path ou o sinal y.")
        y_loaded, sr_loaded = librosa.load(audio_path, sr=None)
        y = y_loaded
        if sr is None:
            sr = sr_loaded
    elif sr is None:
        raise ValueError("Informe sr quando passar o sinal y.")

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    # Característica determinística: média e desvio padrão ao longo do tempo.
    feature = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return feature


def extract_features_from_signal(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Extrai características MFCC a partir de um sinal de áudio em memória.

    Wrapper conveniente sobre :func:`extract_mfcc` para sinais já carregados.

    Args:
        y: sinal de áudio (amostras).
        sr: taxa de amostragem do sinal.
        n_mfcc: número de coeficientes MFCC.
        n_fft: tamanho da janela FFT.
        hop_length: tamanho do salto entre quadros.

    Returns:
        Array NumPy 1D com 2 * n_mfcc características.
    """
    return extract_mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
