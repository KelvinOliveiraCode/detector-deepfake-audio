"""Interface de linha de comando do detector de deepfake de áudio."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import joblib
import numpy as np

from .features import extract_mfcc
from .model import DeepfakeAudioDetector


def build_parser() -> argparse.ArgumentParser:
    """Constrói o parser de argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        prog="detector-deepfake-audio",
        description="Detecta áudio sintético (deepfake) a partir de características MFCC.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser(
        "train",
        help="Treina o classificador a partir de um CSV de características.",
    )
    train_p.add_argument(
        "--features",
        required=True,
        type=Path,
        help="CSV com características e uma coluna 'label' (0=real, 1=sintético).",
    )
    train_p.add_argument(
        "--model-out",
        type=Path,
        default=Path("detector_model.joblib"),
        help="Caminho de saída do modelo treinado (joblib).",
    )

    predict_p = sub.add_parser(
        "predict",
        help="Prediz se um áudio é real ou sintético.",
    )
    predict_p.add_argument(
        "audio",
        type=Path,
        help="Arquivo de áudio a classificar.",
    )
    predict_p.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Caminho do modelo treinado (joblib).",
    )

    return parser


def _load_feature_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Carrega características e rótulos de um CSV.

    Espera uma coluna chamada ``label`` e as demais colunas como características.

    Returns:
        Tupla ``(X, y)`` com as características e os rótulos.
    """
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "label" not in reader.fieldnames:
            raise ValueError("O CSV deve conter uma coluna 'label'.")
        feature_cols = [c for c in reader.fieldnames if c != "label"]
        for row in reader:
            rows.append(([float(row[c]) for c in feature_cols], int(row["label"])))
    if not rows:
        raise ValueError("O CSV não contém linhas de dados.")
    X = np.array([r[0] for r in rows], dtype=float)
    y = np.array([r[1] for r in rows], dtype=int)
    return X, y


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada da CLI.

    Args:
        argv: argumentos da linha de comando (usa sys.argv quando None).

    Returns:
        Código de saída do processo.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        X, y = _load_feature_csv(args.features)
        detector = DeepfakeAudioDetector()
        result = detector.fit(X, y)
        joblib.dump(detector, args.model_out)
        print(
            f"Treinado: acurácia={result.accuracy:.4f} "
            f"amostras={result.n_samples} features={result.n_features} "
            f"saída={args.model_out}"
        )
        return 0

    if args.command == "predict":
        if not args.model.exists():
            print(f"Modelo não encontrado: {args.model}", file=sys.stderr)
            return 1
        detector = joblib.load(args.model)
        features = extract_mfcc(audio_path=str(args.audio))
        pred = int(detector.predict(features.reshape(1, -1))[0])
        proba = detector.predict_proba(features.reshape(1, -1))[0]
        label = "sintético" if pred == 1 else "real"
        print(
            f"Áudio: {args.audio} -> {label} "
            f"(P(real)={proba[0]:.4f}, P(sintético)={proba[1]:.4f})"
        )
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
