"""detector-deepfake-audio — detecção de voz sintética com MFCC + scikit-learn."""

from .features import extract_mfcc
from .model import DeepfakeAudioDetector, TrainResult

__all__ = ["extract_mfcc", "DeepfakeAudioDetector", "TrainResult"]
__version__ = "0.1.0"
