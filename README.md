# detector-deepfake-audio

Detecção de voz sintética (deepfake de áudio) com MFCC + scikit-learn.

Synthetic voice detection (audio deepfake) with MFCC + scikit-learn.

---

## Descrição / Description

**PT-BR.** Pacote Python que extrai coeficientes MFCC de um áudio e treina um
classificador binário para distinguir fala real de fala sintética. O escopo é o
MVP: extração de características com `librosa` e treinamento com `scikit-learn`.
A suíte de testes gera áudio determinístico (senóides com/sem ruído) via NumPy,
sem depender de arquivos de áudio externos.

**EN.** Python package that extracts MFCC coefficients from audio and trains a
binary classifier to tell real speech from synthetic speech. MVP scope: feature
extraction with `librosa` and training with `scikit-learn`. The test suite
generates deterministic audio (sine waves with/without noise) via NumPy, with no
external audio files.

---

## Requisitos / Requirements

Python 3.10+, `librosa`, `scikit-learn`, `numpy`.

---

## Instalação / Installation

```bash
git clone <repo>
cd detector-deepfake-audio
pip install -e .
```

---

## Como usar / How to use

### Extrair características MFCC

**PT-BR.** Extraia o vetor de características de um arquivo de áudio:

```python
from detector_deepfake_audio import extract_mfcc

features = extract_mfcc(audio_path="audio.wav", n_mfcc=13)
print(features.shape)  # (26,)
```

**EN.** Extract the feature vector from an audio file:

```python
from detector_deepfake_audio import extract_mfcc

features = extract_mfcc(audio_path="audio.wav", n_mfcc=13)
print(features.shape)  # (26,)
```

### Treinar o classificador

**PT-BR.** Treine com características `X` e rótulos `y` (0 = real, 1 = sintético):

```python
from detector_deepfake_audio import DeepfakeAudioDetector

detector = DeepfakeAudioDetector(n_mfcc=13, random_state=42)
result = detector.fit(X, y)
print(result.accuracy)   # acurácia de treino
print(result.n_samples)  # número de amostras
print(result.n_features) # número de características
```

**EN.** Train with features `X` and labels `y` (0 = real, 1 = synthetic):

```python
from detector_deepfake_audio import DeepfakeAudioDetector

detector = DeepfakeAudioDetector(n_mfcc=13, random_state=42)
result = detector.fit(X, y)
print(result.accuracy)   # training accuracy
print(result.n_samples)  # number of samples
print(result.n_features) # number of features
```

### Predizer um áudio

**PT-BR.** Classifique se um arquivo é real ou sintético:

```python
from detector_deepfake_audio import DeepfakeAudioDetector
import joblib

detector = joblib.load("detector_model.joblib")
pred = detector.predict(features.reshape(1, -1))[0]
label = "sintético" if pred == 1 else "real"
```

**EN.** Classify whether a file is real or synthetic:

```python
from detector_deepfake_audio import DeepfakeAudioDetector
import joblib

detector = joblib.load("detector_model.joblib")
pred = detector.predict(features.reshape(1, -1))[0]
label = "synthetic" if pred == 1 else "real"
```

### CLI

```bash
# treina a partir de um CSV com coluna 'label'
detector-deepfake-audio train --features features.csv --model-out detector_model.joblib

# prediz um arquivo de áudio
detector-deepfake-audio predict audio.wav --model detector_model.joblib
```

---

## Testes / Tests

**PT-BR.** A suíte usa `pytest` e não requer arquivos de áudio externos: gera
senóides determinísticas (com/sem ruído) com semente fixa, extrai MFCC, treina o
classificador e afirma acurácia > 0.85 em dataset de teste.

```bash
python -m pytest
```

Resultado esperado: todos os testes passam, incluindo
`test_accuracy_on_deterministic_audio` (acurácia > 0.85).

**EN.** The suite uses `pytest` and needs no external audio files: it generates
deterministic sine waves (with/without noise) with a fixed seed, extracts MFCC,
trains the classifier, and asserts accuracy > 0.85 on a test dataset.

```bash
python -m pytest
```

Expected result: all tests pass, including
`test_accuracy_on_deterministic_audio` (accuracy > 0.85).

Critério de aceitação / acceptance criterion: acurácia > 85% no dataset de teste.

---

## Licença / License

MIT. Consulte o arquivo `LICENSE` / see the `LICENSE` file.

---

## Autor / Author

Kelvin Oliveira — área IA/ML. Projeto `detector-deepfake-audio`.

---

Critérios de aceitação / acceptance criteria:
- acurácia > 85% em dataset de teste / accuracy > 85% on test dataset
- suíte pytest que passa / passing pytest suite
- README PT-BR + EN completo / complete PT-BR + EN README
