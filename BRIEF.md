# BRIEF — detector-deepfake-audio

**Projeto:** detector-deepfake-audio (área IA/ML) — detecção de voz sintética
com librosa + scikit-learn.
**Diretório de entrega:**
`C:\Users\Kelvin\AppData\Local\Temp\opencode\github-staging\projects\detector-deepfake-audio\`
**Autor:** Kelvin Oliveira

---

## O que foi entregue

| Arquivo | Função |
|---------|--------|
| `pyproject.toml` | Config do projeto, dependências (librosa, scikit-learn, numpy), entry point CLI, config pytest |
| `src/detector_deepfake_audio/__init__.py` | Pacote, exports públicos (`extract_mfcc`, `DeepfakeAudioDetector`, `TrainResult`) |
| `src/detector_deepfake_audio/features.py` | Extração de MFCC (média + desvio padrão ao longo do tempo) |
| `src/detector_deepfake_audio/model.py` | Classificador binário real vs sintético (StandardScaler + LogisticRegression) |
| `src/detector_deepfake_audio/cli.py` | CLI: `train` (CSV com coluna `label`) e `predict` (arquivo de áudio) |
| `tests/__init__.py` | Pacote de testes |
| `tests/test_detector.py` | Suíte pytest: forma dos MFCC, determinismo, erros de entrada, fit/predict/predict_proba, acurácia > 0.85 em áudio sintético determinístico |
| `README.md` | Documentação bilíngue PT-BR + EN (título, descrição, como usar, testes, licença MIT, autor) |
| `BRIEF.md` | Este arquivo |

---

## Critérios de aceitação (espec) — todos atendidos

- [x] Stack: Python, librosa, scikit-learn, numpy
- [x] Extrai MFCC e treina classificador binário real vs sintético
- [x] Acurácia > 85% em dataset de teste (verificado: `test_accuracy_on_deterministic_audio`)
- [x] Suíte pytest que passa
- [x] README PT-BR + EN completo
- [x] Áudio sintético determinístico no teste (senóides com/sem ruído), sem arquivos externos

---

## Como verificar

```bash
# 1. Entre no diretório do projeto
cd "C:\Users\Kelvin\AppData\Local\Temp\opencode\github-staging\projects\detector-deepfake-audio\"

# 2. Instale o pacote em modo editável
pip install -e .

# 3. Rode a suíde de testes
python -m pytest
```

**Comando de teste usado:** `python -m pytest` (equivalente ao gate `pytest -q` da skill python)

**Resultado dos testes:** 13 passed — todos os testes passam, incluindo
`test_accuracy_on_deterministic_audio` (acurácia > 0.85 no dataset de teste
gerado deterministicamente).

**Caminho do README:**
`C:\Users\Kelvin\AppData\Local\Temp\opencode\github-staging\projects\detector-deepfake-audio\README.md`

**Caminho do BRIEF:**
`C:\Users\Kelvin\AppData\Local\Temp\opencode\github-staging\projects\detector-deepfake-audio\BRIEF.md`

---

## Notas de implementação

- Áudio de teste gerado com NumPy e semente fixa (`RandomState(42)`): senóide
  pura (classe real = 0) vs senóide + ruído gaussiano (classe sintético = 1),
  sem depender de arquivos de áudio externos.
- Divisão treino/teste determinística e estratificada.
- Código com tipos, docstrings e sem comentários com cara de IA (skill antislop-code aplicada ao escrever).
- README revisado contra padrões de AI slop (sem palavras-banidas, sem colões dramáticos, sem puffery).
