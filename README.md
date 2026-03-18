# Regulatory Reference Extraction (REG_ML)

ML-Pipeline zur Erkennung deutscher Rechtsreferenzen (§, Artikel, Anhang, Anlage, ISO-Normen, etc.) in regulatorischen und vertraglichen Texten. Fine-tuned **GBERT-Large** NER-Modell mit BIO-Labeling, evaluiert gegen eine Regex-Baseline.

## Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. Generate │ ──▶ │   2. Train   │ ──▶ │ 3. Evaluate  │ ──▶ │  4. Predict  │
│  (Ollama LLM)│     │  (GBERT NER) │     │  (vs Regex)  │     │  (Inference) │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

## Quickstart

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Daten generieren (Ollama muss lokal laufen)
python3 run.py generate --config config/gpu.yaml

# 2. Training
python3 run.py train --config config/gpu.yaml

# 3. Evaluation
python3 run.py evaluate --config config/gpu.yaml

# 4. Inferenz
python3 run.py predict --text "Gemäß § 25a KWG sind die Anforderungen zu beachten." -c config/gpu.yaml
```

## Architektur

### Modell
- **Backbone**: `deepset/gbert-large` (German BERT, 335M Parameter)
- **Head**: Linear Classifier (3 Labels: O, B-REF, I-REF)
- **Optional**: CRF-Layer, LoRA-Adapter, Gradient Checkpointing
- **Loss**: Class-weighted CrossEntropyLoss (auto-berechnet aus Datendistribution)

### Trainingsdaten
- Synthetisch generiert via lokales **Ollama** (qwen2.5:7b)
- 60+ Dokumenttypen (Gesetze, Verträge, SLAs, Richtlinien, Gutachten, ...)
- `<ref>...</ref>` Tags → BIO-Labels via offset_mapping
- ~40% Negativbeispiele (Texte ohne Referenzen)

### Erkannte Referenztypen
| Typ | Beispiele |
|-----|-----------|
| Paragraphen | § 25a KWG, §§ 305-310 BGB |
| Artikel | Art. 5 DSGVO, Artikel 28 Abs. 1 |
| Absätze/Nummern | Abs. 1, Nr. 3, lit. a, Satz 2 |
| Vertragsklauseln | Ziffer 3.1, Punkt 4.2, Klausel 7 |
| Anhänge/Anlagen | Anhang A, Anlage 3, Anhang 4b |
| Abschnitte | Abschnitt 4.2, Kapitel 3, Teil B |
| Normen/Standards | ISO 27001, DIN EN 62305, IDW PS 330 |
| SLA-Verweise | SLA Ziffer 2.1, Service Level gemäß Anlage 2 |
| Verordnungen | EU-Verordnung 2022/2554 |

### Training Features
- **Class-Weighted Loss**: Auto-berechnete inverse Frequenz-Gewichte (O≈0.4, B-REF≈19, I-REF≈4)
- **Validation Split**: 10% Hold-out für Loss-Monitoring
- **Early Stopping**: Stoppt nach 3 Epochen ohne Val-Loss-Verbesserung
- **Mixed Precision**: fp16 auf CUDA, bf16 auf MPS
- **Differential LR**: Backbone 2e-5, Head 1e-4

## Konfiguration

Zwei YAML-Configs:
- `config/default.yaml` — CPU/Minimal (Tests, Debugging)
- `config/gpu.yaml` — Produktion (RTX 3090)

CLI-Overrides:
```bash
python3 run.py train --config config/gpu.yaml training.num_epochs=5 training.batch_size=8
```

## Projektstruktur

```
REG_ML/
├── run.py                        # CLI (generate, train, evaluate, predict)
├── config/
│   ├── default.yaml              # Basis-Config
│   └── gpu.yaml                  # GPU-Produktion
├── src/
│   ├── model/
│   │   ├── ner_model.py          # GBERT + Classifier + CRF/LoRA
│   │   ├── trainer.py            # Training Loop, Early Stopping, Checkpoints
│   │   └── predictor.py          # Inferenz
│   ├── data/
│   │   ├── generate_dataset.py   # Parallele LLM-Datengenerierung
│   │   ├── llm_client.py         # Ollama Client + Prompt Builder
│   │   ├── bio_converter.py      # Char-Spans → BIO-Labels
│   │   ├── dataset.py            # IterableDataset
│   │   └── cache.py              # JSONL Cache
│   ├── evaluation/
│   │   ├── evaluator.py          # Model vs Baseline Vergleich
│   │   ├── regex_baseline.py     # 10-Typ Regex Baseline
│   │   └── metrics.py            # seqeval P/R/F1
│   └── utils/
│       ├── config.py             # OmegaConf Loader
│       └── device.py             # CUDA/MPS/CPU Detection
├── tests/                        # 150 Tests
├── data/
│   └── gold_test/                # Gold-Testset
├── requirements.txt
└── .env.example
```

## Tech Stack

PyTorch, HuggingFace Transformers, pytorch-crf, PEFT (LoRA), Accelerate, OmegaConf, httpx, seqeval

## Voraussetzungen

- Python 3.10+
- CUDA GPU (empfohlen: RTX 3090+) oder Apple Silicon (MPS)
- [Ollama](https://ollama.ai) lokal mit `qwen2.5:7b` für Datengenerierung

## Lizenz

Internes PoC — nicht zur öffentlichen Verteilung.
