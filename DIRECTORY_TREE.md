# Project Directory Tree

```
Few-Shot-Domain-Adaptation-for-Medical-Image-Classification/
│
├── 📋 Configuration
│   └── configs/
│       └── config.py                      # Central config dataclass
│
├── 📊 Data Pipeline
│   └── data/
│       └── datasets.py                    # CheXpert, NIH loaders + few-shot sampling
│
├── 🧠 Models
│   └── models/
│       ├── vit_backbone.py                # Vision Transformer (timm)
│       └── cnn_backbones.py               # ResNet, DenseNet
│
├── 🔧 Adaptation Modules
│   ├── lora/
│   │   └── lora.py                        # LoRA implementation
│   ├── adapters/
│   │   └── adapter.py                     # Adapter layers
│   └── prompts/
│       └── prompt_tuning.py               # Visual prompt tuning
│
├── 🚂 Training
│   └── train/
│       └── trainer.py                     # PyTorch Lightning module
│
├── 📈 Evaluation
│   └── eval/
│       └── evaluator.py                   # Metrics + bootstrap CIs
│
├── 🛠️ Utilities
│   └── utils/
│       └── utils.py                       # Helper functions
│
├── 📜 Scripts
│   └── scripts/
│       ├── run_train.sh                   # Batch training
│       └── run_eval.sh                    # Evaluation
│
├── 📓 Notebooks
│   └── notebooks/
│       └── colab_end_to_end.ipynb        # Complete Colab tutorial
│
├── 🎯 Main Entry Points
│   ├── run_training.py                    # CLI training driver ⭐
│   └── example_quick_start.py             # Minimal example
│
├── 📚 Documentation
│   ├── README.md                          # Main documentation ⭐
│   ├── WORKFLOW.md                        # Step-by-step guide
│   ├── FILE_SUMMARY.md                    # File-by-file docs
│   ├── PROJECT_SUMMARY.md                 # Complete deliverables
│   └── DIRECTORY_TREE.md                  # This file
│
├── 📦 Dependencies
│   └── requirements.txt                   # Python packages
│
└── 🗂️ Runtime Directories (created during execution)
    ├── checkpoints/                       # Saved model weights
    │   ├── baseline_vit/
    │   ├── lora_adaptation/
    │   ├── adapter_adaptation/
    │   └── prompt_tuning/
    ├── logs/                              # Training logs
    └── data/                              # Dataset files (user-provided)
        ├── chexpert_train.csv
        ├── chexpert_val.csv
        ├── nih_train.csv
        ├── nih_val.csv
        ├── nih_test.csv
        └── images/
            └── ...
```

---

## 📂 Directory Purposes

### Core Implementation (Required)

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `configs/` | Configuration management | `config.py` |
| `data/` | Dataset loading & preprocessing | `datasets.py` |
| `models/` | Neural network architectures | `vit_backbone.py`, `cnn_backbones.py` |
| `lora/` | LoRA adaptation | `lora.py` |
| `adapters/` | Adapter layers | `adapter.py` |
| `prompts/` | Prompt tuning | `prompt_tuning.py` |
| `train/` | Training infrastructure | `trainer.py` |
| `eval/` | Evaluation metrics | `evaluator.py` |
| `utils/` | Helper utilities | `utils.py` |

### Execution & Scripts

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `scripts/` | Automation scripts | `run_train.sh`, `run_eval.sh` |
| `notebooks/` | Interactive tutorials | `colab_end_to_end.ipynb` |
| Root | Main executables | `run_training.py`, `example_quick_start.py` |

### Documentation

| File | Purpose | Lines |
|------|---------|-------|
| `README.md` | Quick start & API reference | ~350 |
| `WORKFLOW.md` | Detailed workflow guide | ~600 |
| `FILE_SUMMARY.md` | File-by-file documentation | ~350 |
| `PROJECT_SUMMARY.md` | Complete deliverables summary | ~450 |
| `DIRECTORY_TREE.md` | This file | ~200 |

### Runtime (Auto-created)

| Directory | Purpose | Created By |
|-----------|---------|------------|
| `checkpoints/` | Model weights | Training scripts |
| `logs/` | Training logs | PyTorch Lightning |
| `data/` | Dataset files | User (manual setup) |

---

## 🎯 Quick Navigation

### Want to...

**Run a quick test?**
→ `example_quick_start.py`

**Train a model?**
→ `run_training.py` or `notebooks/colab_end_to_end.ipynb`

**Understand the code?**
→ `FILE_SUMMARY.md` → Specific module file

**Follow a workflow?**
→ `WORKFLOW.md`

**Learn about the project?**
→ `README.md` → `PROJECT_SUMMARY.md`

**Implement a new method?**
→ `models/` or adaptation folders (`lora/`, `adapters/`, `prompts/`)

**Debug an issue?**
→ `WORKFLOW.md` Troubleshooting section

---

## 📊 Code Statistics

| Category | Files | Lines of Code |
|----------|-------|---------------|
| Core modules | 9 | ~600 |
| Training & eval | 2 | ~150 |
| Main scripts | 2 | ~350 |
| Documentation | 5 | ~2000 |
| Notebooks | 1 | ~500 |
| **Total** | **19** | **~3,600** |

---

## 🔄 File Dependencies

```
run_training.py
├── configs/config.py
├── data/datasets.py
├── models/
│   ├── vit_backbone.py
│   └── cnn_backbones.py
├── lora/lora.py
├── adapters/adapter.py
├── prompts/prompt_tuning.py
├── train/trainer.py
│   └── eval/evaluator.py
│       └── utils/utils.py
└── utils/utils.py
```

---

## 💾 Storage Requirements

| Component | Size |
|-----------|------|
| Source code | <1 MB |
| Documentation | <500 KB |
| Pre-trained ViT weights | ~330 MB |
| CheXpert dataset | ~440 GB |
| NIH ChestX-ray14 | ~45 GB |
| Checkpoints (per experiment) | ~350 MB |
| Training logs | ~10 MB |

**Minimum storage:** ~500 GB (with both datasets)
**Recommended storage:** 1 TB

---

## 🚀 Getting Started Path

```
1. Clone/Download
   ↓
2. Install dependencies (requirements.txt)
   ↓
3. Prepare datasets → data/
   ↓
4. Choose your path:
   ├─→ Quick test: example_quick_start.py
   ├─→ CLI training: run_training.py
   └─→ Colab: notebooks/colab_end_to_end.ipynb
   ↓
5. Results in checkpoints/ and logs/
```

---

## 📱 Accessibility Map

**Level 1: Beginners**
- Start: `README.md`
- Try: `example_quick_start.py`
- Learn: `notebooks/colab_end_to_end.ipynb`

**Level 2: Practitioners**
- Start: `WORKFLOW.md`
- Use: `run_training.py` with CLI args
- Reference: `FILE_SUMMARY.md`

**Level 3: Researchers**
- Study: Implementation files in `lora/`, `adapters/`, `prompts/`
- Extend: Add new modules
- Benchmark: Use `eval/evaluator.py`

---

## 🎨 Color Legend

- 📋 Configuration
- 📊 Data
- 🧠 Models
- 🔧 Adaptation
- 🚂 Training
- 📈 Evaluation
- 🛠️ Utilities
- 📜 Scripts
- 📓 Notebooks
- 🎯 Entry Points
- 📚 Documentation
- 📦 Dependencies
- 🗂️ Runtime

---

**Generated:** November 24, 2025
**Version:** 1.0.0
