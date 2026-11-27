# SARC-FuseNet 🛡️

**Sarcasm-Aware Residual Context Fusion Network for Hindi–English (Hinglish) Code-Mixed Hate Speech, Target, and Severity Detection.**

SARC-FuseNet detects hate speech—specifically identifying nuanced and covert *sarcastic hate*—and simultaneously predicts the **target** and **severity** level of each detected attack with token-level rationales and explainability.

```
                          ┌────────────────────────┐
                          │ Code-Mixed Hinglish    │
                          └───────────┬────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        ┌──────────────────────┐              ┌──────────────────────┐
        │ XLM-RoBERTa Backbone │              │ Pragmatic Sarcasm    │
        │ & Sub-Word Tokenizer │              │ Surface Cues Vector  │
        └──────────┬───────────┘              └──────────┬───────────┘
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Sarcasm Branch Head     │
                        │   P(Sarcasm | Text)       │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ Gated Sarcasm Residual    │
                        │ Fusion Module             │
                        └─────────────┬─────────────┘
                                      │
                   ┌──────────────────┼──────────────────┐
                   ▼                  ▼                  ▼
        ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
        │   Hate Speech    │ │  Target Head     │ │  Severity Head   │
        │   Detection      │ │ (Individual /    │ │ (Low / Moderate /│
        │ (Non-Hate / Hate)│ │  Org/Comm/Society│ │  High)           │
        └──────────────────┘ └──────────────────┘ └──────────────────┘
```

---

## Benchmark Results (Stage 3 Tri-Dataset Benchmark)

SARC-FuseNet was trained and evaluated on the unified **56,907-post multi-task benchmark**, integrating multimodal meme texts from **MultiBully** ($5,718$), real-world code-mixed tweets from **HOT** ($3,189$), and the extensive **Hinglish Toxic Hate corpus** ($48,000$).

### Official Evaluation Scorecard (Held-Out Test Set, $N = 8,513$ posts)

| Task / Objective | Macro-F1 | Accuracy | AUROC | Balanced Acc | Error Rate ($1-\text{Acc}$) | True FNR $\left(\frac{\text{FN}}{\text{FN}+\text{TP}}\right)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Multi-Task Average** | **`0.9714`** | — | — | — | — | — |
| **Hate Speech Detection (`hate`)** | **`0.9665`** | `0.9666` | **`0.9964`** | `0.9665` | `3.34%` | **`3.62%`** |
| **Sarcasm Detection (`sarcasm`)\*** | **`0.9548`** | `0.9550` | **`0.9943`** | `0.9540` | `4.50%` | **`6.95%`** |
| **Target Identification (`target`)** | **`0.9822`** | `0.9813` | — | `0.9820` | `1.87%` | — |
| **Severity Prediction (`severity`)** | **`0.9821`** | `0.9817` | — | `0.9821` | `1.83%` | — |
| **Hate Detection on Sarcastic Posts** | **`0.9699`** | — | — | — | — | **`3.84%`** |
| **Hate Detection on Non-Sarcastic Posts**| **`0.9628`** | — | — | — | — | **`2.87%`** |

*\*Note on Sarcasm Support: 118 posts in the HOT test set lacked sarcasm ground truth and were masked during evaluation ($N_{\text{evaluated}} = 8,395$).*

*For complete training dynamics, cross-dataset split details, and ablation comparisons, see [STAGE_3_FINAL_REPORT.md](STAGE_3_FINAL_REPORT.md) and [SARC-FUSENET_FULL_PROJECT_REPORT.md](reports/SARC-FUSENET_FULL_PROJECT_REPORT.md).*

---

## Dataset Provenance & References

The unified **56,907** row corpus combines three curated sources:

1. **Hinglish Toxic Hate Dataset ($48,000$ rows):**
   - **Repository / URL:** [Kaggle: Hinglish Toxic Hate Dataset](https://www.kaggle.com/datasets/subamraz/hinglish-toxic-hate-dataset/data)
   - **Characteristics:** Large-scale Hinglish social text corpus featuring comprehensive multi-task annotations across hate speech, sarcasm, attack targets (individual, organization, community, society), and fine-grained severity ratings (low, moderate, high).

2. **MultiBully Dataset ($5,718$ rows):**
   - **Repository / URL:** [Kaggle: Multi-Bully Dataset](https://www.kaggle.com/datasets/pranee5/multi-bully)
   - **Reference Paper:** Maity et al., *"MultiBully: Multimodal Cyberbullying Detection in Code-Mixed Social Media Posts"*, ACM SIGIR 2022.
   - **Characteristics:** Multimodal memes with annotations for cyberbullying, sarcasm, victim target, and harm score.

3. **HOT (Hindi Offensive Tweet) Dataset ($3,189$ rows):**
   - **Repository / URL:** [GitHub: HOT_Dataset_modified.csv](https://github.com/ramitsawhney27/OffensiveHinglishTweetClassification/blob/master/HOT_Dataset_modified.csv)
   - **Reference Project:** Ramit Sawhney et al., *Offensive Hinglish Tweet Classification*.
   - **Characteristics:** Code-mixed Twitter/X posts, enriched with Qwen multi-task annotations for target and severity.

---

## Quick Start & Inference

### 1. Installation

```powershell
# Create & activate environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install the SARC-FuseNet package in editable mode
pip install -e .
```

### 2. Run Inference on Any Text

Test the production model using the CLI:

```powershell
python -m sarc_fusenet.cli predict `
  --run-dir runs/exp_tri_dataset_xlmr `
  --text "Congrats bro, strict diet ke beech spicy golgappe ne prove kar diya ki hope pe world kayam hai" `
         "Saale ye lobby ke log itna disgusting aur pathetic harkat karta hai" `
         "Aloo paratha bahut tasty tha yaar"
```

The output returns JSON predictions containing:
- Binary sarcasm classification (`sarcastic` vs `not_sarcastic`)
- Hate speech probability (`non_hate` vs `hate`)
- Hate-conditional target prediction (`individual`, `organization`, `community`, `society`)
- Severity rating (`low`, `moderate`, `high`)
- Gated fusion attention weights and token-level attribution rationales

---

## Reproducing Audited Metrics

To verify and reproduce all official test metrics, confusion matrices, and per-source performance from the held-out test set ($N = 8,513$) without needing GPUs:

```bash
python scripts/audit_metrics.py
python scripts/audit_by_source.py
```

The full test predictions, validation predictions, and training history curves are permanently preserved in `results/stage3_tri_dataset/`.

---

## Reproducing Training

To re-run training on the unified tri-dataset from scratch:

```powershell
python -m sarc_fusenet.cli train --config configs/combined_all_3_datasets.yaml
```

The configuration automatically engages GPU acceleration (CUDA with FP16 AMP) and saves checkpoints and evaluation metrics to `runs/exp_tri_dataset_xlmr/`.

---

## Directory Structure

```
SARC-FuseNet/
├── configs/
│   └── combined_all_3_datasets.yaml
├── data/
│   └── processed/
│       ├── combined_all_3_datasets.csv
│       └── combined_all_3_datasets.meta.json
├── docs/
│   └── SARC-FuseNet_Project_Report.docx
├── reports/
│   ├── STAGE_3_FINAL_REPORT.md
│   ├── SARC-FUSENET_FULL_PROJECT_REPORT.md
│   ├── sarc-fusenet_final_tri_dataset_evaluation.txt
│   └── audit/
├── results/
├── scripts/
├── src/
│   └── sarc_fusenet/
│       └── models/
│           └── sarc_fusenet.py
├── tests/
├── pyproject.toml
└── README.md
```
