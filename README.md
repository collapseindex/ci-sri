# SRI + CI: AG News Public Validation

Reproducible demonstration of **Structural Retention Index (SRI)** alongside Collapse Index (CI) on multi-class text classification.

**Why AG News?** 4-class classification (World, Sports, Business, Sci/Tech) provides richer entropy signals for SRI validation. Multi-class enables detailed failure mode analysis through CSI typing and SRI grading.

## 🎯 Results (Preliminary)

**Reproducible Metrics (Public):**

| Metric | Value | Notes |
|--------|-------|-------|
| **Model** | BERT-AG-News | HuggingFace fine-tuned model |
| **Benchmark Accuracy** | 90.8% | Base examples (clean text) |
| **Overall Accuracy** | 90.4% | Including perturbations |
| **Flip Rate** | 9.2% | 46/500 base examples flip |
| **Dataset Size** | 2,000 rows | 500 base × 4 variants each |
| **Class Balance** | ~25% each | World, Sports, Business, Sci/Tech |
| **Confidence Gap** | 0.028 | Errors vs correct (small) |

**Advanced Metrics (Proprietary Pipeline):**

| Metric | Value | Notes |
|--------|-------|-------|
| **CI Score (avg)** | 0.019 | Prediction instability metric |
| **SRI Score (avg)** | 0.981 | Structural retention metric |
| **CI + SRI** | 1.000 | Perfect complementarity |
| **AUC(CI)** | 0.874 | Error discrimination via instability |
| **AUC(SRI)** | 0.874 | Error discrimination via retention |
| **AUC(Confidence)** | 0.171 | Baseline (vastly outperformed) |
| **CSI Distribution** | 35/20/1/0/0 | Type I/II/III/IV/V breakdown |

*Note: Advanced metrics require commercial licensing. Contact ask@collapseindex.org or visit [collapseindex.org/evals.html](https://collapseindex.org/evals.html)*

## 📊 The SRI Story

**Standard benchmarks say:** "Ship it! 90.8% accuracy."

**Reality under perturbations:** Models exhibit different failure modes classified by the Collapse Severity Index (CSI):
- **Type I:** Stable Collapse - Confidently wrong, no flips (most dangerous)
- **Type II:** Hidden Instability - Internal shifts, same label (hidden brittleness)
- **Type III:** Moderate Flip - Clear label flips under stress
- **Type IV-V:** High/Extreme Flip - Severe instability or chaotic breakdown

*Classification thresholds remain proprietary to prevent adversarial optimization.*

**Why SRI + CI matters:** SRI provides structural quality grading (A-F) orthogonal to CI severity typing. Together they reveal failure modes confidence alone misses.

**Key Insight:** CI + SRI = 1.0 exactly (perfect complementarity). SRI measures precisely what CI does not. Both achieve identical discriminative power (AUC=0.874), vastly outperforming confidence alone (AUC=0.171).

**AG News Results:**
- **479 Type I cases** 35: Stable errors - confidently wrong, no flips
- **20 Type II cases** 20: Hidden instability - internal shifts without label flips
- **1 Type III case** 1: Moderate flip - elevated CI and degraded SRI (Grade C)
- **Overall SRI Grade A** (0.981): Excellent structural retention across all types
- **Error discrimination:** CI distinguishes errors 7.25× better than correct predictions; SRI shows 25.8% separation

## 🔬 Dataset

- **Base:** 500 examples from AG News test set (4-class news classification)
- **Classes:** World, Sports, Business, Sci/Tech
- **Perturbations:** 3 variants per base using:
  - Character-level typos (keyboard distance)
  - Synonym substitution (WordNet)
  - Word swapping (positional)
- **Total:** 2,000 rows (500 × 4 variants)
- **Format:** CSV with columns: `id`, `variant_id`, `text`, `true_label`, `pred_label`, `confidence`, `prob_0`, `prob_1`, `prob_2`, `prob_3`
- **Why 4-class?** Multi-class provides richer entropy signals (ER_ret component of SRI) compared to binary classification

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Dataset (Optional)

The `agnews_ci_sri_demo.csv` is included, but you can regenerate:

```bash
python generate_agnews_demo.py
```

This will:
- Download AG News test set (500 examples)
- Generate 3 perturbations per example
- Run BERT-AG-News inference on all 2,000 rows
- Save to `agnews_ci_sri_demo.csv`

Takes ~5-8 minutes on CPU (4-class model is slightly slower).

### 3. Verify Basic Metrics

Validate flip rate and accuracy independently:

```bash
python validate_metrics.py
```

This verifies metrics that don't require proprietary analysis:
- Flip rate (% of examples with prediction changes)
- Accuracy (base and perturbed)
- Confidence distributions
- Class balance

### 4. Advanced Analysis (Proprietary)

For complete SRI + CI analysis (CSI failure mode typing, SRI grading, AUC curves):

```bash
# Commercial licensing required
# Contact: ask@collapseindex.org
```

**What's included in advanced analysis:**
- Structural Retention Index (SRI) scores
- Collapse Index (CI) scores  
- CSI failure mode classification (Type I-V)
- SRI letter grading (A-F)
- ROC/AUC curves
- Distributions
- Slice Cohorts
- Full HTML report
- Collapse Log raw row level data

## 📁 Files

- `README.md` - This file
- `requirements.txt` - Python dependencies
- `generate_agnews_demo.py` - Dataset generation script
- `validate_metrics.py` - Independent metric verification (flip rate, accuracy)
- `agnews_ci_sri_demo.csv` - Full 2,000-row dataset with predictions

## 🔗 Links

- **SRI Paper (Zenodo):** [https://doi.org/10.5281/zenodo.18016507](https://doi.org/10.5281/zenodo.18016507)
- **Main Repository:** [github.com/collapseindex/collapseindex](https://github.com/collapseindex/collapseindex)
- **Collapse Index Labs:** [collapseindex.org](https://collapseindex.org)
- **Model Used:** [huggingface.co/fabriceyhc/bert-base-uncased-ag_news](https://huggingface.co/fabriceyhc/bert-base-uncased-ag_news)
- **AG News Dataset:** [huggingface.co/datasets/ag_news](https://huggingface.co/datasets/ag_news)

## 📝 Citation

If you use SRI or this validation dataset in your research:

```bibtex
@misc{kwon2025sri,
  title={Structural Retention Index (SRI): A Collapse Index Extension for Orthogonal Stability Assessment},
  author={Kwon, Alex},
  year={2025},
  url={https://github.com/collapseindex/ci-sri},
  doi={10.5281/zenodo.18016507},
  note={Collapse Index Labs}
}
```

**Author:** Alex Kwon ([collapseindex.org](https://collapseindex.org)) · ORCID: [0009-0002-2566-5538](https://orcid.org/0009-0002-2566-5538)

Please also cite the original AG News dataset:

```bibtex
@inproceedings{zhang2015character,
  title={Character-level convolutional networks for text classification},
  author={Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
  booktitle={Advances in neural information processing systems},
  pages={649--657},
  year={2015}
}
```

## ⚖️ License

- **This Repository:** MIT License (code only)
- **SRI Methodology:** Proprietary - © 2025 Collapse Index Labs.
- **AG News Dataset:** Available via HuggingFace Datasets (cite original paper above)
- **BERT Model:** Apache 2.0

**Copyright © 2025 Collapse Index Labs - Alex Kwon. All rights reserved.**

**Note:** This repository provides reproducible validation code for SRI research. The complete SRI implementation is proprietary. For commercial licensing, contact [ask@collapseindex.org](mailto:ask@collapseindex.org).

## 📧 Contact

Questions? Email [ask@collapseindex.org](mailto:ask@collapseindex.org)


