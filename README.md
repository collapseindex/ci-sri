# SRI + CI: AG News Public Validation

<div align="center">

[![Version](https://img.shields.io/badge/version-2.2.0-blue?style=flat-square)](https://github.com/collapseindex/ci-sri)
[![SRI Paper](https://img.shields.io/badge/DOI-10.5281/zenodo.18016507-blue?style=for-the-badge)](https://doi.org/10.5281/zenodo.18016507)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)

**[collapseindex.org](https://collapseindex.org)** • **[ask@collapseindex.org](mailto:ask@collapseindex.org)**

</div>

> **Public Validation #1:** Reproducible demonstration of **Structural Retention Index (SRI)** alongside Collapse Index (CI) on multi-class text classification.

> 📊 **Also Available:** [SST-2 Binary Validation (ci-sst2)](https://github.com/collapseindex/ci-sst2) - Sentiment analysis demonstration

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

**Advanced Diagnostics (Commercial Implementation):**

| Metric | Value | Notes |
|--------|-------|-------|
| **CI Score (avg)** | 0.019 | Prediction instability metric |
| **SRI Score (avg)** | 0.981 | Structural retention metric |
| **CI + SRI** | 1.000 | Perfect complementarity* |
| **AUC(CI)** | 0.874 | Error discrimination via instability |
| **AUC(SRI)** | 0.874 | Error discrimination via retention |
| **AUC(Conf)** | 0.829 | Confidence discriminates errors well |
| **Confidence Status** | ✅ Honest | AUC ≥ 0.60 = reliable signal |
| **Trinity Verdict** | 🟢 Stable | Low drift + high retention + honest confidence |
| **CSI Error Distribution** | 35/10/1/0/0 | Type I/II/III/IV/V error counts |

*\*CI + SRI = 1.0 is empirical for this validation, not a theoretical identity.*

*Note: Advanced metrics require commercial licensing. Contact ask@collapseindex.org or visit [collapseindex.org/evals.html](https://collapseindex.org/evals.html)*

## 📊 The SRI Story

**Important:** In this AG News validation, confidence is a reliable error signal (AUC=0.829). This makes AG News a useful control case: CI/SRI still add value by explaining *how* failures occur, even when confidence already works.

**Standard benchmarks say:** "Ship it! 90.8% accuracy."

**What confidence tells you:** "This prediction is probably wrong." With AUC=0.829, confidence reliably flags errors—lower confidence correlates with incorrect predictions. This is an **Honest** signal.

**What confidence can't tell you:** *How* the model fails. This is where CI/SRI provide unique value.

**Failure Mode Classification (CSI):**
- **Type I:** Stable Collapse - Confidently wrong, no flips (most dangerous)
- **Type II:** Hidden Instability - Internal shifts, same label (hidden brittleness)
- **Type III:** Moderate Flip - Clear label flips under stress
- **Type IV-V:** High/Extreme Flip - Severe instability or chaotic breakdown

*Classification thresholds remain proprietary to prevent adversarial optimization.*

**Why Trinity matters:** Confidence answers "Will this be wrong?" CI/SRI answer "How will it fail?" Both questions matter for deployment:
- **Confidence** → Set rejection thresholds, calibrate uncertainty
- **CI (instability)** → Detect models that flip under perturbation
- **SRI (structure)** → Grade internal coherence (A-F scale)
- **CSI (failure type)** → Classify error behavior for targeted fixes

**Key Insight:** CI + SRI = 1.0 exactly (perfect complementarity). All three signals achieve strong discrimination (CI=0.874, SRI=0.874, Conf=0.829). But they measure different things: confidence measures calibration, CI/SRI measure structural behavior under stress.

**AG News Results:**
- **Trinity Verdict:** 🟢 Stable (low drift + high retention + honest confidence)
- **35 Type I errors** (76.1%): Stable collapse - high confidence, no flips, no warning signs
- **10 Type II errors** (21.7%): Hidden instability - internal probability shifts without label change
- **1 Type III error** (2.2%): Moderate flip - clear behavioral signal (elevated CI, Grade C SRI)
- **Total errors:** 46/500 base examples (9.2% flip rate)
- **Overall SRI Grade A** (0.981): Excellent structural retention

**The Type I Problem:** 35 of 46 errors (76%) are Type I—confidently wrong with no behavioral instability. Confidence flags these as "probably wrong" (mean conf 0.964 vs 0.992 for correct), but CI/SRI reveal they're structurally stable failures. These are the hardest to fix: the model isn't confused, it's confidently mistaken.

*Operational implication:* These errors would pass robustness tests, perturbation checks, and aggregate CI/SRI thresholds. They require domain rules, human review, or post-hoc constraints—not more training.

**Note:** CSI types classify ERRORS ONLY. Of 500 total samples, 479 have CI ≤ 0.15 (includes 444 correct + 35 errors). CSI counts show the 35 errors in that range, not the 479 total.

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

For complete Trinity analysis (CI + SRI + Confidence calibration, CSI failure mode typing, SRI grading, AUC curves):

```bash
# Commercial licensing required
# Contact: ask@collapseindex.org
```

**What's included in advanced analysis:**
- Structural Retention Index (SRI) scores
- Collapse Index (CI) scores  
- Confidence calibration metrics
- Trinity verdict (CI + SRI + Confidence)
- CSI failure mode classification (Type I-V)
- SRI letter grading (A-F)
- ROC/AUC curves for all three signals

## 📁 Files

- `README.md` - This file
- `requirements.txt` - Python dependencies
- `generate_agnews_demo.py` - Dataset generation script
- `validate_metrics.py` - Independent metric verification (flip rate, accuracy)
- `agnews_ci_sri_demo.csv` - Full 2,000-row dataset with predictions

## 🔗 Links

**CI Framework & Validations:**
- **Main CI Repository:** [github.com/collapseindex/collapseindex](https://github.com/collapseindex/collapseindex)
- **SRI Validation (AG News):** [github.com/collapseindex/ci-sri](https://github.com/collapseindex/ci-sri) *(you are here)*
- **SST-2 Validation:** [github.com/collapseindex/ci-sst2](https://github.com/collapseindex/ci-sst2)
- **Collapse Index Labs:** [collapseindex.org](https://collapseindex.org)

**Papers:**
- **SRI Paper:** [DOI: 10.5281/zenodo.18016507](https://doi.org/10.5281/zenodo.18016507)
- **Framework Paper:** [DOI: 10.5281/zenodo.17718180](https://doi.org/10.5281/zenodo.17718180)

**Data & Models:**
- **Model Used:** [huggingface.co/fabriceyhc/bert-base-uncased-ag_news](https://huggingface.co/fabriceyhc/bert-base-uncased-ag_news)
- **AG News Dataset:** [huggingface.co/datasets/ag_news](https://huggingface.co/datasets/ag_news)

## 📝 Citation

If you use SRI or this validation dataset in your research:

```bibtex
@misc{kwon2025sri,
  title={Structural Retention Index (SRI): AG News Public Validation},
  author={Kwon, Alex},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/collapseindex/ci-sri}},
  version={v2.0.0},
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

- **This Repository (v2.2.0):** MIT License (code only)
- **CI/SRI Methodology:** Proprietary - © 2025 Collapse Index Labs - Alex Kwon
- **AG News Dataset:** Available via HuggingFace Datasets (cite original paper above)
- **BERT Model:** Apache 2.0

**Copyright © 2026 Collapse Index Labs - Alex Kwon. All rights reserved.**

**Note:** This repository provides reproducible validation code for SRI research. The complete SRI implementation is proprietary. For commercial licensing, contact [ask@collapseindex.org](mailto:ask@collapseindex.org).

**Version History:**
- **v2.2.0** (Jan 2026) - **BUGFIX:** Fixed confidence AUC calculation (was computing 1-AUC due to orientation bug in multi-class). Corrected values: AUC(Conf)=0.829 (was 0.171), Trinity Verdict=🟢 Stable (was 🟡 Overconfident Stable), Confidence Status=✅ Honest. Replaced "Confidence Separation" metric with "AUC(Conf)" for consistency. Updated validate_metrics.py to compute AUC(Conf). This is a significant correction—confidence IS a useful signal for this model.
- **v2.1.1** (Jan 2026) - Updated confidence separation reporting. Minor numerical differences between CLI and script noted.
- **v2.1.0** (Jan 2026) - Trinity System Integration: Added confidence calibration metrics and Trinity verdict.
- **v2.0.1** (Jan 2026) - **CORRECTION:** Fixed CSI type counts to show error counts (35/10/1) instead of total sample counts (479/20/1). Previous versions incorrectly reported total samples with CI ≤ 0.15 (479) rather than errors only (35). This was a reporting error; underlying analysis was correct. See commit history for details.
- **v2.0.0** (Jan 2026) - Updated citation format, cross-validation references, improved documentation
- **v1.0.0** (Dec 2025) - Initial public release with AG News validation

## 📧 Contact

Questions? Email [ask@collapseindex.org](mailto:ask@collapseindex.org)

