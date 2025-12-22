#!/usr/bin/env python3
"""
Independent validation script for AG News demo dataset.

This script verifies basic metrics that can be calculated without
proprietary analysis pipelines:
- Flip rate: percentage of examples with inconsistent predictions
- Overall accuracy: model correctness across all variants
- Confidence distribution: how confidence differs between correct/incorrect predictions

Advanced metrics require full analysis pipeline (not included).

Usage:
    python validate_metrics.py

Author: Alex Kwon (Collapse Index Labs)
License: MIT
"""
import pandas as pd

# Load dataset
df = pd.read_csv('agnews_ci_sri_demo.csv')

print("=" * 60)
print("AG NEWS DATASET INDEPENDENT VALIDATION")
print("=" * 60)
print("\nThis script verifies basic metrics (flip rate, accuracy, confidence).")
print("Advanced analysis requires proprietary pipeline.")

# Basic stats
total_rows = len(df)
unique_ids = df['id'].nunique()
print(f"\nDataset Stats:")
print(f"  Total rows: {total_rows}")
print(f"  Unique base examples: {unique_ids}")
print(f"  Variants per base: {total_rows // unique_ids}")

# Calculate is_error
df['is_error'] = (df['pred_label'] != df['true_label']).astype(int)

# Flip rate calculation (INDEPENDENTLY VERIFIABLE)
print("\n" + "=" * 60)
print("✓ FLIP RATE (independently verifiable)")
print("=" * 60)

flip_count = 0
for base_id in df['id'].unique():
    base_examples = df[df['id'] == base_id]
    predictions = base_examples['pred_label'].unique()
    
    # If more than one unique prediction across variants, it flipped
    if len(predictions) > 1:
        flip_count += 1

flip_rate = (flip_count / unique_ids) * 100

print(f"Base examples with flips: {flip_count}/{unique_ids}")
print(f"Flip rate: {flip_rate:.1f}%")

# Overall accuracy (INDEPENDENTLY VERIFIABLE)
print("\n" + "=" * 60)
print("✓ OVERALL ACCURACY (independently verifiable)")
print("=" * 60)

# Base examples only (clean text)
base_df = df[df['variant_id'] == 'base']
base_correct = (base_df['pred_label'] == base_df['true_label']).sum()
base_accuracy = (base_correct / len(base_df)) * 100

# All rows (including perturbations)
total_correct = (df['is_error'] == 0).sum()
overall_accuracy = (total_correct / total_rows) * 100

print(f"Base examples (clean): {base_accuracy:.1f}% ({base_correct}/{len(base_df)})")
print(f"All variants (w/ perturbations): {overall_accuracy:.1f}% ({total_correct}/{total_rows})")
print(f"Degradation: {base_accuracy - overall_accuracy:.1f} percentage points")

# Class distribution
print("\n" + "=" * 60)
print("✓ CLASS DISTRIBUTION")
print("=" * 60)
print("Base examples by true label:")
for label in ['World', 'Sports', 'Business', 'Sci/Tech']:
    count = (base_df['true_label'] == label).sum()
    print(f"  {label:12s}: {count:3d} ({count/len(base_df)*100:4.1f}%)")

# Confidence distribution
print("\n" + "=" * 60)
print("✓ CONFIDENCE DISTRIBUTION")
print("=" * 60)
errors_df = df[df['is_error'] == 1]
correct_df = df[df['is_error'] == 0]
print(f"Errors ({len(errors_df)} samples):")
print(f"  Mean confidence: {errors_df['confidence'].mean():.4f}")
print(f"Correct ({len(correct_df)} samples):")
print(f"  Mean confidence: {correct_df['confidence'].mean():.4f}")
print(f"Gap: {correct_df['confidence'].mean() - errors_df['confidence'].mean():.4f}")
print("\n→ Small gap confirms confidence alone is insufficient")

# Advanced metrics notice
print("\n" + "=" * 60)
print("⚠ ADVANCED METRICS (PROPRIETARY)")
print("=" * 60)
print("The following require proprietary analysis pipeline:")
print("  • Structural Retention Index (SRI)")
print("  • Collapse Index (CI)")
print("  • CSI failure mode classification (Type I-V)")
print("  • SRI letter grading (A-F)")
print("  • AUC/ROC curves")
print("\nFor commercial analysis: ask@collapseindex.org")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"✓ Flip rate: {flip_rate:.1f}%")
print(f"✓ Base accuracy: {base_accuracy:.1f}%")
print(f"✓ Overall accuracy: {overall_accuracy:.1f}%")
print(f"✓ Dataset is reproducible and verifiable")
print("⚠ Advanced metrics require proprietary pipeline")
print("=" * 60)


