"""
Generate AG News demo dataset with perturbations for SRI + CI analysis.
Uses HuggingFace datasets + nlpaug for realistic perturbations.

4-class news classification: World, Sports, Business, Sci/Tech
Better for SRI paper because multi-class gives richer entropy signals.

Output: CSV in the format CI pipeline expects.
"""

import pandas as pd
from datasets import load_dataset
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from transformers import pipeline
from tqdm import tqdm
import hashlib
import numpy as np

# Config
N_SAMPLES = 500  # Number of base examples from AG News test set
VARIANTS_PER_SAMPLE = 3  # How many perturbations per base example
MODEL_NAME = "fabriceyhc/bert-base-uncased-ag_news"  # Fine-tuned AG News classifier
OUTPUT_PATH = "agnews_ci_sri_demo.csv"

# AG News label mapping
LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

def generate_base_id(text):
    """Generate deterministic base_id from text."""
    return hashlib.md5(text.encode()).hexdigest()[:12]

def create_perturbations(text, n=3):
    """Generate n perturbations of text using various methods."""
    perturbations = []
    
    # Typo augmenter (keyboard distance)
    typo_aug = nac.KeyboardAug(aug_char_max=2, aug_word_p=0.3)
    
    # Synonym augmenter
    synonym_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.3)
    
    # Word swap augmenter
    swap_aug = naw.RandomWordAug(action='swap', aug_p=0.2)
    
    methods = [typo_aug, synonym_aug, swap_aug]
    
    for i in range(n):
        try:
            aug = methods[i % len(methods)]
            perturbed = aug.augment(text)
            if isinstance(perturbed, list):
                perturbed = perturbed[0]
            perturbations.append(perturbed)
        except Exception as e:
            # Fallback: simple character swap
            perturbed = text.replace('the', 'teh', 1) if 'the' in text else text + '.'
            perturbations.append(perturbed)
    
    return perturbations

def main():
    print("Loading AG News test set...")
    dataset = load_dataset("ag_news", split="test")
    
    # Sample N_SAMPLES from test set
    sampled = dataset.shuffle(seed=42).select(range(N_SAMPLES))
    
    print(f"Generating {VARIANTS_PER_SAMPLE} perturbations per example...")
    
    # Prepare data rows
    rows = []
    
    for idx, example in enumerate(tqdm(sampled)):
        text = example['text']
        true_label = example['label']  # 0=World, 1=Sports, 2=Business, 3=Sci/Tech
        base_id = f"agnews_{idx:04d}"
        
        # Base example (variant_id = 'base')
        rows.append({
            'id': base_id,
            'variant_id': 'base',
            'text': text,
            'true_label': LABEL_MAP[true_label],
            'pred_label': None,  # Will fill after model inference
            'confidence': None,
            'prob_0': None,  # World
            'prob_1': None,  # Sports
            'prob_2': None,  # Business
            'prob_3': None   # Sci/Tech
        })
        
        # Generate perturbations
        perturbations = create_perturbations(text, VARIANTS_PER_SAMPLE)
        
        for var_idx, perturbed_text in enumerate(perturbations, 1):
            rows.append({
                'id': base_id,
                'variant_id': f'v{var_idx}',
                'text': perturbed_text,
                'true_label': LABEL_MAP[true_label],
                'pred_label': None,
                'confidence': None,
                'prob_0': None,
                'prob_1': None,
                'prob_2': None,
                'prob_3': None
            })
    
    print(f"Running model inference on {len(rows)} examples...")
    
    # Load model - returns all class probabilities
    classifier = pipeline("text-classification", model=MODEL_NAME, device=-1, return_all_scores=True)
    
    # Batch inference
    texts = [row['text'] for row in rows]
    predictions = []
    
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        batch_preds = classifier(batch)
        predictions.extend(batch_preds)
    
    # Fill in predictions
    for row, pred in zip(rows, predictions):
        # pred is a list of 4 dicts: [{'label': 'LABEL_0', 'score': 0.8}, ...]
        # Sort by label to ensure consistent ordering
        pred_sorted = sorted(pred, key=lambda x: x['label'])
        
        # Extract probabilities for each class
        row['prob_0'] = pred_sorted[0]['score']  # World
        row['prob_1'] = pred_sorted[1]['score']  # Sports
        row['prob_2'] = pred_sorted[2]['score']  # Business
        row['prob_3'] = pred_sorted[3]['score']  # Sci/Tech
        
        # Find predicted class (highest prob)
        max_idx = np.argmax([pred_sorted[i]['score'] for i in range(4)])
        row['pred_label'] = LABEL_MAP[max_idx]
        row['confidence'] = pred_sorted[max_idx]['score']
    
    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n✓ DONE! Saved to {OUTPUT_PATH}")
    print(f"\nStats:")
    print(f"   - Total examples: {len(df)}")
    print(f"   - Base examples: {N_SAMPLES}")
    print(f"   - Variants per base: {VARIANTS_PER_SAMPLE + 1} (including base)")
    print(f"   - Model accuracy (all): {(df['pred_label'] == df['true_label']).mean():.2%}")
    
    # Quick flip analysis
    flip_count = 0
    for case_id in df['id'].unique():
        group = df[df['id'] == case_id]
        predictions = group['pred_label'].unique()
        if len(predictions) > 1:
            flip_count += 1
    
    flip_rate = flip_count / N_SAMPLES
    print(f"   - Flip rate: {flip_rate:.1%} ({flip_count}/{N_SAMPLES} base examples)")
    
    # Class distribution
    print(f"\n   Class Distribution (base examples only):")
    base_df = df[df['variant_id'] == 'base']
    for class_name in ['World', 'Sports', 'Business', 'Sci/Tech']:
        count = (base_df['true_label'] == class_name).sum()
        print(f"      - {class_name}: {count} ({count/len(base_df):.1%})")
    
    print(f"\n✓ Dataset ready for validation: python validate_metrics.py")
    print(f"✓ Published: https://doi.org/10.5281/zenodo.18016507")

if __name__ == "__main__":
    main()
