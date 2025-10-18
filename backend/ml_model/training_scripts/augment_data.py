# --- File: augment_data.py (Corrected Version) ---

import pandas as pd
from parrot import Parrot
import torch
import warnings
import os

# This line hides the harmless "early_stopping" warning for cleaner output
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

warnings.filterwarnings("ignore")

# Initialize Parrot. Use a GPU if you have one, it's much faster.
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=torch.cuda.is_available())

def augment_dataset(input_csv, output_csv, phrases_to_generate=5):
    """
    Reads a CSV, augments the 'text' column using Parrot, and saves a new, CLEAN CSV.
    """
    print(f"Reading original dataset from {input_csv}...")
    df = pd.read_csv(input_csv)

    if "text" not in df.columns or "category" not in df.columns:
        print("Error: The CSV must contain 'text' and 'category' columns.")
        return

    augmented_rows = []
    total_rows = len(df)
    print(f"Starting augmentation for {total_rows} rows...")

    for index, row in df.iterrows():
        original_text = str(row['text'])
        category = str(row['category'])

        # Add the original row
        augmented_rows.append({'text': original_text, 'category': category})
        print(f"  -> Augmenting row {index + 1}/{total_rows}: '{original_text[:50]}...'")

        try:
            para_phrases = parrot.augment(input_phrase=original_text, max_return_phrases=phrases_to_generate)

            if para_phrases:
                for phrase in para_phrases:
                    # ==========================================================
                    # THE FIX IS HERE!
                    # Before, it was using 'phrase', which is a tuple.
                    # We now correctly use phrase[0] to get only the text string.
                    # ==========================================================
                    clean_text = phrase[0]
                    augmented_rows.append({'text': clean_text, 'category': category})

        except Exception as e:
            print(f"    !! Could not augment phrase. Error: {e}")
            continue

    print("\nAugmentation complete!")
    augmented_df = pd.DataFrame(augmented_rows)
    print(f"Original dataset size: {len(df)} rows")
    print(f"New augmented dataset size: {len(augmented_df)} rows")

    augmented_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Successfully saved CLEAN augmented data to {output_csv}")

if __name__ == "__main__":
    # Use your original, clean CSV as the input
    input_file = "training_dataset_enhanced.csv"
    output_file = "training_dataset_AUGMENTED_CLEAN.csv" # Use a new name
    augment_dataset(input_csv=input_file, output_csv=output_file, phrases_to_generate=5)