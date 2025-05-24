import pandas as pd
from pii_masking.mask import mask_text
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Load dataset
df = pd.read_csv('combined_emails_with_natural_pii.csv')
emails = df['email']

print("[INFO] Applying PII masking to first 100 emails in parallel...")

# Define a wrapper to get only masked text
def mask_wrapper(text):
    return mask_text(text)[0]  # returns only the masked string

if __name__ == '__main__':
    with Pool(processes = max(1, cpu_count() // 2)) as pool:
        masked_emails = list(tqdm(pool.imap(mask_wrapper, emails), total=len(emails)))

    df['email'] = masked_emails

    # Save masked output
    masked_path = 'emails_masked_sample.csv'
    df.to_csv(masked_path, index=False)

    print(f"[INFO] Masked sample saved to {masked_path}")
