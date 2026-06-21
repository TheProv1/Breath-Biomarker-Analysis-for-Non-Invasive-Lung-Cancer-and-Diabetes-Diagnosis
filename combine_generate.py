import pandas as pd
import numpy as np

np.random.seed(42)

MASTER_FILE = 'master_dataset.csv'
BALANCED_FILE = 'balanced_master_dataset.csv'

print(f"Loading '{MASTER_FILE}'...")
df = pd.read_csv(MASTER_FILE)

print("Performing Deep Clean on text labels...")
y_text = df['Class'].astype(str).str.lower().str.strip()
y_text = y_text.str.replace(" ppm", "ppm", regex=False)
y_text = y_text.str.replace(r'\s+', ' ', regex=True)
df['Class'] = y_text

initial_rows = len(df)
# This deletes the redundant idle time where the sensor hasn't reacted yet
df = df.drop_duplicates()
print(f"Removed {initial_rows - len(df)} redundant 'idle sensor' rows!")

class_counts = df['Class'].value_counts()
max_class_count = class_counts.max()

# Ensure every single gas has at least 2000 rows so the AI can train deeply
target_rows = max(max_class_count, 2000) 
print(f"Targeting {target_rows} active reaction rows per class...")

augmented_dataframes = []
numeric_cols = df.select_dtypes(include=[np.number]).columns

print("\nGenerating rich, jittered sensor records...")

for class_name, count in class_counts.items():
    real_class_data = df[df['Class'] == class_name]
    
    # Keep the real, unique reaction rows
    augmented_dataframes.append(real_class_data)
    
    shortfall = target_rows - len(real_class_data)
    if shortfall > 0:
        synthetic_samples = real_class_data.sample(n=shortfall, replace=True).copy()
        
        for col in numeric_cols:
            std_dev = synthetic_samples[col].std()
            if pd.isna(std_dev) or std_dev == 0:
                std_dev = 0.01 
            
            # Tightened noise to 0.5% so it doesn't blur 1ppm vs 2ppm boundaries!
            noise = np.random.normal(loc=0, scale=std_dev * 0.005, size=len(synthetic_samples))
            synthetic_samples[col] = synthetic_samples[col] + noise
            synthetic_samples[col] = synthetic_samples[col].clip(lower=0)
            
        augmented_dataframes.append(synthetic_samples)

balanced_df = pd.concat(augmented_dataframes, ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

balanced_df.to_csv(BALANCED_FILE, index=False)
print(f"\nSuccessfully saved high-density balanced dataset to '{BALANCED_FILE}'!")