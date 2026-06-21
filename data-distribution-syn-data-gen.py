import pandas as pd
import numpy as np
import scipy.stats as st
import warnings

# Suppress warnings for difficult mathematical fits
warnings.filterwarnings("ignore")
np.random.seed(42)

MASTER_FILE = 'master_dataset.csv'
print(f"Loading '{MASTER_FILE}'...")
df = pd.read_csv(MASTER_FILE)

# =====================================================================
# 1. THE DEEP CLEAN & DEDUPLICATION
# =====================================================================
print("Performing Deep Clean on text labels...")
y_text = df['Class'].astype(str).str.lower().str.strip()
y_text = y_text.str.replace(" ppm", "ppm", regex=False)
y_text = y_text.str.replace(r'\s+', ' ', regex=True)
df['Class'] = y_text

df = df.drop_duplicates()

class_counts = df['Class'].value_counts()
target_rows = max(class_counts.max(), 2000)
numeric_cols = df.select_dtypes(include=[np.number]).columns

# =====================================================================
# 2. DEFINE THE 5 REQUESTED DISTRIBUTIONS
# =====================================================================
target_distributions = {
    "Beta": st.beta,
    "Pearson3": st.pearson3,
    "Log-Normal": st.lognorm,
    "Cauchy": st.cauchy,
    "Uniform": st.uniform
}

# =====================================================================
# 3. THE PARAMETRIC GENERATOR FUNCTION
# =====================================================================
def generate_dataset_for_distribution(dist_name, dist_object):
    print(f"\n--- Generating Balanced Dataset using {dist_name} Distribution ---")
    augmented_dataframes = []
    
    for class_name, count in class_counts.items():
        real_class_data = df[df['Class'] == class_name].copy()
        augmented_dataframes.append(real_class_data) # Keep real data
        
        shortfall = target_rows - len(real_class_data)
        
        if shortfall > 0:
            # Create an empty DataFrame to hold the new synthetic points
            synthetic_df = pd.DataFrame(index=range(shortfall), columns=numeric_cols)
            
            # Find sensors actually used for this gas
            valid_cols = real_class_data[numeric_cols].dropna(axis=1, how='all').columns
            
            for col in valid_cols:
                raw_sensor_data = real_class_data[col].dropna().values
                
                # Safety Catch: If we have less than 3 points, we can't fit a complex curve
                if len(raw_sensor_data) < 3:
                    synthetic_df[col] = np.random.choice(raw_sensor_data, size=shortfall)
                    continue
                
                try:
                    params = dist_object.fit(raw_sensor_data)
                    
                    generated_values = dist_object.rvs(*params, size=shortfall)
                    
                    # Hardware sensors can't be negative. 
                    # Cauchy distributions also have "heavy tails" that can accidentally spawn infinity. 
                    # We clip the data between 0 and 2x the maximum real reading to keep it physically possible.
                    max_real_val = raw_sensor_data.max()
                    generated_values = np.clip(generated_values, 0.0, max_real_val * 2.0)
                    
                    synthetic_df[col] = generated_values
                    
                except Exception:
                    # Fallback to standard uniform random if the math crashes on weird data
                    synthetic_df[col] = np.random.uniform(raw_sensor_data.min(), raw_sensor_data.max(), size=shortfall)
            
            # Re-attach the class label
            synthetic_df['Class'] = class_name
            augmented_dataframes.append(synthetic_df)

    # Compile and Save
    balanced_df = pd.concat(augmented_dataframes, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    file_name = f"{dist_name}_balanced_master_dataset.csv"
    balanced_df.to_csv(file_name, index=False)
    print(f"✅ Saved perfectly balanced {dist_name} dataset to '{file_name}'!")

for name, dist in target_distributions.items():
    generate_dataset_for_distribution(name, dist)

print("\n🎉 ALL 5 DATASETS GENERATED SUCCESSFULLY!")