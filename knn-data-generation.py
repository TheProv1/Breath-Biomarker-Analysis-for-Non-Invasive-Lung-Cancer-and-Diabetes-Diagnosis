import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer  # <--- NEW IMPORT
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

MASTER_FILE = 'master_dataset.csv'
KNN_BALANCED_FILE = 'knn_balanced_master_dataset.csv'

print(f"Loading '{MASTER_FILE}'...")
df = pd.read_csv(MASTER_FILE)

# =====================================================================
# 1. THE DEEP CLEAN
# =====================================================================
print("Performing Deep Clean on text labels...")
y_text = df['Class'].astype(str).str.lower().str.strip()
y_text = y_text.str.replace(" ppm", "ppm", regex=False)
y_text = y_text.str.replace(r'\s+', ' ', regex=True)
df['Class'] = y_text

# Remove the idle "Clean Air" rows for maximum accuracy
initial_rows = len(df)
df = df.drop_duplicates()
print(f"Removed {initial_rows - len(df)} redundant 'idle sensor' rows!")

# =====================================================================
# 2. ANALYZE CLASS IMBALANCE
# =====================================================================
class_counts = df['Class'].value_counts()
max_class_count = class_counts.max()
target_rows = max(max_class_count, 2000)

print(f"\nTargeting {target_rows} records per class using KNN interpolation...")

augmented_dataframes =[]
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Initialize the Imputer (Fills NaNs with the average of that specific column)
imputer = SimpleImputer(strategy='mean')

# =====================================================================
# 3. THE CUSTOM KNN SYNTHETIC GENERATOR
# =====================================================================
for class_name, count in class_counts.items():
    real_class_data = df[df['Class'] == class_name].copy()
    augmented_dataframes.append(real_class_data) # Keep the real data
    
    shortfall = target_rows - len(real_class_data)
    
    if shortfall > 0:
        # Step A: Drop columns that are 100% NaN for this specific gas
        valid_cols = real_class_data[numeric_cols].dropna(axis=1, how='all').columns
        
        # SAFETY CATCH 1: If there are literally 0 valid sensor columns (a "Ghost" row)
        if len(valid_cols) == 0:
            print(f"Warning: Class '{class_name}' contains NO sensor data. Skipping KNN.")
            continue
            
        X_raw = real_class_data[valid_cols].values
        
        # SAFETY CATCH 2: If there's only 1 row, we can't do KNN math on it. 
        # We just duplicate the row directly to fill the shortfall.
        if len(X_raw) <= 1:
            print(f"Warning: Only 1 row found for '{class_name}'. Duplicating row directly.")
            synthetic_df = real_class_data.sample(n=shortfall, replace=True).copy()
            augmented_dataframes.append(synthetic_df)
            continue
        
        # THE FIX: Impute (fill) any scattered NaNs with the class mean so KNN doesn't crash
        X_real = imputer.fit_transform(X_raw)
        
        # Step B: Fit the K-Nearest Neighbors Algorithm
        k_neighbors = min(5, len(X_real)) 
            
        knn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
        knn.fit(X_real)
        
        # Step C: Generate Synthetic Points
        synthetic_matrix = np.zeros((shortfall, len(valid_cols)))
        
        for i in range(shortfall):
            # 1. Pick a random existing row
            base_idx = np.random.randint(0, len(X_real))
            base_point = X_real[base_idx]
            
            # 2. Find its K nearest neighbors
            distances, indices = knn.kneighbors([base_point])
            
            # 3. Pick a random neighbor (Skip index 0, because index 0 is the point itself)
            neighbor_idx = indices[0][np.random.randint(1, k_neighbors)]
            neighbor_point = X_real[neighbor_idx]
            
            # 4. Create the new point on the line between them
            step_size = np.random.uniform(0, 1)
            new_point = base_point + step_size * (neighbor_point - base_point)
            
            synthetic_matrix[i] = new_point
            
        # Step D: Put the synthetic data back into a DataFrame
        synthetic_df = pd.DataFrame(synthetic_matrix, columns=valid_cols)
        
        # Re-attach the NaN columns that weren't used for this gas at all
        for col in numeric_cols:
            if col not in valid_cols:
                synthetic_df[col] = np.nan
                
        # Re-attach the class label
        synthetic_df['Class'] = class_name
        
        augmented_dataframes.append(synthetic_df)

# =====================================================================
# 4. COMPILE AND SAVE
# =====================================================================
print("\nMerging and shuffling the KNN-balanced dataset...")
balanced_df = pd.concat(augmented_dataframes, ignore_index=True)

# Ensure no numbers dropped below zero
balanced_df[numeric_cols] = balanced_df[numeric_cols].clip(lower=0)

balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

balanced_df.to_csv(KNN_BALANCED_FILE, index=False)
print(f"\nSuccessfully saved KNN-generated balanced dataset to '{KNN_BALANCED_FILE}'!")