import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib
import os

distributions = ["Beta", "Pearson3", "Log-Normal", "Cauchy", "Uniform", "knn"]

final_scores = {}

for dist in distributions:
    file_name = f"{dist}_balanced_master_dataset.csv"
    
    if not os.path.exists(file_name):
        print(f"\n⚠️ Skipping {dist}: File '{file_name}' not found!")
        continue

    print(f"\n{'='*60}")
    print(f"🚀 TRAINING MODEL ON: {dist.upper()} DISTRIBUTION DATASET")
    print(f"{'='*60}")
    
    print(f"Loading '{file_name}'...")
    df = pd.read_csv(file_name)

    # Separate Features (strictly the raw sensors) and Target
    X = df.drop(columns=['Class'])
    y_text = df['Class'].astype(str)

    # Force all columns to numeric just in case
    X = X.apply(pd.to_numeric, errors='coerce')

    print("Encoding Labels and Splitting Data...")
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    print("\nTraining Max-Power XGBoost Model on GPU (Please wait)...")
    xgb_model = xgb.XGBClassifier(
        tree_method='hist',         
        device='cuda',              
        n_estimators=3000,          
        learning_rate=0.05,         
        max_depth=16,               
        subsample=1.0,              
        colsample_bytree=1.0,       
        gamma=0,                    
        min_child_weight=1,         
        eval_metric='mlogloss', 
        random_state=42,
        missing=float('nan'),
        early_stopping_rounds=100   
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False 
    )

    print("\nEvaluating Model...")
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save the score for the final leaderboard
    final_scores[dist] = accuracy

    print(f"\n--- {dist.upper()} MODEL RESULTS ---")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

    print("Detailed Classification Report:")
    print(classification_report(
        y_test, 
        y_pred, 
        labels=np.arange(len(encoder.classes_)), 
        target_names=encoder.classes_,           
        zero_division=0                          
    ))

    model_save_name = f"{dist.lower()}_xgboost_model.pkl"
    encoder_save_name = f"{dist.lower()}_encoder.pkl"
    joblib.dump(xgb_model, model_save_name)
    joblib.dump(encoder, encoder_save_name)
    print(f"Advanced Model successfully saved to disk as '{model_save_name}'!")


print("\n\n" + "★"*50)
print("🏆 FINAL DISTRIBUTION ACCURACY LEADERBOARD 🏆")
print("★"*50)

# Sort the dictionary by accuracy (highest to lowest)
sorted_scores = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)

for rank, (dist, acc) in enumerate(sorted_scores, start=1):
    print(f"Rank {rank}: {dist.ljust(15)} | Accuracy: {acc * 100:.2f}%")

print("★"*50 + "\n")


print("Generating and saving the leaderboard graph...")

dist_names = [item[0] for item in sorted_scores]
accuracy_percentages = [item[1] * 100 for item in sorted_scores]

# Set up the figure
plt.figure(figsize=(10, 6))

# Create a bar chart with a nice color palette
bars = plt.bar(
    dist_names, 
    accuracy_percentages, 
    color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b'][:len(dist_names)]
)

# Add Titles and Labels
plt.title('🏆 Final Distribution Accuracy Leaderboard 🏆', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Distribution / Data Type', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')

# Set Y-axis limit slightly higher than 100 so the text tags fit well
plt.ylim(0, max(accuracy_percentages) + 10 if accuracy_percentages else 100)

# Add the exact percentage text on top of each bar
for bar in bars:
    yval = bar.get_height()
    # x-position: center of the bar, y-position: slightly above the top of the bar
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        yval + 1, 
        f'{yval:.2f}%', 
        ha='center', 
        va='bottom', 
        fontsize=11, 
        fontweight='bold'
    )

# Aesthetic tweaks
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the graph as an image file in the same directory
graph_filename = "distribution_accuracy_leaderboard.png"
plt.savefig(graph_filename, dpi=300)

# Display the graph in a window
plt.show()

print(f"Graph successfully saved as '{graph_filename}'!")