import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
import joblib

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =====================================================================
# GPU SETUP
# =====================================================================
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU Memory Growth Enabled!")

# The 5 distributions we generated
distributions = ["Beta", "Pearson3", "Log-Normal", "Cauchy", "Uniform", "knn"]

# Dictionary to hold the final scores for the leaderboard
final_scores = {}

for dist in distributions:
    file_name = f"{dist}_balanced_master_dataset.csv"
    
    # Safety check
    if not os.path.exists(file_name):
        print(f"\n⚠️ Skipping {dist}: File '{file_name}' not found!")
        continue

    print(f"\n{'='*60}")
    print(f"🚀 TRAINING CNN ON: {dist.upper()} DISTRIBUTION DATASET")
    print(f"{'='*60}")
    
    print(f"Loading '{file_name}'...")
    df = pd.read_csv(file_name)

    X_raw = df.drop(columns=['Class']).apply(pd.to_numeric, errors='coerce').values
    y_text = df['Class'].astype(str)

    # 1. FIX THE NANS (CRITICAL FOR CNN)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_raw)

    # 2. SCALE THE DATA (CRITICAL FOR CNN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 3. ENCODE LABELS
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_text)
    y_categorical = to_categorical(y_encoded)  

    # 4. SPLIT THE DATA
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.20, random_state=42)

    # 5. RESHAPE FOR 1D-CNN (Samples, Features, Channels)
    X_train_cnn = np.expand_dims(X_train, axis=-1)
    X_test_cnn = np.expand_dims(X_test, axis=-1)

    # 6. BUILD THE 2-LAYER CNN
    print("Building 2-Layer 1D Convolutional Neural Network...")
    model = Sequential()

    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
    model.add(BatchNormalization())

    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(BatchNormalization())

    # Transition to Dense Neural Network
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    # Output Layer
    num_classes = y_categorical.shape[1]
    model.add(Dense(num_classes, activation='softmax'))

    # 7. COMPILE & TRAIN
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience = 5, factor = 0.1, min_lr = 0.00000001)

    print(f"Training CNN on {dist} data... Please wait.")
    history = model.fit(
        X_train_cnn, y_train,
        validation_split=0.2,
        epochs=75, batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
	)

    # 8. EVALUATE
    print("\nEvaluating 2-Layer CNN...")
    test_loss, test_acc = model.evaluate(X_test_cnn, y_test, verbose=0)
    
    # Save the score for the leaderboard
    final_scores[dist] = test_acc

    print(f"\n--- {dist.upper()} CNN RESULTS ---")
    print(f"Overall Accuracy: {test_acc * 100:.2f}%\n")

    y_pred_probs = model.predict(X_test_cnn, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("Detailed Classification Report:")
    print(classification_report(
        y_true_classes, 
        y_pred_classes, 
        labels=np.arange(len(encoder.classes_)), 
        target_names=encoder.classes_,           
        zero_division=0                          
    ))

    # Save each model and its respective preprocessors dynamically
    model_save_name = f"{dist.lower()}_enose_2layer_cnn.keras"
    joblib.dump(scaler, f"{dist.lower()}_cnn_scaler.pkl")
    joblib.dump(imputer, f"{dist.lower()}_cnn_imputer.pkl")
    joblib.dump(encoder, f"{dist.lower()}_cnn_encoder.pkl")
    model.save(model_save_name)
    
    print(f"CNN Model & Preprocessors successfully saved for {dist}!")
    
    # CRITICAL: Clear the GPU memory so the next CNN doesn't crash the graphics card
    tf.keras.backend.clear_session()


# =====================================================================
# THE FINAL SCOREBOARD
# =====================================================================
print("\n\n" + "★"*50)
print("🏆 FINAL CNN DISTRIBUTION ACCURACY LEADERBOARD 🏆")
print("★"*50)

# Sort the dictionary by accuracy (highest to lowest)
sorted_scores = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)

for rank, (dist, acc) in enumerate(sorted_scores, start=1):
    print(f"Rank {rank}: {dist.ljust(15)} | CNN Accuracy: {acc * 100:.2f}%")

print("★"*50 + "\n")

# =====================================================================
# GENERATING THE LEADERBOARD GRAPH
# =====================================================================
import matplotlib.pyplot as plt

print("Generating and saving the CNN leaderboard graph...")

# Extract names and scores in the sorted order
# Multiply by 100 to show percentages on the graph
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
plt.title('🏆 Final CNN Distribution Accuracy Leaderboard 🏆', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Distribution / Data Type', fontsize=12, fontweight='bold')
plt.ylabel('CNN Accuracy (%)', fontsize=12, fontweight='bold')

# Set Y-axis limit slightly higher than the max score so the text tags fit well
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
graph_filename = "75_cnn_distribution_accuracy_leaderboard.png"
plt.savefig(graph_filename, dpi=300)

# Display the graph in a window
plt.show()

print(f"Graph successfully saved as '{graph_filename}'!")
