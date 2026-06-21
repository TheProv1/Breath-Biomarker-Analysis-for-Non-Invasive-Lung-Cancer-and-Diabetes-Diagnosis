import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress annoying scipy warnings
warnings.filterwarnings("ignore")

# 1. LOAD THE ORIGINAL, UNAUGMENTED DATA
MASTER_FILE = 'master_dataset.csv'
print(f"Loading raw, unaugmented data from '{MASTER_FILE}'...")
df = pd.read_csv(MASTER_FILE)

# Deep Clean the text labels (just like before) to make sure we group them correctly
df['Class'] = df['Class'].astype(str).str.lower().str.strip()
df['Class'] = df['Class'].str.replace(" ppm", "ppm", regex=False)
df['Class'] = df['Class'].str.replace(r'\s+', ' ', regex=True)

# 2. SELECT THE DATA TO TEST
# To find the true hardware noise distribution, we isolate ONE sensor and ONE gas.
# Let's test the 'TGS 822' sensor's reaction to 'acetone 1ppm'
TARGET_SENSOR = 'TGS 822'
TARGET_GAS = 'acetone 1ppm'

# Extract only the raw decimal readings for this specific scenario
raw_data = df[df['Class'] == TARGET_GAS][TARGET_SENSOR].dropna().values

print(f"\nAnalyzing {len(raw_data)} real data points for {TARGET_SENSOR} exposed to {TARGET_GAS}...")

# 3. DEFINE THE DISTRIBUTIONS TO TEST
# These are the standard continuous distributions for physical hardware/electrical signals
distributions = [
    ("Normal (Gaussian)", st.norm),
    ("Log-Normal", st.lognorm),
    ("Gamma", st.gamma),
    ("Weibull", st.weibull_min),
    ("Exponential", st.expon)
]

# 4. MATHEMATICAL FITTING (Finding the lowest error)
best_fit_name = ""
best_fit_error = np.inf
best_fit_params = {}

# Create a histogram of the real data to compare against
y_real, x_bins = np.histogram(raw_data, bins=30, density=True)
x_mid = (x_bins[1:] + x_bins[:-1]) / 2

print("\n--- Distribution Fit Results (Lower Error is Better) ---")

for name, distribution in distributions:
    try:
        # Fit the distribution to our real data
        params = distribution.fit(raw_data)
        
        # Generate the theoretical curve based on the fit
        y_theoretical = distribution.pdf(x_mid, *params)
        
        # Calculate Sum of Squared Errors (SSE) between Real Data and Theoretical Curve
        error = np.sum((y_real - y_theoretical) ** 2)
        
        print(f"{name.ljust(20)} : Error = {error:.4f}")
        
        # Keep track of the absolute best fit
        if error < best_fit_error:
            best_fit_error = error
            best_fit_name = name
            best_fit_params = params
            
    except Exception as e:
        print(f"{name} failed to fit.")

print(f"\n🏆 WINNING DISTRIBUTION: {best_fit_name} (Error: {best_fit_error:.4f})")

# =====================================================================
# 5. VISUALIZE FOR YOUR PROFESSOR
# =====================================================================
plt.figure(figsize=(10, 6))

# Plot the real experimental data as a bar chart (Histogram)
sns.histplot(raw_data, bins=30, stat='density', alpha=0.5, color='blue', label='Real Sensor Data')

# Plot the winning theoretical distribution curve
x_axis = np.linspace(min(raw_data), max(raw_data), 1000)
y_winning_curve = [d for n, d in distributions if n == best_fit_name][0].pdf(x_axis, *best_fit_params)

plt.plot(x_axis, y_winning_curve, color='red', linewidth=3, label=f'Best Fit: {best_fit_name}')

plt.title(f'Statistical Distribution Fit\nSensor: {TARGET_SENSOR} | Gas: {TARGET_GAS}')
plt.xlabel('Sensor Response (Resistance/Voltage)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()

# Save the plot for your thesis/presentation!
plt.savefig('distribution_fit.png', dpi=300)
print("\nSaved distribution plot to 'distribution_fit.png'!")
plt.show()