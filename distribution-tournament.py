import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for distributions that fail to fit
warnings.filterwarnings("ignore")

MASTER_FILE = 'master_dataset.csv'
print(f"Loading raw data from '{MASTER_FILE}'...")
df = pd.read_csv(MASTER_FILE)

# 1. DEEP CLEAN LABELS
df['Class'] = df['Class'].astype(str).str.lower().str.strip()
df['Class'] = df['Class'].str.replace(" ppm", "ppm", regex=False)
df['Class'] = df['Class'].str.replace(r'\s+', ' ', regex=True)

# 2. IDENTIFY ALL SENSORS AND GASES
sensors = df.select_dtypes(include=[np.number]).columns.tolist()
gases = df['Class'].unique().tolist()

print(f"Found {len(sensors)} sensors and {len(gases)} unique gas classes.")
print("Starting the Global Distribution Tournament... (This may take a few minutes)\n")

# 3. THE DISTRIBUTION LIBRARY
distributions =[
    ("Normal (Gaussian)", st.norm),
    ("Log-Normal", st.lognorm),
    ("Gamma", st.gamma),
    ("Weibull Min", st.weibull_min),
    ("Exponential", st.expon),
    ("Logistic", st.logistic),
    ("Laplace", st.laplace),
    ("Rayleigh", st.rayleigh),
    ("Pearson3", st.pearson3),
    ("Student's t", st.t),
    ("Gumbel (Right)", st.gumbel_r),
    ("Gumbel (Left)", st.gumbel_l),
    ("Cauchy", st.cauchy),
    ("Beta", st.beta),
    ("Uniform", st.uniform)
]

# 4. THE SCOREBOARD
# Dictionary to track how many times a distribution ranks #1
scoreboard = {name: 0 for name, _ in distributions}
valid_tests_run = 0

# 5. RUN THE TOURNAMENT ACROSS ALL COMBINATIONS
for gas in gases:
    for sensor in sensors:
        # Extract the specific data chunk
        raw_data = df[df['Class'] == gas][sensor].dropna().values
        
        # Skip if there's not enough data to mathematically test (e.g., less than 30 rows)
        if len(raw_data) < 30:
            continue
            
        valid_tests_run += 1
        
        # Setup histogram for SSE calculation
        y_real, x_bins = np.histogram(raw_data, bins=30, density=True)
        x_mid = (x_bins[1:] + x_bins[:-1]) / 2
        
        best_fit_name = ""
        best_fit_error = np.inf
        
        # Test all distributions on this specific chunk
        for name, distribution in distributions:
            try:
                params = distribution.fit(raw_data)
                y_theoretical = distribution.pdf(x_mid, *params)
                error = np.sum((y_real - y_theoretical) ** 2)
                
                # Keep track of the lowest error
                if not np.isnan(error) and error < best_fit_error:
                    best_fit_error = error
                    best_fit_name = name
            except Exception:
                pass
                
        # Award 1 point to the winner of this sensor/gas combination!
        if best_fit_name != "":
            scoreboard[best_fit_name] += 1

# 6. CALCULATE FINAL RESULTS
# Sort the scoreboard from highest wins to lowest wins
ranked_results = sorted(scoreboard.items(), key=lambda x: x[1], reverse=True)

print(f"--- TOURNAMENT COMPLETE ---")
print(f"Total valid Sensor/Gas combinations tested: {valid_tests_run}\n")

print("🏆 GLOBAL LEADERBOARD (Number of 1st Place Wins) 🏆")
for rank, (name, wins) in enumerate(ranked_results):
    if wins > 0:
        win_percentage = (wins / valid_tests_run) * 100
        print(f"Rank {rank+1}: {name.ljust(20)} | Wins: {wins} ({win_percentage:.1f}%)")

absolute_winner = ranked_results[0][0]

# =====================================================================
# 7. VISUALIZE THE LEADERBOARD FOR YOUR PROFESSOR
# =====================================================================
# Filter out distributions that got 0 wins to keep the chart clean
names_to_plot = [x[0] for x in ranked_results if x[1] > 0]
wins_to_plot = [x[1] for x in ranked_results if x[1] > 0]

plt.figure(figsize=(12, 6))
sns.barplot(x=wins_to_plot, y=names_to_plot, palette='viridis')

plt.title(f'Global Distribution Fit Leaderboard\n(Across {valid_tests_run} distinct Sensor/Gas combinations)', fontsize=14)
plt.xlabel('Number of Times Ranked as #1 Best Fit', fontsize=12)
plt.ylabel('Mathematical Distribution', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('global_distribution_tournament.png', dpi=300)
print(f"\nSaved Global Leaderboard plot to 'global_distribution_tournament.png'!")
print(f"\nCONCLUSION: The absolute best overall distribution for the entire dataset is: {absolute_winner}!")
