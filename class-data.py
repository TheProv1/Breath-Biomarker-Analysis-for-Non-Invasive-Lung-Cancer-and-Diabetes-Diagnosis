import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv('Beta_balanced_master_dataset.csv')

# 2. Adjust the figure size
# Make the height larger (e.g., 12 or 15) so all labels have room to breathe
plt.figure(figsize=(10, 12))

# 3. Create a HORIZONTAL plot by setting y='Class'
# This is the key to preventing omitted labels!
sns.countplot(data=df, y='Class', palette='viridis')

# 4. Add titles and labels
plt.title('Number of Records per Class')
plt.xlabel('Count')
plt.ylabel('Sensor Class Target')

# (Optional) If you want to sort the bars by count, you can replace the countplot line with:
# order = df['Class'].value_counts().index
# sns.countplot(data=df, y='Class', palette='viridis', order=order)

# 5. Save the plot
plt.tight_layout()
plt.savefig('beta-dataset.png', dpi=300) # Added dpi=300 for higher resolution
plt.show()