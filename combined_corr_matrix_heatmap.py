# Import libraries: Pandas for data manipulation, Seaborn for heat map, matplotlib for displaying it.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import shutil

# Loading csv file here.
from google.colab import drive
drive.mount('/content/drive')

# db stands for Database (shorthand)
db = pd.read_csv('/content/drive/My Drive/Nick_Paper_Education_Exercise_Study_Data_for_Correlation.csv')

# Reads numeric data only to avoid column and participant number.
numeric_data = db.select_dtypes(include=['float64','int64'])

# Code to exclude specific columns: PainProblems1 /""5.
exclude_columns = ['PainProblems1', 'PainProblems5']

# Filtering data for variables ending in '1' and 'A1', excluding the specified columns when filtering. Baseline, prior to pain education.
data_1 = numeric_data.filter(regex='1$|A1$').drop(columns=exclude_columns, errors='ignore')
# Filter data for variables ending in '5' and 'A2', excluding the specified columns when filtering. This is after pain education.
data_5 = numeric_data.filter(regex='5$|A2$').drop(columns=exclude_columns, errors='ignore')

# Function to create a correlation matrix and calculate p-values.
def create_corr_matrix(data):
    # Calculates the correlation coefficients and p-values using vectorization
    corr, p_values = spearmanr(data, axis=0)
    
    # Converts the numpy arrays to pandas DataFrames
    corr_matrix = pd.DataFrame(corr, index=data.columns, columns=data.columns)
    p_values = pd.DataFrame(p_values, index=data.columns, columns=data.columns)
    
    return corr_matrix, p_values

# Create correlation matrices and p-values for data_1 and data_5.
corr_matrix_1, p_values_1 = create_corr_matrix(data_1)
corr_matrix_5, p_values_5 = create_corr_matrix(data_5)

# Rename the columns and index of the correlation matrix and p-values DataFrame.
name_mapping = {
    'PainDays1': 'Days Manageable Pain',
    'InterfereActive1': 'Interference Activity',
    'InterfereMood1' : 'Interference Mood',
    'InterfereSleep1' : 'Interference Sleep',
    'HowHard1' : 'Hard to Deal',

    'PainDays5': 'Days Manageable Pain',
    'InterfereActive5': 'Interference Activity',
    'InterfereMood5': 'Interference Mood',
    'InterfereSleep5': 'Interference Sleep',
    'HowHard5': 'Hard to Deal',

    'Scale1PSA1': 'Pain Severity',
    'Scale2LIA1': 'Life Interference',
    'Scale3LCA1': 'Life Control',
    'Scale4ADA1': 'Affective Distress',
    'Scale5SA1': 'Support',

    'Scale1PSA2': 'Pain Severity',
    'Scale2LIA2': 'Life Interference',
    'Scale3LCA2': 'Life Control',
    'Scale4ADA2': 'Affective Distress',
    'Scale5SA2': 'Support'
}

corr_matrix_1.rename(columns=name_mapping, index=name_mapping, inplace=True)
p_values_1.rename(columns=name_mapping, index=name_mapping, inplace=True)
corr_matrix_5.rename(columns=name_mapping, index=name_mapping, inplace=True)
p_values_5.rename(columns=name_mapping, index=name_mapping, inplace=True)

# Function to create the heatmap from the correlation matrix. Heatmap will include p-values calculated above.
def plot_heatmap(corr_matrix, p_values, title):
    # The mask is to hide the top half of the heatmap correlations (above r = 1).
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu', fmt='.2f', mask=mask, cbar_kws={'label': 'Spearman Correlation'})
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            p = p_values.iloc[i, j]
            if p < 0.01:
                plt.text(j + 0.5, i + 0.70, '**', ha='center', va='top', color='black')
            elif p < 0.05:
                plt.text(j + 0.5, i + 0.70, '*', ha='center', va='top', color='black')
    plt.title(title)

# Plot heatmaps and save them to a single SVG file. The SVG file gives it a better resolution.
plt.figure(figsize=(8, 12))

plt.subplot(2, 1, 1)
plot_heatmap(corr_matrix_1, p_values_1, 'Spearman Rho Correlation Matrix: Baseline')

plt.subplot(2, 1, 2)
plot_heatmap(corr_matrix_5, p_values_5, 'Spearman Rho Correlation Matrix: After Education')

# Format adjustment layout to prevent overlap between the subplots, saving figure as svg, and display preview of what the end product looks like.
plt.tight_layout()
plt.savefig('combined_correlation_heatmap.svg', dpi=1200)
plt.show()

# Moving the combined image to Google Drive.
shutil.move('combined_correlation_heatmap.svg', '/content/drive/My Drive/Colab Notebooks/combined_correlation_heatmap.svg')
