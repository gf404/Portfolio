# Import libraries: Pandas for data manipulation, Seaborn for heat map, matplotlib for displaying it.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import shutil

# Load csv file
from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv('/content/drive/My Drive/Nick_Paper_Education_Exercise_Study_Data_for_Correlation.csv')

# Numeric data only.
numeric_data = data.select_dtypes(include=['float64','int64'])

# Exclude specific columns
exclude_columns = ['PainProblems1', 'PainProblems5']

# Filter data for variables ending in '1' and 'A1', excluding the specified columns
data_1 = numeric_data.filter(regex='1$|A1$').drop(columns=exclude_columns, errors='ignore')
# Filter data for variables ending in '5' and 'A2', excluding the specified columns
data_5 = numeric_data.filter(regex='5$|A2$').drop(columns=exclude_columns, errors='ignore')

# Function to create correlation matrix and p-values
def create_corr_matrix(data):
    corr_matrix = data.corr(method='spearman')
    p_values = pd.DataFrame(np.zeros_like(corr_matrix), columns=corr_matrix.columns, index=corr_matrix.index)
    for col in data.columns:
        for row in data.columns:
            if col != row:
                corr, p = spearmanr(data[col], data[row])
                p_values.loc[row, col] = p
    return corr_matrix, p_values

# Create correlation matrices and p-values for data_1 and data_5
corr_matrix_1, p_values_1 = create_corr_matrix(data_1)
corr_matrix_5, p_values_5 = create_corr_matrix(data_5)

# Rename the columns and index of the correlation matrix and p-values DataFrame
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

# Incorporates the naming outlined under name_mapping for ease of reading for the reader when viewing the final matrices. 
corr_matrix_1.rename(columns=name_mapping, index=name_mapping, inplace=True)
p_values_1.rename(columns=name_mapping, index=name_mapping, inplace=True)
corr_matrix_5.rename(columns=name_mapping, index=name_mapping, inplace=True)
p_values_5.rename(columns=name_mapping, index=name_mapping, inplace=True)

# Function to plot heatmap
def plot_heatmap(corr_matrix, p_values, title, filename):
    # The mask is to hide the top half of the heatmap correlations (above r = 1).
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    # Sizing of figure.
    plt.figure(figsize=(8, 5))
    # Specs for heatmap. Includes the loop to arrive at significance for each correlation on the heatmap. 
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu', fmt='.2f', mask=mask, cbar_kws={'label': 'Spearman Correlation'})
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            p = p_values.iloc[i, j]
            # When p < 0.01 the ** symbol appears. 
            if p < 0.01:
                plt.text(j + 0.5, i + 0.65, '**', ha='center', va='top', color='black')
            # When p < 0.05 the * symbol appears. 
            elif p < 0.05:
                plt.text(j + 0.5, i + 0.65, '*', ha='center', va='top', color='black')
    # Portrays the title of the image.
    plt.title(title)
    # For saving the file with a resolution of 1200, tight to fit within window in Word. 
    plt.savefig(filename, dpi=1200, bbox_inches='tight')
    # To exhibit a display of the graphic once the program has started. 
    plt.show()

# Plot separate heatmaps, one at baseline (symbolize by ending in 1) and the other after education (symbolizes by ending in 5).
plot_heatmap(corr_matrix_1, p_values_1, 'Spearman Rho Correlation Matrix: Baseline', 'correlation_heatmap_baseline.svg')
plot_heatmap(corr_matrix_5, p_values_5, 'Spearman Rho Correlation Matrix: After Education', 'correlation_heatmap_after_education.svg')

# This moves the images to Google Drive in the appropriate folder. 
shutil.move('correlation_heatmap_baseline.svg', '/content/drive/My Drive/Colab Notebooks/correlation_heatmap_baseline.svg')
shutil.move('correlation_heatmap_after_education.svg', '/content/drive/My Drive/Colab Notebooks/correlation_heatmap_after_education.svg')
