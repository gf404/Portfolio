### 3. Data Summaries and Exploratory Data Analysis (EDA)
# Imports necessary libraries for data manipulation, statistical analysis, and visualization.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
import shutil

# Mounts Google Drive to access the dataset.
from google.colab import drive
drive.mount('/content/drive')

# Loads the CSV file containing pain management before and after an education intervention into a Pandas DataFrame.
try:
    # To read the csv file database.
    data = pd.read_csv('/content/drive/My Drive/Nick_Paper_Education_Exercise_Study_Data_for_Correlation.csv')
except FileNotFoundError:
    print("Error: The specified CSV file was not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()

# Dataset Overview. Identifies size of dataset, and data type of each column (int, float, object, etc.).
print("\nDataset Overview:")
print(f"Shape of the dataset: {data.shape}")
print("\nColumn Data Types:")
print(data.dtypes)

# Checks for any missing values.
print("\nMissing Values Count:")
print(data.isnull().sum())

# Visualizes any missing values.
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Summary Statistics for Numerical Data (Descriptive Stats).
print("\nSummary Statistics for Numerical Data:")
print(data.describe())

# Summary for Categorical Data (if any)
categorical_cols = data.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print("\nFrequency Counts for Categorical Data:")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(data[col].value_counts())

# Selects only numeric data, excluding specific columns.
numeric_data = data.select_dtypes(include=['float64', 'int64'])
exclude_columns = ['PainProblems1', 'PainProblems5']

# Filters baseline and post-education data based on column suffixes.
data_1 = numeric_data.filter(regex='1$|A1$').drop(columns=exclude_columns, errors='ignore')
data_5 = numeric_data.filter(regex='5$|A2$').drop(columns=exclude_columns, errors='ignore')

# Creates a mapping to rename columns for better readability in plots.
name_mapping = {
    'PainDays1': 'Days Manageable Pain',
    'InterfereActive1': 'Interference Activity',
    'InterfereMood1': 'Interference Mood',
    'InterfereSleep1': 'Interference Sleep',
    'HowHard1': 'Hard to Deal',

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

# Applies column renaming to both datasets.
data_1.rename(columns=name_mapping, inplace=True)
data_5.rename(columns=name_mapping, inplace=True)

# Generates Histograms for Numerical Data.
numeric_cols = numeric_data.columns
print("\nGenerating histograms for numerical data...")
numeric_data[numeric_cols].hist(figsize=(12, 10), bins=20, color='#4E79A7', edgecolor='black')
plt.suptitle("Distributions of Numeric Data")
plt.tight_layout()
plt.show()

# Boxplots for Detecting Outliers.
plt.figure(figsize=(12, 8))
sns.boxplot(data=numeric_data, orient="h", palette="Set2")
plt.title("Boxplots of Numeric Data")
plt.show()

# Correlation Analysis of entire dataset.
print("\nCorrelation Analysis:")
corr_matrix = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot for Key Variables.
# Adjusts the number of columns for pairplot
selected_columns = data_1.columns[:4]  
if len(selected_columns) > 1:
    print("\nGenerating pairplot for selected data...")
    sns.pairplot(data_1[selected_columns])
    plt.show()

# Statistical Analysis: Mean, Std Dev, and Wilcoxon Test
means_1 = data_1.mean()
means_5 = data_5.mean()
std_devs_1 = data_1.std()
std_devs_5 = data_5.std()

# Performs the Wilcoxon signed-rank tests and extracts p-values.
p_values = pd.Series({col: wilcoxon(data_1[col], data_5[col]).pvalue for col in means_1.index})

print("\nP-values for each comparison:")
print(p_values)

# Creates a DataFrame for Plotting Results.
plot_data = pd.DataFrame({
    "Variable": means_1.index,
    "Baseline": means_1.values,
    "After Education": means_5.values,
    "p-value": p_values.values,
    "std_dev_baseline": std_devs_1.values,
    "std_dev_after": std_devs_5.values
})

# Saves Cleaned Data.
output_path = '/content/drive/My Drive/cleaned_processed_pain_data.csv'
data.to_csv(output_path, index=False)
print(f"\nCleaned data saved to: {output_path}")
