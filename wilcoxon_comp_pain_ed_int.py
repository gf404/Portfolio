# Import necessary libraries for data manipulation, statistical analysis, and visualization.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
import shutil

# Mounts my Google Drive to access the dataset.
from google.colab import drive
drive.mount('/content/drive')

# Loads the CSV file containing pain management before and after an education intervention into a Pandas DataFrame.
try:
    data = pd.read_csv('/content/drive/My Drive/Nick_Paper_Education_Exercise_Study_Data_for_Correlation.csv')
except FileNotFoundError:
    print("Error: The specified CSV file was not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()

# Selects only numeric data, excluding subject IDs and specified columns.
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

# Applies the column renaming to both datasets
data_1.rename(columns=name_mapping, inplace=True)
data_5.rename(columns=name_mapping, inplace=True)

# Calculates means and standard deviations for baseline and after education datasets.
means_1 = data_1.mean()
means_5 = data_5.mean()
std_devs_1 = data_1.std()
std_devs_5 = data_5.std()

# Performs Wilcoxon signed-rank tests to compare baseline and after education scores and extracts, p-values to assess statistical significance.
p_values = pd.Series({col: wilcoxon(data_1[col], data_5[col]).pvalue for col in means_1.index})

# Prints p-values for each comparison to assess statistical significance.
print("P-values for each comparison:")
print(p_values)

# Creates a DataFrame combining means, standard deviations, and p-values for plotting.
plot_data = pd.DataFrame({
    "Variable": means_1.index,
    "Baseline": means_1.values,
    "After Education": means_5.values,
    "p-value": p_values.values,
    "std_dev_baseline": std_devs_1.values,
    "std_dev_after": std_devs_5.values
})

# Reshapes the DataFrame from a wide to long format using melt, making it suitable for seaborn plotting
plot_data_melted = plot_data.melt(id_vars=["Variable", "p-value"], value_vars=["Baseline", "After Education"],
                                  var_name="Condition", value_name="Score")

# Sets the seaborn theme, this is entirely for aesthetics.
sns.set_theme(style="whitegrid")
# Determines the size of the entire graphic.
plt.figure(figsize=(20, 15))
# Creates a grouped bar chart using seaborn to compare the baseline and after education scores.
g = sns.catplot(
    data=plot_data_melted, kind="bar",
    x="Variable", y="Score", hue="Condition",
    palette={"Baseline": "white", "After Education": "gray"},
    alpha=.6, height=6, ci=None, edgecolor="black"
)

# Iterates over each variable to add error bars and significance markers.
for i, variable in enumerate(plot_data["Variable"]):
    # Calculates positions for the bars.
    # Position for the Baseline bar.
    pos_baseline = i - 0.17  
    # Position for the After Education bar.
    pos_after = i + 0.17     

    # Retrieves the mean scores for both conditions. Calculated above.
    mean_baseline = plot_data["Baseline"][i]
    mean_after = plot_data["After Education"][i]

    # Retrieves standard deviations for both conditions. Calculated above.
    std_dev_baseline = plot_data["std_dev_baseline"][i]
    std_dev_after = plot_data["std_dev_after"][i]

    # Plots error bars for the Baseline condition.
    plt.errorbar(
        pos_baseline, mean_baseline,
        yerr=std_dev_baseline,
        fmt='none', c='black', capsize=5
    )

    # Plots error bars for the After Education condition.
    plt.errorbar(
        pos_after, mean_after,
        yerr=std_dev_after,
        fmt='none', c='black', capsize=5
    )

    # Checks if the p-value indicates statistical significance
    p_value = plot_data["p-value"][i]
    if p_value < 0.05:
        # Determines the highest point of the error bars for annotation placement.
        y_max_baseline = mean_baseline + std_dev_baseline
        y_max_after = mean_after + std_dev_after
        y_max = max(y_max_baseline, y_max_after)
        # Position above the highest error bar.
        y_annotation = y_max + 0.05  

        # Adds an asterisk to denote statistical significance.
        plt.text(
            i, y_annotation, '*',
            ha='center', va='bottom', color='black'
        )

        # Draws lines to connect the two bars indicating significance.
        # Left vertical line from top of Baseline error bar to annotation.
        plt.plot(
            [pos_baseline, pos_baseline],
            [y_max_baseline, y_annotation],
            color='black', linestyle='-', linewidth=0.75
        )

        # Horizontal line connecting both vertical lines at the annotation level.
        plt.plot(
            [pos_baseline, pos_after],
            [y_annotation, y_annotation],
            color='black', linestyle='-', linewidth=0.75
        )

        # Right vertical line from annotation to top of After Education error bar.
        plt.plot(
            [pos_after, pos_after],
            [y_annotation, y_max_after],
            color='black', linestyle='-', linewidth=0.75
        )

# Customizes plot aesthetics by removing left spine, setting labels, rotating x-tick labels, adjusting the legend.
g.despine(left=True)
g.set_axis_labels("Quantitative Measures", "Mean Scores")
g.set_xticklabels(rotation=45, ha="right")
g.legend.set_title("")
g.legend.set_bbox_to_anchor((1, 0.86))

# Adjusts the layout to ensure the labels and legend are fully visible.
plt.tight_layout(rect=[0, 0, 1.2, 1.2])

# Adds a title to the bar graph.
plt.title('Comparison of Quantitative Measures')

# Adjusts the top margin to make space for the legend.
plt.subplots_adjust(top=0.9)

# Saves the plot as a high-resolution SVG file and displays it.
plt.savefig('comparison_wilcoxon_bar_plot.svg', dpi=1200)
plt.show()

# Moves the saved plot to Google Drive for access.
try:
    shutil.move('comparison_wilcoxon_bar_plot.svg', '/content/drive/My Drive/Colab Notebooks/comparison_wilcoxon_bar_plot.svg')
except FileNotFoundError:
    print("Error: The plot file was not found. Please ensure it was saved correctly.")
except Exception as e:
    print(f"An unexpected error occurred while moving the file: {e}")
