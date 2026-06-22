## DevelopersHub AI/ML Engineering Internship

# Task 1: Exploring and Visualizing a Simple Dataset (Iris Dataset)

## Objective

The goal of this project is to build a foundational understanding of Exploratory Data Analysis (EDA). By loading, inspecting, and visualizing the classic Iris dataset, we aim to uncover underlying distributions, feature relationships, and detect any structural anomalies or outliers before running machine learning models.

## Dataset Overview

Dataset Name: Iris Dataset (loaded via Seaborn's built-in manager)

Shape: 150 rows, 5 columns

Columns:

sepal_length (Numerical - cm)

sepal_width (Numerical - cm)

petal_length (Numerical - cm)

petal_width (Numerical - cm)

species (Categorical - setosa, versicolor, virginica)

## Installation & Quick Start

Clone this repository:

git clone https://github.com/numanabubakar/ai-ml-internship-task.git
cd Task 1


Install the required dependencies:

pip install pandas matplotlib seaborn


Run the analysis script:

python task1_iris_analysis.py


## Methodology & Project Structure

The project code follows a modular, three-step approach:

### 1. Load and Inspect

Loaded the dataset into a pandas.DataFrame.

Used .shape, .head(), and columns lookup to inspect data integrity.

Utilized .info() to check for missing values (non-null counts) and verified that there are no missing fields in this dataset.

### 2. Summary Statistics

Generated descriptive metrics using .describe().

Assessed range, mean, and standard deviation across features to understand overall scales.

3. Data Visualization

Three primary visualizations are generated and saved locally in high-definition (300 DPI):

iris_scatter_plot.png: A scatter plot of Sepal Length vs. Sepal Width colored by species. This reveals clear linear clustering patterns, showing that setosa is highly distinct, while versicolor and virginica have mild overlap.

iris_histograms.png: Histograms representing the frequency distribution of all four features. This identifies bimodal distributions in petal features, further emphasizing species differences.

iris_box_plots.png: A comprehensive box plot comparing numerical variables across species to flag data variance and outliers.

### Key Findings & Analytical Insights

Species Separability: The setosa species can be easily separated from the other two species based on its significantly smaller petal length and width.

Outlier Analysis (from Box Plots):

Sepal Width Outliers: The box plot visualization reveals that there are noticeable outliers in the sepal_width feature, specifically within the virginica species distribution (points extending beyond the upper/lower whiskers of the box plot).

Feature Stability: Petal measurements exhibit tight clusters for setosa, whereas versicolor and virginica display broader ranges of variance, indicating higher morphological diversity in these species.

### Key Skills Demonstrated

Data Ingestion and Preprocessing (Pandas)

Statistical Summary and Profiling

Advanced Charting Aesthetics (Seaborn & Matplotlib)

Analytical Interpretation of Statistics (outlier detection, clustering intuition)