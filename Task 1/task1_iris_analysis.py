#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install pandas seaborn matplotlib


# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


print("--- STEP 1: LOADING DATASET ---")
# We load the dataset directly using seaborn's built-in dataset manager
df = sns.load_dataset('iris')

# Print the shape of the dataset
print(f"Dataset Shape (Rows, Columns): {df.shape}\n")
print("Column Names:")
print(list(df.columns), "\n")

# Print the first 5 rows
print("First 5 Rows of the Dataset:")
print(df.head(), "\n")


# In[4]:


# STEP 2: Summary Statistics
# ==========================================
print("--- STEP 2: SUMMARY STATISTICS ---")
# Get basic information about columns, non-null values, and data types
print("Dataset Info:")
df.info()
print("\n")

# Get descriptive statistics for numerical columns
print("Descriptive Statistics:")
print(df.describe(), "\n")


# In[5]:


print("--- STEP 3: GENERATING VISUALIZATIONS ---")

# Apply a clean, modern aesthetic s"tyle
sns.set_theme(style="whitegrid")

# 1. Scatter Plot: Sepal Length vs Sepal Width colored by Species
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, 
    x='sepal_length', 
    y='sepal_width', 
    hue='species', 
    palette='viridis', 
    s=70, 
    alpha=0.8
)
plt.title('Sepal Length vs Sepal Width by Species', fontsize=14, pad=15)
plt.xlabel('Sepal Length (cm)', fontsize=12)
plt.ylabel('Sepal Width (cm)', fontsize=12)
plt.tight_layout()
# Save the plot locally
plt.savefig('iris_scatter_plot.png', dpi=300)
plt.close()
print("- Saved: iris_scatter_plot.png")


# In[6]:


plt.figure(figsize=(12, 8))
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[feature], kde=True, color='royalblue', bins=15)
    plt.title(f'Distribution of {feature.replace("_", " ").title()}', fontsize=12)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
plt.tight_layout()
# Save the plot locally
plt.savefig('iris_histograms.png', dpi=300)
plt.close()
print("- Saved: iris_histograms.png")


# In[7]:


plt.figure(figsize=(10, 6))
# Melt dataframe to long format for easier plotting of multiple features alongside each other
df_melted = pd.melt(df, id_vars='species', value_vars=features)
sns.boxplot(
    data=df_melted, 
    x='variable', 
    y='value', 
    hue='species', 
    palette='Set2'
)
plt.title('Feature Distributions & Outlier Check by Species', fontsize=14, pad=15)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Values (cm)', fontsize=12)
plt.xticks(ticks=range(4), labels=[f.replace("_", " ").title() for f in features])
plt.tight_layout()
# Save the plot locally
plt.savefig('iris_box_plots.png', dpi=300)
plt.close()
print("- Saved: iris_box_plots.png")


# In[ ]:




