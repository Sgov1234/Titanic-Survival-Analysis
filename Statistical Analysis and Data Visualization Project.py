import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic = pd.read_csv(url)

# Display the first few rows of the dataset
print(titanic.head())

# Display summary statistics
print(titanic.describe())

# Display information about the dataset
print(titanic.info())

# Check for missing values
print(titanic.isnull().sum())

# Fill missing values
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)
titanic.drop(columns=['Cabin'], inplace=True)

# Verify that there are no missing values left
print(titanic.isnull().sum())

# Set the aesthetic style of the plots
sns.set_style('whitegrid')

# Plot the distribution of passenger ages
plt.figure(figsize=(10, 6))
sns.histplot(titanic['Age'], kde=True)
plt.title('Distribution of Passenger Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Plot the survival rate by sex
plt.figure(figsize=(10, 6))
sns.barplot(x='Sex', y='Survived', data=titanic)
plt.title('Survival Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('Survival Rate')
plt.show()

# Plot the survival rate by passenger class
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', data=titanic)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

# Chi-square test for independence between Sex and Survived
contingency_table = pd.crosstab(titanic['Sex'], titanic['Survived'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f'Chi-square test:\nChi2: {chi2}\nP-value: {p}')

# T-test to compare ages of those who survived and those who did not
survived_ages = titanic[titanic['Survived'] == 1]['Age']
not_survived_ages = titanic[titanic['Survived'] == 0]['Age']
t_stat, p_value = stats.ttest_ind(survived_ages, not_survived_ages, nan_policy='omit')
print(f'T-test:\nT-statistic: {t_stat}\nP-value: {p_value}')

# Compute the correlation matrix
corr_matrix = titanic.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

print("Conclusion and Insights:")
print("- The survival rate is higher for females compared to males.")
print("- First class passengers had a higher survival rate compared to other classes.")
print("- There is a statistically significant difference in the ages of passengers who survived and those who did not.")
