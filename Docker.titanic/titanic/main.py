# -*- coding: utf-8 -*-
"""Titanic EDA Script"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class TitanicEDA:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Loads the Titanic dataset into a pandas DataFrame."""
        if not os.path.exists(self.file_path):
            print(f"File not found: {self.file_path}")
            return 

        try:
            self.df = pd.read_csv(self.file_path)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def generate_summary_statistics(self):
        """Generates summary statistics for the Titanic dataset."""
        if self.df is not None:
            summary = self.df.describe(include='all')
            print("Summary statistics:")
            print(summary)
        else:
            print("Data is not loaded yet.")

    def visualize_survival_by_feature(self, feature):
        """Visualizes survival distribution by a given feature."""
        if self.df is not None:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=self.df, x=feature, hue='Survived')
            plt.title(f"Survival Distribution by {feature}")
            plt.xlabel(feature)
            plt.ylabel('Count')
            plt.savefig(f"survival_by_{feature}.png")
            plt.close()  # Close the plot to prevent display
        else:
            print("Data is not loaded yet.")

    def visualize_age_distribution(self):
        """Visualizes age distribution and survival rates."""
        if self.df is not None:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.df, x='Age', hue='Survived', kde=True, bins=30)
            plt.title("Age Distribution with Survival Rates")
            plt.xlabel('Age')
            plt.ylabel('Density')
            plt.savefig("age_distribution_survival.png")
            plt.close()  # Close the plot to prevent display
        else:
            print("Data is not loaded yet.")

    def visualize_sex_survival(self):
        """Visualizes survival rates based on Sex."""
        if self.df is not None:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=self.df, x='Sex', hue='Survived')
            plt.title("Survival Distribution by Sex")
            plt.xlabel('Sex')
            plt.ylabel('Count')
            plt.savefig("survival_by_sex.png")
            plt.close()  # Close the plot to prevent display
        else:
            print("Data is not loaded yet.")

if __name__ == "__main__":
    # Update the path to your dataset
    dataset_path = os.path.join("Data", "train.csv")

    # Initialize the TitanicEDA object with the correct dataset path
    titanic_eda = TitanicEDA(dataset_path)

    # Run the analysis steps
    titanic_eda.load_data()
    titanic_eda.generate_summary_statistics()
    titanic_eda.visualize_survival_by_feature('Pclass')
    titanic_eda.visualize_survival_by_feature('Sex')
    titanic_eda.visualize_age_distribution()
    titanic_eda.visualize_sex_survival()