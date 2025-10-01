import seaborn as sns
import matplotlib.pyplot as plt

def plot_attrition_distribution(df):
    sns.countplot(x="Attrition", data=df)
    plt.title("Employee Attrition Distribution")
    plt.show()

def plot_job_satisfaction(df):
    sns.countplot(x="JobSatisfaction", hue="Attrition", data=df)
    plt.title("Job Satisfaction vs Attrition")
    plt.show()
