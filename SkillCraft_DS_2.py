import pandas as pd

# Load datasets
train_df = pd.read_csv(r"titanic/train.csv")
test_df = pd.read_csv(r"titanic/test.csv")
gender_df = pd.read_csv(r"titanic/gender_submission.csv")

print(train_df.head())
print(train_df.info())
print(train_df.isnull().sum())

# Fill Age with median
train_df["Age"].fillna(train_df["Age"].median(), inplace=True)

# Fill Embarked with most common value
train_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace=True)

# Drop Cabin (too many missing values)
train_df.drop(columns=["Cabin"], inplace=True)

# Convert Sex to numeric (male=0, female=1)
train_df["Sex"] = train_df["Sex"].map({"male": 0, "female": 1})

import seaborn as sns
import matplotlib.pyplot as plt

# Survival count
sns.countplot(x="Survived", data=train_df)
plt.title("Survival Count")
plt.show()

# Survival by Gender
sns.countplot(x="Sex", hue="Survived", data=train_df)
plt.title("Survival by Gender")
plt.show()

# Survival by Class
sns.countplot(x="Pclass", hue="Survived", data=train_df)
plt.title("Survival by Passenger Class")
plt.show()

# Age distribution
sns.histplot(train_df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Survival by Age
sns.histplot(data=train_df, x="Age", hue="Survived", multiple="stack")
plt.title("Survival by Age")
plt.show()
