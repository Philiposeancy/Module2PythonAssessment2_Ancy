import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from google.colab import files
uploaded=files.upload()
df=pd.read_csv("Titanic-Dataset (1).csv")
print('Passenger Count by Embarked Port')
print(df.head(5))
print(df.columns)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Embarked', hue='Sex', palette='Set2')
plt.title('Passenger Count by Embarked Port')
plt.xlabel('Embarked Port')
plt.ylabel('Number of Passengers')
plt.show()
print("Age Distribution across Classes")
plt.figure(figsize=(8, 5))

sns.swarmplot( data=df,x='Pclass',y='Age',hue='Sex',palette='Set2')
plt.title('Age Distribution across Passenger Classes')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.show()
gender_count = df['Sex'].value_counts()

plt.figure(figsize=(7, 7))
plt.pie(gender_count,labels=gender_count.index,autopct='%1.1f%%',explode=(0, 0.1), startangle=90
)
plt.title('Gender Distribution of Titanic Passengers')
plt.show()
print("Correlation Matrix Heatmap....")
data=df[['Age','Fare','Pclass','SibSp','Parch','Survived']]
corr=data.corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr,annot=True,cmap='coolwarm',fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()
print("Fare Distribution by Class and Gender")
plt.figure(figsize=(8, 5))
sns.violinplot(data=df,x='Pclass',y='Fare',hue='Sex')
plt.title('Fare Distribution by Class and Gender')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()

print("Scatter Plot: Age vs Fare")
sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived')
plt.title('Age vs Fare by Survival Status')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(title='Survived')
plt.show()



