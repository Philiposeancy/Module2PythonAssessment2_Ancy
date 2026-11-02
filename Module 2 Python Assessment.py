Module 2: Python Assessment 2


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
uploaded = files.upload()
df=pd.read_excel("ABC Company.xlsx")
print(df.head(5))
df.shape
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
df = df.dropna(subset=['Name', 'Team', 'Position', 'Age', 'Salary'])
np.random.seed(42)
print(df[['Name', 'Team', 'Position', 'Age', 'Salary', 'Height']].isnull().sum())
df['height']=np.random.randint(150,181,size=len(df))
print("Data after changing")
display(df[['Height']].head(11))


team_counts=df['Team'].value_counts()
print("Team counts:")
print(team_counts)
team_percentage=(df['Team'].value_counts(normalize=True)*100).round(2)
team_distribution=pd.DataFrame({'Team counts':team_counts,'Team percentage':team_percentage})
print("Team distribution:")
print(team_distribution)
plt.figure(figsize=(9, 5))
ax = sns.barplot(
    x=team_counts.index,
    y=team_counts.values,
    hue=team_counts.index,
    palette='Set2',
    legend=False
)
plt.title("Employee Distribution Across Teams")
plt.xlabel("Team")
plt.ylabel("Number of Employees")
plt.xticks(rotation=45)

print("..............")
position_counts=df['Position'].value_counts()
print("Position counts:")
print(position_counts)
group_by_position=df.groupby('Position')['Name'].apply(list)
print("Group by position:")
print(group_by_position)
plt.figure(figsize=(9, 5))
sns.barplot(
    x=position_counts.index,
    y=position_counts.values,
    hue=position_counts.index,
    palette='pastel',
    legend=False
)
plt.title("Employee Distribution by Position")
plt.xlabel("Position")
plt.ylabel("Number of Employees")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("........................")
bins=[20,30,40,50,60]
labels=['20-30','30-40','40-50','50-60']
df['age_group']=pd.cut(df['Age'],bins=bins,labels=labels)
age_group_counts=df['age_group'].value_counts()
print("Age group counts:")
print(age_group_counts)
predominent_age_group=age_group_counts.idxmax()
print("Predominent age group:",predominent_age_group)

plt.figure(figsize=(8, 5))
ax = sns.barplot(
    x=age_group_counts.index.astype(str),
    y=age_group_counts.values,
    hue=age_group_counts.index.astype(str),
    palette='viridis',
    legend=False
)
plt.title("Employee Distribution by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Employees")

team_salary=df.groupby('Team')['Salary'].mean()
print("Team salary:")
print(team_salary)
highest_team=team_salary.idxmax()
print("Highest paid team:",highest_team)
highest_amount_team=team_salary.max()
print("Highest paid amount:",highest_amount_team)
print("-"*40)
position_salary=df.groupby('Position')['Salary'].mean()
print("Position salary:")
print(position_salary)
highest_position=position_salary.idxmax()
print("Highest paid position:",highest_position)
highest_position_amount=position_salary.max()
print("Highest paid position amount:",highest_position_amount)
plt.figure(figsize=(9, 5))
sns.barplot(
    x=team_salary.index,
    y=team_salary.values,
    hue=team_salary.index,
    palette='Blues_r',
    legend=False
)
plt.title("Total Salary Expenditure by Team")
plt.xlabel("Team")
plt.ylabel("Total Salary Expenditure")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print(".................")
correlation=df['Age'].corr(df['Salary'])
print("Correlation:",correlation)
if correlation>0.5:
    print("There is a strong positive correlation between Age and Salary")
elif correlation<-0.5:
    print("There is a strong negative correlation between Age and Salary")
else:
    print("There is no strong correlation between Age and Salary")
plt.figure(figsize=(8,6))
sns.regplot(x='Age',y='Salary',data=df,color="purple",scatter_kws={"alpha":0.6})
plt.title("Correlation between Age vs Salary",fontsize=14,fontweight="bold")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.grid(True,linestyle="--",alpha=0.5)
plt.show()


