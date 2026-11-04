import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.preprocessing import StandardScaler, MinMaxScaler
uploaded=files.upload()
df=pd.read_csv("Employee.csv")

print("Data Exploration")
df.head(5)
df
df.shape
df.columns
df.info()
df.describe()

print("......................................")
print("Total number of unique values")
print(df.nunique())

print(df['Company'].unique())
print(df['Age'].unique())
print(df['Place'].unique())
print(df['Country'].unique())
print(df['Gender'].unique())

df['Company']=df['Company'].replace(['TCS','Tata Consultancy Services'],'TCS')
df['Company']=df['Company'].replace(['Infosys','Infosys Pvt Lmt'],'Infosys')
print(df['Company'].unique())
print("......................................")
print("Total number of unique values")
print(df.nunique())
df.describe()
print(">>>>>>>>>>>>>>>>>>")
print("Data Cleaning")
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df['Age']=df['Age'].replace(0,np.nan)
df.head(10)
null_count=df.isnull().sum()
null_percentage=df.isnull().sum()/len(df)* 100
print("Null Count:",null_count)
print("Null Percentage:",null_percentage)
sns.boxplot(x=df['Age'])
plt.show()

sns.boxplot(x=df['Salary'])
plt.show()
print("No outliers")
df.describe()
print("Skewness check.......")
print("Age Skewness:", df['Age'].skew())
print("Salary Skewness:", df['Salary'].skew())
print("......................................")
print("IQR method")
Q1=df['Age'].quantile(0.25)
Q3=df['Age'].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
age_outliers=df[(df['Age']>=lower_bound)&(df['Age']<=upper_bound)]
print("Number of Outliers:", len(age_outliers))
Q1=df['Salary'].quantile(0.25)
Q3=df['Salary'].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
Salary_outliers=df[(df['Salary']>=lower_bound)&(df['Salary']<=upper_bound)]
print("Number of Outliers:", len(Salary_outliers))
print("Mean:", df['Salary'].mean())
print("Median:", df['Salary'].median())

print("Replace missing Age &Salary values with Mean")
df['Age']=df['Age'].fillna(df['Age'].mean())
df['Salary']=df['Salary'].fillna(df['Salary'].mean())

print(df[['Age','Salary']].isnull().sum())
print("......................................")
df['Company']=df['Company'].fillna(df['Company'].mode()[0])

df['Place']=df['Place'].fillna(df['Place'].mode()[0])

df['Country']=df['Country'].fillna(df['Country'].mode()[0])

df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])

print(df.isnull().sum())
print("Drop column...")
df.drop(columns=['Country'],inplace=True)
df
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("Data Analysis................")
filtered_data= df[(df['Age']>40)&(df['Salary']<5000)]
print(filtered_data)


plt.scatter(df['Age'],df['Salary'])
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Age vs Salary')
plt.show()
place_count=df['Place'].value_counts()
print(place_count)
place_count.plot(kind='bar')
plt.xlabel('Place')
plt.ylabel('Number of Employees')
plt.title('Number of Employees from Each Place')
plt.show()
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("Data Encoding..........................")
print("One hot encoding..")
df=pd.get_dummies(df,columns=['Company','Place'],dtype=int)
df.head()
print("Standard scaling..............")
numerical_cols=df.select_dtypes(include=np.number).columns
standard_scaler=StandardScaler()
standard_scaled=df.copy()
standard_scaled[numerical_cols]=standard_scaler.fit_transform(df[numerical_cols])
minmax_scaler=MinMaxScaler()
minmax_scaled=df.copy()
minmax_scaled[numerical_cols] = minmax_scaler.fit_transform(df[numerical_cols])
