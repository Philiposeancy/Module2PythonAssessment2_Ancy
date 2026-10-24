Module2PythonAssessment2_Ancy

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("customer_behavior.csv")
print(df.head(5))
print("............................")
print("Data Cleaning")
print("No of rows before cleaning:",len(df))
df=df.dropna(subset=['PurchaseAmount','Gender'])
df['PurchaseAmount']=pd.to_numeric(df['PurchaseAmount'],errors='coerce')
df=df.dropna(subset=['PurchaseAmount'])
df=df[df['PurchaseAmount']>=0]
print("No of rows after cleaning:",len(df))
print("............................")
mean_purchase=df['PurchaseAmount'].mean()
print("Mean Purchase Amount:",mean_purchase)
median_purchase=df['PurchaseAmount'].median()
print("Median Purchase Amount:",median_purchase)
mode_purchase=df['PurchaseAmount'].mode()[0]
print("Mode Purchase Amount:",mode_purchase)
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['PurchaseAmount'], color='skyblue')
plt.title("Overall Boxplot of Purchase Amount")
plt.show()
print("............................")
Q1=df['PurchaseAmount'].quantile(0.25)
Q3=df['PurchaseAmount'].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
outliers=df[(df['PurchaseAmount']<lower_bound)|(df['PurchaseAmount']>upper_bound)]
print("Outliers:")
print(outliers)
print("lower bound:",lower_bound)
print("upper bound:",upper_bound)
print("No of Outliers",len(outliers))
print("............................")
purchase_skew=df['PurchaseAmount'].skew()
purchase_kurt=df['PurchaseAmount'].kurt()
print(f"Skewness..: {purchase_skew:.2f}")
print(f"Kurtosis....: {purchase_kurt:.2f}")
print("............................")
male_purchase=df[df['Gender']=='Male']['PurchaseAmount']
female_purchase=df[df['Gender']=='Female']['PurchaseAmount']
t_stat,p_value=stats.ttest_ind(male_purchase,female_purchase)
print("P value..:",p_value)
if p_value<0.05:
    print("There is a significant difference in spending between Male and Female customers..")
else:
    print(" There is no significant difference in spending between Male and Female customers.")
plt.figure(figsize=(6, 4))
sns.boxplot(x='Gender',y='PurchaseAmount',data=df,palette='Set2')
plt.title('Purchase Amount Distribution by Gender')
plt.show()
print("............................")
contigency_table=pd.crosstab(df['ProductCategory'],df['Churn'])
result = stats.chi2_contingency(contigency_table)
p_value = result[1] 
print("P value..:",p_value)
if p_value<0.05:
    print("There is a significant association between Product Category and Churn")
else:
    print("There is no significant association between Product Category and Churn")
    contigency_table.plot(kind='bar',stacked=True,figsize=(8, 5),color=['#4c72b0', '#c44e52'])
plt.title('Customer Churn Status by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.legend(['Not Churned(0)','Churned(1)'])
plt.tight_layout()
plt.show()
print("............................")
regions=df['Region'].unique()
print("Regions:",regions)
groups=[df[df['Region']==region]['PurchaseAmount'] for region in regions]
f_stat,p_value=stats.f_oneway(*groups)
print("P value..:",p_value)
if p_value<0.05:
    print("There is a significant difference in spending across different regions")
else:
    print("There is no significant difference in spending across different regions")
plt.figure(figsize=(8, 5))
sns.barplot(x='Region', y='PurchaseAmount', data=df, estimator=np.mean, errorbar=None, palette='viridis')
plt.title('Average Purchase Amount Across Different Regions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("............................")
print("............................")
Campaign_A=df[df['CampaignGroup']=='A']['PurchaseAmount']
Campaign_B=df[df['CampaignGroup']=='B']['PurchaseAmount']
mean_A=Campaign_A.mean()
mean_B=Campaign_B.mean()
print("Mean Purchase Amount in Campaign A:",mean_A)
print("Mean Purchase Amount in Campaign B:",mean_B)
t_stat,p_value=stats.ttest_ind(Campaign_A,Campaign_B)
print("P value..:",p_value)
if p_value<0.05:
  print("Reject the Null Hypothesis")
  if(mean_A>mean_B):
    print("Campaign A has higher Significance Perfomance")
  else:
    print("Campaign B has higher Significance Perfomance")
else:
   print("Fail to reject null hypothesis")
plt.figure(figsize=(5, 4))
sns.barplot(x='CampaignGroup', y='PurchaseAmount', data=df[df['CampaignGroup'].isin(['A', 'B'])], 
            estimator=np.mean, palette='pastel', capsize=0.1)
plt.title('Campaign Performance: Group A vs Group B')
plt.show()
print("............................")
print("............................")
stst,p_value=stats.normaltest(df['PurchaseAmount'].dropna())
print("P value..:",p_value)
if p_value<0.05:
    print("The data is not normally distributed")
else:
    print("The data is normally distributed")
print("............................")
sample_means = []
for _ in range(1000):
  sample=df['PurchaseAmount'].dropna().sample(n=50,replace=True)
  sample_means.append(sample.mean())
plt.figure(figsize=(8,5))
sns.histplot(sample_means,kde=True)
plt.title("Distribution of Sample Means")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.show()
plt.figure(figsize=(8, 5))
sns.histplot(sample_means, kde=True, color='teal')
plt.title("Distribution of Sample Means (Central Limit Theorem)")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.show()
print("............................")
print("............................")
data_ci = df['PurchaseAmount'].dropna()
mean_ci = np.mean(data_ci)
sem_ci = stats.sem(data_ci)
confidence_interval = stats.t.interval(0.95, df=len(data_ci)-1, loc=mean_ci, scale=sem_ci)
print(f"Mean Purchase Amount: {mean_ci:.2f}")
print(f"95% Confidence Interval: {confidence_interval}")
print("............................")







