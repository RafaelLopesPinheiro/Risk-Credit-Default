#!/usr/bin/env python
# coding: utf-8

# # Summary
#     This project is build using dataset of customer behavior from Kaggle, looking for predict the capability of payment, given a new client data.
#     Using various types of categorical models, spliting data set to better implement and deploy.
#     

# # Libraries 

# In[64]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
sns.set_theme(color_codes=True, style='dark', font='sans-serif')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[148]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# # Getting Data

# In[4]:


# reading data from csv file
df = pd.read_csv("../Python for DS/Bank Dataset.csv")
df


# In[78]:


# Loking through the variables types
df.info()
# Looking for NA in Dataset
df.isnull().sum()


# In[79]:


df.describe() 


# # EDA

# Deeper looking on variables, searching for patterns and anomalies that could help us understand better how they affect the target variable

# In[40]:


sns.countplot(x='House_Ownership', hue='Risk_Flag', data=df)


# In[150]:


plt.figure(figsize=(8,7))
fig, ax = plt.subplots(1,2)
sns.countplot(df['Married/Single'], ax=ax[0], hue=df['Risk_Flag'])
sns.countplot(df['Car_Ownership'], ax=ax[1], hue=df['Risk_Flag'])
plt.tight_layout()


# In[151]:


sns.distplot(a=df['Age']) 
#sns.displot(x=df['Age']);


# In[152]:


sns.distplot(a=df['Income'])  
#well distributed income data


# In[77]:


sns.heatmap(data=df.corr(), annot=True)  

#variables looking good, not much correlated, excluding 'Experience' and 'current_job_years', that make sense being correlated.


# Checking for Outliers

# In[84]:


sns.boxplot(data=df['Age'])  #no outliers


# In[87]:


sns.boxplot(data=df['Income']) #no outliers


# In[98]:


p = df.groupby('Risk_Flag')['Risk_Flag'].count()
plt.pie(p, explode=[0.05, 0.1], labels=['Non-Defaulters', 'Defaulter'], radius=2, autopct='%1.1f%%', shadow=True)


# In[102]:


#Counting how many unique rows have on that columns
print(len(df['CITY'].unique()))
print(len(df['STATE'].unique()))
print(len(df['Profession'].unique()))


# # Summary of data visualization
#     The target variable are skewed heavly (88% - 12%) 
#     No Outliers but need to scale age and income
#     Experience and job years are correlated, drop then on feature selection or use Principal Component Analysis(PCA)
#     Put married and car ownership on binare form
#     Find the relationship with categorical variables and target variable using chi-square test

# In[106]:


#build chi-square test function

def chi_square_test(data):
    stat, p, dof, expected = chi2_contingency(car_ownership_risk_flag)
    alpha = 0.05
    print('p values is'+ str(p))
    if p <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (H0 holds true)')


# In[107]:


car_ownership_risk_flag = pd.crosstab(df['Car_Ownership'], df['Risk_Flag'])
car_ownership_risk_flag


# In[108]:


chi_square_test(car_ownership_risk_flag)


# In[109]:


marital_status_risk_flag = pd.crosstab(df['Married/Single'], df['Risk_Flag'])
marital_status_risk_flag


# In[110]:


chi_square_test(marital_status_risk_flag)


# In[111]:


house_ownership_risk_flag = pd.crosstab(df['House_Ownership'], df['Risk_Flag'])
house_ownership_risk_flag


# In[112]:


chi_square_test(house_ownership_risk_flag)


# Building PCA

# In[113]:


features = ['CURRENT_JOB_YRS','Experience']

df_for_pca = df[features]
scaled_df_for_pca = (df_for_pca - df_for_pca.mean(axis=0)/df_for_pca.std())  #normalizate(scaled) data on df
scaled_df_for_pca


# In[114]:


pca = PCA()
df_pca = pca.fit_transform(scaled_df_for_pca)
component_names = [f"PC{i+1}" for i in range(df_pca.shape[1])]
df_pca = pd.DataFrame(df_pca, columns=component_names)

df_pca.head()


# In[115]:


df1 = pd.concat([df, df_pca], axis=1)  #new df with PCA variables 
df1.head()


# In[116]:


features = ['Married/Single', 'Car_Ownership', 'Profession', 'CITY', 'STATE']
label_encoder = LabelEncoder()

for col in features:
    df1[col] = label_encoder.fit_transform(df1[col])


# In[117]:


df2= pd.get_dummies(df1, columns=["House_Ownership"])
df2.drop(['Id'], axis=1, inplace=True)


# In[119]:


X = df2.drop(['Risk_Flag'], axis=1)
y = df2.Risk_Flag

#setting the train and test dataframes to use in various models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[120]:





# In[121]:


sm = SMOTE(random_state = 500) 
X_res, y_res = sm.fit_resample(X_train, y_train)


# # TESTING AND EVALUATING MODELS

# In[124]:


### REGRESSÃO LOGISTICA

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter = 500000)
model.fit(X_res, y_res)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
accuracy


# In[153]:


print(classification_report(y_test,y_pred))


# In[128]:


### KNN Model

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
model.fit(X_res, y_res)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
accuracy


# In[129]:


print(classification_report(y_test, y_pred))


# In[130]:


### RANDOM FOREST CLASSIFICATION

from sklearn.ensemble import RandomForestClassifier

mode = RandomForestClassifier(criterion='gini', bootstrap=True, random_state=420)
model.fit(X_res, y_res)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
accuracy


# In[131]:


print(classification_report(y_test, y_pred))


# In[132]:


### DECISION TREE

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='entropy', random_state=420)
model.fit(X_res, y_res)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
accuracy


# In[133]:


print(classification_report(y_test, y_pred))


# In[136]:


### XGBOOST 

from xgboost import XGBClassifier

model = XGBClassifier(learning_rate=0.1, n_estimators=1000, use_label_encoder=False, random_state=420)
model.fit(X_res,y_res)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
accuracy


# In[137]:


print(classification_report(y_test, y_pred))


# In[138]:


### AdaBoost Classifier

from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(random_state=15000)
model.fit(X_res,y_res)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
accuracy


# In[139]:


print(classification_report(y_test, y_pred))


# In[140]:


### Passive Agressive Classifier

from sklearn.linear_model import PassiveAggressiveClassifier

model = PassiveAggressiveClassifier(random_state=14500)
model.fit(X_res, y_res)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
accuracy


# In[141]:


print(classification_report(y_test, y_pred))


# In[142]:


### Bagging Classifier
from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier(random_state=14500)
model.fit(X_res,y_res)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
accuracy


# In[143]:


print(classification_report(y_test, y_pred))


# In[144]:


### Gradient Boosting 
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_res,y_res)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
accuracy


# In[145]:


print(classification_report(y_test, y_pred))


# In[146]:


### Extra Tree
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(criterion='entropy', random_state=15000)
model.fit(X_res,y_res)
y_pred = model.predict(X_test)
accuracy = model.score(X_test,y_test)
accuracy


# In[147]:


print(classification_report(y_test, y_pred))


# # Conclusion
#     Training all the models with a training data set, we predicted the data from test dataset getting close to real data income, we could conclude that XGBoost would be selected, with approximately 89% accuracy.
#     The Random Forest and Extra Tree model performed well too, and could also be chosed.
#     

# Reference
# Link dataset https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior

# In[ ]:




