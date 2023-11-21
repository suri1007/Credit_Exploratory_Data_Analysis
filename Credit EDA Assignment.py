#!/usr/bin/env python
# coding: utf-8

# # Credit EDA Assignment

# # 1. Importing the Necessary Libraries

# In[125]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from plotly.subplots import make_subplots
import plotly.graph_objects as go
warnings.filterwarnings(action='ignore')





# In[126]:


AD = pd.read_csv(r"C:\D DRIVE\EPGP-DS\Project data\EDA\Assignment\application_data.csv")
PD = pd.read_csv(r"C:\D DRIVE\EPGP-DS\Project data\EDA\Assignment\previous_application.csv")


# # 2. Check the structure of data

# 2.1 Examining Application Data

# In[127]:


AD.head()


# In[128]:


AD.shape


# In[129]:


AD.info()


# In[130]:


AD.describe()


# 2.2 Examining Previous Application Data

# In[131]:


PD.head()


# In[132]:


PD.shape


# In[133]:


PD.info()


# In[134]:


AD.dtypes.value_counts()


# # 3.1 Data Qaulity Check and Missing Values

# 3.1.1 Checking missing values in Application Data

# In[135]:


(AD.isnull().mean()*100).sort_values(ascending=False)


# # 3.2 Dropping columns where missing values

# Note : Missing Values Consideration:
# Drop of the values of higher percentage -using certain criteria
#                  Columns Above 100 -->40%
#                  Columns in Range 50-100--> - 25%
#                  Columns in Range <50--> - 10%
# 

# 

# In[136]:


s1= (AD.isnull().mean()*100).sort_values(ascending=False)[AD.isnull().mean()*100 > 40]
s1


# In[137]:


cols = (AD.isnull().mean()*100 > 40)[AD.isnull().mean()*100 >40].index.tolist()

cols


# In[138]:


len(cols)


# We are good to delete 49 columns because Null percentage for the columns is greater than 40%

# In[139]:


#Dropping 49 columns

AD.drop(columns=cols,inplace=True)


# In[140]:


AD.shape


# Null Value Percentage in new Data set 

# In[141]:


S2 = (AD.isnull().mean()*100).sort_values(ascending=False)
S2


# In[142]:


S2.head(10)


# # 3.3 Imputation of Missing Values 

# Imputation in Categorical variables

# In[143]:


AD.head()


# Imputation in numerical Variables

# Impute the Missing Values of below columns with mode
# 
# - AMT_REQ_CREDIT_BUREAU_MON
# -AMT_REQ_CREDIT_BUREAU_WEEK
# -AMT_REQ_CREDIT_BUREAU_DAY
# -AMT_REQ_CREDIT_BUREAU_HOUR
# -AMT_REQ_CREDIT_BUREAU_QRT
# 

# In[144]:


for i in S2.head(10).index.to_list():
    if 'AMT_REQ_CREDIT' in i:
        print ('Most frequent value in {0} is : {1}' .format(i,AD[i].mode()[0]))
        
        print ('Imputing the missing value with :{0}' .format(i,AD[i].mode()[0]))
        
        print ('Null Values in {0} after imputation :{1}' .format(i,AD[i].mode()[0]))


# Missing Value in percentage of missing columns

# In[145]:


(AD.isnull().mean()*100).sort_values(ascending = False)


# Impute Missing Value for occuption_type

# In[146]:


#We can impute missing values in OCCUPTION_TYPE column with 'Laborers'

fig = px.bar(AD.OCCUPATION_TYPE.value_counts(),color=AD.OCCUPATION_TYPE.value_counts())
fig.update_traces(textposition ='outside', marker_coloraxis=None)
fig.update_xaxes(title='Occuption Type')
fig.update_yaxes(title='Count')
fig.update_layout(title=dict(text="Occuption Type Frequency" , x=0.5 , y=.95),
                 title_font_size=20,
                 showlegend=False,
                 height=450)

fig.show()


# Impute Missing Values (XNA) in CODE_GENDER with mode

# In[147]:


AD['CODE_GENDER'].value_counts()


# In[148]:


AD['CODE_GENDER'].replace(to_replace='XNA',value=AD ['CODE_GENDER'].mode()[0],inplace=True)


# In[149]:


AD['CODE_GENDER'].value_counts()


# Impute Missing Values for EXT_SOURCE_3
# 

# In[150]:


AD.EXT_SOURCE_3.dtype


# In[151]:


AD.EXT_SOURCE_3.fillna(AD.EXT_SOURCE_3.median(),inplace=True)


# Percentage of missing values after Imputation

# In[152]:


(AD.isnull().mean()*100).sort_values(ascending=False)


# Repalce 'XNA with NaN

# In[153]:


AD = AD.replace('XNA' , np.NaN)


# # Delete All flag columns

# In[154]:


AD.columns


# Flag Columns

# In[155]:


col = []
for i in AD.columns:
    if 'FLAG' in i:
        col.append(i)
        
col


# Delete all flag columns as they won't be much useful in our analysis

# In[156]:


AD= AD[[i for i in AD.columns if 'FLAG' not in i]]
AD.head()


# # Impute Missing values for AMT_ANNUITY & AMT_GOODS_PRICE

# In[157]:


col = ['AMT_INCOME_TOTAL' , 'AMT_CREDIT' , 'AMT_ANNUITY' , 'AMT_GOODS_PRICE']
for i in col:
    print('Null Values in {0} : {1}' . format (i,AD[i].isnull().sum()))


# In[158]:


AD['AMT_ANNUITY'].fillna(AD['AMT_ANNUITY'].median(),inplace=True)
AD['AMT_GOODS_PRICE'].fillna(AD['AMT_GOODS_PRICE'].median(),inplace=True)
AD['AMT_ANNUITY'].isnull().sum()
AD['AMT_GOODS_PRICE'].isnull().sum()


# # Correcting Data

# In[159]:


days= []
for i in AD.columns:
    if 'DAYS' in i:
        days.append(i)
        print('Unique values in {0} column : {1}' .format (i, AD[i].unique()))
        
        print('NULL Values in {0} column : {1}' .format (i, AD[i].isnull().sum()))
        
        print()
                                


# In[160]:


AD[days]


# Use Absolute Values in DAYS columns
# 
# 

# In[161]:


AD[days] = abs(AD[days])
AD[days]


# # Binning 

# Lets do binning of these variables

# In[162]:


for i in col:
    AD[i+'_Range'] = pd.qcut(AD[i], q = 5, labels = ['Very Low' , 'Low' , 'Medium' , 'High' , 'Very High'])
    print (AD[i+'_Range'].value_counts())
    print()


# In[163]:


AD['YEARS_EMPLOYED'] = AD['DAYS_EMPLOYED']/365
AD['Client_Age'] = AD['DAYS_BIRTH']/365


# Drop 'DAYS_EMPLOYED' & 'DAYS_BIRTH' column as we will be performing analysis on Year basis

# In[164]:


AD.drop(columns= ['DAYS_EMPLOYED', 'DAYS_BIRTH' ],inplace=True)


# In[165]:


AD['Age Group'] = pd.cut (x = AD['Client_Age'], bins= [0,20,30,40,50,60,100], labels = ['0-20' , '20-30', '30-40' ,'40-50' , '50-60' , '60-100'])


# In[166]:


AD[['SK_ID_CURR' , 'Client_Age' , 'Age Group']]


# In[167]:


AD['Work Experience'] = pd.cut(x = AD['YEARS_EMPLOYED'], bins = [0,5,10,15,20,25,30,100], labels = ['0-5' , '5-10' , ' 10-15' , '15-20' , '20-25' , '25-30' , '30-100'])


# In[168]:


AD[['SK_ID_CURR' , 'YEARS_EMPLOYED' , 'Work Experience']]


# # OUTLIER DETECTION

# Analyzing AMT columns for outliers

# In[169]:


cols = ['AMT_INCOME_TOTAL' , 'AMT_CREDIT' , 'AMT_ANNUITY' , 'AMT_GOODS_PRICE']

fig,axes = plt.subplots(ncols = 2 , nrows = 2 , figsize = (15,15))
count = 0 
for i in range (0,2):
    for j in range (0,2):
        sns.boxenplot(y = AD[cols[count]] , ax = axes [i,j])
        count+=1
        
plt.show()


# Below Columns have Outliers and those values can be dropped:-
# 
# -- AMT_INCOME_TOTAL
# -AMT_ANNUITY

# REMOVE OUTLIERS FOR 'AMT_COLUMNS_TOTAL' COLUMN

# In[170]:


AD = AD[AD['AMT_INCOME_TOTAL'] < AD['AMT_INCOME_TOTAL'].max()]


# REMOVE OUTLIERS FOR 'AMT_ANNUITY' COLUMN

# In[171]:


AD = AD[AD['AMT_INCOME_TOTAL'] < AD['AMT_INCOME_TOTAL'].max()]


# Analysing CNT_CHILDREN column for Outliers

# In[172]:


fig = px.box(AD['CNT_CHILDREN'])
fig.update_layout( title = dict (text = "Number of children" , x = 0.5 , y = 0.95), title_font_size = 20, showlegend = False, width = 400, height = 400,)

fig.show()


# In[173]:


AD['CNT_CHILDREN'].value_counts()


# In[174]:


AD.shape[0]


# Remove all data points where CNT_CHILDREN is greater than 10 

# In[175]:


AD = AD [AD['CNT_CHILDREN'] <= 10]
AD.shape[0]


# Around Eight values are dropped where number of children are greater than 10 

# # Analysing YEARS_EMPLOYED columns for outliers

# In[176]:


sns.boxplot(y = AD['YEARS_EMPLOYED'])
plt.show()


# In[177]:


AD['YEARS_EMPLOYED'].value_counts()


# In[178]:


AD.shape[0]


# In[179]:


AD['YEARS_EMPLOYED'][AD['YEARS_EMPLOYED']>1000]=np.NaN


# In[180]:


sns.boxplot(y = AD['YEARS_EMPLOYED'])
plt.show()


# In[181]:


AD.isnull().sum().sort_values(ascending= False).head(10)


# # Analysing AMT_REQ_CREDIT columns for outliers

# In[182]:


cols = [i for i in AD.columns if 'AMT_REQ' in i ]
cols


# In[183]:


fig,axes = plt.subplots(ncols = 3 , nrows = 2 , figsize = (15,15))
count = 0 
for i in range (0,2):
    for j in range (0,3):
        sns.boxenplot(y = AD[cols[count]] , ax = axes [i,j])
        count+=1
        
plt.show()


# AMT_REQ_CREDIT_BUREAU_QRT contains an outlier

# Remove Outlier for AMT_REQ_CREDIT_BUREAU_QRT

# In[184]:


AD = AD[AD['AMT_REQ_CREDIT_BUREAU_QRT'] <AD['AMT_REQ_CREDIT_BUREAU_QRT'].max()]


# # UNIVARIATE ANALYSIS

# In[185]:


AD.columns


# In[186]:


fig1 = px.bar(AD['OCCUPATION_TYPE'].value_counts() ,color = AD['OCCUPATION_TYPE'].value_counts())
fig1.update_traces(textposition ='outside', marker_coloraxis=None)
fig1.update_xaxes(title='Occuption Type')
fig1.update_yaxes(title='Count')
fig1.update_layout(title=dict(text="Occuption Typ" , x=0.5 , y=.95),
                 title_font_size=20,
                 showlegend=False,
                 height=450)

fig1.show()


# In[187]:


fig2 = px.bar(AD['ORGANIZATION_TYPE'].value_counts() ,color = AD['ORGANIZATION_TYPE'].value_counts())
fig2.update_traces(textposition ='outside', marker_coloraxis=None)
fig2.update_xaxes(title='ORGANIZATION TYPE')
fig2.update_yaxes(title='Count')
fig2.update_layout(title=dict(text="Organization Type Frequency" , x=0.5 , y=.95),
                 title_font_size=20,
                 showlegend=False,
                 height=450)

fig2.show()


# Insights 
# 
# - Most People who applied for Loan Appliaction are Laborers
# - Most People who applied for Loan Application belong to either Business Entity Type3 or Self - Employed Organization Type.

# In[188]:


cols = ['Age Group', 'NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'CODE_GENDER', 'Work Experience']

fig = make_subplots(rows=4, cols=2, subplot_titles=cols, horizontal_spacing=0.1, vertical_spacing=0.13)

count = 0
for i in range(4):
    for j in range(2):
        fig.add_trace(go.Bar(
            x=AD[cols[count]].value_counts().index,
            y=AD[cols[count]].value_counts(),
            name=cols[count],
            textposition='auto',
            text=[str(i) + '%' for i in (AD[cols[count]].value_counts(normalize=True) * 100).round(1).tolist()],
        ), row=i + 1, col=j + 1)
        count += 1

fig.update_layout(
    title="Analyze Categorical variables (Frequency / Percentage)",
    title_font_size=20,
    showlegend=False,
    width=960,
    height=1600,
)
fig.show()


# Insights 
# 
# Banks has recieved majority of the loan application from 30-40 & 40-50Age groups.
# 
# More than 50% of clients who have applied for the loan belong to Working Income Type.
# 
# 88.7% clients with Secondary /Secondary Special education type have applied for the loan.
# 
# Married people tend to apply more for loan i.e 63.9% clients who are have applied for loan are married.
# 
# Female loan Application are more as compare to males. this may be because bank charges less rate of interest of females.
# 
# Majority of the clients who have applied for the loan have their own house/apartment. Around 88.7% clients are owning either a house or an apartment.
# 
# client with work expiernce between 0-5 years have applied most for loan application.
# 
# 90.5% application have requested for a Cash Loans
# 

# In[189]:


AD.nunique () .sort_values()


# # Checking Imbalance

# In[190]:


AD['TARGET'].value_counts(normalize=True)


# In[191]:


fig = px.pie(values = AD['TARGET'].value_counts(normalize=True),
            names = AD['TARGET'].value_counts(normalize=True).index,
            hole =0.5)
fig.update_layout( title=dict(text = "Target Imbalance" , x =0.5, y = 0.95),
                 title_font_size = 20,
                 showlegend=False
                 )

fig.show()


# In[192]:


app_target0 = AD.loc[AD.TARGET == 0]
app_target1 = AD.loc[AD.TARGET == 1]


# In[193]:


app_target0.shape


# In[194]:


app_target1.shape


# In[195]:


cols = ['Age Group' , 'NAME_CONTRACT_TYPE' , 'NAME_INCOME_TYPE' , 'NAME_EDUCATION_TYPE']

title  = [None]*(2*len(cols))
title[::2]=[i+ '(Non - Payment Difficulties)' for i in cols]
title[1::2]=[i+ '(Payment Difficulties)' for i in cols]

fig = make_subplots( rows = 4,
                   cols = 2,
                   subplot_titles=title,
                   )

count = 0 
for i in range (1,5):
    for j in range (1,3):
        if j ==1:
             fig.add_trace(go.Bar(x=app_target0[cols[count]].value_counts().index, 
                             y=app_target0[cols[count]].value_counts(),
                             name=cols[count],
                             textposition='auto',
                             text= [str(i) + '%' for i in (app_target0[cols[count]].value_counts(normalize=True)*100).round(1).tolist()],
                            ),
                      row=i,col=j)
        else:
            fig.add_trace(go.Bar(x=app_target1[cols[count]].value_counts().index, 
                             y=app_target1[cols[count]].value_counts(),
                             name=cols[count],
                             textposition='auto',
                             text= [str(i) + '%' for i in (app_target1[cols[count]].value_counts(normalize=True)*100).round(1).tolist()],
                            ),
                      row=i,col=j)
            count+=1 
fig.update_layout(
                    title=dict(text = "Analyze Categorical variables (Payment/ Non-Payment Difficulties)",x=0.5,y=0.99),
                    title_font_size=20,
                    showlegend=False,
                    height = 1600,
                  )
fig.show()


# In[196]:


cols = ['NAME_FAMILY_STATUS' , 'NAME_HOUSING_TYPE' , 'CODE_GENDER' , 'Work Experience']

title  = [None]*(2*len(cols))
title[::2]=[i+ '(Non - Payment Difficulties)' for i in cols]
title[1::2]=[i+ '(Payment Difficulties)' for i in cols]

fig = make_subplots( rows = 4,
                   cols = 2,
                   subplot_titles=title,
                   )

count = 0 
for i in range (1,5):
    for j in range (1,3):
        if j ==1:
             fig.add_trace(go.Bar(x=app_target0[cols[count]].value_counts().index, 
                             y=app_target0[cols[count]].value_counts(),
                             name=cols[count],
                             textposition='auto',
                             text= [str(i) + '%' for i in (app_target0[cols[count]].value_counts(normalize=True)*100).round(1).tolist()],
                            ),
                      row=i,col=j)
        else:
            fig.add_trace(go.Bar(x=app_target1[cols[count]].value_counts().index, 
                             y=app_target1[cols[count]].value_counts(),
                             name=cols[count],
                             textposition='auto',
                             text= [str(i) + '%' for i in (app_target1[cols[count]].value_counts(normalize=True)*100).round(1).tolist()],
                            ),
                      row=i,col=j)
            count+=1 
fig.update_layout(
                    title=dict(text = "Analyze Categorical variables (Payment/ Non-Payment Difficulties)",x=0.5,y=0.99),
                    title_font_size=20,
                    showlegend=False,
                    height = 1600,
                  )
fig.show()


# # Bivariate / Multivarite Analysis

# Group data by 'AMT_CREDIT_Range' & 'CODE_GENDER'

# In[197]:


df1 = AD.groupby(by=['AMT_CREDIT_Range' , 'CODE_GENDER']).count().reset_index()[['AMT_CREDIT_Range' , 'CODE_GENDER', 'SK_ID_CURR']]

df1


# Group data by 'AMT_INCOME_TOTAL_Range' & 'CODE_GENDER'

# In[198]:


fig1=px.bar(data_frame=df1,
       x='AMT_CREDIT_Range',
       y='SK_ID_CURR',color='CODE_GENDER',
       barmode='group',
       text='SK_ID_CURR'
      )
fig1.update_traces(textposition='outside')
fig1.update_xaxes(title='Day') 
fig1.update_yaxes(title='Transaction count')
fig1.update_layout(
                    title=dict(text = "Loan Applications by Gender & Credit Range",x=0.5,y=0.95),
                    title_font_size=20,
                  )
fig1.show()


# Insights 
# 
# Females are mostly applying for Very Low credit Loans.
# 
# Males are applying for Medium & High credit loans.
# 
# 

# # Income Vs Credit Amount (Payment / Non Payment Difficulties)

# In[199]:


fig = px.box (app_target0, x = "AMT_INCOME_TOTAL_Range", y = "AMT_CREDIT" , color = 'NAME_FAMILY_STATUS',
             title = "Income Range Vs Credit Amount (Non- Payment Difficulties)")
fig.show()


# In[200]:


fig = px.box (app_target1, x = "AMT_INCOME_TOTAL_Range", y = "AMT_CREDIT" , color = 'NAME_FAMILY_STATUS',
             title = "Income Range Vs Credit Amount (Payment Difficulties)")
fig.show()


# # Age Group VS Credit Amount (Payment / Non Payment Diffculties)

# In[201]:


fig = px.box (app_target0, x = "Age Group", y = "AMT_CREDIT" , color = 'NAME_FAMILY_STATUS',
             title = "Age Group Vs Credit Amount (Non-Payment Difficulties)")
fig.show()


# In[202]:


fig = px.box (app_target1, x = "Age Group", y = "AMT_CREDIT" , color = 'NAME_FAMILY_STATUS',
             title = "Age Group Vs Credit Amount (Payment Difficulties)")
fig.show()


# # Numerical Vs Numerical Variables

# In[203]:


sns.pairplot(AD[['AMT_INCOME_TOTAL' , 'AMT_GOODS_PRICE' , 'AMT_CREDIT' , 'AMT_ANNUITY' , 'Client_Age' , 'YEARS_EMPLOYED']].fillna(0))
plt.show()


# # Correlation in Target0 & Target1

# In[204]:


plt.figure(figsize=(12,8))
sns.heatmap(app_target0[['AMT_INCOME_TOTAL' , 'AMT_GOODS_PRICE', 'AMT_CREDIT' , 'AMT_ANNUITY' , 'Client_Age', 'YEARS_EMPLOYED', 'DAYS_ID_PUBLISH' ,'DAYS_REGISTRATION' , 'EXT_SOURCE_2' ,'EXT_SOURCE_3' , 'REGION_POPULATION_RELATIVE' ]].corr(),annot= True , cmap ="RdYlGn")

plt.title('Correlation matrix for Non-Payment Difficulties')
plt.show()


# In[205]:


plt.figure(figsize=(12,8))
sns.heatmap(app_target1[['AMT_INCOME_TOTAL' , 'AMT_GOODS_PRICE', 'AMT_CREDIT' , 'AMT_ANNUITY' , 'Client_Age', 'YEARS_EMPLOYED', 'DAYS_ID_PUBLISH' ,'DAYS_REGISTRATION' , 'EXT_SOURCE_2' ,'EXT_SOURCE_3' , 'REGION_POPULATION_RELATIVE' ]].corr(),annot= True , cmap ="RdYlGn")

plt.title('Correlation matrix for Payment Difficulties')
plt.show()


# # Data Analysis on Previous Appliaction Data Set

# In[206]:


PD.head()


# In[207]:


s1= (PD.isnull().mean()*100).sort_values(ascending=False)[PD.isnull().mean()*100 > 40]
s1


# In[208]:


PD.shape


# In[209]:


PD.drop(columns=s1.index,inplace=True)


# In[210]:


PD.shape


# # Changing Negative Values in the Days columns to positive Values

# In[211]:


days= []
for i in PD.columns:
    if 'DAYS' in i:
        days.append(i)
        print('Unique values in {0} column : {1}' .format (i, PD[i].unique()))
        
        print('NULL Values in {0} column : {1}' .format (i, PD[i].isnull().sum()))
        
        print()


# In[212]:


PD[days] =abs(PD[days])


# In[213]:


PD[days]


# In[214]:


PD = PD.replace ('XNA' ,np.NaN)
PD = PD.replace ('XAP' ,np.NaN)


# # Univariate Analysis on Previous App Data

# In[215]:


PD.columns


# In[216]:


cols = ['NAME_CONTRACT_STATUS','WEEKDAY_APPR_PROCESS_START', 
        'NAME_PAYMENT_TYPE','CODE_REJECT_REASON', 
        'NAME_CONTRACT_TYPE','NAME_CLIENT_TYPE']

#Subplot initialization
fig = make_subplots(
                     rows=3, 
                     cols=2,
                     subplot_titles=cols,
                     horizontal_spacing=0.1,
                     vertical_spacing=0.17 
                   )
# Adding subplots
count=0
for i in range(1,4):
    for j in range(1,3):
        fig.add_trace(go.Bar(x=PD[cols[count]].value_counts().index, 
                             y=PD[cols[count]].value_counts(),
                             name=cols[count],
                             textposition='auto',
                             text= [str(i) + '%' for i in (PD[cols[count]].value_counts(normalize=True)*100).round(1).tolist()],
                            ),
                      row=i,col=j)
        count+=1
fig.update_layout(
                    title=dict(text = "Analyze Categorical variables (Frequency / Percentage)",x=0.5,y=0.99),
                    title_font_size=20,
                    showlegend=False,
                    width = 960,
                    height = 1200,
                  )
fig.show()


# # Approved Loan 

# In[217]:


Approved = PD[PD['NAME_CONTRACT_STATUS']=='Approved']


# In[218]:


cols = ['NAME_PORTFOLIO','NAME_GOODS_CATEGORY', 
        'CHANNEL_TYPE','NAME_YIELD_GROUP', 
        'NAME_PRODUCT_TYPE','NAME_CASH_LOAN_PURPOSE']

#Subplot initialization
fig = make_subplots(
                     rows=3, 
                     cols=2,
                     subplot_titles=cols,
                     horizontal_spacing=0.1,
                     vertical_spacing=0.17 
                   )
# Adding subplots
count=0
for i in range(1,4):
    for j in range(1,3):
        fig.add_trace(go.Bar(x=PD[cols[count]].value_counts().index, 
                             y=PD[cols[count]].value_counts(),
                             name=cols[count],
                             textposition='auto',
                             text= [str(i) + '%' for i in (PD[cols[count]].value_counts(normalize=True)*100).round(1).tolist()],
                            ),
                      row=i,col=j)
        count+=1
fig.update_layout(
                    title=dict(text = "Analyze Categorical variables (Frequency / Percentage)",x=0.5,y=0.99),
                    title_font_size=20,
                    showlegend=False,
                    width = 960,
                    height = 1200,
                  )
fig.show()


# # Refused Loans

# In[219]:


Refused = PD[PD['NAME_CONTRACT_STATUS'] =='Refused']


# In[220]:


cols = ['NAME_PORTFOLIO','NAME_GOODS_CATEGORY', 
        'CHANNEL_TYPE','NAME_YIELD_GROUP', 
        'NAME_PRODUCT_TYPE','NAME_CASH_LOAN_PURPOSE']

#Subplot initialization
fig = make_subplots(
                     rows=3, 
                     cols=2,
                     subplot_titles=cols,
                     horizontal_spacing=0.1,
                     vertical_spacing=0.17 
                   )
# Adding subplots
count=0
for i in range(1,4):
    for j in range(1,3):
        fig.add_trace(go.Bar(x=PD[cols[count]].value_counts().index, 
                             y=PD[cols[count]].value_counts(),
                             name=cols[count],
                             textposition='auto',
                             text= [str(i) + '%' for i in (PD[cols[count]].value_counts(normalize=True)*100).round(1).tolist()],
                            ),
                      row=i,col=j)
        count+=1
fig.update_layout(
                    title=dict(text = "Analyze Categorical variables (Frequency / Percentage)",x=0.5,y=0.99),
                    title_font_size=20,
                    showlegend=False,
                    width = 960,
                    height = 1200,
                  )
fig.show()


# # Merging Application & Previous Application Data

# In[221]:


# Example of a custom function named mergereturn
def mergereturn(df1, df2, how='inner', on=None):
    # Your function implementation here
    pass


# In[ ]:





# In[222]:


merged_df = pd.merge(AD, PD, on='SK_ID_CURR', how='inner')
merged_df


# In[223]:


appdata_merge = AD.merge(PD,on='SK_ID_CURR', how='inner')
appdata_merge.shape


# In[224]:


merged_df = pd.merge(df1, df2, on='ID', how='inner')  # You can choose 'how' to specify the type of join (inner, outer, left, right)
print(merged_df)


# In[ ]:


df_combine = AD.merge(PD, left_on = 'SK_ID_CURR', right_on= 'SK_ID_CURR', how = 'left' )


# In[ ]:


df_combine


# In[ ]:


df_combine.shape


# In[ ]:


def plot_merge(df_combine, column_name):
    col_value = ['Refused' , 'Approved' , 'Canceled' , 'Unused offer']
    
    fig = make_subplots(
                     rows=2, 
                     cols=2,
                     subplot_titles=col_value,
                     horizontal_spacing=0.1,
                     vertical_spacing=0.3
                   )
    # Adding subplots
    count=0
    for i in range(1,3):
        for j in range(1,3):
            fig.add_trace(go.Bar(x=df_combine[df_combine['NAME_CONTRACT_STATUS']==col_value[count]][column_name].value_counts().index, 
                             y=df_combine[df_combine['NAME_CONTRACT_STATUS']==col_value[count]][column_name].value_counts(),
                             name=cols[count],
                             textposition='auto',
                             text= [str(i) + '%' for i in (df_combine[df_combine['NAME_CONTRACT_STATUS']==col_value[count]][column_name].value_counts(normalize=True)*100).round(1).tolist()],
                            ),
                      row=i,col=j)
            count+=1
    fig.update_layout(
                    title=dict(text = "NAME_CONTRACT_STATUS VS "+column_name,x=0.5,y=0.99),
                    title_font_size=20,
                    showlegend=False,
                    width = 960,
                    height = 960,
                  )
    fig.show()


# In[ ]:


def plot_pie_merge(df_combine,column_name):
    col_value = ['Refused','Approved', 'Canceled' , 'Unused offer']
    
    #Subplot initialization
    fig = make_subplots(
                     rows=2, 
                     cols=2,
                     subplot_titles=col_value,
                     specs=[[{"type": "pie"}, {"type": "pie"}],[{"type": "pie"}, {"type": "pie"}]],
                   )
    # Adding subplots
    count=0
    for i in range(1,3):
        for j in range(1,3):
            fig.add_trace(go.Pie(labels=df_combine[df_combine['NAME_CONTRACT_STATUS']==col_value[count]][column_name].value_counts().index, 
                             values=df_combine[df_combine['NAME_CONTRACT_STATUS']==col_value[count]][column_name].value_counts(),
                             textinfo='percent',
                             insidetextorientation='auto',
                             hole=.3
                            ),
                      row=i,col=j)
            count+=1
    fig.update_layout(
                    title=dict(text = "NAME_CONTRACT_STATUS VS "+column_name,x=0.5,y=0.99),
                    title_font_size=20,
                    width = 960,
                    height = 960,
                  )
    fig.show()


# In[ ]:


plot_pie_merge(df_combine,'NAME_CONTRACT_TYPE_y')


# Insights 
# 
# - Banks Mostly approve Consumer Loans
# - Most of the Refused_& Cancelled loans are cash loans.

# In[ ]:


plot_pie_merge(df_combine,'NAME_CLIENT_TYPE')


# Insights 
# 
# - Most of the Approved, refused & cancelled loans belong to the old clients.
# 
# - Almost 27.4% loans were provided to new customers.

# In[ ]:


plot_pie_merge(df_combine,'CODE_GENDER')


# Insights 
# 
# - Approved percentage of loans provided to females is more as compared to refused percentage.

# In[ ]:


plot_merge(df_combine,'NAME_EDUCATION_TYPE')


# Insights 
# 
# - Most of the approved loans belong to applicants with Secondary / Secondary Special education type.

# In[ ]:


plot_merge(df_combine,'OCCUPATION_TYPE')


# In[ ]:


plot_merge(df_combine,'NAME_GOODS_CATEGORY')


# In[ ]:


plot_merge(df_combine,'PRODUCT_COMBINATION')


# Insights
# 
# - Most of the approved loans belongs to POS household with  interest & POS mobile with interest product combination.
# 
# - 15% refused loans belong to Cash X - Sell:low product combination.
#     
# - Most of the cancelled loan belong to cash category.
# 
# 81.3% Unused offer loans belong to POS mobile with interest.

# In[ ]:


plot_merge(df_combine,'AMT_INCOME_TOTAL_Range')


# Final Insight of this Analysis is following:
#     
#     - Most of the loans are getting approved for Applicants with Low income range.
#       May be they are opting for low credit loans.
#         
#     - Almost 28% loan applications are either getting rejected or cancelled even though applicant belong to HIGH Income range. May be they have requested for quite HIGH credit range.

# # Here is the End of this Analysis

# In[ ]:





# In[ ]:





# In[ ]:




