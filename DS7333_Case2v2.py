#!/usr/bin/env python
# coding: utf-8

# # DS7333 Case Study 2
# ## Predicting Hospital Readmittance Utilizing Logistic Regression Models
#
# #### John Girard, Shijo Joseph, Douglas Yip

# Installing and setting up flake8

# if you do not have flake8 installed
# then uncomment the pip line below and run it.

# pip install flake8 pycodestyle_magic

# #### Objective

# Your case study is to build a classifier using logistic regression to
# predict hospital readmittance. There is missing data that must be imputed.
# Once again, discuss variable importances as part of your submission..

# In[3]:

# Importing Libraries that will be used to
# ingest data and complete our regression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ## 1) Import data
# This process will ingest the data into dataframe

# In[4]:


# Import data
path = 'https://raw.githubusercontent.com/dk28yip/MSDS7333/main/'
diabetic_f = path + 'diabetic_data.csv'
df = pd.read_csv(diabetic_f)


# In[5]:


# Examine the shape of the data
df.shape


# Data contains 101,766 lines of data with 50 sets of variables.

# In[6]:


# check initial import dataframe
df.info()
df.head()


# In[7]:


# ids as intgers are changed to string since the are identifyers
# Convert variables from numeric into strings
cols_convert = ['encounter_id', 'patient_nbr',
                'admission_type_id', 'discharge_disposition_id',
                'admission_source_id']
df[cols_convert] = df[cols_convert].astype('str')


# In[8]:


# Check the summary of statistics
df.describe().T


# ## 2) Clean data
# This process will clean the data before proceeding to EDA

# Reviewing the excel document there are columns that have "?"
# where python will read these as objects that are not null.
# We will need to remove the ?

# In[9]:


# will replace ? with NAN
imput_df = df.replace('?', np.nan)


# In[10]:


# look at the isna after changing ? to NAN
imput_df.isna().sum()


# In[11]:


# create missing dataframe to build bar graph
missing = imput_df.isna().sum()
missing = pd.DataFrame(missing,
                       columns=['missing'])
missing = missing[(missing.T != 0).any()]

missing = missing.sort_values('missing',
                              ascending=False)
missing

# create bar graph from missing data
missing.plot(kind='bar',
             legend=False, color='green')

# displaying the title
plt.title("NAN by Category",
          fontsize=18)

# diplay labels
plt.ylabel('NAN Count', fontsize=14)
plt.xlabel('Categories', fontsize=14)
for index, data in enumerate(missing['missing']):
    plt.text(x=index, y=data+2,
             s=f"{data}",
             fontdict=dict(fontsize=12),
             ha='center')
plt.figure(figsize=(10, 25))
plt.show


# ## 3) Imputtation of data
# In this section we wil evaluate what to do with
# the the missing data for these columns;
# - __a.__ Weight
# - __b.__ Payer_code/medical_speciallty
# - __d.__ race
# - __e.__ diag_1, 2 and 3

# #### 3a) Weight Imputtation
# We impute weight data by using age as an indicator.
# Given we have some domain knowledge and can research
# valid weight values within specific age ranges, we used
# that knowledge to substitute missing values with the
# weight range that has the highest frequency within that range.

# In[12]:


imput_df['weight'].unique()


# In[13]:


imput_df1 = imput_df.loc[df['weight'] == 'nan', :]


# In[14]:


imput_df1.loc[imput_df['admission_source_id'] == 11].shape[0]


# In[15]:


imput_df.loc[(imput_df['age'] == '[90-100)') &
             (imput_df['weight'] != 'nan'),
             'weight'].value_counts()


# In[16]:


imput_df.loc[(imput_df['age'] == '[10-20)') &
             (imput_df['weight'].isna()),
             'weight'] = '[50-75)'

imput_df.loc[(imput_df['age'] == '[0-10)') &
             (imput_df['weight'].isna()),
             'weight'] = '[0-25)'

imput_df.loc[(imput_df['age'] == '[20-30)') &
             (imput_df['weight'].isna()),
             'weight'] = '[50-75)'

imput_df.loc[(imput_df['age'] == '[30-40)') &
             (imput_df['weight'].isna()),
             'weight'] = '[75-100)'

imput_df.loc[(imput_df['age'] == '[40-50)') &
             (imput_df['weight'].isna()),
             'weight'] = '[75-100)'

imput_df.loc[(imput_df['age'] == '[50-60)') &
             (imput_df['weight'].isna()),
             'weight'] = '[75-100)'

imput_df.loc[(imput_df['age'] == '[60-70)') &
             (imput_df['weight'].isna()),
             'weight'] = '[75-100)'

imput_df.loc[(imput_df['age'] == '[70-80)') &
             (imput_df['weight'].isna()),
             'weight'] = '[75-100)'

imput_df.loc[(imput_df['age'] == '[80-90)') &
             (imput_df['weight'].isna()),
             'weight'] = '[50-75)'

imput_df.loc[(imput_df['age'] == '[90-100)') &
             (imput_df['weight'].isna()),
             'weight'] = '[50-75)'


# In[17]:


# check if NAN is still there
imput_df['weight'].unique()


# #### 3b) Payer_code/medical_speciallty
# Looking at Payer_code and medical_specialiaty
# we will look at how much data is missing.

# In[18]:


imput_df.isnull().sum()/len(df)*100


# Given that we have no domanin knowledge and
# that >40% of the data for the two columns are
# missing, we will remove the columns for our model.

# In[19]:


# removing data columns where more than 30% of the data is missing.
imput_df = imput_df.drop(['weight', 'payer_code',
                          'medical_specialty'], axis=1)


# #### 3c) Race Imputtation
# We impute weight data by using age as an indicator.
# We examine valid weight values within specific age ranges and
# substitute missing values with the weight range that has the
# highest frequency within that range.

# In[20]:


# We will look at is the race column.
race_graph = imput_df['race'].value_counts().plot(
    kind='bar', figsize=(14, 8), color='green')
# displaying the title
plt.title("Race Count", fontsize=18)
# diplay labels
plt.ylabel('Count', fontsize=14)
plt.xlabel('Race', fontsize=14)

# We will use the mode for the race column
imput_df['race'] = imput_df['race'].fillna(
    imput_df['race'].mode()[0])

# It stands to reason that older patients, over weight patients,
# and patients with chronic illness that spend more time in the
# hospital will have a more likely chance to be readmitted to the hospital.

# #### 3c) Diag 1,2 and 3 Imputtation
# Given that we have <1.5% of examples without data, given its
# immateraility, we removed the rows that did not have values

# In[21]:

imput_df = imput_df.loc[~imput_df.diag_3.isna()]

# In[22]:

imput_df = imput_df.loc[~imput_df.diag_2.isna()]

# In[23]:

imput_df = imput_df.loc[~imput_df.diag_1.isna()]

# ## 3) EDA - Look at ID mapping

# One area that we observed was the discharged IDs.
# Give that 4 ids are related to death, the likelyhood of
# readmission is 0 and therefore we remove the row.
# ###### Discharge IDs
# - __11)__ Expired
# - __20)__ Expired at home. Medicaid only, hospice.
# - __21)__ Expired in a medical facility. Medicaid only, hospice.
# - __22)__ Expired, place unknown. Medicaid only, hospice.

# In[24]:

# Looking at the IDs_mapping.csv we can see that 11 -
# Expired ,19,20,21 are related to death or hospice.
# These samples are removed from the predictive model.
imput_df = imput_df.loc[
    ~imput_df.discharge_disposition_id
    .isin(['11', '19', '20', '21'])
]

# In[25]:

# remove unique identifiers
imput_df = imput_df.drop(['encounter_id',
                          'patient_nbr'], axis=1)

# In[26]:

imput_df.shape

# The number of records that were removed was 1,652 rows
# and 3 columns from the original data set.

# ## 4) Model LDA
# This process will clean the data before proceeding to EDA

# In[27]:

# create the target dataframe
y = imput_df['readmitted']

# Change readmittance to numeric - ordinal
# and remove encounter id and patient number
y = y.replace(to_replace='NO', value=0)
y = y.replace(to_replace='<30', value=1)
y = y.replace(to_replace='>30', value=2)

y

# In[28]:

# create df for hotcode less the y varaible
imput_df_lessY = imput_df.drop('readmitted', axis=1)

# In[29]:

# One Hot Encode Categorical Variables
# https://datagy.io/sklearn-one-hot-encode/
# https://datascience.stackexchange.com/questions/71804/
# how-to-perform-one-hot-encoding-on-multiple-categorical-columns
# list the caegorical columns less the Y variable
categorical_cols = imput_df_lessY.select_dtypes(
    exclude=np.number).columns.tolist()
categorical_cols


# In[30]:


imput_df_lessY


# In[31]:

scaler = StandardScaler()
# Get the Numeric columns for the X
X_Numeric = imput_df_lessY.loc[
    :, imput_df_lessY.columns.difference(categorical_cols)]

# Scale the Numerics
X_Numeric_scaled = pd.DataFrame(scaler.fit_transform(X_Numeric),
                                columns=X_Numeric.columns)
X_Numeric

# In[32]:

X_Numeric_scaled

# In[33]:

X_Numeric_scaled.describe()

# In[34]:

X_cat = imput_df_lessY[categorical_cols]
X_cat = X_cat.reset_index()
X_cat = X_cat[categorical_cols]
X_cat

# In[35]:

result = pd.concat([X_Numeric_scaled,
                    X_cat],
                   axis=1)
result

# In[36]:

# create one hot code df and create a df_final to be used for the model
X = pd.get_dummies(result, columns=categorical_cols)
X


# In[37]:

X.describe()

# set the target y .. Then find the priors, % of being readmitted
# less than 30 days and greater than 30 days (grouping together)
# and no being opposite
# remove the target from big dataset.

# use cross validation technique with the priors in mind.
# Use logistic regression.

# Fit Model
lr_model = LogisticRegression(multi_class='multinomial',
                              random_state=1,
                              n_jobs=-1)
lr_model.fit(X, y)

# check accuracy with same data
lr_model.score(X, y)

# In[40]:

# find tune model and find the best regularization C for our model

splits = KFold(n_splits=5, shuffle=True)

best = -1000
alpha = np.logspace(0.1, 1, 10)
for i in alpha:
    lr_model.C = i
    out = cross_val_score(lr_model, X, y,
                          scoring='neg_mean_squared_error',
                          cv=splits, n_jobs=-1).mean()
    if out > best:
        best = out
        best_alpha = i
print('Best Alpha:', best_alpha)
print('Best MSE:', best)

best_alpha

best

# In[43]:

# split data for training

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2,
                     shuffle=True, stratify=y, random_state=1)


# In[51]:


# Create a model with the best hyper parameter for C and look at the accruacy
best_lr = LogisticRegression(C=3.1622776601683795,
                             multi_class='multinomial',
                             random_state=1, n_jobs=-1)
pred_ncv = best_lr.fit(X_train, y_train)
pred_ncv.score(X_test, y_test)


# create prediction modelscores
pred = best_lr.predict(X_test)
pred

# ### 4) Results from model
# Confusion matrix and fscore to evaluate model accruacy

# In[53]:

print(classification_report(y_test, pred,
                            target_names=['No_Readmission',
                                          'Readmission <30days',
                                          'Readmission +30days']))


# In[54]:

cm0 = confusion_matrix(y_test, pred)
x_axis_labels = ['NO', '<30', '>30']
y_axis_labels = ['NO', '<30', '>30']
ax = plt.axes()
sns.heatmap(cm0, cmap='Blues', annot=True, fmt='d',
            xticklabels=x_axis_labels,
            yticklabels=y_axis_labels,
            ax=ax,
            cbar_kws={'label': 'Readmission Count', })
ax.set_title('Hospital Readmission Confusion Matrix Heatmap')
plt.xlabel("Predicted readmission")
plt.ylabel("True readmission")
plt.show

# In[55]:


diab_LSR = pd.concat((X, y), axis=1)
diab_LSR = pd.DataFrame(diab_LSR)
diab_LSR.info(1)


# In[56]:


# Feature Importance
feature_names = diab_LSR.drop('readmitted', axis=1).copy().columns.values
feature_importances = pd.DataFrame(
    pred_ncv.coef_[0],
    index=feature_names,
    columns=['importance']).sort_values('importance',
                                        ascending=False)


# In[57]:


# Feature Importance plot
num = 10
ylocs = np.arange(num)
# get the feature importance for top num and sort in reverse order
values_to_plot = feature_importances.iloc[:num].values.ravel()[::-1]
feature_labels = list(feature_importances.iloc[:num].index)[::-1]

plt.figure(num=None, figsize=(8, 15),
           dpi=80, facecolor='w', edgecolor='k')
plt.barh(ylocs, values_to_plot, align='center')
plt.ylabel('Features', fontsize=20)
plt.xlabel('Importance Score', fontsize=20)
plt.title(['Feature Importance Score - Readmission Classification'
           ', Logistic Regression'],
          fontsize=20)
plt.yticks(ylocs,
           feature_labels,
           fontsize=11)


plt.show()
