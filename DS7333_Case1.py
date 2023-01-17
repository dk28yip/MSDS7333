#!/usr/bin/env python
# coding: utf-8

# # DS7333 Case Study 1
# ## Predicting the Critical Temperature of Various Superconductors Utilizing Regression Models
# 
# #### John Girard, Shijo Joseph, Douglas Yip

# Installing and setting up flake8 

# In[1]:


# if you do not have flake8 installed
# then uncomment the pip line below and run it.

# pip install flake8 pycodestyle_magic


# In[2]:





# #### Objective
# 
# We have been tasked with creating a linear regression model to predict critical temperature of various super conductors.  We will also examine which variables are most important in our model.

# In[3]:


# Importing Libraries that will be used to
# ingest data and complete our regression
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model \
    import Lasso, LinearRegression, Ridge, RidgeCV, LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection \
    import train_test_split, cross_val_score
from statsmodels.stats.outliers_influence \
    import variance_inflation_factor
from statistics import mean





# ## 1) Import and Check data
# This process will ingest the data into one dataframe and will validate integrity of data before proceeding to EDA

# In[4]:


# Import data
path = 'https://raw.githubusercontent.com/dk28yip/MSDS7333/main/'
unique_f = path + 'unique_m.csv'
train_f = path + 'train.csv'
df1 = pd.read_csv(unique_f)
df2 = pd.read_csv(train_f)

# drop critical temp since column exist in both sets
df1 = df1.drop(['critical_temp'], axis=1)

# Merge the two data frames
df = pd.concat([df1, df2], axis=1)


# In[5]:


# check dataframe
df.info()
df.head()


# In[6]:


# Examine the shape of the data
df.shape


# #### Dropped material column due to redudancy of data given that the invididual elements are showing in the data.  

# In[7]:


df = df.drop(['material'], axis=1)


# In[8]:


# Any missing values in the dataset
def plot_missingness(df: pd.DataFrame = df) -> None:
    nan_df = pd.DataFrame(df.isna().sum()).reset_index()
    nan_df.columns = ['Column', 'NaN_Count']
    nan_df['NaN_Count'] = nan_df['NaN_Count'].astype('int')
    nan_df['NaN_%'] = round(nan_df['NaN_Count']/df.shape[0] * 100, 1)
    nan_df['Type'] = 'Missingness'
    nan_df.sort_values('NaN_%', inplace=True)

    # Add completeness
    for i in range(nan_df.shape[0]):
        complete_df = pd.DataFrame([nan_df.loc[i, 'Column'],
                                    df.shape[0]
                                    - nan_df.loc[i, 'NaN_Count'], 100
                                    - nan_df.loc[i, 'NaN_%'],
                                    'Completeness']).T
        complete_df.columns = ['Column', 'NaN_Count', 'NaN_%', 'Type']
        complete_df['NaN_%'] = complete_df['NaN_%'].astype('int')
        complete_df['NaN_Count'] = complete_df['NaN_Count'].astype('int')
        nan_df = pd.concat([nan_df, complete_df], sort=True)
    nan_df = nan_df.rename(columns={"Column": "Feature", "NaN_%": "Missing %"})

    # Missingness Plot
    fig = px.bar(nan_df, x='Feature', y='Missing %',
                 title=f"Missingness Plot (N={df.shape[0]})",
                 color='Type', opacity=0.6,
                 color_discrete_sequence=['red', '#808080'],
                 width=800, height=400)
    fig.show()


# In[9]:


plot_missingness(df)


# #### Based on our missing NA check, there are no NAs in the data.

# In[10]:


# Print the column names
for col_names in df.columns:
    print(col_names)


# In[11]:


# Examine the column names and data types
pd.set_option('display.max_rows', None)
df.describe().T


# ## 2) Scale data
# Given that the box plot shows a wide range of variance between columns we will scale the data

# In[12]:


# Create the response and dependent variable sets
X = df.drop(labels=['critical_temp'], axis=1)
y = df.critical_temp


scaler = StandardScaler()  # Create the standard scaler object

# Fit the data and transform it, leaving an array
X_scaled = scaler.fit_transform(X)

# Recreate the dataframe
X_scaled = pd.DataFrame(data=X_scaled, columns=X.columns)


# In[13]:


#Re-examine the plots again to see the data normalized. 
for i in X_scaled.columns:
    sns.displot(X_scaled, x = X_scaled[i])
    plt.show()


# #### Evaluate two ways of collinearity with VIF and Corr

# In[14]:


vif_data = pd.DataFrame()
vif_data["feature"] = X_scaled.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X_scaled.values, i)
                   for i in range(len(X_scaled.columns))]

print(vif_data)


# In[15]:


print(vif_data.sort_values('VIF', ascending=False))


# In[16]:


# Remove variables that are correlated
# columns that start with wtd and gmean
# columns that contain fie, radius, ThermalConducitivity and Density

X_scaled_clean = \
    X_scaled.loc[:, ~X_scaled.columns.str.startswith('wtd')]
X_scaled_clean = \
    X_scaled_clean.loc[:, ~X_scaled_clean.columns.str.startswith('gmean')]
X_scaled_clean = \
    X_scaled_clean.loc[:, ~X_scaled_clean.columns.str.contains('fie')]
X_scaled_clean = \
    X_scaled_clean.loc[:, ~X_scaled_clean.columns.str.contains('radius')]
X_scaled_clean = \
    X_scaled_clean.loc[:,
                       ~X_scaled_clean
                       .columns.str.contains('ThermalConductivity')]
X_scaled_clean = \
    X_scaled_clean.loc[:, ~X_scaled_clean.columns.str.contains('Density')]


# In[17]:


# Examine the correlation matrix of the variables
corr = X_scaled_clean.corr()
corr.style.background_gradient()
corr.style.background_gradient().set_precision(2)


# In[18]:


# remove columns based on high correlation
if 'number_of_elements' in X_scaled_clean:
    del X_scaled_clean['number_of_elements']
if 'std_FusionHeat' in X_scaled_clean:
    del X_scaled_clean['std_FusionHeat']
if 'entropy_ElectronAffinity' in X_scaled_clean:
    del X_scaled_clean['entropy_ElectronAffinity']
if 'entropy_FusionHeat' in X_scaled_clean:
    del X_scaled_clean['entropy_FusionHeat']
if 'entropy_Valence' in X_scaled_clean:
    del X_scaled_clean['entropy_Valence']
if 'std_ElectronAffinity' in X_scaled_clean:
    del X_scaled_clean['std_ElectronAffinity']
if 'std_atomic_mass' in X_scaled_clean:
    del X_scaled_clean['std_atomic_mass']
if 'std_Valence' in X_scaled_clean:
    del X_scaled_clean['std_Valence']
if 'range_FusionHeat' in X_scaled_clean:
    del X_scaled_clean['range_FusionHeat']


# In[19]:


# Reevaluate correlation values with the remaining columns
corr_clean = X_scaled_clean.corr()
corr_clean.style.background_gradient()
corr_clean.style.background_gradient().set_precision(2)


# In[20]:


# Review data frame after the removing correlated variables
X_scaled_clean.columns


# In[21]:


X_scaled_clean.shape


# #### After clean data of multicollinearity we are left with 94 columns to evaluate our models

# ## 3) Modeling

# #### Linear Regression Model

# In[22]:


# Create a linear regression model
lin_reg = LinearRegression()

# Perform 5-fold cross validation
splits = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(lin_reg, X_scaled_clean, y,
                         scoring='neg_mean_squared_error',
                         cv=splits, n_jobs=3)

# Print the mean and standard deviation of the MSE
print("Mean MSE: ", scores.mean())


# #### Model Lasso L1 Regression

# In[23]:


# Feature importance. Idea here would be to use L1 Regression
# to pick out feature importance and those features for L2 Ridge Regression
# For the best results we will cross validate the data first

splits = KFold(n_splits=5, shuffle=True)
L1_model = Lasso()

# Create error alerts then find the best
# hyperparameter for the Lasso regression
best = -10000
alpha = np.logspace(-10, 10, 100)
for i in alpha:
    L1_model.alpha = i
    out = cross_val_score(L1_model,
                          X_scaled_clean, y,
                          scoring='neg_mean_squared_error',
                          cv=splits, n_jobs=3).mean()
    if out > best:
        best = out
        best_alpha = i
print('Best Alpha:',  best_alpha)
print('Best MSE:', best)


# In[24]:


lasso_best = Lasso(alpha=0.31)
lasso_best.fit(X_scaled_clean, y)

idx_lasso = np.argpartition(abs(lasso_best.coef_), -10)[-10:]
indices_lasso = idx_lasso[
    np.argsort(abs(lasso_best.coef_)[idx_lasso])].tolist()
print('Top 10 coefficients that LASSO chose are:\n',
      X_scaled_clean.columns[indices_lasso])


# In[25]:


(lasso_best.coef_)[indices_lasso]


# #### Model 2 L2 Ridge

# In[27]:


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


# In[28]:


model = Ridge()
param_dist = {
    "alpha": np.logspace(-10, 10, 100)
}
grid_search = GridSearchCV(model,
                           param_grid=param_dist,
                           scoring='neg_mean_squared_error')
grid_search.fit(X_scaled_clean, y)
report(grid_search.cv_results_)


# In[29]:


ridge_best = Ridge(alpha=2154.4346900318865)
ridge_best.fit(X_scaled_clean, y)

idx_ridge = np.argpartition(abs(ridge_best.coef_), -10)[-10:]
indices_ridge = idx_ridge[
    np.argsort(abs(ridge_best.coef_)[idx_ridge])].tolist()
print('Top 10 coefficients that RIDGE chose are:\n',
      X_scaled_clean.columns[indices_ridge])


# In[30]:


(ridge_best.coef_)[indices_ridge]

