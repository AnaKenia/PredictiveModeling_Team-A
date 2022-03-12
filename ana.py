from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

script_dir = Path(__file__).parent
df = pd.DataFrame()

# Read all files and create a unique data frame containing all info, for train set
for i in range(1,3):
    file = 'train/data_0{}.csv'.format(i)
    df_new = pd.read_csv(file)
    df = df.append(df_new,ignore_index=True)
    
# Read file for test set
file = 'test/test_data_123.csv'
df_test = pd.read_csv(file)
df_test = df_test.set_index('date')

# Create a df grouped by date and generating the mean of all cols
cols = df.columns
df = df.groupby('date')[cols].mean()

# Info for checking nulls and dtype
print(df.info())

# Describe: mean, std, min, max, etc
print(df.describe())


"""Correlation to POWER"""
# Calculate correlation to target variable
features = df
target = df['POWER']
corr_target = abs(features.corrwith(target)).to_frame().reset_index()
corr_target = corr_target.rename(columns= {'index': 'var', 0:'corr'})

# Display relation to turbine or ambient
var = pd.DataFrame(df.columns)
dicti = {'date': 0, 'T_AMB': 0, 'P_AMB': 0, 'CMP_SPEED': 1, 'CDP': 1, 'GGDP': 1, \
         'HPT_IT': 1, 'CDT': 1, 'LPT_IT': 1, 'EXH_T': 1, 'RH': 0, 'WAR': 0, 'POWER': 1}
# Map values
var['rel'] = var[0].map(dicti)
# Create df with correlation value and relation
corr_target = corr_target.merge(var, left_on = 'var', right_on = 0)

# Bar plot of correlation and relation
sns.barplot(data = corr_target, y = 'var', x = 'corr', orient = 'h', hue = 'rel', palette="Blues_d")
plt.title('Correlation to POWER. 0: amb, 1: turb')
plt.xlabel('Correlation'); plt.ylabel('Variable')
plt.show()
    

"""Handling missing values"""
# Summary of nans per col
print(df.isna().sum())

# Fill df with mean value of cols
for col in df.columns:
    df[col].fillna(np.mean(df[col]), inplace = True)
for col in df_test.columns:
    df_test[col].fillna(np.mean(df_test[col]), inplace = True)
    
# Re-check for nan values
print(df.isna().sum())


"""Splitting the dataset into train and test sets. All train set"""

# Split feature and target variable

x = df.drop(columns = ['POWER']) 
y = df['POWER']

# # Split intro train and test sets
# # TEST SIZE V IMPORTANT BC TONS OF MISSING VALUES FOR POWER, AROUND 40%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33)

# Initialize a Linear Reggression model
linreg = LinearRegression()

# Fit (train) linreg to the train set
linreg.fit(x, y) 

# Predictions using the test set
y_pred = linreg.predict(x_test)

# Coeff
coef = linreg.coef_

# R2 score
r2score = r2_score(y_test,y_pred)
print("R2 score is ", r2score)

# Expected vs. actual values
plt.scatter(y_pred, y_test, label = 'data') 
x=np.linspace(min(y_pred), max(y_pred))      
plt.plot(x, x, label = 'slope = 1')
plt.title("Expected vs. actual values")
plt.xlabel('Predicted values'); plt.ylabel('Actual values')
plt.legend()
plt.show()





"""Predictions using test set"""
y_predictions = linreg.predict(df_test)
