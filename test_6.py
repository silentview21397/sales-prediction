# from __future__ import division
from datetime import datetime, timedelta,date
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

import plotly.io as pio

import warnings
warnings.filterwarnings("ignore")

import chart_studio.plotly as py
# import plotly.offline as pyoff
import plotly.graph_objs as go

#import Keras
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split

#initiate plotly
# pyoff.init_notebook_mode()

abr_month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
            'aug', 'sep', 'oct', 'nov', 'dec']

abr_month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
            'aug', 'sep', 'oct', 'nov', 'dec']

def load_monthly_data():
    """Loads monthly sales data from  csv"""

    monthly = pd.read_csv(
        'SELL_1.csv',
        delimiter=';',
        decimal=',',
        encoding='latin-1')
    monthly = monthly.rename(columns={
        'Date': 'date',
        'PKod': 'sku',
        'Pgroup': 'group',
        'Pname': 'name',
        'Pquantity': 'quantity',
        'pce_zn': 'unit_cogs',
        'pwa_zn': 'agg_cogs',  # aggregate cogs for this sku
        'pce_sn': 'unit_revenue',
        'pwn_sn': 'agg_rev',  # aggregate revenue for this sku
        'pmarza': 'gross_margin',
        'pmarzajedn': 'unit_contribution',
        'pkwmarza': 'agg_dollar_contribution',
        'pudzmarza': 'share_of_margin'
    })

    monthly = monthly.drop(['pwa_sn', 'pce_sb', 'pwa_sb', 'pudzsb'], axis=1)
    monthly.group = monthly.group.str.lower()
    monthly.name = monthly.name.str.lower()
    monthly.date = pd.to_datetime(monthly.date, format="%d.%m.%Y")
    monthly.unit_cogs = monthly.unit_cogs.str.replace(
        ',', '.').str.replace(
        ' ', '')

    monthly.group = monthly.group.replace(
        {"ketch_concetrate_mustard_majo_horseradish": "sauce"})

    return monthly

monthly = load_monthly_data()
# print(monthly.head())
# monthly = load_monthly_data()

# Drop empty rows
idx_to_drop = monthly.unit_cogs[monthly.unit_cogs.str.len() == 0].index
monthly = monthly.drop(idx_to_drop)

# Convert to numeric
monthly.unit_cogs =  monthly.unit_cogs.astype('float')

# Add aggregate revenue column
# monthly['agg_rev'] = monthly['quantity']
monthly['sales'] = monthly['quantity'] * monthly['unit_revenue']
monthly.isnull().sum()
monthly.duplicated().sum()
# print(monthly.columns)
categorical_cols = ['group']
numeric_cols = list(monthly.columns)
numeric_cols.remove('group')
monthly.describe(include = ['O'])
monthly.describe(include = [np.number])
print(monthly.head())
# print(monthly.date)
final_list=monthly.groupby(['date'])["sales"].apply(lambda x : x.astype(int).sum()).reset_index()
final_list['store']=monthly['sku']
final_list['item']=monthly['quantity']
print(final_list.head())
# d=final_list['date'].dt.strftime('%m').values
# print(d)
# prices=final_list['agg_rev']
# print(prices.head())
#  date  store  item  sales
df_sales=final_list
#read the data in csv
# df_sales = pd.read_csv('train.csv')
# print(df_sales.head())
#convert date field from string to datetime
# df_sales['date'] = pd.to_datetime(df_sales['date'])

#show first 10 rows
# df_sales.head(10)
# print("df_sales.head(10)",df_sales.head())
#represent month in date field as its first day
df_sales['date'] = df_sales['date'].dt.year.astype('str') + '-' + df_sales['date'].dt.month.astype('str') + '-01'
df_sales['date'] = pd.to_datetime(df_sales['date'])
#groupby date and sum the sales
df_sales = df_sales.groupby('date').sales.sum().reset_index()
# print("df_sales",df_sales.head(12))
#plot monthly sales
plot_data = [
    go.Scatter(
        x=df_sales['date'],
        y=df_sales['sales'],
    )
]
plot_layout = go.Layout(
        title='Montly Sales'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)

#create a new dataframe to model the difference
df_diff = df_sales.copy()
#add previous sales to the next row
df_diff['prev_sales'] = df_diff['sales'].shift(1)
#drop the null values and calculate the difference
df_diff = df_diff.dropna()
df_diff['diff'] = (df_diff['sales'] - df_diff['prev_sales'])
# df_diff.head(10)
# print("df_diff.head(8)",df_diff.head(8))
#plot sales diff
plot_data = [
    go.Scatter(
        x=df_diff['date'],
        y=df_diff['diff'],
    )
]
plot_layout = go.Layout(
        title='Montly Sales Diff'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pio.show(fig)
#create dataframe for transformation from time series to supervised
df_supervised = df_diff.drop(['prev_sales'],axis=1)
print("df_supervised.head()",df_supervised.head(12))
#adding lags
for inc in range(1,13):
    try:
        field_name = 'lag_' + str(inc)
        # print("field_name",field_name)
        # print(df_supervised['diff'].shift(inc))

        df_supervised[field_name] = df_supervised['diff'].shift(inc)
    except Exception as e:
        print(e)
#drop null values
# df_supervised = df_supervised.reset_index(drop=True)
# df_supervised = df_supervised.dropna().reset_index(drop=True)
# avg=df_supervised.avg()

df_supervised = df_supervised.fillna(1).reset_index(drop=True)

print("df_supervised",df_supervised.head)
# Import statsmodels.formula.api
import statsmodels.formula.api as smf
# Define the regression formula
model = smf.ols(formula='diff ~ lag_1', data=df_supervised)
print('Fit the regression')
# Fit the regression
model_fit = model.fit()
# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)
#import MinMaxScaler and create a new dataframe for LSTM model
from sklearn.preprocessing import MinMaxScaler
df_model = df_supervised.drop(['sales','date'],axis=1)
#split train and test set
train_set, test_set = df_model[0:-10].values, df_model[-10:].values
#apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)
# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)
# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)
X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
model = Sequential()
model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, nb_epoch=100, batch_size=1, verbose=1, shuffle=False)
y_pred = model.predict(X_test,batch_size=1)
#for multistep prediction, you need to replace X_test values with the predictions coming from t-1
#reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])
#rebuild test set for inverse transform
pred_test_set = []
for index in range(0,len(y_pred)):
    print (np.concatenate([y_pred[index],X_test[index]],axis=1))
    pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))
#reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
#inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)
#create dataframe that shows the predicted sales
result_list = []
sales_dates = list(df_sales[-10:].date)
print("sales_dates",sales_dates)
act_sales = list(df_sales[-10:].sales)
print("act_sales",act_sales)
for index in range(0,len(pred_test_set_inverted)):
    result_dict = {}
    try:
        result_dict['date'] = sales_dates[index + 1]
        result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
        result_list.append(result_dict)

    except Exception as e:
        print(e)
        # result_dict['pred_value']=0
        # result_dict['date']="2018-02-01"
        # print(e)

df_result = pd.DataFrame(result_list)
#for multistep prediction, replace act_sales with the predicted sales
# merge with actual sales dataframe
df_sales_pred = pd.merge(df_sales, df_result, on='date', how='left')
# df_sales_pred = pd.concat(df_sales, df_result)
# plot actual and predicted
plot_data = [
    go.Scatter(
        x=df_sales_pred['date'],
        y=df_sales_pred['sales'],
        name='actual'
    ),
    go.Scatter(
        x=df_sales_pred['date'],
        y=df_sales_pred['pred_value'],
        name='predicted'
    )

]
plot_layout = go.Layout(
    title='Sales Prediction'
)
fig = go.Figure(data=plot_data, layout=plot_layout)
pio.show(fig)