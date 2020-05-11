import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
abr_month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
            'aug', 'sep', 'oct', 'nov', 'dec']


def load_monthly_data():


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


idx_to_drop = monthly.unit_cogs[monthly.unit_cogs.str.len() == 0].index
monthly = monthly.drop(idx_to_drop)


monthly.unit_cogs =  monthly.unit_cogs.astype('float')

monthly['agg_rev'] = monthly['quantity']

monthly.isnull().sum()
monthly.duplicated().sum()

categorical_cols = ['group']
numeric_cols = list(monthly.columns)
numeric_cols.remove('group')
monthly.describe(include = ['O'])
monthly.describe(include = [np.number])
print(monthly.head())




final_list=monthly.groupby(['date'])["agg_rev"].apply(lambda x : x.astype(int).sum()).reset_index()
print(final_list.head(12))
d=final_list['date'].dt.strftime('%m').values
print(d)
prices=final_list['agg_rev']
print(prices.head())

import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

plt.switch_backend('TkAgg')



from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
def predict_price(d,x):


    print("dates" + str(d))
    print("prices=" + str(prices))
    print("x" + str(x))


    d = [int(x) for x in d]

    print(d)

    '''
    Builds predictive model and graphs it
    This function creates 3 models, each of them will be a type of support vector machine.
    A support vector machine is a linear seperator. It takes data that is already classified and tries
    to predict a set of unclassified data.
    So if we only had two data classes it would look like this
    It will be such that the distances from the closest points in each of the two groups is farthest away.
    When we add a new data point in our graph depending on which side of the line it is we could classify it
    accordingly with the label. However, in this program we are not predicting a class label, so we don't
    need to classify instead we are predicting the next value in a series which means we want to use regression.
    SVM's can be used for regression as well. The support vector regression is a type of SVM that uses the space between
    data points as a margin of error and predicts the most likely next point in a dataset.
    The predict_prices returns predictions from each of our models

    '''
    dates = np.reshape(d, (len(d), 1))
    print("dates1=" + str(dates))

    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    svr_rbf.fit(dates, prices)
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    random_forest = RandomForestClassifier(5)

    random_forest.fit(dates, prices)

    extratrees = ExtraTreesClassifier(n_estimators=100, random_state=0)
    #extratrees.fit(dates,prices)
    gradient_boost=GradientBoostingClassifier()
    gradient_boost.fit(dates,prices)

    plt.plot(dates, prices, color='pink', label='Real Value')
    plt.plot(dates,  random_forest.predict(dates), color='Black', label='Random Forest')
    #plt.plot(dates, extratrees.predict(dates), color='Brown', label='Extratrees')
    plt.plot(dates,  gradient_boost.predict(dates), color='Orange', label='GradientBoost')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')  # plotting the line made by the RBF kernel
    plt.plot(dates, svr_lin.predict(dates), color='green',
             label='Linear model')  # plotting the line made by linear kernel
    plt.plot(dates, svr_poly.predict(dates), color='blue',
             label='Polynomial model')

    plt.xlabel('Date')  # Setting the x-axis
    plt.ylabel('Price')  # Setting the y-axis
    plt.title('Support Vector Regression')  # Setting title
    plt.legend()  # Add legend
    plt.show()  # To display result on screen
    print(svr_rbf.predict(np.reshape([x], (len([x]), 1)))[0])
    return svr_rbf.predict(np.reshape([x], (len([x]), 1)))[0]


predicted_price = predict_price(d,8)


print('The predicted prices are:', predicted_price)
