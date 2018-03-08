#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 16:58:55 2018

@author: vlad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model, Input
from keras.layers import Dense
from keras.layers import LSTM
from pandas import Series, DataFrame, concat
from keras.models import Sequential
from math import sqrt


def build_dataset():
    #LOAD DATA
    features_data = pd.read_csv('~/Projects/Store_sales/Features_data_set.csv').fillna('VK')
    sales_data = pd.read_csv('~/Projects/Store_sales/sales_data_set.csv').fillna('VK')
    stores_data = pd.read_csv('~/Projects/Store_sales/stores_data_set.csv').fillna('VK')

    #MERGE DATA INTO A SINGLE DATA SET
    features_data.drop('IsHoliday', axis = 1, inplace = True)
    data = pd.merge(sales_data, stores_data, on='Store')
    data = pd.merge(data, features_data, on=['Store', 'Date'])
    del(features_data, sales_data, stores_data)

    #REPLACE DATE STRING WITH DATETIME
    data['Date'] = pd.Series([datetime.strptime(d, '%d/%m/%Y') for d in data['Date']])
    
    #ADD YEAR, MONTH AND WEEK COLUMNS
    data['Year'] = pd.Series([t.year for t in data['Date']])
    data['Month'] = pd.Series([t.month for t in data['Date']])
    '''there is a slight inconsistency here because of a leap year and the 
    defnintion of the sales week depending on what week day the year starts;
    however given the dates are all fridays the datetime week function does
    a good job assigning week numbers'''
    data['Week'] = pd.Series([t.week for t in data['Date']])

    return data

#LET'S DO SOME EXPLORATORY DATA ANALYSIS. THE DATASET CONTAINS SEVERAL STORES.
#SELECT AN EXAMPLE STORE AND DEPARTMENT TO EXAMINE TIMELINE CHANGES
data.describe(include = 'all')
data.dtypes

select_store = 1
select_dept = 1
data_df = data.loc[(data['Store'] == select_store) & (data['Dept'] == select_dept)]



#PLOT SELECTED COLUMNS
values = data_df.values
to_plot = [ 3, 4, 5, 6, 7, 8, 9]
i = 1
# plot each column
plt.figure()
for item in to_plot:
	plt.subplot(len(to_plot), 1, i)
	plt.plot(values[:, item])
	plt.title(data_df.columns[item], y=0.5, loc='right')
	i += 1
plt.show()


to_plot = [10, 11, 12, 13, 14, 15]
i = 1
# plot each column
plt.figure()
for item in to_plot:
	plt.subplot(len(to_plot), 1, i)
	plt.plot(values[:, item])
	plt.title(data.columns[item], y=0.5, loc='right')
	i += 1
plt.show()


#SOME MORE EXPLORATORY DATA ANALSYSIS
x = data['Store'].unique()
data['Store'].nunique()
data['Store'].value_counts()

#PLOT STORES VS NUMBER OF DEPARTMENTS FOR EACH STORE
def store_plot():
    
    x = data['Store'].unique()
    y = []
    for i in range(len(x)):
        temp_df = data.loc[data['Store'] == i+1]
        y.append(temp_df['Dept'].nunique())
        i += 1


    plt.figure()
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel('stores')
    plt.ylabel('departments')
    plt.show()


#ANNUAL SALES PER STORE
    
df_2010 = data[data['Date'].isin(pd.date_range("2010-01-01", "2010-12-31"))]
df_2011 = data[data['Date'].isin(pd.date_range("2011-01-01", "2011-12-31"))]
df_2012 = data[data['Date'].isin(pd.date_range("2012-01-01", "2012-12-31"))]

x = data['Week'].unique()
x = df_2010['Week'].unique()
y = df_2010.groupby('Week')['Weekly_Sales'].sum()

#build table aggregating sales by year and week
annual_sales_df = pd.concat([df_2010.groupby('Week')['Weekly_Sales'].sum(), df_2011.groupby('Week')['Weekly_Sales'].sum()], axis = 1)    
annual_sales_df = pd.concat([annual_sales_df, df_2012.groupby('Week')['Weekly_Sales'].sum()], axis = 1)
annual_sales_df.columns = ['2010', '2011', '2012']



plt.figure()
plt.plot(annual_sales_df['2010'])
plt.plot(annual_sales_df['2011'])
plt.plot(annual_sales_df['2012'])
plt.legend(loc='upper left')
plt.show()


#and also let's take a look at how sales vary by stores in a given year
#say 2011. This gives us a hairy graph, but it makes clear what sort of 
#trends we are seeing
df_2011.groupby(['Week', 'Store']).sum()['Weekly_Sales'].unstack().plot(legend = None)
plt.title('Weekly sales by store in 2011')
plt.xlim(1,52)

#and now do the same for 2010 which is an incomplete year
df_2010.groupby(['Week', 'Store']).sum()['Weekly_Sales'].unstack().plot(legend = None)
plt.title('Weekly sales by store in 2010')
plt.xlim(1,52)

#and 2012 which is also incomplete
df_2012.groupby(['Week', 'Store']).sum()['Weekly_Sales'].unstack().plot(legend = None)
plt.title('Weekly sales by store in 2012')
plt.xlim(1,52)

'''     
#LET'S BUILD DATA SERIES READY FOR A MODEL NOW
#START WITH TOTAL ANUAL SALES
def build_data_set():
    sales_df = pd.concat([df_2010.groupby('Week')['Weekly_Sales'].sum(), df_2011.groupby('Week')['Weekly_Sales'].sum()], axis = 0)
    sales_df = pd.concat([sales_df, df_2012.groupby('Week')['Weekly_Sales'].sum()], axis = 0)
    sales_df = pd.DataFrame(sales_df)

    #take changes in sequence values
    sales_df = sales_df.diff()
    
    n_vars = sales_df.shape[1]
    shift_by = 1
    cols, names = list(), list()

    for i in range(shift_by, 0, -1):
        cols.append(sales_df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    cols.append(sales_df)
    names += [('var%d(t)' % (j+1)) for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)

    return agg
'''




#LET'S BUILD A MODEL; LSTM IS AN OBVIOUS CANDIDATE. I wouldn't bother with
#removal of trend, especially because I can't see an obvious one

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 
    

sales_df = sales_df.as_matrix()
X, y = sales_df[:, 0:-1], sales_df[:, -1]
#lstm needs 3 dim
X = X.reshape(X.shape[0], 1, X.shape[1])
#set this to 1
batch_size = 1


# scale data to [-1, 1]
def scale(train, test):
	
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
	
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
	
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecast
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

'''
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df
'''

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, shuffle=False)
		model.reset_states()
	return model


 # make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]



sales_df = build_data_set()
raw_values = sales_df['var1(t-1)'].values
diff_values = sales_df['var1(t-1)'].diff()
#drop na
diff_values = diff_values[np.isfinite(diff_values)]

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values
 
# split data into train and test-sets
#train, test = supervised_values[0:-20], supervised_values[-20:]
train = supervised_values


# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)
 
# fit the model
lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
lstm_model.save('/Users/vlad/Projects/Store_sales/lstm_model_sales.h5')
        
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)
 



# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
	# make one-step forecast
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = forecast_lstm(lstm_model, 1, X)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# store forecast
	predictions.append(yhat)
	expected = raw_values[len(train) + i + 1]
	print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))


# line plot of observed vs predicted
plt.plot(test[:,1])
plt.plot(predictions)
plt.xlim(0,40)
plt.show()

lstm_model.load_weights('/Users/vlad/Projects/Store_sales/lstm_model_sales.h5', by_name=True)
#now try to predict 35 weeks that should cover the holiday shopping season
predictions = list()
X = test_scaled[0, 0:-1]
raw_anchor = raw_values[-21]
for i in range(35):
    # make one-step forecast
    yhat = forecast_lstm(lstm_model, 1, X)
    remember_X = np.asarray(yhat).reshape(1,)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    raw_anchor = yhat
    yhat = raw_anchor + yhat
    # store forecast
    predictions.append(yhat)
    X = remember_X
    
	



plt.plot(test)
plt.show()

predictions = list()
for i in range(len(test)):
	# make prediction
	predictions.append(history[-1])
	# observation
	history.append(test[i])
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(test)
pyplot.plot(predictions)
pyplot.show()





# repeat experiment
repeats = 30
error_scores = list()
for r in range(repeats):
	# fit the model
	lstm_model = fit_lstm(train_scaled, 1, 100, 4)
	# forecast the entire training dataset to build up state for forecasting
	train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
	lstm_model.predict(train_reshaped, batch_size=1)
	# walk-forward validation on the test data
	predictions = list()
	for i in range(len(test_scaled)):
		# make one-step forecast
		X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
		yhat = forecast_lstm(lstm_model, 1, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# invert differencing
		yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
		# store forecast
		predictions.append(yhat)
	# report performance
	rmse = sqrt(mean_squared_error(raw_values[-20:], predictions))
	print('%d) Test RMSE: %.3f' % (r+1, rmse))
	error_scores.append(rmse)



#LET'S BUILD WEEKLY AGGREGATE SALES SERIES
def build_series():
    
    #build separate data frames for each year
    df_2010 = data[data['Date'].isin(pd.date_range("2010-01-01", "2010-12-31"))]
    df_2011 = data[data['Date'].isin(pd.date_range("2011-01-01", "2011-12-31"))]
    df_2012 = data[data['Date'].isin(pd.date_range("2012-01-01", "2012-12-31"))]

    #aggregate and concatenate annual data frames
    sales_df = pd.concat([df_2010.groupby('Week')['Weekly_Sales'].sum(), df_2011.groupby('Week')['Weekly_Sales'].sum()], axis = 0)
    sales_df = pd.concat([sales_df, df_2012.groupby('Week')['Weekly_Sales'].sum()], axis = 0)
    sales_df = pd.DataFrame(sales_df)

    return sales_df


# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)


# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    raw_values = series.values
    #build a supervised data set for training
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    #in this case, let's use the full data set for training

    return scaler, train, test



def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
	# reshape training into [samples, timesteps, features]
	X, y = train[:, 0:n_lag], train[:, n_lag:]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	# design network
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer='adam')
	# fit network
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
		model.reset_states()
	return model


# make one forecast
def forecast_lstm(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]


# invert differenced forecast
def inverse_difference(last_ob, forecast):
	# invert first forecast
	inverted = list()
	inverted.append(forecast[0] + last_ob)
	# propagate difference forecast using inverted first value
	for i in range(1, len(forecast)):
		inverted.append(forecast[i] + inverted[i-1])
	return inverted

# plot the forecast
def plot_forecast(series, forecast, n_test):
    # plot the entire dataset in blue
    plt.plot(series.values)
    # plot the forecast in red
    off_s = len(series) - n_test - 1
    off_e = off_s + len(forecast) + 1
    xaxis = [x for x in range(off_s, off_e)]
    yaxis = [series.values[off_s]] + forecast
    #yaxis = forecast
    plt.plot(xaxis, yaxis, color='red')
    # show the plot
    plt.show()


n_test = 0
n_lag = 1
n_seq = 15
n_epochs = 2000
n_batch = 1
n_neurons = 10


data = build_dataset()
series = build_series()
#!!!because n_test is zero, swap train and test to address prepare_data quirk
scaler, _, train = prepare_data(series, n_test, n_lag, n_seq)

model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
model.save('/Users/vlad/Projects/Store_sales/lstm_model_sales.h5')

#build a forecast
X = train[126, 0:n_lag]

forecast = forecast_lstm(model, X, n_batch)
# inverse transform forecasts and test
forecast = np.array(forecast)
forecast = forecast.reshape(1, len(forecast))
inv_scale = scaler.inverse_transform(forecast)
inv_scale = inv_scale[0, :]
index = len(series) - n_test - 1
last_ob = series.values[index]
forecast = inverse_difference(last_ob, inv_scale)
#forecast = forecast[0]
#forecast = pd.DataFrame(forecast)

plot_forecast(series, forecast, n_test)


