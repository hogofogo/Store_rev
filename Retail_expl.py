#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 11:19:27 2018

@author: vlad
"""

#EXPLORATORY ANALYSIS ON RETAIL STORE DATA SET
#ASSUME ALL DATA LOADED FROM RETAIL.PY

#LET'S DO SOME EXPLORATORY DATA ANALYSIS. THE DATASET CONTAINS SEVERAL STORES.
#SELECT AN EXAMPLE STORE AND DEPARTMENT TO EXAMINE TIMELINE CHANGES
data.describe(include = 'all')
data.dtypes

select_store = 1
select_dept = 1
data_df = data.loc[(data['Store'] == select_store) & (data['Dept'] == select_dept)]



#PLOT SELECTED COLUMNS FOR THE SELECTED STORE
values = data_df.values
to_plot = [ 3, 4, 5, 6, 7, 8, 9]
i = 1
# plot each column
plt.figure()
for item in to_plot:
    plt.subplot(len(to_plot), 1, i)
    plt.plot(values[:, item])
    plt.yticks([])
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
    plt.yticks([])
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


'''So, what we got is three incomplete years; 2010 is missing the beginning,
but only a few weeks, 2011 is complete, and 2012 is missing the last 8 or so
weeks, including the holidays. It would be interesting to see if a model
can be built that predicts the usual seasonal fluctuation based on the 
observations from 2010 and 2011. 

I am not going to bother with building a train/test split this time, but
instead will use the full data set as for training. It's pretty clear what
we can expect to see in that gap at the end of 2012 graph. 

The purpose here is only to set up a model that would correctly infer the 
typical sales patterns surrounding the holiday season. LSTM is an ovious model
candidate. Given LSTM choice, it would be intresting to:

    - make a series of predictions in one go, i.e. make one prediction of
    several weeks, as opposed to a week by week prediction
    - experiment with adding different independent variables and their
    effect on the model performance
'''