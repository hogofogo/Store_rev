# Store sales
## Overview

Data is from Kaggle:
https://www.kaggle.com/manjeetsingh/retaildataset

In this particular case, I wanted to run an RNN to see its ability to predict the usual holiday season spike and also do some useful exploratory analysis.


## Architecture

In this case, I was interested in building a model that would be able to predict simultaneously a series of values, e.g. say based on sales in week 43, predict sales in the following 15 weeks. This prediction time range includes the holiday season and just by looking at the exploratory analysis Retail_exploratory.pynb.pdf we will know what to expect for this year.

The model itself is a simple LTSM. The complication is in this case data preparation and prediction of several time steps, as opposed to one.

## Data cleaning

Retail_exploratory.pynb.pdf explains what has been done and why


## Training

So far I have built two variations: first based on sales value as independent variable, second on sales value and week number. Next, I will probably add a holiday/non-holiday factor to the independent variables and perhaps other. I would have been useful to have results for the full year in 2012 to examine accuracy, but the data for the last few weeks is not available.

## Results

Trained on sales value only:
![alt text](screenshots/filename.png "Description goes here")
See Screen Shot 2018-03-08 at 1.14.12 PM; the model picked the holiday spike, but it did not pick the post-holiday drop. However, this is a very simple model that only looks at revenue and neglects any contributing factors (such as week of the year, for example), and I haven't done much model tuning. I will try building a more complicated model now that would take several parameters into account.

Now the model has been trained on both sales value and week number:
Nod bad: Screen Shot 2018-03-09 at 10.53.28 AM. The model correctly picks two holiday spikes in the correct weeks, and the subsequent post-holiday dropoff. The predicted sales level is lower than I would have anticipated, and I am not yet sure whether the model is reacting the slight downwards y-o-y trend, or it needs more training/data/etc. I will try to add other parameters.