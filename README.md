# Store sales
## Overview

Data is from Kaggle:
[_____]

In this particular case, I wanted to run an RNN to see its ability to predict the usual holiday season spike and also do some useful exploratory analysis.


## Architecture


## Data cleaning


## Training


## Results

preliminary: the good: Screen Shot 2018-03-08 at 1.14.12 PM shows that the model picked up the holiday spike. the bad: it did not pick the post-holiday drop. However, this is a very simple model that only looks at revenue and neglects any contributing factors (such as week of the year, for example), and I haven't done much model tuning. I will try building a more complicated model now that would take several parameters into account.