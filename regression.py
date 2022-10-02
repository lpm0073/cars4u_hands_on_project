import warnings                                                  # Used to ignore the warning given as output of the code
warnings.filterwarnings('ignore')

import numpy as np                                               # Basic libraries of python for numeric and dataframe computations
import pandas as pd

import matplotlib.pyplot as plt                                  # Basic library for data visualization
import seaborn as sns                                            # Slightly advanced library for data visualization

from sklearn.model_selection import train_test_split             # Used to split the data into train and test sets.

from sklearn.linear_model import LinearRegression, Ridge, Lasso  # Import methods to build linear model for statistical analysis and prediction

from sklearn.tree import DecisionTreeRegressor                   # Import methods to build decision trees.
from sklearn.ensemble import RandomForestRegressor               # Import methods to build Random Forest.

from sklearn import metrics                                      # Metrics to evaluate the model

from sklearn.model_selection import GridSearchCV                 # For tuning the model
