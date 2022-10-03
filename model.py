#------------------------------------------------------------------------------
# written by:   Lawrence McDaniel
#               https://lawrencemcdaniel.com
#
# date:         oct-2022
#
# usage:        transcription of Cars4U hands-on session 2-Oct-2022
#               for Regression Analysis.
#------------------------------------------------------------------------------

# native Python libraries
import sys
import math
import warnings                                                  # Used to ignore the warning given as output of the code
warnings.filterwarnings('ignore')

# from this repo
from utils import (
    encode_cat_vars, 
    get_model_score,
    build_ols_model,
    model_pref,
    checking_vif,
    treating_multicollinearity
    )

from qc import (
    process_column_mileage, 
    process_column_engine, 
    process_column_power, 
    process_column_new_price, 
    missing_value_treatment, 
    feature_engineering_name,
    plot_distribution_km_driven,
    plot_distribution_price,
    drop_redundant_columns
    )

# from MIT IDSS Data Science & Machine Learning Jupyter Notebook
import matplotlib.pyplot as plt                                  # Basic library for data visualization
import numpy as np                                               # Basic libraries of python for numeric and dataframe computations
import pandas as pd
import pylab
import scipy.stats as stats
import seaborn as sns                                            # Slightly advanced library for data visualization

from sklearn.model_selection import train_test_split             # Used to split the data into train and test sets.
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # Import methods to build linear model for statistical analysis and prediction
from sklearn.tree import DecisionTreeRegressor                   # Import methods to build decision trees.
from sklearn.ensemble import RandomForestRegressor               # Import methods to build Random Forest.
from sklearn import metrics                                      # Metrics to evaluate the model
from sklearn.model_selection import GridSearchCV                 # For tuning the model

# these imports were coded in-line 
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.compat import lzip

# open and clean the data file
df = pd.read_csv("used_cars_data.csv")
process_column_mileage()
process_column_engine()
process_column_power()
process_column_new_price()
missing_value_treatment()
feature_engineering_name()
plot_distribution_km_driven()
plot_distribution_price()
drop_redundant_columns()

"""
Potential techniques: Since it is a regression problem we will first start with the parametric model - linear regression and Ridge Regression.

Overall solution design: The potential solution design would look like this:
- Checking the data description to get the idea of basic statistics or summary of data.
- Univariate analysis to see how data is spread out, getting to know about the outliers.
- Bivariate analysis to see how different attributes vary with the dependent variable.
- Outlier treatment if needed - In this case, outlier treatment is not necessary as outliers are the luxurious cars and in real-world scenarios, such cars would appear in data and we would want our predictive model to capture the underlying pattern for them.
- Missing value treatment using appropriate techniques.
- Feature engineering - transforming features, creating new features if possible.
- Choosing the model evaluation technique - 1) R Squared 2) RMSE can be any other metrics related to regression analysis.
- Splitting the data and proceeding with modeling.
- Model tuning to see if the performance of the model can be improved further.

Measures of success:
- R-squared and RMSE can be used as a measure of success.
- R-squared: This will tell us how much variation our predictive model can explain in data.
- RMSE: This will give us a measure of how far off the model is predicting the original values on average.

Model Building
1. What we want to predict is the "Price". We will use the normalized version 'price_log' for modeling.
2. Before we proceed to the model, we'll have to encode categorical features. We will drop categorical features like Name.
3. We'll split the data into train and test, to be able to evaluate the model that we build on the train data.
4. Build Regression models using train data.
5. Evaluate the model performance.
"""

def model():
    ind_vars = df.drop(["Price", "price_log"], axis = 1)
    dep_var = df[["price_log", "Price"]]
    ind_vars_num = encode_cat_vars(ind_vars)
    ind_vars_num.head()

    x_train, x_test, y_train, y_test = train_test_split(
        ind_vars_num, dep_var, test_size = 0.3, random_state = 1
    )
    print("Number of rows in train data =", x_train.shape[0])
    print("Number of rows in test data =", x_test.shape[0])

    """
    Fitting a linear model - Linear Regression
    Linear Regression can be implemented using:

    1) Sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    2) Statsmodels: https://www.statsmodels.org/stable/regression.html
    """
    # Statsmodel api does not add a constant by default. We need to add it explicitly.
    x_train = sm.add_constant(x_train)
    # Add constant to test data
    x_test = sm.add_constant(x_test)

    olsmodel1 = build_ols_model(y_train, train=x_train)    # internal def
    print(olsmodel1.summary())

    # Notes:
    # [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    # [2] The smallest eigenvalue is 1.13e-29. This might indicate that there are
    # strong multicollinearity problems or that the design matrix is singular.

    # - Both the R-squared and Adjusted R squared of our model are very high. This is a clear indication that we have been able to create a very good model that can explain the variance in the price of used cars for up to 95%.
    # - The model is not an underfitting model.
    # - To be able to make statistical inferences from our model, we will have to test that the linear regression assumptions are followed.
    # - Before we move on to assumption testing, we'll do a quick performance check on the test data.

    # Checking model performance
    model_pref(olsmodel1, x_train, x_test, y_train, y_test) # High Overfitting.

    # - Root Mean Squared Error of train and test data is starkly different, indicating that our model is overfitting the train data.
    # - Mean Absolute Error indicates that our current model can predict used car prices within a mean error of 1.9 lakhs on test data.
    # - The units of both RMSE and MAE are the same - Lakhs in this case. But RMSE is greater than MAE because it penalizes the outliers more.
    # - Mean Absolute Percentage Error is ~15% on the test data.

    # Checking the Linear Regression Assumptions
    # 1. No Multicollinearity
    # 2. Mean of residuals should be 0
    # 3. No Heteroscedasticity
    # 4. Linearity of variables
    # 5. Normality of error terms

    # Checking Assumption 1: No Multicollinearity
    # - We will use VIF, to check if there is multicollinearity in the data.
    # - Features having a VIF score >5 will be dropped/treated till all the features have a VIF score <5.

    print(checking_vif(x_train))    # internal def
    # observations:
    # ---------------
    # - There are a few variables with high VIF.
    # - Our current model is extremely complex. Let us first bin the Brand and Model columns.
    # - This wouldn't essentially reduce multicollinearity in the data, but it will help us make the dataset more manageable
    df.groupby(["Brand", "Model"])["new_price_num"].mean().sort_values(ascending = False)

    # We will create a new variable Car Category by binning the new_price_num
    # Create a new variable - Car Category
    df1 = df.copy()
    df1["car_category"] = pd.cut(
        x = df["new_price_num"],
        bins = [0, 15, 30, 50, 200],
        labels = ["Budget_Friendly", "Mid-Range", "Luxury_Cars", "Ultra_luxury"],
    )
    # car_category.value_counts()

    # Drop the Brand and Model columns.
    df1.drop(columns = ["Brand", "Model"], axis = 1, inplace = True)

    # We will have to create the x and y datasets again
    ind_vars = df1.drop(["Price", "price_log"], axis = 1)
    dep_var = df1[["price_log", "Price"]]

    # Dummy encoding
    ind_vars_num = encode_cat_vars(ind_vars)

    # Splitting data into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        ind_vars_num, dep_var, test_size = 0.3, random_state = 1
    )

    print("Number of rows in train data =", x_train.shape[0])
    print("Number of rows in test data =", x_test.shape[0], "\n\n")

    # Statsmodel api does not add a constant by default. We need to add it explicitly.
    x_train = sm.add_constant(x_train)
    # Add constant to test data
    x_test = sm.add_constant(x_test)

    # Fit linear model on new dataset
    olsmodel2 = build_ols_model(y_train, train=x_train)
    print(olsmodel2.summary())

    # The R squared and adjusted r squared values have decreased, but are still 
    # quite high indicating that we have been able to capture most of the 
    # information of the previous model even after reducing the number 
    # of predictor features.
    #
    # As we try to decrease overfitting, the r squared of our train model is expected to decrease.
    print(checking_vif(x_train))

    # Checking model performance
    model_pref(olsmodel2, x_train, x_test, y_train, y_test)  # No Overfitting.
    # - The RMSE on train data has increased now but has decreased on test data.
    # - The RMSE values on both datasets being close to each other indicate that the model is not overfitting the training data anymore.
    # - Reducing overfitting has caused the MAE to increase on training data but the test MAE has in fact reduced.
    # - MAPE on test data has increased to ~20%

    # LOOKS GOOD: NO OVERFITTING.
    #
    # NOW: remove multicollinearity from the model.

    # Removing Multicollinearity
    # To remove multicollinearity
    # 1. Drop every column that has VIF score greater than 5, one by one
    # 2. Look at the adjusted R square of all these models
    # 3. Drop the Variable that makes the least change in Adjusted-R square
    # 4. Check the VIF Scores again
    # 5. Continue till you get all VIF scores under 5

    high_vif_columns = [
        "engine_num",
        "power_num",
        "new_price_num_log",
        "Fuel_Type",
        "car_category",
    ]
    treating_multicollinearity(high_vif_columns, x_train, x_test, y_test)

    # Dropping cars_category would have the maximum impact on predictive power of the model (amongst the variables being considered)
    # We'll drop engine_num and check the vif again

    # Drop 'engine_num' from train and test
    col_to_drop = "engine_num"
    x_train = x_train.loc[:, ~x_train.columns.str.startswith(col_to_drop)]
    x_test = x_test.loc[:, ~x_test.columns.str.startswith(col_to_drop)]

    # Check VIF now
    vif = checking_vif(x_train)
    print("VIF after dropping ", col_to_drop)
    print(vif)

    # observations:
    # ---------------
    # Dropping engine_num has brought the VIF of power_num below 5
    # new_price_num, Fuel_Type and car_category still show some multicollinearity

    # Checking which one of these should we drop next
    high_vif_columns = [
        "new_price_num",
        "Fuel_Type",
        "car_category",
    ]
    treating_multicollinearity(high_vif_columns, x_train, x_test, y_test)

    # Drop 'new_price_num' from train and test
    col_to_drop = "new_price_num"
    x_train = x_train.loc[:, ~x_train.columns.str.startswith(col_to_drop)]
    x_test = x_test.loc[:, ~x_test.columns.str.startswith(col_to_drop)]

    # Check VIF now
    vif = checking_vif(x_train)
    print("VIF after dropping ", col_to_drop)
    print(vif)

    # observations:
    # -----------------
    # We have removed multicollinearity from the data.
    # Fuel_Type variables are showing high vif because most cars are either diesel or petrol.
    # These two features are correlated with each other. It is not a good practice to consider VIF values for dummy variables as they are correlated to other categories and hence have a high VIF usually. So we built the model and calculate the p-values.
    # We will not drop this variable from the model because this will not affect the interpretation of other features in the model.

    # Let's look at the model with the data that does not have multicollinearity
    # Fit linear model on new dataset
    olsmodel3 = build_ols_model(y_train, train=x_train)
    print(olsmodel3.summary())

    print("\n\n")

    # Checking model performance
    model_pref(olsmodel3, x_train, x_test, y_train, y_test)

    # Checking Assumption 2: Mean of residuals should be 0
    residuals = olsmodel3.resid
    np.mean(residuals)
    # Mean of redisuals is very close to 0. The second assumption is also satisfied.

    # Checking Assumption 3: No Heteroscedasticity
    # Homoscedacity - If the residuals are symmetrically distributed across the 
    # regression line, then the data is said to be homoscedastic.
    #
    # Heteroscedasticity- - If the residuals are not symmetrically distributed 
    # across the regression line, then the data is said to be heteroscedastic. 
    # In this case, the residuals can form a funnel shape or any other 
    # non-symmetrical shape.

    # We'll use Goldfeldquandt Test to test the following hypothesis
    name = ["F statistic", "p-value"]
    test = sms.het_goldfeldquandt(residuals, x_train)
    lzip(name, test)

    # observations:
    # ---------------
    # - Since the p-value > 0.05 we cannot reject the Null Hypothesis that the residuals are homoscedastic.
    # - Assumptions 3 is also satisfied by our olsmodel3.

    # Checking Assumption 4: Linearity of variables
    # Predictor variables must have a linear relation with the dependent variable.
    # To test the assumption, we'll plot residuals and fitted values on a plot 
    # and ensure that residuals do not form a strong pattern. 
    # They should be randomly and uniformly scattered on the x-axis.
    # Predicted values
    fitted = olsmodel3.fittedvalues

    # sns.set_style("whitegrid")
    sns.residplot(fitted, residuals, color = "purple", lowess = True)

    plt.xlabel("Fitted Values")
    plt.ylabel("Residual")
    plt.title("Residual PLOT")
    plt.show()

    # observations:
    # ---------------
    # Assumptions 4 is satisfied by our olsmodel3. There is no pattern in the residual vs fitted values plot.

    # Checking Assumption 5: Normality of error terms
    # The residuals should be normally distributed.
    # Plot histogram of residuals
    sns.distplot(residuals)

    stats.probplot(residuals, dist = "norm", plot = pylab)
    plt.show()

    # observations: 
    # ---------------
    # - The residuals have a close to normal distribution. Assumption 5 is also satisfied.
    # - We should further investigate these values in the tails where we have made huge residual errors.    

    print(olsmodel3.summary())

    """
    Observations from the model
    1. With our linear regression model we have been able to capture ~90 variations in our data.

    2. The model indicates that the most significant predictors of the price of used cars are -
        - The year of manufacturing
        - Number of seats in the car
        - Power of the engine
        - Mileage
        - Kilometers Driven
        - Location
        - Fuel_Type
        - Transmission - Automatic/Manual
        - Car Category - budget brand to ultra-luxury

    The p-values for these predictors are <0.05 in our final model.

    3. Newer cars sell for higher prices. 1 unit increase in the year of manufacture 
    leads to [exp(0.1170) = 1.12 Lakh] increase in the price of the vehicle when everything else is constant. 
    It is important to note here that the predicted values are log(price) and therefore coefficients have to be converted accordingly to understand that influence in Price.
    
    4. As the number of seats increases, the price of the car increases - exp(0.0343) = 1.03 Lakhs.
    
    5. Mileage is inversely correlated with Price. Generally, high Mileage cars are the lower budget cars. 
    It is important to note here that correlation is not equal to causation. 
    That is to say, an increase in Mileage does not lead to a drop in prices. 
    It can be understood in such a way that the cars with high mileage do not 
    have a high power engine and therefore have low prices.

    6. Kilometers Driven have a negative relationship with the price which is intuitive. 
    A car that has been driven more will have more wear and tear and hence 
    sell at a lower price, everything else being constant.

    7. The categorical variables are a little hard to interpret. But it can be seen 
    that all the car_category variables in the dataset have a positive relationship 
    with the Price and the magnitude of this positive relationship increases 
    as the brand category moves to the luxury brands. Here the dropped car_category 
    variable for budget-friendly cars serves as a reference variable to other 
    car_category variables when everything else is constant. 1 unit increase in 
    car_category_Luxury_Cars leads to [exp(0.4850) = 1.62 Lakh] increase in the 
    price of the vehicle than the car_category_budget_friendly_Cars that serves 
    as a reference variable when everything else is constant.
    """

def analyzing_off_the_mark_predictions():
    ind_vars = df.drop(["Price", "price_log"], axis = 1)
    dep_var = df[["price_log", "Price"]]
    ind_vars_num = encode_cat_vars(ind_vars)
    ind_vars_num.head()

    x_train, x_test, y_train, y_test = train_test_split(
        ind_vars_num, dep_var, test_size = 0.3, random_state = 1
    )

    ##  Function to calculate r2_score and RMSE on train and test data

    # Extracting the rows from original data frame df where indexes are same as the training data
    original_df = df[df.index.isin(x_train.index.values)].copy()

    # Extracting predicted values from the final model
    olsmodel3 = build_ols_model(y_train, train=x_train)

    residuals = olsmodel3.resid
    fitted_values = olsmodel3.fittedvalues

    # Add new columns for predicted values
    original_df["Predicted price_log "] = fitted_values
    original_df["Predicted Price"] = fitted_values.apply(math.exp)
    original_df["residuals"] = residuals
    original_df["Abs_residuals"] = residuals.apply(math.exp)
    original_df["Difference in Lakhs"] = np.abs(
        original_df["Price"] - original_df["Predicted Price"]
    )

    # Let us look at the top 20 predictions where our model made highest extimation errors (on train data)
    original_df.sort_values(by = ["Difference in Lakhs"], ascending = False).head(100)

    # observations:
    # ----------------
    # - A 2017 Land Rover, whose new model sells at 230 Lakhs and the used 
    #   version sold at 160 Lakhs was predicted to be sold at 32L. It is not 
    #   apparent after looking at numerical predictors, why our model predicted 
    #   such a low value here. This could be because all other land rovers in 
    #   our data seem to have sold at lower prices.
    #
    # - The second one on the list here is a Porsche cayenne that was sold at 
    #   2 Lakhs but our model predicted the price as 85.4. This is most 
    #   likely a data entry error. A 2019 manufactured Porsche selling 
    #   for 2 Lakh is highly unlikely. With all the information we have, 
    #   the predicted price of 85L seems much more likely. We will be 
    #   better off dropping this observation from our current model. 
    #   If possible, the better route would be to gather more information here.
    #
    # - There are a few instances where the model predicts lesser than the 
    #   actual selling price. These could be a cause for concern. The 
    #   model predicting lesser than potential selling price is not 
    #   good for business.

    sns.scatterplot(
        original_df["Difference in Lakhs"],
        original_df["Price"],
        hue=original_df["Fuel_Type"],
    )

    # observations:
    # --------------
    # Most outliers are the Petrol cars. Our model predicts that resale value 
    # of diesel cars is higher compared to petrol cars. This is probably 
    # the cause of these outliers.

def ridge_regression_model():
    # Splitting data into train and test
    ind_vars = df.drop(["Price", "price_log"], axis = 1)
    dep_var = df[["price_log", "Price"]]
    ind_vars_num = encode_cat_vars(ind_vars)

    x_train, x_test, y_train, y_test = train_test_split(
        ind_vars_num, dep_var, test_size = 0.3, random_state = 1
    )

    # Create a Ridge regression model
    rdg = Ridge()

    # Fit Ridge regression model.
    rdg.fit(x_train, y_train["price_log"])

    # Get score of the model.
    Ridge_score = get_model_score(x_train, x_test, y_train, y_test, model=rdg)

    # Observations
    # ----------------
    # Ridge regression is able to produce better results compared to Linear Regression.
    return rdg

def decision_tree():
    # see https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html

    ind_vars = df.drop(["Price", "price_log"], axis = 1)
    dep_var = df[["price_log", "Price"]]
    ind_vars_num = encode_cat_vars(ind_vars)

    x_train, x_test, y_train, y_test = train_test_split(
        ind_vars_num, dep_var, test_size = 0.3, random_state = 1
    )

    # Create a decision tree regression model
    dtree = DecisionTreeRegressor(random_state = 1)

    # Fit decision tree regression model.
    dtree.fit(x_train, y_train["price_log"])

    # Get score of the model.
    Dtree_model = get_model_score(x_train, x_test, y_train, y_test, model=dtree)
    # Observations
    # ---------------
    # Decision Tree is overfitting on the training set and hence not able to generalize well on the test set.

    # Feature Importance
    # Print the importance of features in the tree building ( The importance of
    # a feature is computed as the (normalized) total reduction of the criterion 
    # brought by that feature. It is also known as the Gini importance )
    print(
        pd.DataFrame(
            dtree.feature_importances_, columns = ["Imp"], index = x_train.columns
        ).sort_values(by = "Imp", ascending = False)
    )

    # Observations
    # --------------
    # Power, Year and km_per_unit_fuel are the top 3 important features of decision tree model.
    return dtree

def random_forest():
    # see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

    ind_vars = df.drop(["Price", "price_log"], axis = 1)
    dep_var = df[["price_log", "Price"]]
    ind_vars_num = encode_cat_vars(ind_vars)

    x_train, x_test, y_train, y_test = train_test_split(
        ind_vars_num, dep_var, test_size = 0.3, random_state = 1
    )

    # Create a Random Forest regression model 
    rf = RandomForestRegressor(random_state = 1, oob_score = True)

    # Fit Random Forest regression model.
    rf.fit(x_train, y_train["price_log"])

    # Get score of the model.
    RandomForest_model = get_model_score(x_train, x_test, y_train, y_test, model=rf)

    # Observations:
    # --------------
    # Random Forest model has performed well on training and test set and we can see the model has overfitted slightly.

    # Feature Importance
    # Print the importance of features in the tree building ( The importance 
    # of a feature is computed as the (normalized) total reduction of the 
    # criterion brought by that feature. It is also known as the Gini importance )
    print(
        pd.DataFrame(
            rf.feature_importances_, columns = ["Imp"], index = x_train.columns
        ).sort_values(by = "Imp", ascending = False)
    )

    # Observations
    # --------------
    # Power, Year and km_per_unit_fuel are some of the important features of random forest model.
    return rf

def hyperparameter_tuning_decision_tree():

    ind_vars = df.drop(["Price", "price_log"], axis = 1)
    dep_var = df[["price_log", "Price"]]
    ind_vars_num = encode_cat_vars(ind_vars)

    x_train, x_test, y_train, y_test = train_test_split(
        ind_vars_num, dep_var, test_size = 0.3, random_state = 1
    )

    # Choose the type of regressor.
    dtree_tuned = DecisionTreeRegressor(random_state = 1)

    # Grid of parameters to choose from
    parameters = {
        "max_depth": list(np.arange(2, 25, 5)) + [None],
        "min_samples_leaf": [1, 3, 5, 7],
        "max_leaf_nodes": [2, 5, 7] + [None],
    }

    # Type of scoring used to compare parameter combinations
    scorer = metrics.make_scorer(metrics.r2_score)

    # Run the grid search
    grid_obj = GridSearchCV(dtree_tuned, parameters, scoring = scorer, cv = 5)
    grid_obj = grid_obj.fit(x_train, y_train["price_log"])

    # Set the clf to the best combination of parameters
    dtree_tuned = grid_obj.best_estimator_

    # Fit the best algorithm to the data.
    dtree_tuned.fit(x_train, y_train["price_log"])

    # Get score of the dtree_tuned
    dtree_tuned_score = get_model_score(x_train, x_test, y_train, y_test, model=dtree_tuned)

    # Observations:
    # --------------
    # Overfitting in decision tree is still there.
    pd.DataFrame(dtree_tuned.feature_importances_, columns = ["Imp"], index = x_train.columns).sort_values(by = "Imp", ascending = False).plot(kind = 'bar')

    # Observations:
    # --------------
    # Power, Year and new_price_num are the top 3 important features of decision tree model.

    return dtree_tuned

def hyperparameter_tuning_random_forest():
    ind_vars = df.drop(["Price", "price_log"], axis = 1)
    dep_var = df[["price_log", "Price"]]
    ind_vars_num = encode_cat_vars(ind_vars)

    x_train, x_test, y_train, y_test = train_test_split(
        ind_vars_num, dep_var, test_size = 0.3, random_state = 1
    )

    # Choose the type of regressor
    rf_tuned = RandomForestRegressor(random_state = 1)

    # Grid of parameters to choose from
    parameters = {
        "max_depth": [5, 6],
        "max_features": ["sqrt", "log2"],
        "n_estimators": [300, 500, 900, 1000],
    }

    # Type of scoring used to compare parameter combinations
    scorer = metrics.make_scorer(metrics.r2_score)

    # Run the grid search
    grid_obj = GridSearchCV(rf_tuned, parameters, scoring = scorer, cv = 5)
    grid_obj = grid_obj.fit(x_train, y_train["price_log"])

    # Set the clf to the best combination of parameters
    rf_tuned = grid_obj.best_estimator_

    # Fit the best algorithm to the data.
    rf_tuned.fit(x_train, y_train["price_log"])

    # Get score of the model.
    rf_tuned_score = get_model_score(x_train, x_test, y_train, y_test, model=rf_tuned)

    # Observations:
    # --------------
    # There's still scope for improvement with tuning the hyperparameters of the Random Forest.

    # Feature Importance
    pd.DataFrame(rf_tuned.feature_importances_, columns = ["Imp"], index = x_train.columns).sort_values(by = 'Imp', ascending = False).plot(kind = 'bar')

    # Observations:
    # ---------------
    # new_price_num, power_num, engine_num, and Year are the top 4 important 
    # variables in predicting car price according to Random Forest.

    return rf_tuned

def model_comparison():
    # Defining list of models
    rdg = ridge_regression_model()
    dtree = decision_tree()
    dtree_tuned = hyperparameter_tuning_decision_tree()
    rf = random_forest()
    rf_tuned = hyperparameter_tuning_random_forest()
    models = [rdg, dtree, dtree_tuned, rf, rf_tuned]

    # Defining empty lists to add train and test results
    r2_train = []
    r2_test = []
    rmse_train = []
    rmse_test = []

    # Looping through all the models to get the rmse and r2 scores
    for model in models:
        # accuracy score
        j = get_model_score(x_train=None, x_test=None, y_train=None, y_test=None, model=model, flag = False)
        r2_train.append(j[0])
        r2_test.append(j[1])
        rmse_train.append(j[2])
        rmse_test.append(j[3])

    comparison_frame = pd.DataFrame(
        {
            "Model": [
                "Ridge Regression",
                "Decision Tree",
                "Tuned Decision Tree",
                "Random Forest",
                "Tuned Random Forest",
            ],
            "Train_r2": r2_train,
            "Test_r2": r2_test,
            "Train_RMSE": rmse_train,
            "Test_RMSE": rmse_test,
        }
    )
    comparison_frame

    # observations:
    # ---------------
    # Ridge Regression and Linear Regression have performed very well on data. 
    # However, tuned Decision tree has performed better on training and test set. 
    # There is slight overfitting, if we can tune it better we can remove it.

"""
Refined insights:
Name: 
- The Name column has 2041 unique values and this column would not be very useful in our analysis. But the name contains both the brand name and the model name of the vehicle and we can process this column to extract Brand and Model names to reduce the number of levels.

Extracting the car brands: 
- After extracting the car brands from the name column we find that the most frequent brand in our data is Maruti and Hyundai.

Extracting car model name:
- After extracting the car name it gets clear that our dataset contains used cars from luxury as well as budget-friendly brands.
- The mean price of a used Lamborghini is 120 Lakhs and that of cars from other luxury brands follow in descending order and this output is very close to our expectation (domain knowledge), in terms of brand order. Towards the bottom end, we have more budget-friendly brands.

Important variable with Random Forest:
- According to the Random Forest model the most significant predictors of the price of used cars are
    Power of the engine
    The year of manufacturing
    Engine
    New_Price

Business Insights and Recommendations
- Some southern markets tend to have higher prices. It might be a good strategy to plan growth in southern cities using this information. Markets like Kolkata (coeff = -0.2) are very risky and we need to be careful about investments in this area.
- We will have to analyze the cost side of things before we can talk about profitability in the business. We should gather data regarding that.
- The next step would be to cluster different sets of data and see if we should make multiple models for different locations/car types.

Proposal for the final solution design:
- Our final tuned Decision model has an R-squared of ~0.89 on the test data, which means that our model can explain 89% variation in our data also the RMSE on test data is ~3.75 which means we can predict very closely to the original values. This is a very good model and we can use this model in production.
"""

# Python execution main program entry point
def main():
    args = sys.argv[1:]
    # args is a list of the command line args
