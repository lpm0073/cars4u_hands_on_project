#------------------------------------------------------------------------------
# written by:   Lawrence McDaniel
#               https://lawrencemcdaniel.com
#
# date:         oct-2022
#
# usage:        transcription of Cars4U hands-on session 2-Oct-2022
#               for Regression Analysis.
#------------------------------------------------------------------------------
import sys
import re
import warnings                                                  # Used to ignore the warning given as output of the code
warnings.filterwarnings('ignore')

from utils import (
    encode_cat_vars, 
    get_model_score,
    build_ols_model,
    rmse,
    mape,
    mae,
    model_pref,
    checking_vif,
    treating_multicollinearity
    )

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
from sklearn.impute import KNNImputer

import statsmodels.api as sm
import math
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
import pylab
import scipy.stats as stats

# module-level declarations
df = pd.read_csv("used_cars_data.csv")

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


def explore_the_data():
    # 1. Exploring the Data
    # Loading the data
    # Loading the data into Python to explore and understand it

    # keep in mind that you can open this file in MS Excel, if its
    # installed on your computer. In general, if the data file is 
    # less than say, 100,000 rows then this is usually the quickest
    # way to review the data file contents.
    print(f"There are {df.shape[0]} rows and {df.shape[1]} columns.")  # f-string

    # dump the first 10 rows of the data file
    df.head(10)

    # generate a report of the structure and meta data of the data file
    df.info()

    # Let us check if the units correspond to the fuel types as expected.
    df.groupby(by = ["Fuel_Type", "mileage_unit"]).size()

    # Notes:
    # -------------------
    # - S.No. is just an index for the data entry. In all likelihood, this 
    #   column will not be a significant factor in determining the price of the car. 
    #   Having said that, there are instances where the index of the data entry contains 
    #   the information about the time factor (an entry with a smaller index 
    #   corresponds to data entered years ago). Therefore, we will not drop 
    #   this variable just yet. Let us see if there is any relationship with 
    #   the price when we do a bivariate analysis.
    #
    # - Car names contain a lot of model information. Let us check how many
    #   individual names we have. If they are too many, we can process this 
    #   column to extract important information.
    #
    # - Mileage, Engine, and Power will also need some processing before we 
    #   are able to explore them. We'll have to extract numerical information 
    #   from these columns.
    #
    # - New Price column also needs some processing. This one also contains 
    #   strings and a lot of missing values.

def process_column_mileage():
    # - We have car mileage in two units, kmpl and km/kg.
    # - After quick research on the internet it is clear that these 2 units are used for cars of 2 different fuel types.
    # - kmpl - kilometers per litre - is used for petrol and diesel cars. -km/kg - kilometers per kg - is used for CNG and LPG-based engines.
    # - We have the variable Fuel_type in our data.

    # Create 2 new columns after splitting the mileage values.
    km_per_unit_fuel = []
    mileage_unit = []

    for observation in df["Mileage"]:
        if isinstance(observation, str):
            if (
                observation.split(" ")[0]
                .replace(".", "", 1)
                .isdigit()  # First element should be numeric
                and " " in observation  # Space between numeric and unit
                and (
                    observation.split(" ")[1]
                    == "kmpl"  # Units are limited to "kmpl" and "km/kg"
                    or observation.split(" ")[1] == "km/kg"
                )
            ):
                km_per_unit_fuel.append(float(observation.split(" ")[0]))
                mileage_unit.append(observation.split(" ")[1])
            else:
                # To detect if there are any observations in the column that do not follow
                # The expected format [number + ' ' + 'kmpl' or 'km/kg']
                print(
                    "The data needs further processing. All values are not similar ",
                    observation,
                )
        else:
            # If there are any missing values in the mileage column,
            # We add corresponding missing values to the 2 new columns
            km_per_unit_fuel.append(np.nan)
            mileage_unit.append(np.nan)
    
    # No print output from the function above. The values are all in the expected format or NaNs
    # Add the new columns to the data
    df["km_per_unit_fuel"] = km_per_unit_fuel
    df["mileage_unit"] = mileage_unit

    # Checking the new dataframe
    df.head(5)  # looks good!



def process_column_engine():
    # The data dictionary suggests that Engine indicates the displacement 
    # volume of the engine in CC. We will make sure that all the observations 
    # follow the same format - [numeric + " " + "CC"] and create a new 
    # numeric column from this column.

    # This time, lets use a regex to make all the necessary checks.
    # Regular Expressions, also known as “regex”, are used to match strings 
    # of text such as particular characters, words, or patterns of characters. 
    # It means that we can match and extract any string pattern from the text 
    # with the help of regular expressions.
    # regex_engine

    # Create a new column after splitting the engine values.
    engine_num = []

    # Regex for numeric + " " + "CC"  format
    regex_engine = "^\d+(\.\d+)? CC$"

    for observation in df["Engine"]:
        if isinstance(observation, str):
            if re.match(regex_engine, observation):
                engine_num.append(float(observation.split(" ")[0]))
            else:
                # To detect if there are any observations in the column that do not follow [numeric + " " + "CC"]  format
                print(
                    "The data needs furthur processing. All values are not similar ",
                    observation,
                )
        else:
            # If there are any missing values in the engine column, we add missing values to the new column
            engine_num.append(np.nan)

    # No print output from the function above. The values are all in the same format - [numeric + " " + "CC"] OR NaNs
    # Add the new column to the data
    df["engine_num"] = engine_num

    # Checking the new dataframe
    df.head(5)

def process_column_power():
    # The data dictionary suggests that Power indicates the maximum power of the 
    # engine in bhp. We will make sure that all the observations follow the 
    # same format - [numeric + " " + "bhp"] and create a new numeric column 
    # from this column, like we did for Engine.

    # Create a new column after splitting the power values
    power_num = []

    # Regex for numeric + " " + "bhp"  format
    regex_power = "^\d+(\.\d+)? bhp$"

    for observation in df["Power"]:
        if isinstance(observation, str):
            if re.match(regex_power, observation):
                power_num.append(float(observation.split(" ")[0]))
            else:
                power_num.append(np.nan)
        else:
            # If there are any missing values in the power column, we add missing values to the new column
            power_num.append(np.nan)

    # Add the new column to the data
    df["power_num"] = power_num

    # Checking the new dataframe
    df.head(10)  # Looks good now

def process_column_new_price():
    # We know that New_Price is the price of a new car of the same model in 
    # INR Lakhs (1 Lakh = 100, 000).
    #
    # This column clearly has a lot of missing values. We will impute the 
    # missing values later. For now we will only extract the numeric 
    # values from this column.
    new_price_num = []

    # Regex for numeric + " " + "Lakh"  format
    regex_power = "^\d+(\.\d+)? Lakh$"

    for observation in df["New_Price"]:
        if isinstance(observation, str):
            if re.match(regex_power, observation):
                new_price_num.append(float(observation.split(" ")[0]))
            else:
                # Converting values in Crore to lakhs
                new_price_num.append(float(observation.split(" ")[0]) * 100)
        else:
            # If there are any missing values in the New_Price column, we add missing values to the new column
            new_price_num.append(np.nan)

    # Add the new column to the data
    df["new_price_num"] = new_price_num

    # Checking the new dataframe
    df.head(5)  # Looks ok

def feature_engineering_name():
    # Extract Brand Names
    df["Brand"] = df["Name"].apply(lambda x: x.split(" ")[0].lower())
    df["Brand"].value_counts()
    plt.figure(figsize = (15, 7))
    sns.countplot(y = "Brand", data = df, order = df["Brand"].value_counts().index)

    # Extract Model Names
    df["Model"] = df["Name"].apply(lambda x: x.split(" ")[1].lower())
    df["Model"].value_counts()
    plt.figure(figsize = (15, 7))
    sns.countplot(y = "Model", data = df, order = df["Model"].value_counts().index[0:30])

def feature_engineering_category():
    # It is clear from the above charts that our dataset contains used cars 
    # from luxury as well as budget-friendly brands.
    # We can create a new variable using this information. 
    # We will bin all our cars in 3 categories:
    # - Budget-Friendly
    # - Mid Range
    # - Luxury Cars
    df.groupby(["Brand"])["Price"].mean().sort_values(ascending = False)

    # The output is very close to our expectation (domain knowledge), in terms 
    # of brand ordering. The mean price of a used Lamborghini is 120 Lakhs and 
    # that of cars from other luxury brands follow in descending order.
    #
    # Towards the bottom end we have the more budget-friendly brands.
    #
    # We can see that there is some missingness in our data. 
    # Let us come back to creating this variable once we have removed 
    # missingness from the data.

    df.describe().T
    # Observations:
    # ----------------------
    # 1. S.No. has no interpretation here but as discussed earlier let us drop it only after having looked at the initial linear model.
    # 2. Kilometers_Driven values have an incredibly high range. We should check a few of the extreme values to get a sense of the data.
    # 3. Minimum and the maximum number of seats in the car also warrant a quick check. On average a car seems to have 5 seats, which is right.
    # 4. We have used cars being sold at less than a lakh rupees and as high as 160 lakh, as we saw for Lamborghini earlier. 
    #    We might have to drop some of these outliers to build a robust model.
    # 5. Min Mileage being 0 is also concerning, we'll have to check what is going on.
    # 6. Engine and Power mean and median values are not very different. Only someone with more domain knowledge would be able to comment further on these attributes.
    # 7. New price range seems right. We have both budget-friendly Maruti cars and Lamborghinis in our stock. 
    #    Mean being twice that of the median suggests that there are only a few very high range brands, which again makes sense.

    # Check Kilometers_Driven Extreme values
    df.sort_values(by = ["Kilometers_Driven"], ascending = True).head(10)

    # Let us check if we have a similar car in our dataset.
    df[df["Name"].str.startswith("Audi A4")]

    # Let us replace #seats in row index 3999 form 0 to 5
    df.loc[3999, "Seats"] = 5.0

    # Check Mileage - km_per_unit_fuel extreme values
    df.sort_values(by = ["km_per_unit_fuel"], ascending = True).head(10)

    # Check Mileage - km_per_unit_fuel extreme values
    df.sort_values(by = ["km_per_unit_fuel"], ascending = False).head(10)

def non_numeric_features():
    # Looking at value counts for non-numeric features
    num_to_display = 10  # Defining this up here so it's easy to change later

    for colname in df.dtypes[df.dtypes == "object"].index:
        val_counts = df[colname].value_counts(dropna = False)  # Will also show the NA counts
        
        print(val_counts[:num_to_display])
        
        if len(val_counts) > num_to_display:
            print(f"Only displaying first {num_to_display} of {len(val_counts)} values.")
        print("\n\n")  # Just for more space in between

def exception_electric_cars():
    # We had checked cars of different **`Fuel_Type`** earlier, but we did not 
    # encounter the 2 electric cars. Let us check why.
    df.loc[df["Fuel_Type"] == "Electric"]

    # observations:
    # ---------------
    # Mileage values for these cars are NaN, that is why we did not encounter 
    # these earlier with groupby.
    #
    # Electric cars are very new in the market and very rare in our dataset. 
    # We can consider dropping these two observations if they turn out to be 
    # outliers later. There is a good chance that we will not be able to 
    # create a good price prediction model for electric cars, with the 
    # currently available data.

    # Checking missing values in the dataset
    df.isnull().sum()

    # observations:
    # ---------------------
    # - 2 Electric car variants don't have entries for Mileage.
    # - Engine displacement information of 46 observations is missing and a maximum power of 175 entries is missing.
    # - Information about the number of seats is not available for 53 entries.
    # - New Price as we saw earlier has a huge missing count. We'll have to see if there is a pattern here.
    # - Price is also missing for 1234 entries. Since price is the response variable that we want to predict, we will have to drop these rows when we build a model. These rows will not be able to help us in modeling or model evaluation. But while we are analyzing the distributions and doing missing value imputations, we will keep using information from these rows.
    # - New Price for 6247 entries is missing. We need to explore if we can impute these or if we should drop this column altogether.

def drop_redundant_columns():
    df.drop(
        columns=["Mileage", "mileage_unit", "Engine", "Power", "New_Price"], inplace = True
    )

def plot_distribution_price():
    sns.distplot(df["Price"])
    # observation: This is a highly skewed distribution. 
    # Let us use log transformation on this column to see if that helps normalize the distribution.

    sns.distplot(np.log(df["Price"]), axlabel = "Log(Price)")

    # Creating a new column with the transformed variable.
    df["price_log"] = np.log(df["Price"])

    # price vs location
    plt.figure(figsize = (15, 7))
    sns.boxplot(x = "Location", y = "Price", data = df)
    # observation: Price of used cars has a large IQR in Coimbatore and Bangalore

def plot_distribution_km_driven():
    sns.distplot(df["Kilometers_Driven"])

    # Log transformation
    sns.distplot(np.log(df["Kilometers_Driven"]), axlabel = "Log(Kilometers_Driven)")
    df["kilometers_driven_log"] = np.log(df["Kilometers_Driven"])

def plot_distribution_bivariate():
    sns.pairplot(df, hue = "Fuel_Type")
    # We can get a lot of information by zooming in on these charts.
    # --------------
    # - Kilometers Driven does not appear to be related to price, contrary to popular belief.
    # - Price has a positive relationship with Year. The more recent the car, the higher the price.
    # - S.No. does not capture any information that we were hoping for. The temporal element of variation is captured in the year column.
    # - 2-seater cars are all luxury variants. Cars with 8-10 seats are exclusively mid to high range.
    # - Mileage does not seem to show much relationship with the price of used cars.
    # - Engine displacement and Power of the car have a positive relationship with the price.
    # - New Price and Used Car Price are also positively correlated, which is expected.
    # - Kilometers Driven has a peculiar relationship with the Year variable. Generally, the newer the car lesser the distance it has traveled, but this is not always true.
    # - CNG cars are conspicuous outliers when it comes to Mileage. The mileage of these cars is very high.
    # - Mileage and power of newer cars are increasing owing to advancements in technology.
    # - Mileage has a negative correlation with engine displacement and power. More powerful the engine, the more fuel it consumes in general.

def correlation_between_numeric_variables():
    plt.figure(figsize = (12, 7))
    sns.heatmap(df.corr(), annot = True, cmap = "coolwarm")
    # observations:
    # --------------
    # Power and engine are important predictors of price
    # We will have to work on imputing New Price missing values because this is 
    # a very important feature in predicting used car price accurately

def missing_value_treatment():
    # Checking missing values again
    df.isnull().sum()

    # Look at a few rows where seats is missing
    df[df["Seats"].isnull()]

    # We'll impute these missing values one by one, by taking median number of 
    # seats for the particular car, using the Brand and Model name.
    df.groupby(["Brand", "Model"], as_index = False)["Seats"].median()

    # Impute missing Seats
    df["Seats"] = df.groupby(["Brand", "Model"])["Seats"].transform(
        lambda x: x.fillna(x.median())
    )

    # Check missing values in 'Seats'
    df[df["Seats"].isnull()]

    # Maruti Estilo can accomodate 5
    df["Seats"] = df["Seats"].fillna(5.0)

    # We will use similar methods to fill missing values for engine, power and new price
    df["engine_num"] = df.groupby(["Brand", "Model"])["engine_num"].transform(
        lambda x: x.fillna(x.median())
    )
    df["power_num"] = df.groupby(["Brand", "Model"])["power_num"].transform(
        lambda x: x.fillna(x.median())
    )
    df["new_price_num"] = df.groupby(["Brand", "Model"])["new_price_num"].transform(
        lambda x: x.fillna(x.median())
    )

    df.isnull().sum()
    # observations:
    # -----------------
    # There are still some NAs in power and new_price_num.
    # There are a few car brands and models in our dataset that do not contain the new price information at all.

def knn_imputation_from_():
    # Now we'll have to estimate the new price using the other features. KNN 
    # imputation is one of the imputation methods that can be used for this. 
    # This sklearn method requires us to encode categorical variables if we 
    # are using them for imputation. In this case, we'll use only selected 
    # numeric features for imputation.
    #
    # see: https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
    imputer = KNNImputer(n_neighbors = 3, weights = "uniform")  # 3 Nearest Neighbours
    temp_df_for_imputation = [
        "engine_num",
        "power_num",
        "Year",
        "km_per_unit_fuel",
        "new_price_num",
        "Seats",
    ]

    temp_df_for_imputation = imputer.fit_transform(df[temp_df_for_imputation])
    temp_df_for_imputation = pd.DataFrame(
        temp_df_for_imputation,
        columns=[
            "engine_num",
            "power_num",
            "Year",
            "km_per_unit_fuel",
            "new_price_num",
            "Seats",
        ],
    )

    # Add imputed columns to the original dataset
    df["new_price_num"] = temp_df_for_imputation["new_price_num"]
    df["power_num"] = temp_df_for_imputation["power_num"]
    df["km_per_unit_fuel"] = temp_df_for_imputation["km_per_unit_fuel"]

    df.isnull().sum()
    # Drop the redundant columns.
    df.drop(columns = ["Kilometers_Driven", "Name", "S.No."], inplace = True)

    # Drop the rows where 'Price' == NaN and proceed to modelling
    df = df[df["Price"].notna()]

"""
-------------------------------------------------------------------------------
                           -- MODEL BEGINS HERE --
-------------------------------------------------------------------------------
"""
def model():
    ind_vars = df.drop(["Price", "price_log"], axis = 1)
    dep_var = df[["price_log", "Price"]]
    ind_vars_num = encode_cat_vars(pd, ind_vars)
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

    olsmodel1 = build_ols_model(sm, y_train, train=x_train)    # internal def
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
    model_pref(olsmodel1, x_train, x_test)  # High Overfitting.

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

    print(checking_vif(pd, x_train))    # internal def
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
    ind_vars_num = encode_cat_vars(pd, ind_vars)

    # Splitting data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        ind_vars_num, dep_var, test_size = 0.3, random_state = 1
    )

    print("Number of rows in train data =", x_train.shape[0])
    print("Number of rows in test data =", x_test.shape[0], "\n\n")

    # Statsmodel api does not add a constant by default. We need to add it explicitly.
    x_train = sm.add_constant(X_train)
    # Add constant to test data
    x_test = sm.add_constant(X_test)

    # Fit linear model on new dataset
    olsmodel2 = build_ols_model(sm, y_train, train=x_train)
    print(olsmodel2.summary())

    # The R squared and adjusted r squared values have decreased, but are still 
    # quite high indicating that we have been able to capture most of the 
    # information of the previous model even after reducing the number 
    # of predictor features.
    #
    # As we try to decrease overfitting, the r squared of our train model is expected to decrease.
    print(checking_vif(pd, x_train))

    # Checking model performance
    model_pref(olsmodel2, x_train, x_test)  # No Overfitting.
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
    treating_multicollinearity(pd, high_vif_columns, x_train, x_test, y_test)

    # Dropping cars_category would have the maximum impact on predictive power of the model (amongst the variables being considered)
    # We'll drop engine_num and check the vif again

    # Drop 'engine_num' from train and test
    col_to_drop = "engine_num"
    x_train = x_train.loc[:, ~x_train.columns.str.startswith(col_to_drop)]
    x_test = x_test.loc[:, ~x_test.columns.str.startswith(col_to_drop)]

    # Check VIF now
    vif = checking_vif(pd, x_train)
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
    treating_multicollinearity(pd, high_vif_columns, x_train, x_test, y_test)

    # Drop 'new_price_num' from train and test
    col_to_drop = "new_price_num"
    x_train = x_train.loc[:, ~x_train.columns.str.startswith(col_to_drop)]
    x_test = x_test.loc[:, ~x_test.columns.str.startswith(col_to_drop)]

    # Check VIF now
    vif = checking_vif(pd, x_train)
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
    olsmodel3 = build_ols_model(sm, y_train, train=x_train)
    print(olsmodel3.summary())

    print("\n\n")

    # Checking model performance
    model_pref(olsmodel3, x_train, x_test)

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
    ind_vars_num = encode_cat_vars(pd, ind_vars)
    ind_vars_num.head()

    x_train, x_test, y_train, y_test = train_test_split(
        ind_vars_num, dep_var, test_size = 0.3, random_state = 1
    )

    ##  Function to calculate r2_score and RMSE on train and test data

    # Extracting the rows from original data frame df where indexes are same as the training data
    original_df = df[df.index.isin(x_train.index.values)].copy()

    # Extracting predicted values from the final model
    olsmodel3 = build_ols_model(sm, y_train, train=x_train)

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
    ind_vars_num = encode_cat_vars(pd, ind_vars)

    X_train, X_test, y_train, y_test = train_test_split(
        ind_vars_num, dep_var, test_size = 0.3, random_state = 1
    )

    # Create a Ridge regression model
    rdg = Ridge()

    # Fit Ridge regression model.
    rdg.fit(X_train, y_train["price_log"])

    # Get score of the model.
    Ridge_score = get_model_score(X_train, X_test, y_train, y_test, np, metrics, model=rdg)

    # Observations
    # ----------------
    # Ridge regression is able to produce better results compared to Linear Regression.
    return rdg

def decision_tree():
    # see https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html

    ind_vars = df.drop(["Price", "price_log"], axis = 1)
    dep_var = df[["price_log", "Price"]]
    ind_vars_num = encode_cat_vars(pd, ind_vars)

    X_train, X_test, y_train, y_test = train_test_split(
        ind_vars_num, dep_var, test_size = 0.3, random_state = 1
    )

    # Create a decision tree regression model
    dtree = DecisionTreeRegressor(random_state = 1)

    # Fit decision tree regression model.
    dtree.fit(X_train, y_train["price_log"])

    # Get score of the model.
    Dtree_model = get_model_score(X_train, X_test, y_train, y_test, np, metrics, model=dtree)
    # Observations
    # ---------------
    # Decision Tree is overfitting on the training set and hence not able to generalize well on the test set.

    # Feature Importance
    # Print the importance of features in the tree building ( The importance of
    # a feature is computed as the (normalized) total reduction of the criterion 
    # brought by that feature. It is also known as the Gini importance )
    print(
        pd.DataFrame(
            dtree.feature_importances_, columns = ["Imp"], index = X_train.columns
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
    ind_vars_num = encode_cat_vars(pd, ind_vars)

    X_train, X_test, y_train, y_test = train_test_split(
        ind_vars_num, dep_var, test_size = 0.3, random_state = 1
    )

    # Create a Random Forest regression model 
    rf = RandomForestRegressor(random_state = 1, oob_score = True)

    # Fit Random Forest regression model.
    rf.fit(X_train, y_train["price_log"])

    # Get score of the model.
    RandomForest_model = get_model_score(rf)

    # Observations:
    # --------------
    # Random Forest model has performed well on training and test set and we can see the model has overfitted slightly.

    # Feature Importance
    # Print the importance of features in the tree building ( The importance 
    # of a feature is computed as the (normalized) total reduction of the 
    # criterion brought by that feature. It is also known as the Gini importance )
    print(
        pd.DataFrame(
            rf.feature_importances_, columns = ["Imp"], index = X_train.columns
        ).sort_values(by = "Imp", ascending = False)
    )

    # Observations
    # --------------
    # Power, Year and km_per_unit_fuel are some of the important features of random forest model.
    return rf

def hyperparameter_tuning_decision_tree():

    ind_vars = df.drop(["Price", "price_log"], axis = 1)
    dep_var = df[["price_log", "Price"]]
    ind_vars_num = encode_cat_vars(pd, ind_vars)

    X_train, X_test, y_train, y_test = train_test_split(
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
    grid_obj = grid_obj.fit(X_train, y_train["price_log"])

    # Set the clf to the best combination of parameters
    dtree_tuned = grid_obj.best_estimator_

    # Fit the best algorithm to the data.
    dtree_tuned.fit(X_train, y_train["price_log"])

    # Get score of the dtree_tuned
    dtree_tuned_score = get_model_score(dtree_tuned)

    # Observations:
    # --------------
    # Overfitting in decision tree is still there.
    pd.DataFrame(dtree_tuned.feature_importances_, columns = ["Imp"], index = X_train.columns).sort_values(by = "Imp", ascending = False).plot(kind = 'bar')

    # Observations:
    # --------------
    # Power, Year and new_price_num are the top 3 important features of decision tree model.

    return dtree_tuned

def hyperparameter_tuning_random_forest():
    ind_vars = df.drop(["Price", "price_log"], axis = 1)
    dep_var = df[["price_log", "Price"]]
    ind_vars_num = encode_cat_vars(pd, ind_vars)

    X_train, X_test, y_train, y_test = train_test_split(
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
    grid_obj = grid_obj.fit(X_train, y_train["price_log"])

    # Set the clf to the best combination of parameters
    rf_tuned = grid_obj.best_estimator_

    # Fit the best algorithm to the data.
    rf_tuned.fit(X_train, y_train["price_log"])

    # Get score of the model.
    rf_tuned_score = get_model_score(rf_tuned)

    # Observations:
    # --------------
    # There's still scope for improvement with tuning the hyperparameters of the Random Forest.

    # Feature Importance
    pd.DataFrame(rf_tuned.feature_importances_, columns = ["Imp"], index = X_train.columns).sort_values(by = 'Imp', ascending = False).plot(kind = 'bar')

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
        j = get_model_score(model, False)
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
