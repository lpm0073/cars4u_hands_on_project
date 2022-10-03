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

import numpy as np                                               # Basic libraries of python for numeric and dataframe computations
import pandas as pd

import matplotlib.pyplot as plt                                  # Basic library for data visualization
import seaborn as sns                                            # Slightly advanced library for data visualization

from sklearn.impute import KNNImputer

# module-level declarations
df = pd.read_csv("used_cars_data.csv")

"""
Project Part I: explore the data. Identify and remedy QC and completeness problems with the data file.
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


# Python execution main program entry point
def main():
    args = sys.argv[1:]
    # args is a list of the command line args
