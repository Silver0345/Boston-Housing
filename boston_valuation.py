import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Gather Data
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
values = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
price = raw_df.values[1::2, 2]

features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']
# truning new values array into a dataframe
values = pd.DataFrame(data= values, columns=features)

# turning price into a dataframe
price = pd.DataFrame(data=price, columns=['PRICE'])
# merging data and target into a new dataframe
data = pd.concat([values, price], axis=1)
new_features = values.drop(['INDUS','AGE'],axis=1)
new_features.head()

log_prices = np.log(price)
target = log_prices

property_stats = new_features.mean().values.reshape(1,11)

ln_reg = LinearRegression().fit(new_features,target)
fitted_val = ln_reg.predict(new_features)

# MSE and RMSE 
mse = mean_squared_error(target, fitted_val)
rmse = np.sqrt(mse)

CRIM_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8
ZILLOW_MEDIAN_PRICE = 583.3
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE / np.median(price)

def get_log_estimate(numb_rooms,
                    students_per_classroom, next_to_river=False,
                    high_confidence=True):
    
    # Configure property
    property_stats[0][RM_IDX] = numb_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classroom
    
    if next_to_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0
    
    
    # Make prediction
    log_estimate = ln_reg.predict(property_stats)
    
    # Calc Range
    
    if high_confidence:
        upper_bound = log_estimate + 2*rmse
        lower_bound = log_estimate - 2*rmse
        interval = 95
    else:
        upper_bound = log_estimate + 2*rmse
        lower_bound = log_estimate - 2*rmse
        interval = 68
        
    return log_estimate, upper_bound, lower_bound, interval


def get_dollar_estimate(rm, ptratio, chas=False, large_range=True):
    """ Estiamte Price of Property In Boston
        
        Keyword Arguments:
        
        rm --- number of rooms in the property
        ptratios --- number of students per teacher in the classroom
        chas --- True if the property is next to Charles river, False otherwise.
        large_range --- True for a 95% prediction interval, False for a 68% prediction interval
    """
    
    if rm < 1 or ptratio < 1:
        print("The Value is Unrealistic. Please Try Again!")
        return
    
    log_est, upper, lower, conf = get_log_estimate(rm, 
                                                   students_per_classroom=ptratio, 
                                                   next_to_river=chas, 
                                                   high_confidence=large_range)

    # Convert to today's dollars
    dollar_est = np.e**log_est * 1000 * SCALE_FACTOR
    dollar_hi = np.e**upper * 1000 * SCALE_FACTOR
    dollar_low = np.e**lower * 1000 * SCALE_FACTOR


    # Round the dollar values to the nearest thousand
    round_est = np.around(dollar_est,-3)
    round_hi = np.around(dollar_hi,-3)
    round_low = np.around(dollar_low,-3)

    print(f"The estimated property value is ${round_est[0][0]}.")
    print(f"At {conf}% confidance the valuation range is.")
    print(f"USD ${round_low[0][0]} at the lower end to USD ${round_hi[0][0]} at the high end.")