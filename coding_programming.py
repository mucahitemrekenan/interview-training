"""
Subtask 2.1: Write a Python function that takes a list of numbers and returns the mean, median, and standard deviation.
You can use libraries like NumPy if needed.
"""
import numpy as np


def calculator():
    user_input = input('please enter your numbers seperated by commas like 1,4,5,6,4: ')
    my_list = [float(x) for x in user_input.split(',')]
    mean = np.mean(my_list)
    median = np.median(my_list)
    std_dev = np.std(my_list)
    return mean, median, std_dev


mean_val, median_val, std_val = calculator()

"""
Subtask 2.2:Write a brief description of a data science project you would like to work on or have worked on in the past.
Include the problem statement, data used, and the approach you would take or have taken.

My Answer:
I worked 3 years on electricity demand prediction on a company. 
We predicted that the hourly electricity demand of a city for 1-4 days ahead. 
We had 4-5 years of hourly demand data as numerical. And this data was growing day by day as data came.
We cleaned the outlier points of data. 
And made lots of statistical analyses like mean, median, std deviation, linear trend etc. 
We applied these analyses for some periods as daily, 4 hours, 12 hours, weekly, monthly, and seasonally. 
Then extract lots of features from our target variable by using the moving average method. 
We used weather features location by location differently. 
After extracting lots of features we eliminated them by using statistical approaches like correlation,
and variance with target variable and correlation between features. 
Then we applied backward and forward elimination methods. 
Finally we used different machine learning methods like boosting, arima models, and neural networks. 
"""
