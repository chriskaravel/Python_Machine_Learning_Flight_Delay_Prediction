import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import  style
import pickle

## NOTE : This is an example of delay prediction with pd.DataFrames

# load the data from the csv file.if our data is seperated by semicolons we need to do sep=";"
data = pd.read_csv("Lots_of_flight_data1.csv")

data = data[["CRS_DEP_TIME","DEP_TIME","DEP_DELAY","CRS_ARR_TIME","ARR_TIME","ARR_DELAY","CRS_ELAPSED_TIME","ACTUAL_ELAPSED_TIME","AIR_TIME","DISTANCE"]]

# data with no NaN values
data_no_nulls = data.dropna()

# X is our features we use to try and do our prediction
X = data_no_nulls.loc[:,["CRS_DEP_TIME","DEP_TIME","DEP_DELAY","CRS_ARR_TIME","ARR_TIME","CRS_ELAPSED_TIME","ACTUAL_ELAPSED_TIME","AIR_TIME","DISTANCE"]]

# y is the value we try to predict
y = data_no_nulls.loc[:,["ARR_DELAY"]]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE

best=0
for x in range(10):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    regressor = LinearRegression()
    # train the model using the training data
    regressor.fit(x_train, y_train)
    acc = regressor.score(x_test, y_test)
    predictions = regressor.predict(x_test)
    # print("accuracy: \n", acc)
    # print("r^2: \n", metrics.r2_score(y_test, predictions))
    if acc > best:
        best=acc
        # save our model with pickle
        # you save model if u have hundreds of thousand data you dont want to retrain the model every time
        # so you save that model if it has a good accuracy
        with open("flight_model.pickle", 'wb') as f:
            pickle.dump(regressor, f)

#you load your saved model
pickle_in = open("flight_model.pickle","rb")
regressor=pickle.load(pickle_in)

# load the data from the csv file
data = pd.read_csv("Lots_of_flight_data2.csv")

data = data[["CRS_DEP_TIME","DEP_TIME","DEP_DELAY","CRS_ARR_TIME","ARR_TIME","ARR_DELAY","CRS_ELAPSED_TIME","ACTUAL_ELAPSED_TIME","AIR_TIME","DISTANCE"]]

# data with no NaN values
data_no_nulls = data.dropna()

# X is our features we use to try and do our prediction
X = data_no_nulls.loc[:,["CRS_DEP_TIME","DEP_TIME","DEP_DELAY","CRS_ARR_TIME","ARR_TIME","CRS_ELAPSED_TIME","ACTUAL_ELAPSED_TIME","AIR_TIME","DISTANCE"]]

# y is the value we try to predict
y = data_no_nulls.loc[:,["ARR_DELAY"]]

predictions = regressor.predict(X)
predictions_df = pd.DataFrame(predictions)
predictions_df.columns=['Predicted Delay']
# Reset the index values to the second dataframe appends properly
y_test_df = pd.DataFrame(y).reset_index(drop=True)

#Concat the two dataframes
merged_df = pd.concat([predictions_df, y_test_df],axis=1)
#show all rows
pd.set_option('display.max_rows', None)
print(merged_df)



