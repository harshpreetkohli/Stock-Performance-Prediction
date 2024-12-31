"""
Script is used to
1. Create data sets for training the models
2. Create test and validatoin sets
3. Train the ML models 
4. Predict the stock prices for the last 25 days
5. Store the preds for each stok in the DB table called preds
"""
#imports
import pandas as pd
import numpy as np
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


DB_PATH = "scripts/articles.db"
TRAINING_DATA_SPLIT = 0.90 # 90 percent of the data will be used as training set and 10 as validation
SYMBOLS = ["TSLA", "AAPL", "IBM"]

def create_data_sets(data: pd.DataFrame):
    """
    Create the training data set for training the models.
    Args:
        data(pd.DataFrame): data frame containing all the close_price data per day
    """
    data_set = data.filter(['close_price']).values
    training_data_len = int(len(data_set) * TRAINING_DATA_SPLIT) # input length of the traing data
    scalar = MinMaxScaler(feature_range=(0, 1))
    scaled_data  = scalar.fit_transform(data_set)
    train_data = scaled_data[: training_data_len, :]

    x_train = []
    y_train = []
    val_input_size = int(training_data_len * (1 - TRAINING_DATA_SPLIT)) # 25 is the val_input size
    #Take the training data from the fisrt 225 entries
    for i in range(val_input_size, len(train_data)):
        x_train.append(train_data[i-val_input_size:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    return x_train, y_train, scaled_data, scalar, training_data_len


def make_preds(model, scaled_data, scalar, training_data_len):
    """
    Will create the validation test set and run the prediction based on the model provided.
    """
    val_window_size = int(training_data_len * (1 - TRAINING_DATA_SPLIT))
    test_data = scaled_data[training_data_len - val_window_size:, :]
    x_test = []
    for i in range(val_window_size, len(test_data)):
        x_test.append(test_data[i-val_window_size:i, 0])

    x_test = np.array(x_test)

    preds = model.predict(x_test)
    preds = scalar.inverse_transform(preds.reshape(-1, 1))
    return preds


def write_to_db(data: pd.DataFrame, conn: sqlite3.Connection) -> None:
    """
    Write the predictions made by the models in the preds table of the DB
    """
    cursor = conn.cursor()
    for _, row in data.iterrows():
        query =  f"""
        INSERT INTO preds (date, stock_symbol, sector, close_price, linear_predictions, xgb_predictions, rfr_predictions) VALUES(
        '{row['date']}', '{row['stock_symbol']}', '{row['sector']}',
        {row['close_price']}, {row['linear_predictions']}, {row['xgb_predictions']}, {row['rfr_predictions']})"""
        cursor.execute(query)
    conn.commit()

def delete_previous_preds(conn: sqlite3.Connection) -> None:
    """
    Whenever the models are retrained we overwrite the old predictions.
    This function removes the old predictions from the table
    """
    cursor = conn.cursor()
    cursor.execute("DELETE FROM preds")
    conn.commit()
    cursor.close()
        

if __name__ == '__main__':
    conn = sqlite3.connect(DB_PATH)
    delete_previous_preds(conn)
    for symbol in SYMBOLS:
        data = pd.read_sql(f"SELECT * FROM stock_data WHERE stock_symbol = '{symbol}'", conn)
        x_train, y_train, scaled_data, scalar, training_data_len = create_data_sets(data)

        ## model definition
        Linear_regression_model = LinearRegression()
        XGBRegressor_model = XGBRegressor()
        RFR_model = RandomForestRegressor(n_estimators=100, random_state=42)


        print("---------------Training Linear Regression model-------------------")
        Linear_regression_model.fit(x_train, y_train)
        print("Accuracy: ", Linear_regression_model.score(x_train, y_train))
        print("---------------Training XGBoostRegressor Regression model-------------------")
        XGBRegressor_model.fit(x_train, y_train)
        print("Accuracy: ", XGBRegressor_model.score(x_train, y_train))
        print("---------------Training RFR Regression model-------------------")
        RFR_model.fit(x_train, y_train)
        print("Accuracy: ", XGBRegressor_model.score(x_train, y_train))
        print("---------------Finished Training----------------------")

        print("---------------Making Predictions---------------------")
        linear_preds = make_preds(Linear_regression_model, scaled_data, scalar, training_data_len)
        xgb_preds = make_preds(XGBRegressor_model, scaled_data, scalar, training_data_len)
        rfr_preds = make_preds(RFR_model, scaled_data, scalar, training_data_len)
        train = data[:training_data_len]
        validation = data[training_data_len:]

        validation["linear_predictions"] = linear_preds
        validation["xgb_predictions"] = xgb_preds
        validation["rfr_predictions"] = rfr_preds

        write_to_db(validation, conn)
