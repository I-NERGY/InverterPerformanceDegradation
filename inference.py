import numpy as np
from numpy import nan

from numpy import split
from numpy import array
from joblib import load
import pandas as pd

from tensorflow.keras.models import load_model


def gen_window(df):
    # split data into 7-day sequences
    try:
        # select the last 7*model_weeks[-1] observations to make forecasts
        df = df[-7 * model_weeks[-1]:]

        # load the sklearn preprocessing parameters, then transform the affected columns
        cc = ["PR", "G_M0", "T", "E_N"]
        trans = load(SCALER_PATH)

        # suppress the SettingWithCopyWarning from pandas
        pd.options.mode.chained_assignment = None

        df.loc[:, cc[-(len(cc) - 1):]] = trans.transform(np.array(df[cc[-(len(cc) - 1):]]))
        df.loc[:, cc[0]] = df.loc[:, cc[0]].values / 100

        df = array(split(df, len(df) // 7))

        return df

    except Exception as e:
        print("Error: supply dataframe with length divisible by 7 and has at least ", 7 * model_weeks[-1], " observations \n")
        print(e)


def gen_forecast(test_df, train_weeks=(1, 2, 3)):
    col_len = len(test_df.columns)
    # select the last 7*model_weeks[-1] observations to make forecasts
    test_ = gen_window(test_df)
    print("test input shape: ", test_.shape)

    test_data = [test_[:ind, ...].reshape((1, ind, 1, LEN, col_len)) for ind in train_weeks]

    input_shapes = [arr.shape for arr in test_data]
    print("model input shapes: ", input_shapes)

    model = load_model(MODEL_PATH)

    output = model.predict(test_data)
    output *= 100
    print("\n performance ratio forcast: ", output.flatten().tolist())


if __name__ == '__main__':
    TEST_PATH = "./test.csv"
    MODEL_PATH = "./models/model"
    SCALER_PATH = "./models/std_scaler.bin"

    # days to use with CONVLSTM
    LEN = 7
    COLS_CHECK = ["PR", "G_M0", "T", "E_N"]
        
    testDf = pd.read_csv(TEST_PATH)
    testDf = testDf[COLS_CHECK]
    testDf.replace('?', nan, inplace=True)
    testDf = testDf.fillna(method="pad")
    testDf = testDf.astype('float32')

    # weeks used to stack models
    model_weeks = (1, 2)

    gen_forecast(testDf, model_weeks)