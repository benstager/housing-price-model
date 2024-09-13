import pandas as pd
from skimpy import skim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from neuralforecast import NeuralForecast
from mlforecast import MLForecast
from statsforecast.models import Naive, SeasonalNaive, SimpleExponentialSmoothing, RandomWalkWithDrift
from statsforecast import StatsForecast
from neuralforecast.models import LSTM, GRU, NBEATS, TFT, MLP
from neuralforecast.losses.pytorch import MAE, MSE
from dateutil.relativedelta import relativedelta
import time

def plot_ts(df):
    code = df['postcode'].values[0]
    property_type = df['propertyType'].values[0]
    plt.plot(df['full_date'], df['price/bedroom'])
    plt.title(f'Price/Bedroom vs. Date Sold for Post Code {code} and Housing Type {property_type}')
    plt.xlabel('Date Sold')
    plt.ylabel('Price/Bedroom')

def predict_and_visualize_forecast(ts, year, bedrooms):
    date1 = pd.to_datetime(ts['full_date'].values[-1:][0])
    date2 = pd.to_datetime(f'{year+1}-12-1') 

    difference = relativedelta(date2, date1)
    months_difference = difference.years * 12 + difference.months
    model_str = 'Neural Forecast'
    horizon = months_difference
    preds = run_NeuralForecast(ts, horizon)['predictions']
    code = ts['postcode'].values[0]
    prop = ts['propertyType'].values[0]

    best_price = min(preds[preds['ds'].apply(lambda x: x.year == year)]['MLP'])
    best_date = pd.to_datetime(preds[preds['MLP'] == best_price]['ds'].values[0])
    total_price = best_price*bedrooms

    print()
    print()
    print()
    print()
    print('--------------------------------------------------')
    print(f'Best month to buy: {best_date.month}/{best_date.day}/{best_date.year}')
    print(f'Price per bedroom: ${np.round(best_price, 2)}')
    print(f'Total cost for {bedrooms} bedrooms: ${np.round(total_price,2)}')
    print(best_date)
    print('--------------------------------------------------')

    plt.plot(ts['full_date'], ts['price/bedroom'])
    plt.plot(preds['ds'], preds['MLP'])
    plt.xlabel('date')
    plt.ylabel('price/bedroom')
    plt.title(f'Forecast for Price/Bedroom for Postcode {code} and Housing Type {prop}')
    plt.legend(['history', 'forecast'])
    plt.show()

def prepare_and_clean_frames(master_df, post_code, housing_type):

    master_df['datesold'] = master_df['datesold'].apply(lambda x: pd.to_datetime(x))

    master_df['datesold_year'] = master_df['datesold'].apply(lambda x: x.year)
    master_df['datesold_month'] = master_df['datesold'].apply(lambda x: x.month)
    master_df['datesold_day'] = master_df['datesold'].apply(lambda x: x.day)
    
    unit_code = master_df[(master_df['postcode'] == post_code) & (master_df['propertyType'] == housing_type)].sort_values('datesold')

    unit_code_final = unit_code.groupby(['datesold_year', 'datesold_month']).agg({
    'postcode':pd.Series.mode, 'price':'sum', 'propertyType':pd.Series.mode, 'bedrooms':'sum', 
    }).reset_index()

    unit_code_final['full_date'] = pd.to_datetime(
        unit_code_final['datesold_year'].astype(str) + '-' + unit_code_final['datesold_month'].astype(str) + '-01'
        )
    unit_code_final['price/bedroom'] = unit_code_final['price']/unit_code_final['bedrooms']

    unit_code_final = unit_code_final.drop(['datesold_year', 'datesold_month','price', 'bedrooms'], axis=1)

    date_range = pd.date_range(start=unit_code_final['full_date'].min(), end=unit_code_final['full_date'].max(), freq='MS')
    missing_dates = date_range.difference(unit_code_final['full_date'])

    missing_df = pd.DataFrame({
        'postcode':post_code,
        'propertyType':housing_type,
        'full_date': missing_dates,
        'price/bedroom':np.nan
    })

    full_data = pd.concat([missing_df,unit_code_final]).sort_values('full_date')

    full_data['price/bedroom'] = full_data['price/bedroom'].ffill()
    
    if np.inf in full_data['price/bedroom'].values:
        with_inf = full_data[full_data['price/bedroom'] == np.inf]
        without_inf = full_data[full_data['price/bedroom'] != np.inf]
        with_inf['price/bedroom'] = np.mean(without_inf['price/bedroom'])
        full_data = pd.concat([with_inf, without_inf])

    return full_data

def run_BaselineForecast(df):

    X_train, X_test, y_train, y_test = train_test_split(df['full_date'], df['price/bedroom'], test_size=.25, shuffle=False)
    y_preds_test = np.tile(y_train.values[-1:], X_test.shape[0])
    error = mean_absolute_error(y_preds_test, y_test)
    model = 'Naive'
    
    kwargs = {
        'model':model,
        'error':error,
        'X_train':X_train,
        'X_test':X_test,
        'y_train':y_train,
        'y_test':y_test,
        'y_test_preds':y_preds_test,
        'df':df
    }

    return kwargs

def run_NeuralForecast(df, months):
    X_train, X_test, y_train, y_test = train_test_split(df['full_date'], df['price/bedroom'], test_size=.25, shuffle=False)

    df_train_NIXTLA = pd.DataFrame({
        'unique_id':0,
        'ds': X_train,
        'y': y_train
    })

    df_test_NIXTLA = pd.DataFrame({
        'unique_id':0,
        'ds':X_test,
        'y':y_test
    })

    models = [MLP(
        h=df_test_NIXTLA.shape[0],
        input_size=12,
        max_steps=100,
        random_seed=22,
        loss=MAE(),
    )]

    model = NeuralForecast(models=models, freq='MS')
    model.fit(df_train_NIXTLA)
    preds_test = model.predict()

    error = mean_absolute_error(df_test_NIXTLA['y'].values, preds_test['MLP'])

    models = [MLP(
        h = months,
        input_size=12,
        max_steps=100,
        random_seed=22,
        loss=MAE(),
    )]

    model = NeuralForecast(models=models, freq='MS')
    model.fit(pd.concat([df_train_NIXTLA, df_test_NIXTLA]))
    y_preds = model.predict()

    kwargs = {
        'model':model,
        'error':error,
        'df_train':df_train_NIXTLA,
        'df_test':df_test_NIXTLA,
        'y_test_preds':preds_test,
        'df':df,
        'predictions':y_preds
    }
    
    return kwargs

if __name__ == '__main__':

    df = pd.read_csv('raw_sales.csv')
    all_postcodes = df['postcode'].unique()
    housing_type_1 = 'unit'
    housing_type_2 = 'house'

    ts_sets = {}

    for code in all_postcodes:
        try:
            code_house = prepare_and_clean_frames(df, code, 'house')
            code_unit = prepare_and_clean_frames(df, code, 'unit')
            ts_sets[code] = [code_house, code_unit]
        except:
            ts_sets[code] = 'NOT ENOUGH DATA'

    print('---------------------------------')
    print('Entering forecasting model.....')
    print('---------------------------------')

    print('DIRECTORY OF POSTCODES:')
    print()
    for code in all_postcodes:
        print(code)

    user_post = int(input('Enter a postcode (press q to quit): '))
    user_type = input('Enter a housing type (unit or house): ')
    user_hrzn = int(input('What year would you like to buy a home?: '))
    user_bedrooms = int(input('How many bedrooms?: '))

    while user_post != 'q':
        if user_type == 'unit':
            idx = 1
        else:
            idx = 0

        predict_and_visualize_forecast(ts_sets[user_post][idx], user_hrzn, user_bedrooms)

        user_post = int(input('Enter a postcode (press q to quit): '))
        if user_post == 'q':
            quit()
        user_type = input('Enter a housing type (unit or house): ')
        user_hrzn = int(input('What year would you like to buy a home?: '))
        user_bedrooms = int(input('How many bedrooms?: '))