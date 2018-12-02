import pandas as pd
import numpy  as np

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation


#  Take in true and predicted values then calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


dataFile = pd.read_csv('files/Mar-Nov-18-eCom-SOQty.csv')
dataFile.head()

prophet = Prophet()
prophet.fit(dataFile)

future = prophet.make_future_dataframe(periods=365)
future.tail()

forecast = prophet.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = prophet.plot(forecast)
fig1.savefig('forecastSOQty.png')

fig2 = prophet.plot_components(forecast)
fig2.savefig('forecastComponentsSOQty.png')

cross_validation_results = cross_validation(prophet, initial='30 days', period='15 days', horizon='5 days')
# map_baseline = mean_absolute_percentage_error(cross_validation_results, cross_validation_results.yhat)
# print map_baseline
