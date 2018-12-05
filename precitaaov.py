import pandas as pd

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics

dataFile = pd.read_csv('files/Mar-Nov-18-eCom-aov.csv')
dataFile.head()
# adding the outliers into the model
dataFile.loc[(dataFile['ds'] == '20-10-2018'), 'y'] = None
dataFile.loc[(dataFile['ds'] == '26/11/2018'), 'y'] = None
dataFile.loc[(dataFile['ds'] == '27/11/2018'), 'y'] = None

prophet = Prophet(
    growth='linear',
    seasonality_mode='additive')

prophet.fit(dataFile)

future = prophet.make_future_dataframe(freq='D', periods=30*6)
future.tail()

forecast = prophet.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = prophet.plot(forecast)
fig1.savefig('forecastAOV.png')

fig2 = prophet.plot_components(forecast)
fig2.savefig('forecastComponentsAOV.png')

cross_validation_results = cross_validation(prophet, initial='210 days', period='15 days', horizon='70 days')
print cross_validation_results

performance_metrics_results = performance_metrics(cross_validation_results)
print performance_metrics_results

