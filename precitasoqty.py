import pandas as pd
from fbprophet import Prophet

dataFile = pd.read_csv('files/Mar-Nov-18-eCom-SOQty.csv')
dataFile.head()


m = Prophet()
m.fit(dataFile)

future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


fig1 = m.plot(forecast)
fig1.savefig('forecastSOQty.png')

fig2 = m.plot_components(forecast)
fig2.savefig('forecastComponentsSOQty.png')


