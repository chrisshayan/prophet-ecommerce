import pandas as pd
from fbprophet import Prophet

aovDataFile = pd.read_csv('files/Mar-Nov-18-eCom-aov.csv')
aovDataFile.head()


m = Prophet()
m.fit(aovDataFile)

future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


fig1 = m.plot(forecast)
fig1.savefig('forecastAOV.png')

fig2 = m.plot_components(forecast)
fig2.savefig('forecastComponentsAOV.png')


