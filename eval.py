import json


import numpy as np
import pandas as pd

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

network_analytics = pd.read_csv('./data/Network_Analytics.csv')
series = pd.Series(network_analytics['OutboundUtilzation (%)'])

X = series.values
size = int(len(X) * 0.80)
train, test = X[0:size], X[size:len(X)]

def worker_func(id : int):
    model = ARIMA(endog=train, order=(0, 0, id))
    result = model.fit()
    aic = result.aic
    bic = result.bic
    residuals = result.resid
    result = acorr_ljungbox(residuals, lags=[id])
    p_value = result.iloc[0,1]
    return [id, aic, bic, p_value]

dir = 'output/information_criteria/MA/new'

for x in range(2000, 2001):
    print(f"Processing {x}")
    with open(f"{dir}/criterions_{x}.json", 'w') as json_file:
        json.dump(worker_func(x), json_file)
