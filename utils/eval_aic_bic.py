###########################
#  Master thesis
# 
# UNIVERSITY: 
#  Faculty of Information Technology, Brno University of Technology
# 
# AUTHOR:
#  Aleš Kašpárek          <xkaspa48@stud.fit.vutbr.cz>
###########################

import os
import json
import pandas as pd

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

network_analytics = pd.read_csv('../data/Network_Analytics.csv')
series = pd.Series(network_analytics['OutboundUtilzation (%)'])

X = series.values
size = int(len(X) * 0.80)
train, test = X[0:size], X[size:len(X)]

def worker_func_ma(id : int):
    model = ARIMA(endog=train, order=(0, 0, id))
    result = model.fit()
    aic = result.aic
    bic = result.bic
    residuals = result.resid
    result = acorr_ljungbox(residuals, lags=[id])
    p_value = result.iloc[0,1]
    return [id, aic, bic, p_value]

dir = '../output/information_criteria/MA/'

for x in range(1, 61):
    print(f"Processing {x}")
    with open(f"{dir}/criterions_{x}.json", 'w') as json_file:
        json.dump(worker_func_ma(x), json_file)

data = []

for file in os.listdir(dir):
    full_file_name = dir + file
    with open(full_file_name, 'r') as json_file:
        new_data = json.load(json_file)
    data.extend(new_data)

data = sorted(data, key=lambda x : x[0])

with open(f'{dir}/criteria.json', 'w') as json_file:
    json.dump(data, json_file) 

dir = '../output/information_criteria/AR/'

def worker_func_ar(id : int):
    model = AutoReg(endog=train, lags=id)
    result = model.fit()
    aic = result.aic
    bic = result.bic
    residuals = result.resid
    result = acorr_ljungbox(residuals, lags=[id])
    p_value = result.iloc[0,1]
    return [id, aic, bic, p_value]

for x in range(1, 2001):
    print(f"Processing {x}")
    with open(f"{dir}/criterions_{x}.json", 'w') as json_file:
        json.dump(worker_func_ar(x), json_file)

data = []

for file in os.listdir(dir):
    full_file_name = dir + file
    with open(full_file_name, 'r') as json_file:
        new_data = json.load(json_file)
    data.extend(new_data)

data = sorted(data, key=lambda x : x[0])

with open(f'{dir}/criteria.json', 'w') as json_file:
    json.dump(data, json_file) 