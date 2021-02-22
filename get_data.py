from binance.client import Client
import config
import mplfinance as mpf
import numpy as np
import pandas as pd
import datetime
import json
from utils import *

client = Client(config.API_KEY, config.API_SECRET)
symbol = 'ETHUSDT'

startTime = "1 Jan, 2015"
print('downloading')
klines = client.get_historical_klines(
    symbol=symbol, interval=client.KLINE_INTERVAL_15MINUTE, start_str=startTime)
print('download finished')

print('processing')
df = klines_to_dataframe(klines)
print(df)

df.to_csv('data/%s_historical_klines_15m.csv'%symbol)