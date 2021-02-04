from binance.client import Client
import config
import mplfinance as mpf
import numpy as np
import pandas as pd
import datetime

client = Client(config.API_KEY, config.API_SECRET)

# get market depth
depth = client.get_order_book(symbol='BNBBTC')


# place a test market buy order, to place an actual order use the create_order function
order = client.create_test_order(
    symbol='BNBBTC',
    side=Client.SIDE_BUY,
    type=Client.ORDER_TYPE_MARKET,
    quantity=100)


klines = client.get_klines(
    symbol="ETHUSDT", interval=client.KLINE_INTERVAL_30MINUTE, limit=100)


klines = np.array(klines)
klines = klines[:, :6]


data = pd.DataFrame(data=klines, columns=[
                    "Time", "Open", "High", "Low", "Close", "Volume"])
data["Time"] = data["Time"].astype('float')/1000.0
data["Open"] = data["Open"].astype('float')
data["High"] = data["High"].astype('float')
data["Low"] = data["Low"].astype('float')
data["Close"] = data["Close"].astype('float')
data["Volume"] = data["Volume"].astype('float')

data["Time"] = [datetime.datetime.fromtimestamp(x) for x in data["Time"]]
data.reset_index(drop=True, inplace=True)
data.set_index('Time', inplace=True)
print(data)

mpf.plot(data, type='candle', mav=(6,9), volume=True)
