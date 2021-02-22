from binance.client import Client
import config
import mplfinance as mpf
import numpy as np
import pandas as pd
import datetime
from utils import *

client = Client(config.API_KEY, config.API_SECRET)
symbol = 'ETHBUSD'

# get market depth
# depth = client.get_order_book(symbol='BNBBTC')

# place a test market buy order, to place an actual order use the create_order function
# order = client.create_test_order(
#     symbol='ETHBUSD',
#     side=Client.SIDE_BUY,
#     type=Client.ORDER_TYPE_MARKET,
#     quantity=256)


klines = client.get_klines(
    symbol=symbol, interval=client.KLINE_INTERVAL_30MINUTE, limit=512)

draw_klines(klines, print_df=True, mav=(6, 9))