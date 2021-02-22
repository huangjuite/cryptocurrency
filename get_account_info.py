from binance.client import Client
import config
import mplfinance as mpf
import numpy as np
import pandas as pd
import datetime
import json


client = Client(config.API_KEY, config.API_SECRET)

# info = client.get_account()
# print(json.dumps(info, indent=4))

balance = client.get_asset_balance(asset='ETH')
print(json.dumps(balance, indent=4))

# details = client.get_asset_details()
# print(json.dumps(details, indent=4))
