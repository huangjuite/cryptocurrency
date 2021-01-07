from binance.client import Client
import config

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
    symbol="ETHUSDT", interval=client.KLINE_INTERVAL_1MINUTE)

for k in klines:
    print(k)
