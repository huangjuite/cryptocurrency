import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import io
import cv2


def klines_to_dataframe(klines):
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

    return data


def draw_klines(klines, print_df=False, mav=None):

    data = klines_to_dataframe(klines)

    if print_df:
        print(data)

    if mav is None:
        mpf.plot(data, type='candle', style='charles', volume=True)
    else:
        mpf.plot(data, type='candle', style='charles', mav=mav, volume=True)


def draw_klines_np(klines, start, end, periods):
    datelist = pd.date_range(start, end, periods=periods)
    data = pd.DataFrame(data=klines, columns=[
                        "Open", "High", "Low", "Close", "Volume"])
    data = data.set_index(datelist)
    mpf.plot(data, type='candle', style='charles', volume=True)


def draw_dataset(x, y, seq, save=False):
    t0 = datetime.date.today()
    t1 = t0 + datetime.timedelta(days=1)
    t2 = t1 + datetime.timedelta(days=1)
    t3 = t2 + datetime.timedelta(days=1)

    lines = []
    datelist = pd.date_range(t2, t3, seq)
    for v, t in zip(y, datelist):
        lines.append([str(t), v])

    datelist = pd.date_range(t0, t3, periods=seq*3)
    data = pd.DataFrame(data=x, columns=[
                        "Open", "High", "Low", "Close", "Volume"])
    data = data.set_index(datelist)

    if save:
        buf = io.BytesIO()
        mpf.plot(
            data,
            type='candle',
            style='charles',
            volume=True,
            savefig=buf,
            alines=dict(
                alines=lines,
                colors=['b'],
                linewidths=[0.1]
            )
        )
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.close()
        return img
    else:
        mpf.plot(
            data,
            type='candle',
            style='charles',
            volume=True,
            alines=dict(
                alines=lines,
                colors=['b'],
                linewidths=[0.1]
            )
        )
