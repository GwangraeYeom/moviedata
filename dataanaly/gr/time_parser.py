import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime

def timetodate():
    file = pd.read_csv('../data/ratings.csv')
    frame = pd.DataFrame(file)

    # print( datetime.datetime.fromtimestamp(964982400 / 1000))
    # print( datetime.datetime.fromtimestamp(964982224 / 1000))
    # print(datetime.datetime.fromtimestamp(964984100 / 1000))

    # print(file)
    # print(frame)
    date_data = []
    time_data = []
    hour_data = []
    obj = frame.loc[0:, "timestamp"]
    for i in obj:
        # print(i)
        timetodate = datetime.datetime.fromtimestamp(i / 1000)
        # 연 월 일
        date_data.append(timetodate.strftime("%Y%m%d"))
        # 시간 분
        time_data.append(timetodate.strftime("%H%M"))
        #  시간
        hour_data.append(timetodate.strftime("%H"))
        # print(time_data)
        # print(i)
        # print(i['timestamp'])

    date_series = pd.Series(date_data)
    time_series = pd.Series(time_data)
    hour_series = pd.Series(hour_data)
    # print(date_series)
    # print(time_series)
    frame['date_series'] = date_series
    frame['time_series'] = time_series
    frame['hour_series'] = hour_series
    # print(frame)
    frame.to_csv("../saved_data/saved_rating.csv", index=False)
    #print(obj)
    # date = datetime.datetime.fromtimestamp(964982703/1000)

    # print(date)
    # print(date.ctime())
    # print(date.isoformat())
    print("끝")

if __name__ ==  "__main__":
    timetodate()