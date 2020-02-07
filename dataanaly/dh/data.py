import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime

def timetodate():
    file = pd.read_csv('../data/ratings.csv')
    frame = pd.DataFrame(file)

    print( datetime.datetime.fromtimestamp(964982400 / 1000))
    print( datetime.datetime.fromtimestamp(964982224 / 1000))
    print(datetime.datetime.fromtimestamp(964984100 / 1000))

    # print(file)
    # print(frame)
    date_data = []
    time_data = []
    hour_data = []
    obj = frame.loc[0:, "timestamp"]
    for i in obj:
        #print(i)
        timetodate = datetime.datetime.fromtimestamp(i / 1000)
        # 연 월 일
        date_data.append(timetodate.strftime("%Y%m%d"))
        # 시간 분
        time_data.append(timetodate.strftime("%H%M"))
        #  시간
        hour_data.append(timetodate.strftime("%H"))
        #print(time_data)
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


def datashow(movieId):
    # 영화별, 장르별
    file = pd.read_csv('../saved_data/saved_rating.csv')
    frame = pd.DataFrame(file)
    # obj1 = frame.loc[:, ["userId", "rating", "movieId", "time_series", "timestamp", "hour_series"]]

    frame = frame[(frame['movieId'] == 296)]

    frame.to_csv("../saved_data/obj1.csv", index=False)

    hour_avg = frame.groupby(frame['hour_series'])[['rating']].mean()
    hour_count = frame.groupby(frame['hour_series'])[['rating']].count()

    user_avg = frame.groupby(['userId', 'hour_series'])[['rating']].mean()

    # temp = obj1[(obj1['time_series'] < a_time) & (obj1['time_series'] >= b_time)]
    # temp = temp.groupby(temp['time_series'])[['rating']].mean()
    # temp.to_csv("../saved_data/temp.csv")
    # temp = pd.read_csv("../saved_data/temp.csv")

    hour_avg.to_csv("../saved_data/hour_temp.csv")
    hour_avg = pd.read_csv("../saved_data/hour_temp.csv")

    hour_count.to_csv("../saved_data/hour_count.csv")
    hour_count = pd.read_csv("../saved_data/hour_count.csv")



    user_avg.to_csv("../saved_data/user_avg.csv")
    user_avg = pd.read_csv("../saved_data/user_avg.csv")
    user_avg = user_avg[(user_avg['userId'] == 1)]

    print(user_avg)
    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3, 1, 3)
    #ax1.plot(temp['time_series'],temp['rating'], 'g--')
    #ax1.set_xticks(range(60))
    #ax2.scatter(temp['time_series'],temp['rating'])

    ax1.plot(hour_avg['hour_series'], hour_avg['rating'], 'g--')
    ax1.set_xticks(range(24))
    ax2.plot(hour_count['hour_series'], hour_count['rating'], 'g--')
    #ax2.scatter(hour_avg['hour_series'], hour_avg['rating'])
    ax3.scatter(user_avg['hour_series'], user_avg['rating'])
    ax2.set_xticks(range(24))
    ax3.set_xticks(range(24))
    plt.show()


if __name__ ==  "__main__":
    # timetodate()
    datashow(13)