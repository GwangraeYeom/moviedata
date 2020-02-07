import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime

def datashow(movieId):
    # 영화별, 장르별
    file = pd.read_csv('../saved_data/saved_rating.csv')
    frame = pd.DataFrame(file)

    print(frame.head(5))
    # 특정 영화에 대한 데이터 프레임 생성
    movie_frame = frame[frame['movieId'] == movieId]
    print(movie_frame)

    hour_avg = movie_frame.groupby(movie_frame['hour_series'])[['rating']].mean()
    print(hour_avg.head(30))
    hour_count = movie_frame.groupby(movie_frame['hour_series'])[['rating']].count()
    print(hour_count.head(30))
    # hour_avg.plot(color='b', marker='o', linestyle='--')
    fig = plt.figure()

    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax1.set_xticks(hour_avg.index)
    ax1.plot(hour_avg)
    ax2.set_xticks(hour_avg.index)
    ax2.plot(hour_count)
    plt.show()




if __name__ == '__main__':
    datashow(1)
