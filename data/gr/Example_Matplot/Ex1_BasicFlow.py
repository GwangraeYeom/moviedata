# https://blog.naver.com/codingspecialist/221368322641
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from os import path
from IPython.display import display
import seaborn

path_prefix = path.dirname(path.abspath(__file__))
print(path_prefix)
battles_csv = path.join(path_prefix,'Data\\battles.csv')
deaths_csv = path.join(path_prefix,'Data\\character-deaths.csv')

battles = pd.read_csv(battles_csv, sep=',')
deaths = pd.read_csv(deaths_csv, sep=',')

print('---전쟁---')
print(battles.shape) #numpy패키지의 함수 (행,열)의 튜플로 반환
print()

print('---죽음---')
print(deaths.shape)
print()

print('---전체 칼럼---')
print(deaths.columns)
print()

print('---deaths 데이터프레임.head() 5행 반환---')
print(deaths.head())
print()

print('---Book of Death 칼럼만 출력---')
print(deaths["Book of Death"])
print()

#Book of Deat컬럼의 데이터별 개수 합을 반환 ex)1장에서 죽은 사람수 49
book_of_death_count = deaths['Book of Death'].value_counts().sort_index()
print(book_of_death_count)
print(type(book_of_death_count))
print(book_of_death_count.index)

print(book_of_death_count.loc[:, ])
ax1 = book_of_death_count.plot(color='b', marker='o', linestyle='--')
# ax1.set_xticks(book_of_death_count.index)

ax1.set_xlim([0,6])
ax1.set_ylim([1,120])
save_path = path_prefix +'\\Figure\\plot1.png'
plt.savefig(save_path)
plt.show()