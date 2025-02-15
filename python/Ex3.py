import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data = pd.read_csv('tarantino.csv', delimiter=',')
print('\nИсходные данные: \n',
      data.head())

nodeath_data = data.loc[data['type'] != 'death']
print('\nКоличество произнесенных проклятий по фильмам: \n',
      pd.concat([nodeath_data.groupby('movie')['word'].count(),
                 nodeath_data.groupby('movie')['word'].nunique()],
                keys=['Всего', 'Уникальных'], axis=1))

death_count_data = data.loc[data['word'].isna()].groupby('movie')['type'].count()
print('\nКоличество смертей по фильмам и % соотношения: \n',
      pd.concat([death_count_data,
                 death_count_data / data.word.isna().sum() * 100,
                 death_count_data / nodeath_data.groupby('movie')['word'].count() * 100],
                keys=['Кол-во смертей', 'count(death)/all_filmes_death (%)', 'count(death)/count(word) (%)'],
                axis=1))

print('\nКоличество произнесенных проклятий по фильмам: \n',
      nodeath_data.groupby(['movie', 'word'])['word'].count())

fig1 = plt.figure(figsize=(10, 6))
subdata = nodeath_data.groupby('word')['word'].count()
plt.bar(subdata.index, subdata.values)
plt.grid(visible=True)
plt.title('Диаграмма частот употребления проклятий относительно всех фильмов')
plt.xticks(rotation=90)
plt.tight_layout()

movies = data.movie.unique()
fig2, im = plt.subplots(2, 4, figsize=(12, 7))
fig2.suptitle('Диаграмма частот употребления проклятий для всех фильмов')
for i in range(2):
    for j in range(4):
        if i * j == 3:
            print('')
        else:
            subdata = nodeath_data.loc[data['movie'] == movies[4 * i + j]].groupby('word')['word'].count()
            im[i][j].bar(subdata.index, subdata.values)
            im[i][j].set_title(movies[(i + 3) * i + j])
            im[i][j].set_xticklabels(im[i][j].get_xticklabels(), rotation=90, size=8)
plt.tight_layout()

fig3, im = plt.subplots(2, 4, figsize=(12, 7))
fig3.suptitle('Распределение интервалов между проклятиями по фильмам')
for i in range(2):
    for j in range(4):
        if i * j == 3:
            print('')
        else:
            intervals = np.diff(sorted(data.loc[data['movie'] == movies[4 * i + j]]['minutes_in'].to_list()))
            im[i][j].hist(intervals)
            im[i][j].set_title(movies[(i + 3) * i + j])
plt.show()
