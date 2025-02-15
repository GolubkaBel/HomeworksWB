import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')
import scipy.stats as sts
import matplotlib.pyplot as plt


data = pd.read_csv('drinks.csv', delimiter=',')
print('Статистика по данным: \n', data.describe())


fig1, im = plt.subplots(1, 3, figsize=(10, 6), constrained_layout=True)
fig1.suptitle("Страны-самые большие любители по напиткам(10%, где больше всего пьют)", fontsize=14)

beer_data = data.loc[data['beer_servings']
                     > np.percentile(data['beer_servings'], 90, interpolation='midpoint')]
beer_data = beer_data.sort_values(['beer_servings'], ascending=False)
im[0].bar(beer_data['country'], beer_data['beer_servings'], color='pink', edgecolor='black')
im[0].set_title('Пиво')
im[0].set_xticklabels(im[0].get_xticklabels(), rotation=90, size=8)

spirit_data = data.loc[data['spirit_servings']
                       > np.percentile(data['spirit_servings'], 90, interpolation='midpoint')]
spirit_data = spirit_data.sort_values(['spirit_servings'], ascending=False)
im[1].bar(spirit_data['country'], spirit_data['spirit_servings'], color='purple', edgecolor='black')
im[1].set_title('Крепкое')
im[1].set_xticklabels(im[1].get_xticklabels(), rotation=90, size=8)

wine_data = data.loc[data['wine_servings'] >
                     np.percentile(data['wine_servings'], 90, interpolation='midpoint')]
wine_data = wine_data.sort_values(['wine_servings'], ascending=False)
im[2].bar(wine_data['country'], wine_data['wine_servings'], color='blue', edgecolor='black')
im[2].set_title('Вино')
im[2].set_xticklabels(im[2].get_xticklabels(), rotation=90, size=8)
plt.tight_layout()

fig2 = plt.figure(figsize=(7, 5))
plt.title('box-plot для понимания распределения данных')
data_without = data[['country', 'beer_servings', 'spirit_servings', 'wine_servings']]
data_without.boxplot()
'Можно заметить, что wine_servings имеют большое кол-во выбросов, поэтому далее выведены страны,' \
' соответствующие выбросам'

print('\nВыбросы в данных wine_servings: \n',
      data.loc[data['wine_servings']>np.percentile(data['wine_servings'], 75, interpolation='midpoint') +
               1.5*sts.iqr(data['wine_servings'], interpolation='midpoint')].country)

fig3 = plt.figure(figsize=(9, 4))
plt.boxplot(data['total_litres_of_pure_alcohol'],vert=False)
plt.title('box-plot для понимания распределения данных согласно total_litres_of_pure_alcohol')
plt.yticks(rotation=90)
plt.tight_layout()
plt.show()
