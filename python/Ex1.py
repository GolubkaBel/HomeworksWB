import pandas
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use('TkAgg')
import scipy.stats as sts
import matplotlib.pyplot as plt
from collections import Counter

data = pandas.read_csv('shopping_trends_updated.csv', delimiter=',')
print(data.iloc[1])

corr_matrix = data[['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']].corr()
fig1 = plt.figure()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
'По данной матрице можно сказать, что зависимости между данными нет'

# Гипотеза 1. Сумма покупки зависит от возраста, чем старше человек, тем дороже его заказ
print('\nТест на нормальность(ks): %.2f' % sts.kstest(data['Purchase Amount (USD)'], 'norm').pvalue)
'Данные [Age] и [Purchase Amount (USD)] не имеют нормального распределения'
correlation, p_value = sts.spearmanr(data['Age'], data['Purchase Amount (USD)'])
print(f'\nГипотеза 1 (Корреляция Спирмена): corr = {correlation:.2f}, p = {p_value:.3f}')
'Из (r = -0.01, p = 0.514) можно сделать выводы, что нет оснований отвергнуть H0, ' \
'а это означает, что корееляции между данными нет.'

# Гипотеза 2. Товары теплых цветов более дорогие относительно товаров других цветов
# print('\nВозможные цвета по данным: \n', data['Color'].unique())
warm_data = data.loc[data['Color'].isin(['Red', 'Magenta', 'Orange', 'Gold', 'Violet', 'Peach', 'Maroon'])][
    'Purchase Amount (USD)']
no_warm_data = data.loc[~data['Color'].isin(['Red', 'Magenta', 'Orange', 'Gold', 'Violet', 'Peach', 'Maroon'])][
    'Purchase Amount (USD)']
statistic, p_value = sts.mannwhitneyu(warm_data, no_warm_data, alternative='greater')
print(f'\nГипотеза 2 (Тест Манн-Уитни): stat = {statistic:.2f}, p = {p_value:.3f}')
# Зависимости между цветом и ценной нет

# Гипотеза 3. Зависимость между ценой и методом оплаты
statistic, p_value = sts.kruskal(*[group['Purchase Amount (USD)'] for name, group in data.groupby('Payment Method')])
print(f'\nГипотеза 3 (Тест Краскела-Уоллиса): stat = {statistic:.2f}, p = {p_value:.3f}')
'Зависимости вновь нет'

# Гипотеза 4. Зависимость размера одежда и возраста у мужчин
men_data = data.loc[data['Gender'] == 'Male']


def func(size):
    if size == 'XS':
        return 46
    elif size == 'S':
        return 48
    elif size == 'M':
        return 50
    elif size == 'L':
        return 52
    else:
        return 54


men_data.loc[:, 'Size'] = men_data['Size'].apply(func)
correlation, p_value = sts.spearmanr(men_data['Age'], men_data['Size'])
print(f'\nГипотеза 4 (Корреляция Спирмена): corr = {correlation:.2f}, p = {p_value:.3f}')
'Зависимости вновь нет'

# Гипотеза 5. Рейтинг зависит от наличия описания
desc_data = data.loc[data['Subscription Status'] == 'Yes']['Review Rating']
nodesc_data = data.loc[data['Subscription Status'] != 'Yes']['Review Rating']
statistic, p_value = sts.mannwhitneyu(desc_data, nodesc_data)
print(f'\nГипотеза 5 (Тест Манн-Уитни): stat = {statistic:.2f}, p = {p_value:.3f}')
'Зависимости вновь нет'

# Найдите самый популярный товар
print('\nСамые популярные товар(ы):\n', data['Item Purchased'].mode())
print('Количество самого популярного товара: ', len(data.loc[data['Item Purchased'] == 'Pants']['Item Purchased']))
'Самый популярный товар - это самый часто встречающийся, поэтому используем моду из мер центральной тенденции'

# Постройте распределение покупателей по полу
gender_data = data[['Gender', 'Customer ID']].drop_duplicates()
fig2 = plt.figure(figsize=(5.5, 5))
plt.hist(gender_data['Gender'], color='green', edgecolor='black')
plt.title('Распределение покупателей по гендеру')
plt.xlabel('Гендер')
plt.ylabel('Количество пользователей', )
# plt.show()

# Определите, какой пол (и отдельно возраст) покупает больше всего, чаще всего, самые дорогие товары
print('\nСтатистическая сводка по стоимости товаров \n', data['Purchase Amount (USD)'].describe())
expensive_data = data.loc[data['Purchase Amount (USD)'] >
                          np.percentile(data['Purchase Amount (USD)'], 75, interpolation='midpoint')]
'Дорогими товарами считаем те, которые выходят за пределы Q3 (75% данных) справа'
print('\nПол, покупающий самые дорогие товары: ', expensive_data['Gender'].mode().iloc[0])
print('Возраст, покупающий самые дорогие товары: ', sts.mode(expensive_data['Age']))

# Определите, есть ли зависимость между цветом одежды и сезоном
color_season_data = pd.crosstab(data.loc[data['Category'] == 'Clothing']['Color'],
                                data.loc[data['Category'] == 'Clothing']['Season'])
chi2, p, *rest = sts.chi2_contingency(color_season_data)
print('\nCтатистика хи-квадрат (одежда-сезон): %f, \np-значение: %f' % (chi2, p))
'Т.к. p-value>0.05, нет оснований отвергнуть Н0 => данные не имееют статистическо значимой зависимости'

# Посчитайте наш сезонный mau (уникальных пользователей за сезон) и его динамику
season_customer_data = data.groupby('Season')['Customer ID'].nunique()
print('\nСезонный mau и его динамика: \n',
      pd.concat([season_customer_data, season_customer_data.pct_change() * 100], axis=1))

# Найдите самую популярную букву в названии одежды
clothes_list = data.loc[data['Category'] == 'Clothing']['Item Purchased'].to_list()
clothes_monoword = ''.join(clothes_list).lower()
print('\nСамая популярная буква: ', Counter(clothes_monoword).most_common(1), '\n')
