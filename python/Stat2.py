import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')
import scipy.stats as sts
import matplotlib.pyplot as plt

data = pd.read_csv('experiment_lesson_4.csv', delimiter=',')
# print(data)

test = data.loc[data['experiment_group'] == 'test']
# print(test)

control = data.loc[data['experiment_group'] == 'control']
# print(control)

fig1, im = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
im[0].hist(test['delivery_time'])
im[0].axvline(x=45, color='r', label='axvline - full height')
im[0].axvline(x=sts.mode(test['delivery_time'])[0], color='g', label='axvline - full height')
im[0].set_title('Test')

im[1].hist(control['delivery_time'])
im[1].axvline(x=45, color='r', label='axvline - full height')
im[1].axvline(x=sts.mode(control['delivery_time'])[0], color='g', label='axvline - full height')
im[1].set_title('Control')
#plt.show()

print('Разница кол-ва наблюдений: ', np.abs(test.shape[0]-control.shape[0]))

print('Тест на нормальность для тестовых данных: ', sts.normaltest(test['delivery_time'], nan_policy='omit'))
print('Тест на нормальность для конрольных данных: ', sts.normaltest(control['delivery_time'], nan_policy='omit'))

D = 0
meanControl = control['delivery_time'].mean()
for x in control['delivery_time']:
    D = D + (x-meanControl)**2
D = D/(len(control['delivery_time'])-1)
stdControl = np.sqrt(D)
print('Стандартное отклонение в котрольной группе: ', stdControl)

print('Стандартное отклонение в тестовой группе: ', test['delivery_time'].std())

print('Быстрая сводка по тестовым: ')
print(test.describe())

print('Быстрая сводка по контрольным: ')
print(control.describe())

print('Сравнение средних: ', sts.ttest_ind(test['delivery_time'], control['delivery_time']))

print('Процентное изменение среднего: ', ((test['delivery_time'].mean()-meanControl)/meanControl)*100)
