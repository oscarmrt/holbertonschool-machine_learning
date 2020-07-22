#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

fruits = np.append(np.zeros((1, 3)), fruit.cumsum(axis=0), axis=0)
fruits_name = ['apples', 'bananas', 'orange', 'peaches']
fruits_color = ['red', 'yellow', '#ff8000', '#ffe5b4']


for f in range(len(fruit)):
    plt.bar(
        range(3),
        height=fruit[f],
        bottom=fruits[f],
        width=0.5,
        color=fruits_color[f],
        label=fruits_name[f]
    )

plt.yticks(np.linspace(0, 80, 9))
plt.xticks(range(3), ['Farrah', 'Fred', 'Felicia'])
plt.ylabel('Quantity of Fruit')
plt.legend(loc='best')
plt.title('Number of Fruit per Person')
plt.show()
