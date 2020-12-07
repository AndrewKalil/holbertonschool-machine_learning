#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

labels = ['Farrah', 'Fred', 'Felicia']
X = [0, 1, 2]
barWidth = 0.5

# arg[0]: to do three simultaniously, 
# arg[1]: stack to be added, 
# arg[2](bottom): matrix with added previous stacks 
plt.bar(X, fruit[0], width=barWidth, label='apples', color='red')
plt.bar(X, fruit[1], bottom=fruit[0], width=barWidth, label='bananas',
        color='yellow')
height1 = np.add(fruit[0], fruit[1]).tolist()
plt.bar(X, fruit[2], bottom=height1, width=barWidth, label='oranges',
        color='#ff8000')
height2 = np.add(height1, fruit[2]).tolist()
plt.bar(X, fruit[3], bottom=height2, width=barWidth, label='peaches',
        color='#ffe5b4')

plt.xticks(X, labels)
plt.ylim([0, 80])
plt.yticks(range(0, 81, 10))
plt.legend(loc='upper right')
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')

plt.show()
plt.savefig('6-bars.png')