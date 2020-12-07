#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig, ax = plt.subplots()
ax.hist(student_grades, bins=10, range=(0, 100),
        edgecolor='black')
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.xlim([0, 100])
plt.ylim([0, 30])
plt.xticks(range(0, 101, 10))

plt.show()
# plt.savefig('4-frequency.png')