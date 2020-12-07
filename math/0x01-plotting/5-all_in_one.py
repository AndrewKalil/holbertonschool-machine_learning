#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

mpl.rcParams['axes.titlesize'] = 'x-small'
mpl.rcParams['axes.labelsize'] = 'x-small'
mpl.rcParams['legend.fontsize'] = 'x-small'
fig = plt.figure(constrained_layout=True)

gs = GridSpec(3, 2, figure=fig)
fig.suptitle('All in One')

ax1 = fig.add_subplot(gs[0, :-1])
ax1.plot(y0, 'r')

ax2 = fig.add_subplot(gs[0, 1:])
ax2.scatter(x1, y1, c='magenta', s=5)
plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
plt.title("Men's Height vs Weight")

ax3 = fig.add_subplot(gs[1, :-1])
ax3.plot(x2, y2)
plt.yscale('log')
plt.xlim([0, 28651])
plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.title("Exponential Decay of C-14")

ax4 = fig.add_subplot(gs[1, 1:])
ax4.plot(x3, y31, 'r--', label='C-14')
plt.xlim([0, 20000])
plt.ylim([0, 1])
ax4.plot(x3, y32, 'g-', label='Ra-226')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of Radioactive Elements')
plt.legend(loc='upper right')

ax5 = fig.add_subplot(gs[2, :])
ax5.hist(student_grades, bins=10, range=(0, 100),
         edgecolor='black')
plt.xticks(range(0, 101, 10))
plt.xlim([0, 100])
plt.ylim([0, 30])
plt.xlabel("Grades")
plt.ylabel('Number of Students')
plt.title('Project A')

plt.savefig('5-all_in_one.png')