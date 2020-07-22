#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

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

plt.rcParams['axes.labelsize'] = 'x-small'
plt.rcParams['axes.titlesize'] = 'x-small'
draw = plt.figure(constrained_layout=True)
grid = plt.matplotlib.gridspec.GridSpec(3, 2, figure=draw)
draw.suptitle('All in One')

ax0 = draw.add_subplot(grid[0, 0])
ax0.set_xlim(0, 10)
ax0.plot(y0, 'r')

ax1 = draw.add_subplot(grid[0, 1])
ax1.scatter(x1, y1, 9, c='magenta')
ax1.set_xlabel('Height (in)')
ax1.set_ylabel('Weight (lbs)')
ax1.set_title('Men\'s Height vs Weight')

ax2 = draw.add_subplot(grid[1, 0])
ax2.plot(x2, y2)
ax2.set_yscale('log')
ax2.set_xlim(0, 28650)
ax2.set_xlabel('Time (years)')
ax2.set_ylabel('Fraction Remaining')
ax2.set_title('Exponential Decay of C-14')

ax3 = draw.add_subplot(grid[1, 1])
ax3.plot(x3, y31, '--r', label='C-14')
ax3.plot(x3, y32, 'g', label='Ra-226')
ax3.set_xlim(0, 20000)
ax3.set_ylim(0, 1)
ax3.set_xlabel('Time (years)')
ax3.set_ylabel('Fraction Remaining')
ax3.set_title('Exponential Decay of Radioactive Elements')
ax3.legend()

ax4 = draw.add_subplot(grid[2, :])
ax4.hist(student_grades, bins=np.linspace(40, 100, 7), edgecolor='black')
ax4.set_xlim(0, 100)
ax4.set_ylim(0, 30)
ax4.set_xticks(np.linspace(0, 100, 11))
ax4.set_xlabel('Grades')
ax4.set_ylabel('Number of Students')
ax4.set_title('Project A')

plt.show()
