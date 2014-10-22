import matplotlib.pyplot as plt
import numpy as np
import pylab

SENTENCE_NAME = './sentence3.txt'
TIME_NAME = './time3.txt'

x_range=[]
with open(SENTENCE_NAME, 'r') as sen:
	for line in sen:
		x_range.append(line.split()[2])

y_range=[]
with open(TIME_NAME, 'r') as time:
	for line in time:
		y_range.append(line.split()[2])

ylim = 2.50
xlim = 21

x_range = np.array(x_range).astype('float')
y_range = np.array(y_range).astype('float')
zeros = np.zeros(100)

z = np.polyfit(np.append(zeros ,x_range),np.append(zeros ,y_range), 3)
f = np.poly1d(z)

x_new = np.linspace(0, xlim, 50)
y_new = f(x_new)

fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(0,xlim,1))
ax.set_yticks(np.arange(0,ylim,0.25))
plt.grid()

plt.plot(x_range,y_range,'bo', x_new, y_new,'r-', linewidth=2.5)

plt.xlabel('Sentence Length (words)')
plt.ylabel('Run Time (s)')
plt.title('Parser run time with respect to sentence length')
pylab.xlim([0,xlim])
pylab.ylim([0,ylim])
plt.show()
