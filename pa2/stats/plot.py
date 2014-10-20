import matplotlib.pyplot as plt
import numpy as np
import pylab

x_range=[]

with open('./sentence.txt','r') as sen:
	for line in sen:
		x_range.append(line.split()[2])

y_range=[]
with open('./time.txt','r') as time:
	for line in time:
		y_range.append(line.split()[2])

ylim = 3.25
xlim = 21

x_range = np.array(x_range).astype('float')
y_range = np.array(y_range).astype('float')
zeros = np.zeros(1000)

z = np.polyfit(np.append(zeros ,x_range),np.append(zeros ,y_range), 2)
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
