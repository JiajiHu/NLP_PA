import numpy as np

x_range=[]

with open('./sentence.txt','r') as sen:
	for line in sen:
		x_range.append(int(line.split()[2]))

y_range=[]
with open('./time.txt','r') as time:
	for line in time:
		y_range.append(float(line.split()[2]))

print "total:\t", sum(y_range)
print "avg:\t", (sum(y_range)+0.0)/len(y_range)

twentySum = 0;
twentyCount = 0;

for i in range(len(x_range)):
	if (x_range[i] == 20):
		twentySum += y_range[i]
		twentyCount += 1

print "twenty:\t", (twentySum + 0.0)/twentyCount