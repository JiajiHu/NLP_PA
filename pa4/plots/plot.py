import matplotlib.pyplot as plt
import sys

x_range = []
y_range = []
counter = 0
with open("./final_train.txt") as f:
	for line in f:
		if counter > 0:
			x_range.append(counter)
			y_range.append(float(line.split()[-1]))
		counter += 1
x2_range = []
y2_range = []
maxy2 = 0;
counter2 = 0
with open("./final_dev.txt") as f:
	for line in f:
		if counter2 > 0:
			x2_range.append(counter2)
			y2 = float(line.split()[-1])
			if y2 > maxy2:
				maxy2 = y2
			y2_range.append(y2)
		counter2 += 1

plt.plot(x_range,y_range,'b-o',label='Train F1')
plt.plot(x2_range,y2_range,'r-o',label='Dev F1')
plt.xlabel('SGD iterations')
plt.ylabel('F1 Score')
plt.title('Final System Learning Curve')
plt.legend(loc=7)
plt.show()

print maxy2
