import sys

counter = 0
maxy = 0
maxc = 0
with open('final_dev.txt') as f:
	for line in f:
		if counter > 0:
			if float(line.split()[-1]) > maxy:
				maxy = float(line.split()[-1])
				maxc = counter
		counter+=1
print maxy
print maxc