import sys

def getLabel(l):
	if l == 'O':
		return 0
	if l == 'I-LOC':
		return 1
	if l == 'I-MISC':
		return 2
	if l == 'I-ORG':
		return 3
	if l == 'I-PER':
		return 4


cm = [[0,0,0,0,0] for i in range(5)]
with open('test_output.txt') as f:
	for line in f:
		_, label, pred = line.split()
		cm[getLabel(label)][getLabel(pred)] += 1

print cm[0]
print cm[1]
print cm[2]
print cm[3]
print cm[4]
