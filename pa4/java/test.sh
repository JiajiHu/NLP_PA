#!/bin/bash

Iters=40

ant
eval "java -Xmx1g -cp 'extlib/*:classes' cs224n.deep.NER ../data/train ../data/dev ../output/output.out $Iters ../data/test"

echo "************************** NEW RUN **************************" >> ../results/deeper_train.txt
echo "3, 60, 0.001, true, 0.0005, true, true, 20, false" >> ../results/deeper_train.txt

echo "************************** NEW RUN **************************" >> ../results/deeper_dev.txt
echo "3, 60, 0.001, true, 0.0005, true, true, 20, false" >> ../results/deeper_dev.txt

echo "************************** NEW RUN **************************" >> ../results/deeper_test.txt
echo "3, 60, 0.001, true, 0.0005, true, true, 20, false" >> ../results/deeper_test.txt

for (( i=0; i <= Iters; i++ ))
do
	echo "-------------------- Iteration $i --------------------" >> ../results/deeper_train.txt
	eval "../conlleval -r -d '\t' < ../output/output.out_train_iter${i}deeper >> ../results/deeper_train.txt"
	echo "-------------------- Iteration $i --------------------" >> ../results/deeper_dev.txt
	eval "../conlleval -r -d '\t' < ../output/output.out_dev_iter${i}deeper >> ../results/deeper_dev.txt"
	echo "-------------------- Iteration $i --------------------" >> ../results/deeper_test.txt
	eval "../conlleval -r -d '\t' < ../output/output.out_test_iter${i}deeper >> ../results/deeper_test.txt"
done