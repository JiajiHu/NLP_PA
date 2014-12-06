#!/bin/bash

Iters=20

ant
eval "java -Xmx1g -cp 'extlib/*:classes' cs224n.deep.NER ../data/train ../data/dev ../output/output.out $Iters"

echo "************************** NEW RUN **************************" >> ../results/train_results.txt
echo "************************** NEW RUN **************************" >> ../results/dev_results.txt
for (( i=0; i <= Iters; i++ ))
do
	echo "-------------------- Iteration $i --------------------" >> ../results/train_results.txt
	eval "../conlleval -r -d '\t' < ../output/output.out_train_iter$i >> ../results/train_results.txt"
	echo "-------------------- Iteration $i --------------------" >> ../results/dev_results.txt
	eval "../conlleval -r -d '\t' < ../output/output.out_dev_iter$i >> ../results/dev_results.txt"
done