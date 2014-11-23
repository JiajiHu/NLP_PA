#!/bin/bash

ant
java -cp classes cs224n/deep/NER ../data/train ../data/dev ../output/baselineOutput
../conlleval -d '\t' < ../output/baselineOutput