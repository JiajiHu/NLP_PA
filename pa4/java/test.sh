#!/bin/bash

ant
java -cp "extlib/*:classes" cs224n.deep.NER ../data/train ../data/dev ../output/baselineOutput