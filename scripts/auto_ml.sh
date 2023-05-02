#!/bin/bash

for ((classA=4; classA<10; classA++))
do
    for ((classB=$classA+1; classB<10; classB++))
    do
        python auto_ml.py --classA $classA --classB $classB
    done
done
