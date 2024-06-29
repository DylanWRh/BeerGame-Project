#!/bin/bash

python -u main.py --gameConfig=3 --maxEpisodesTrain=5000 --batchSize=128 --DQNckpt=chptexp1log1 > exp1log1
python -u main.py --gameConfig=7 --maxEpisodesTrain=5000 --batchSize=128 --DQNckpt=chptexp1log2 > exp1log2
python -u main.py --gameConfig=11 --maxEpisodesTrain=5000 --batchSize=128 --DQNckpt=chptexp1log3 > exp1log3