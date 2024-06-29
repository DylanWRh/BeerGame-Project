#!/bin/bash

# python -u main.py --gameConfig=3 --maxEpisodesTrain=5000 --batchSize=128 --DQNckpt=chptexp1log1 > exp1log1
python -u main.py --gameConfig=4 --maxEpisodesTrain=5000 --batchSize=128 --DQNckpt=chptexp2log2 > exp2log2
python -u main.py --gameConfig=5 --maxEpisodesTrain=5000 --batchSize=128 --DQNckpt=chptexp2log3 > exp2log3
python -u main.py --gameConfig=6 --maxEpisodesTrain=5000 --batchSize=128 --DQNckpt=chptexp2log4 > exp2log4