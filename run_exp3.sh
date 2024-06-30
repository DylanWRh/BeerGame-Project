#!/bin/bash

# python -u main.py --gameConfig=6 --maxEpisodesTrain=5000 --batchSize=128 --DQNckpt=chptexp3log1 --actionUp=2  --actionLow=-2 > exp3log1.txt
python -u main.py --gameConfig=6 --maxEpisodesTrain=5000 --batchSize=128 --DQNckpt=chptexp3log2 --actionUp=1  --actionLow=-1 > exp3log2.txt
python -u main.py --gameConfig=6 --maxEpisodesTrain=5000 --batchSize=128 --DQNckpt=chptexp3log3 --actionUp=3  --actionLow=-3 > exp3log3.txt

