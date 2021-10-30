#!/usr/bin/env bash
for year in `seq 2016 2017`;
	do
	for region in `seq 1 4`;
        	do
                	# $i
			python inference.py -m '/home/annus/Desktop/palsar/palsar_models_focal/trained_separately/train_on_2015/model-4.pt' -b 20 -y $year -r $region
	        done
	done
