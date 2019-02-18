for year in `seq 2007 2010`;
	do
	for region in `seq 1 4`;
        	do
                	# $i
			python inference.py -m /home/annus/Desktop/palsar/palsar_models_focal/trained_on_2007-10/model-12.pt -b 20 -y $year -r $region
	        done
	done
