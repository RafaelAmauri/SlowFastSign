# BEFORE RUNNING, GENERATE THE FULL GLOSS DICT!!

# For more details, read: https://github.com/VIPL-SLP/VAC_CSLR/issues/50

n_replications=$1 # How many replications to run

for i in 10 20; do 
	for j in $(seq 1 $n_replications); do
        currentFolder="phoenix$i-run$j"
		
		cd preprocess

		python3 dataset_preprocess.py --dataset $currentFolder --dataset-root ../dataset/$currentFolder/phoenix-2014-multisigner

		cd ../

		python3 main.py --device 0 --dataset $currentFolder --loss-weights Slow=0.25 Fast=0.25 --work-dir work_dir/$currentFolder 
	done; 
done
