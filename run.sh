#!/bin/sh
# This is a comment!



#cleanup
F_cleanup() {
    echo "cleanup"
    rm -r ./data/raw/all_files > /dev/null
    rm -r ./data/processed > /dev/null
    rm -r ./data/out_tg/tmp > /dev/null
    exit
}


#delete traces of previous runs
rm -fr ./data/raw/all_files/* > /dev/null
rm -fr ./data/processed/* > /dev/null
rm -fr ./data/out_tg/* > /dev/null

#creating dirs
mkdir ./data/ command
mkdir ./data/raw
mkdir ./data/processed
mkdir ./data/raw/all_files
mkdir ./data/out_tg/tmp

echo "============"
echo "Step 0: Installing dependencies in a virtual environment(It doesn't change your settings)"
echo "============"
pipenv install

echo "============"
echo "Step 1: Preparing the data"
echo "============"
pipenv run python ./process_data/prepare_wav_dir.py --input_dir ./data/raw --output_dir ./data/raw/all_files
if [ $? -eq 1 ]; then
    echo "Failed to collect the data, check log.txt"
    F_cleanup
fi

echo "============"
echo "Step 2: Processing sound files...(may take a while - approx. 1 sec per file)"
echo "============"
pipenv run python process_data_pipeline.py --input_dir ./data/raw/all_files --output_dir ./data/processed
if [ $? -eq 1 ]; then
    echo "Failed to process the data, check log.txt"
    F_cleanup
fi

echo "============"
echo "Step 3: Running Dr.VOT"
echo "============"
pipenv run python predict.py --inference ./data/processed/ --out_dir ./data/out_tg/tmp
if [ $? -eq 1 ]; then
    echo "Failed to run Dr.VOT system, check log.txt"
    F_cleanup
fi

echo "============"
echo "Step 4: Organizing output"
echo "============"
pipenv run python post_predict_script.py --summary ./data/out_tg/tmp/summary.csv --tg_dir ./data/out_tg/tmp/ --filenames ./data/raw/all_files/files.txt
if [ $? -eq 1 ]; then
    echo "Failed to post-process the data, check log.txt"
    F_cleanup
fi

mv ./data/out_tg/tmp/new_summary.csv ./data/out_tg/summary.csv
mv ./data/out_tg/tmp/hierarchical_tg/* ./data/out_tg/

echo "============"
echo "============"
echo "============"
echo "Finished: Dr.VOT predictions can be found at : ./data/out_tg/"
echo "============"
echo "============"
echo "============"


F_cleanup

