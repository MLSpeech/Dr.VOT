#!/bin/sh
date > log.txt
python ./helpers/check_req.py >> log.txt
if [ $? -eq 1 ]; then
    echo "Missing requirements"
    exit
fi
sh run.sh | tee -a log.txt
