#!/bin/bash
sudo -S /usr/sbin/sshd
source /opt/docker/bin/packages.sh
source /opt/conda/etc/profile.d/conda.sh 
conda activate rapids 

nohup jupyter-lab --port 8888 --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.allow_origin="*" > /dev/null 2>&1 &
mlflow server --host 0.0.0.0
tail -f /dev/null
