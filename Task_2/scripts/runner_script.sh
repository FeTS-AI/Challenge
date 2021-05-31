#!/bin/bash
 
# load default .bashrc. This is never loaded automatically
source /home/m167k/.bashrc_nnunet   # TODO

GPU_NUM=1
GPU_MEM=10.7G
CONTAINER="container.sif"
DATA_DIR="/gpu/data/OE0441/m167k/datasets/test_nnunet"
OUT_DIR="/gpu/data/OE0441/m167k/datasets/test_nnunet_results"

# TODO select appropriate nodes
bsub -B -N -R "tensorcore" -R "hname!='e230-dgx2-1' && hname!='e230-dgx2-2'" -gpu num=$GPU_NUM:j_exclusive=yes:mode=exclusive_process:gmem=$GPU_MEM -q gpu \
    -o /home/m167k/job_outputs/%J.out python test_container.py $CONTAINER -i $DATA_DIR -o $OUT_DIR --timeout 300 --logfile "$OUT_DIR/${CONTAINER%.sif}.log"
