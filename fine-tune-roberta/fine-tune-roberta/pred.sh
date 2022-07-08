#!/bin/bash

#set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=1:00:00

# set name of job
#SBATCH --job-name=jy_02

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL


#SBATCH --partition=devel

# run the application
module load python/anaconda3

source activate qust

python predict.py --pred_data /jmain02/home/J2AD003/txk60/yxw50-txk60/ye/fine-tune-roberta/data/expmrc-main/data/cmrc2018/expmrc-cmrc2018-dev.json --ans_model ./answer_large.model --evi_model ./evidence_large.model
