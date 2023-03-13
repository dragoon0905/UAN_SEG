#!/bin/bash

#SBATCH --job-name=UANSEG
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH -o main_UAN.out
#SBATCH -w aurora-g8
#SBATCH --partition batch_sw_ugrad
#SBATCH --time=3-0



source /data/dragoon0905/init.sh
conda activate pixmatch

#rm -rf /local_datasets/CityScapes
#cp -r /data/dragoon0905/datasets/Mapillary  /local_datasets/Mapillary
#cp -r /data/dragoon0905/datasets/CityScapes  /local_datasets/CityScapes


HYDRA_FULL_ERROR=1 python main_UAN.py --config-name=OP_gta52CS lam_aug=0.0 name=gta5
