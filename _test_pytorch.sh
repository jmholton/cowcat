#!/bin/bash
#SBATCH --job-name=test_pytorch
#SBATCH --account=pc_als831
#SBATCH --partition=es1
#SBATCH --qos=es_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A40:1
#SBATCH --time=00:05:00
#SBATCH --output=slurm-pytorch-test.out

source /etc/profile.d/modules.sh
export MODULEPATH=$MODULEPATH:/global/software/rocky-8.x86_64/modfiles/Core
echo "MODULEPATH=$MODULEPATH"
module load gcc/10.5.0 && echo "gcc ok"
module load cuda/11.8.0 && echo "cuda ok"
module load cudnn/8.7.0.84-11.8 && echo "cudnn ok"
module load --force ml/pytorch/2.3.1-py3.11.7-mf && echo "pytorch ok"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
