#!/bin/bash
#
#SBATCH --job-name=prediction_based_propagation
#SBATCH --output=output.log
#SBATCH --error=output.log
#SBATCH --partition=batch
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1

echo "Job started."
echo "Date : $(date)"
echo "Node : $(hostname)"
echo "Job ID : $SLURM_JOB_ID"

source ../python/env/bin/activate
module load Java/21.0.7
module load CUDA/12.8.0

# Verify CUDA installation
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo "Starting Java application."
srun java --enable-native-access=ALL-UNNAMED -cp ./maxicp.jar org.maxicp.RunXCSP3.RunXSCP3

echo "Job ended."
