#!/bin/bash
#SBATCH --job-name=unet-falcon
#SBATCH --account=me_jiechen
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_base
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/andrew03/pythonRepos/UNet-Cars/logs/%x_%j.out
#SBATCH --error=/home/andrew03/pythonRepos/UNet-Cars/logs/%x_%j.err
#SBATCH --chdir=/home/andrew03/pythonRepos/UNet-Cars

set -euo pipefail

# --- Environment ---
module reset
module load Python/3.11.3-GCCcore-12.3.0
source /home/andrew03/pythonRepos/UNet-Cars/arc_venvs/unet_311_a30_normal_q/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# --- Run naming ---
# Priority: 1st CLI arg > RUN_NAME env > SLURM job name
USER_RUN_NAME="${1:-${RUN_NAME:-$SLURM_JOB_NAME}}"
TS=$(date +%Y%m%d-%H%M%S)
RUN_TAG="${USER_RUN_NAME}_${SLURM_JOB_ID}"

# Optional timestamp suffix: export ADD_TS=1 to enable
: "${ADD_TS:=0}"
if [[ "$ADD_TS" == "1" ]]; then
  RUN_TAG="${RUN_TAG}_${TS}"
fi

# Expose for training code if desired
export RUN_TAG

# --- TensorBoard setup (runs on the compute node) ---
: "${TB_PORT:=6006}"                                        # override via: sbatch --export=ALL,TB_PORT=16006 ...
: "${TB_LOGROOT:=$HOME/pythonRepos/UNet-Cars/runs}"         # parent directory for runs
TB_LOGDIR="${TB_LOGROOT}/${RUN_TAG}"
mkdir -p "$TB_LOGDIR"

# Start TensorBoard in the background, bound to localhost only
nohup tensorboard \
  --logdir "$TB_LOGDIR" \
  --port "$TB_PORT" \
  --host 127.0.0.1 \
  > "$TB_LOGDIR/tensorboard.log" 2>&1 &

TB_PID=$!
NODE=$(hostname)

# set defaults safely even with `set -u`
: "${LOCAL_PORT:=6006}"   # local browser port (left side of -L)

cat<<EOF
================ TensorBoard Info =================
TB_NODE=$NODE
TB_PORT=$TB_PORT
TB_LOGDIR=$TB_LOGDIR
TB_PID=$TB_PID

TB_TUNNEL_CMD: 
ssh -N -L 6006:localhost:$TB_PORT -J $USER@falcon1.arc.vt.edu $USER@$NODE

BROWSER_URL:
http://localhost:${LOCAL_PORT}

(If 6006 is busy locally, change the first number; falcon2.arc.vt.edu also works.)
===================================================
EOF

# --- Training ---
python -u src/train.py --logdir "$TB_LOGDIR"

# Keep TB alive until job ends (training may exit first)
wait $TB_PID || true
