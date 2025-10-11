#!/bin/bash
#PBS -P gcg51557
#PBS -q rt_HG
#PBS -v RTYPE=rt_HG,USE_SSH=1
#PBS -l select=1:ngpus=1
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -N 0162_espnet_ipu

set -eu

echo "JOB_ID : $PBS_JOBID"
echo "WORKDIR: $PBS_O_WORKDIR"
cd "$PBS_O_WORKDIR"

module list

source ~/miniforge3/etc/profile.d/conda.sh
conda activate llmJudge310

echo "==== which python ===="
which python
python --version

mkdir -p logs

exec python -m tools.espnet_ipu_jsonl \
  --wav-glob "/home/acg17145sv/experiments/0162_dialogue_model/NISQA/data_audio/tmp_csj_20s_head50_1219655.pbs1/*.wav" \
  --output data_real/csj/ipu.jsonl \
  --device cuda \
  --skip-non-stereo \
  --overwrite
