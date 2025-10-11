#!/bin/bash
#PBS -P gcg51557
#PBS -q rt_HG
#PBS -v RTYPE=rt_HG,USE_SSH=1
#PBS -l select=1:ngpus=1
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -N 0162_espnet_asr

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

exec python -m tools.espnet_asr \
  --wav-glob "/home/acg17145sv/experiments/0162_dialogue_model/moshi-finetune/output/moshi_stage3_new_jchat_tabidachi/step_498_fp32/continuation_tabidachi_full/generated_wavs/*.wav" \
  --output "data_tabidachi/moshi_stage3_new_jchat_tabidachi/transcripts_dialog.jsonl" \
  --device "cuda" \
  --overwrite \
  > "logs/0162_espnet_asr_${PBS_JOBID}.log" 2>&1
