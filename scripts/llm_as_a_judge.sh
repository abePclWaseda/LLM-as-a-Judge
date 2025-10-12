#!/bin/bash
#PBS -P gcg51557
#PBS -q rt_HG
#PBS -v RTYPE=rt_HG,USE_SSH=1
#PBS -l select=1:ngpus=1
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -N 0162_llm_as_a_judge

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

exec python -m tools.llm_as_a_judge \
  --input  "data_real/csj/transcripts_dialog.jsonl" \
  --output "data_real/csj/evaluated_dialog_multi.jsonl" \
  --model  "llm-jp/llm-jp-3.1-13b-instruct4" \
  --dtype  "bfloat16" \
  --max-new-tokens 128 \
  --overwrite \
  > "logs/0162_llmJudge_${PBS_JOBID}.log" 2>&1
