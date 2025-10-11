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

python3 tools/ipu_metrics.py \
  --input data_real/csj/ipu.jsonl \
  --min_silence 0.2 \
  --outdir data_real/csj
