# =======================
# One-click Ablation Runner 
# =======================
#
#   chmod +x run_all_ablation.sh
#   ./run_all_ablation_v3.sh /path/to/leaf_dataset
#
#
#   ABLATE_PY=/abs/path/to/main.py
#   MODEL=resnet50
#   EPOCHS=50  BS=32  LR=1e-4  WORKERS=4
#   DEVICE=cuda|cpu   GPU=0
#   SEED=42

# ----------- Positional arg -----------
DATA=${1:-"/abs/path/to/JYB"}

# ----------- Script & Python -----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY=${PYTHON:-python}
ABLATE_PY="${ABLATE_PY:-${SCRIPT_DIR}/main.py}"
if [[ ! -f "$ABLATE_PY" ]]; then
  echo "[Error] Ablation script not found: $ABLATE_PY" >&2
  exit 1
fi

# ----------- Config -----------
MODEL=${MODEL:-resnet50}
EPOCHS=${EPOCHS:-50}
BS=${BS:-32}
LR=${LR:-1e-4}
WORKERS=${WORKERS:-4}
DEVICE=${DEVICE:-cuda}
GPU=${GPU:-0}
SEED=${SEED:-42}

# Determinism-related envs
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:16:8 || true
export NVIDIA_TF32_OVERRIDE=0
export TORCH_SHOW_CPP_STACKTRACES=0

STAMP=$(date +"%Y%m%d_%H%M%S")
ROOT=${ROOT:-"run/ablation_${STAMP}"}
export ROOT
mkdir -p "$ROOT/logs"
export CUDA_VISIBLE_DEVICES="$GPU"

# ----------- Experiments list -----------
EXP_LIST=(
  "A0|Baseline CE|--use_ce"
  "A1|+ CORAL (Ordinal)|--use_ce --use_coral --ce_weight 0.5 --coral_weight 0.5"
  "A2|+ Mask 4ch|--use_ce --use_coral --ce_weight 0.5 --coral_weight 0.5 --add_mask_channel"
  "A3|+ Weighted Sampler|--use_ce --use_coral --add_mask_channel --use_weighted_sampler"
  "A4|+ EMA|--use_ce --use_coral --add_mask_channel --use_weighted_sampler --use_ema"
  "A5|+ Attention: ECA|--use_ce --use_coral --add_mask_channel --use_weighted_sampler --use_ema --attention eca"
  "A6|+ Attention: CBAM|--use_ce --use_coral --add_mask_channel --use_weighted_sampler --use_ema --attention cbam"
)

# ----------- Run loop -----------
idx=0
for LINE in "${EXP_LIST[@]}"; do
  IFS='|' read -r TAG DESC FLAGS <<<"$LINE"
  echo -e "\n================== ${TAG}: ${DESC} =================="
  LOG_FILE="$ROOT/logs/${TAG}.log"

  # shellcheck disable=SC2206
  FLAG_ARR=($FLAGS)

  EXP_SEED=$((SEED + idx))
  echo "[Info] Using SEED=${EXP_SEED} for ${TAG}"

  set -x
  "$PY" "$ABLATE_PY" \
    --data "$DATA" \
    --model_name "$MODEL" \
    --epochs "$EPOCHS" \
    --batchsize "$BS" \
    --lr "$LR" \
    --num_workers "$WORKERS" \
    --device "$DEVICE" \
    --log_root "$ROOT" \
    ${FLAG_ARR+"${FLAG_ARR[@]}"} \
    --seed "$EXP_SEED" | tee "$LOG_FILE"
  { set +x; } 2>/dev/null

  echo "[Done] ${TAG} -> log: $LOG_FILE"
  idx=$((idx+1))
  sleep 2
done

# ----------- Summarize to Markdown -----------
MD_OUT="$ROOT/ablation_summary.md"
CSV_PATH="$ROOT/ablation_results.csv"

"$PY" - "$ROOT" <<'PY'
import sys, os, pandas as pd
root = sys.argv[1]
csv_path = os.path.join(root, 'ablation_results.csv')
if not os.path.exists(csv_path):
    raise SystemExit(f'[Error] CSV not found: {csv_path}')
df = pd.read_csv(csv_path)

# ensure columns & normalize
if 'attention' not in df.columns: df['attention'] = 'none'
df['attention'] = df['attention'].fillna('none').str.lower()
for col in ['ce','coral','mask4ch','sampler','ema','tta']:
    if col not in df.columns: df[col] = 0
    df[col] = df[col].astype(int)

def infer_id(r):
    key = (int(r['ce']), int(r['coral']), int(r['mask4ch']), int(r['sampler']), int(r['ema']), int(r['tta']), str(r['attention']).lower())
    mapping = {
        (1,0,0,0,0,0,'none'):'A0',
        (1,1,0,0,0,0,'none'):'A1',
        (1,1,1,0,0,0,'none'):'A2',
        (1,1,1,1,0,0,'none'):'A3',
        (1,1,1,1,1,0,'none'):'A4',
        (1,1,1,1,1,0,'eca') :'A6'
    }
    return mapping.get(key,'-')

df['exp_id'] = df.apply(infer_id, axis=1)
order = ['A0','A1','A2','A3','A4','A6','A7','-']
df['__ord__'] = df['exp_id'].apply(lambda x: order.index(x) if x in order else 999)
df = df.sort_values(['__ord__','test_qwk'], ascending=[True, False])

cols = ['exp_id','attention','ce','coral','mask4ch','sampler','ema','tta',
        'test_acc','test_f1','test_qwk','test_mAP','exp_dir']
lines = []
lines.append('# Ablation Summary')
lines.append(f'Root: {root}\n')
lines.append('| ID | Attention | CE | CORAL | Mask4Ch | WeightedSampler | EMA | TTA | Test Acc | F1 | QWK | mAP | Log Dir |')
lines.append('|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
for _, r in df[cols].iterrows():
    lines.append(f"| {r['exp_id']} | {str(r['attention'])} | {int(r['ce'])} | {int(r['coral'])} | {int(r['mask4ch'])} | {int(r['sampler'])} | {int(r['ema'])} | {int(r['tta'])} | {r['test_acc']:.4f} | {r['test_f1']:.4f} | {r['test_qwk']:.4f} | {r['test_mAP']:.4f} | {r['exp_dir']} |")

open(os.path.join(root, 'ablation_summary.md'), 'w', encoding='utf-8').write('\n'.join(lines))
print('[Summary] wrote', os.path.join(root, 'ablation_summary.md'))
PY

echo "All experiments finished. See:"
echo "  CSV : ${CSV_PATH}"
echo "  MD  : ${MD_OUT}"
echo "  LOGS: ${ROOT}/logs/"
