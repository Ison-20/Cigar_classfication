#!/bin/bash
# =======================
# Cross-Model Validation Runner (Response to Reviewer)
# Tests Baseline (A0) vs. Optimized Framework (A4) across multiple backbones
# =======================

# ----------- Positional arg -----------
DATA=${1:-"/abs/path/to/JYB"}

# ----------- Script & Python -----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY=${PYTHON:-python}
# 确保这里指向你刚刚修改过加入了 timm 的 V7 脚本！
ABLATE_PY="${ABLATE_PY:-${SCRIPT_DIR}/main.py}" 

if [[ ! -f "$ABLATE_PY" ]]; then
  echo "[Error] Script not found: $ABLATE_PY" >&2
  exit 1
fi

# ----------- Config -----------
EPOCHS=${EPOCHS:-50}
BS=${BS:-32}
LR=${LR:-1e-4}
WORKERS=${WORKERS:-4}
DEVICE=${DEVICE:-cuda}
GPU=${GPU:-0}
SEED=${SEED:-42}

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:16:8 || true
export NVIDIA_TF32_OVERRIDE=0

STAMP=$(date +"%Y%m%d_%H%M%S")
ROOT=${ROOT:-"run/cross_model_${STAMP}"}
export ROOT
mkdir -p "$ROOT/logs"
export CUDA_VISIBLE_DEVICES="$GPU"

# ----------- Models to Test -----------
#MODELS=("resnet50" "convnext_t" "swin_t")
MODELS=("densenet" "se_resnet" "resnext")

# ----------- Experiments list (Baseline vs. A4) -----------
# 注意：Ours(A4) 严格对齐了论文中的描述：0.7/0.3比例，两阶段采样(epoch 11关闭)
EXP_LIST=(
  "Baseline|Standard CE only|--use_ce"
  "Ours(A4)|Mask4Ch+WRS(to_ep11)+EMA+CE/CORAL(0.7/0.3)|--use_ce --use_coral --ce_weight 0.7 --coral_weight 0.3 --add_mask_channel --use_weighted_sampler --switch_off_sampler_epoch 11 --use_ema"
  "ablationV3(A4)|CE+EMA+CORAL|--use_ce --use_coral --add_mask_channel --use_weighted_sampler --use_ema"
)

# ----------- Run loop -----------
idx=0
for MODEL in "${MODELS[@]}"; do
    for LINE in "${EXP_LIST[@]}"; do
        IFS='|' read -r TAG DESC FLAGS <<<"$LINE"
        
        # 组合实验名称，例如: resnet50_Baseline, swin_t_Ours(A4)
        EXP_NAME="${MODEL}_${TAG}"
        echo -e "\n================== Running: ${EXP_NAME} =================="
        echo "Description: ${DESC}"
        LOG_FILE="$ROOT/logs/${EXP_NAME}.log"

        # shellcheck disable=SC2206
        FLAG_ARR=($FLAGS)

        EXP_SEED=$((SEED + idx))
        
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

        echo "[Done] ${EXP_NAME} -> log: $LOG_FILE"
        idx=$((idx+1))
        sleep 2
    done
done

# ----------- Summarize to Markdown -----------
MD_OUT="$ROOT/cross_model_summary.md"
CSV_PATH="$ROOT/ablation_results.csv"

"$PY" - "$ROOT" <<'PY'
import sys, os, pandas as pd
root = sys.argv[1]
csv_path = os.path.join(root, 'ablation_results.csv')
if not os.path.exists(csv_path):
    raise SystemExit(f'[Error] CSV not found: {csv_path}')
df = pd.read_csv(csv_path)

# Ensure columns exist
for col in ['ce','coral','mask4ch','sampler','ema']:
    if col not in df.columns: df[col] = 0
    df[col] = df[col].astype(int)

# Identify if it's Baseline or Ours based on flags
def infer_method(r):
    if r['mask4ch'] == 1 and r['ema'] == 1 and r['coral'] == 1:
        return 'Ours (Framework)'
    return 'Baseline (CE)'

df['Method'] = df.apply(infer_method, axis=1)

# Sort for better reading
df = df.sort_values(by=['model', 'Method'], ascending=[True, True])

cols = ['model', 'Method', 'test_acc', 'test_f1', 'test_qwk', 'test_mAP', 'params_M']
lines = []
lines.append('# Cross-Model Validation Summary')
lines.append(f'Root: {root}\n')
lines.append('| Backbone Model | Config / Method | Test Acc | Macro-F1 | QWK | mAP | Params (M) |')
lines.append('|---|---|---:|---:|---:|---:|---:|')

for _, r in df[cols].iterrows():
    lines.append(f"| **{r['model']}** | {r['Method']} | {r['test_acc']:.4f} | {r['test_f1']:.4f} | {r['test_qwk']:.4f} | {r['test_mAP']:.4f} | {r['params_M']} |")

open(os.path.join(root, 'cross_model_summary.md'), 'w', encoding='utf-8').write('\n'.join(lines))
print('[Summary] wrote', os.path.join(root, 'cross_model_summary.md'))
PY

echo "All 6 experiments finished! Review the summary at:"
echo "  MD  : ${MD_OUT}"