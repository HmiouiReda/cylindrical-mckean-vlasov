#!/usr/bin/env bash
# task 5: every run in one go, one at a time, results packed into one file
#
#   ./run_task5.sh --smoke     # tiny versions of every step, a few minutes
#   nohup ./run_task5.sh > task5.out 2>&1 &     # the real thing, ~6 h
#
# a failed step is logged and the script carries on with the next one.
# results: outputs/task5/ and outputs/task5_results.tar.gz
# progress: outputs/task5/status.txt (one line per finished step)

set -u -o pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-.venv/bin/python}

SMOKE=0
[ "${1:-}" = "--smoke" ] && SMOKE=1

if [ $SMOKE = 1 ]; then
  OUT=outputs/task5_smoke
  MAIN_EXTRA=(--quick --M 300 --mdn-epochs 10 --psi-epochs 10)
  SAVE=(--save-model 10)
  ES_EXTRA=(--k 10 --n-hi 2000 --refine 2 --copies 4 --plugin-sizes 50,200
            --J-list 1,3 --em-iters 30)
  KL_EXTRA=(--k 5 --kl-samples 5000 --J-list 1,6)
  M_LIST="300 400"
else
  OUT=outputs/task5
  MAIN_EXTRA=()
  SAVE=(--save-model)
  ES_EXTRA=()
  KL_EXTRA=()
  M_LIST="1000 2000 8000"
fi
mkdir -p "$OUT/logs"
STATUS="$OUT/status.txt"
: > "$STATUS"

# run one step: name, then the command; output goes to logs/<name>.log
step() {
  local name=$1; shift
  local start=$(date +%s)
  echo "$(date '+%F %T')  start  $name" | tee -a "$OUT/progress.log"
  "$@" > "$OUT/logs/$name.log" 2>&1
  local rc=$?
  local mins=$(( ($(date +%s) - start) / 60 ))
  local word=ok; [ $rc = 0 ] || word="FAILED (exit $rc)"
  echo "$name  $word  ${mins} min" | tee -a "$STATUS"
}

# environment, for the record
{ date; hostname; nproc; free -g 2>/dev/null; $PY --version; git rev-parse HEAD; } \
  > "$OUT/logs/environment.log" 2>&1

step tests $PY -m pytest tests -q

# 1. models (also task 8's cost runs)
for b in double_well multiplicative kuramoto; do
  step "model_$b" $PY main.py --benchmark $b "${SAVE[@]}" "${MAIN_EXTRA[@]}" \
    --outdir "$OUT/$b"
done

# 2. the reference check
for b in double_well multiplicative kuramoto; do
  step "check_$b" $PY error_sources.py --run "$OUT/$b" "${ES_EXTRA[@]}"
done

# 3. particle number in training
for n in 50 800; do
  step "dw_N$n" $PY main.py --benchmark double_well --N $n "${MAIN_EXTRA[@]}" \
    --outdir "$OUT/dw_N$n"
done

# 4. number of scenarios
for m in $M_LIST; do
  step "dw_M$m" $PY main.py --benchmark double_well "${MAIN_EXTRA[@]}" --M $m \
    --outdir "$OUT/dw_M$m"
done

# 5. ou2d
step ou2d_e1200 $PY main.py --benchmark ou2d "${MAIN_EXTRA[@]}" \
  --mdn-epochs $([ $SMOKE = 1 ] && echo 20 || echo 1200) --outdir "$OUT/ou2d_e1200"
step ou2d_kl $PY error_sources.py --kl-only --benchmark ou2d "${KL_EXTRA[@]}" \
  --out "$OUT/ou2d_kl.json"

# pack everything into one file to copy home
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "$(date '+%F %T')  done -> $OUT.tar.gz" | tee -a "$OUT/progress.log"
cat "$STATUS"
