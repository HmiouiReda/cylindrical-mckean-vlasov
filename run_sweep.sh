#!/usr/bin/env bash
# hyperparameter sweep before the freeze: ou and kuramoto, one factor at a
# time, two data seeds, then the combined winners (tasks_plan.html, sweep)
#
#   ./run_sweep.sh --smoke     # tiny version of every stage, a few minutes
#   nohup ./run_sweep.sh > sweep.out 2>&1 &     # the real thing, ~1-3 h
#   ./run_sweep.sh --list      # write the job lists of the real thing, run nothing
#
# runs JOBS jobs at once (default 6), each with THREADS torch threads
# (default 4). a failed run is logged and the script carries on. a run whose
# metrics.json exists is skipped, so rerunning the script resumes.
# results: outputs/sweep/ and outputs/sweep.tar.gz
# progress: outputs/sweep/status.txt (one line per finished run)
# read first: outputs/sweep/sweep_report.txt

set -u -o pipefail
cd "$(dirname "$0")"
export PY=${PYTHON:-.venv/bin/python}
JOBS=${JOBS:-6}
export THREADS=${THREADS:-4}

SMOKE=0; LIST=0
[ "${1:-}" = "--smoke" ] && SMOKE=1
[ "${1:-}" = "--list" ] && LIST=1

if [ $SMOKE = 1 ]; then
  export OUT=outputs/sweep_smoke
  export EXTRA="--quick --M 300 --mdn-epochs 10 --psi-epochs 10"
  V_J="2"; V_n_fourier="2"; V_mdn_depth="1"; V_psi_depth="1"
  V_hidden="64"; V_lr="0.003"; V_sig_ou="1"; V_sig_kuramoto="1"
  TRAIN_SEEDS="1 2"
else
  export OUT=outputs/sweep
  export EXTRA=""
  V_J="1 2 3 5 8"; V_n_fourier="1 2 4 6 8"; V_mdn_depth="1 2 3 4"
  V_psi_depth="1 2 3 4"; V_hidden="64 128 256"; V_lr="0.0001 0.0003 0.001 0.003"
  V_sig_ou="1 2 3 4"; V_sig_kuramoto="1 2 3 4 5 6"
  TRAIN_SEEDS="1 2 3 4"
fi
export CACHE=$OUT/sim_cache
export STATUS=$OUT/status.txt
mkdir -p "$OUT/logs" "$OUT/runs" "$OUT/jobs"
touch "$STATUS"

log() { echo "$(date '+%F %T')  $*" | tee -a "$OUT/progress.log"; }

# one run: name, then main.py arguments; output goes to logs/<name>.log
run_one() {
  local name=$1; shift
  local dir=$OUT/runs/$name
  if [ -f "$dir/metrics.json" ]; then
    echo "$name  skipped (done before)" >> "$STATUS"; return 0
  fi
  local start=$(date +%s)
  # shellcheck disable=SC2086
  OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
    "$PY" main.py "$@" $EXTRA --baselines none --sim-cache "$CACHE" \
    --outdir "$dir" > "$OUT/logs/$name.log" 2>&1
  local rc=$?
  local mins=$(( ($(date +%s) - start) / 60 ))
  local word=ok; [ $rc = 0 ] || word="FAILED (exit $rc)"
  echo "$name  $word  ${mins} min" >> "$STATUS"
  return 0
}
export -f run_one

# run every line of a job file, JOBS at a time
run_jobs() {
  local file=$1
  log "start  $(basename "$file")  $(grep -c . "$file") runs"
  xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _ < "$file"
  log "done   $(basename "$file")"
}

# the current value of each factor, from configs/<b>.json over DEFAULT_RUN
defaults() {
  "$PY" -c "
import json, sys
from main import DEFAULT_RUN
c = json.load(open('configs/' + sys.argv[1] + '.json'))
r = {**DEFAULT_RUN, **c.get('run', {})}
print(' '.join(f'{k}={r[k]}' for k in
      ['J', 'n_fourier', 'sig_level', 'mdn_depth', 'psi_depth', 'hidden', 'lr']))
" "$1"
}

same() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 == b + 0) }'; }

# one-factor jobs of one benchmark and seed, plus the train-seed replicates
factor_jobs() {
  local b=$1 s=$2
  local -A def
  for kv in $(defaults "$b"); do def[${kv%%=*}]=${kv#*=}; done
  local sig_values; [ "$b" = ou ] && sig_values=$V_sig_ou || sig_values=$V_sig_kuramoto
  local -A vals=([J]=$V_J [n_fourier]=$V_n_fourier [sig_level]=$sig_values
                 [mdn_depth]=$V_mdn_depth [psi_depth]=$V_psi_depth
                 [hidden]=$V_hidden [lr]=$V_lr)
  local -A flag=([J]=--J [n_fourier]=--n-fourier [sig_level]=--sig-level
                 [mdn_depth]=--mdn-depth [psi_depth]=--psi-depth
                 [hidden]=--hidden [lr]=--lr)
  for f in J n_fourier sig_level mdn_depth psi_depth hidden lr; do
    for v in ${vals[$f]}; do
      same "$v" "${def[$f]}" && continue
      echo "${b}__s${s}__${f}__${v} --benchmark $b --seed $s ${flag[$f]} $v"
    done
  done
  if [ "$s" = 0 ]; then
    for t in $TRAIN_SEEDS; do
      echo "${b}__s0__ts__${t} --benchmark $b --seed 0 --train-seed $t"
    done
  fi
}

# stage A: the kuramoto defaults fill the cache (~20 min each, simulation);
# all ou runs go alongside, since ou simulates in seconds
A=$OUT/jobs/stage_a.txt
{
  for s in 0 1; do echo "kuramoto__s${s}__default --benchmark kuramoto --seed $s"; done
  for s in 0 1; do echo "ou__s${s}__default --benchmark ou --seed $s"; done
  for s in 0 1; do factor_jobs ou $s; done
  [ $SMOKE = 1 ] && echo "smoke__s0__forced_failure --benchmark no_such_benchmark"
} > "$A"

# stage B: kuramoto one-factor runs and replicates, all from the cache
B=$OUT/jobs/stage_b.txt
{ for s in 0 1; do factor_jobs kuramoto $s; done; } > "$B"

if [ $LIST = 1 ]; then
  echo "stage A: $(grep -c . "$A") runs ($A)"
  echo "stage B: $(grep -c . "$B") runs ($B)"
  echo "stage C: at most 8 runs, chosen after B"
  exit 0
fi

# environment, for the record
{ date; hostname; nproc; free -g 2>/dev/null; uptime; "$PY" --version;
  git rev-parse HEAD; git status --short; } > "$OUT/logs/environment.log" 2>&1

log "tests"
if "$PY" -m pytest tests -q > "$OUT/logs/tests.log" 2>&1; then
  echo "tests  ok" >> "$STATUS"
else
  echo "tests  FAILED (see logs/tests.log), carrying on" >> "$STATUS"
fi

run_jobs "$A"
run_jobs "$B"

# stage C: pick the winners by the rule, run them combined, order check
log "select"
"$PY" sweep_table.py "$OUT" --stage select > "$OUT/logs/select.log" 2>&1 \
  || echo "select  FAILED (see logs/select.log)" >> "$STATUS"
cat "$OUT/logs/select.log"
if [ -f "$OUT/stage_c_jobs.txt" ]; then
  cp "$OUT/stage_c_jobs.txt" "$OUT/jobs/stage_c.txt"
  run_jobs "$OUT/jobs/stage_c.txt"
fi

log "final tables"
"$PY" sweep_table.py "$OUT" --stage final > "$OUT/logs/final.log" 2>&1 \
  || echo "final  FAILED (see logs/final.log)" >> "$STATUS"

# pack everything but the cache into one file to copy home
tar -czf "$OUT.tar.gz" --exclude="$(basename "$OUT")/sim_cache" \
  -C "$(dirname "$OUT")" "$(basename "$OUT")"
log "done -> $OUT.tar.gz"
grep -c ' ok' "$STATUS" | sed 's/^/runs ok: /'
grep FAILED "$STATUS" || echo "no failures"
cat "$OUT/sweep_report.txt" 2>/dev/null
