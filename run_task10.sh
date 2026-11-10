#!/usr/bin/env bash
# task 10: the final campaign from the frozen configuration (f74c3a4).
# five seeds on every functional, the stale numbers of section 4, task 5's
# network rows, task 8's cost runs, and reda's task 3 runs (tasks_plan.html)
#
#   ./run_task10.sh --smoke     # tiny version of every stage, ~15 min
#   nohup ./run_task10.sh > task10.out 2>&1 &     # the real thing, ~4-6 h
#   ./run_task10.sh --list      # write the job lists, run nothing
#
# stages: 1 cost runs, seed 0, one at a time (they fill the cache);
# 2 moments seeds 1-4 (fill the cache); 3 everything else; R reda's runs from
# task10_reda_jobs.txt; F tables and tarball.
# a failed job is logged and the script carries on. a job with a .done marker
# is skipped, so rerunning resumes; after filling task10_reda_jobs.txt, rerun
# to run only reda's stage. keep outputs/task10/sim_cache for that.
# results: outputs/task10/ and outputs/task10.tar.gz (without the cache)
# read first: outputs/task10/status.txt, then paper/checks.txt

set -u -o pipefail
cd "$(dirname "$0")"
export PY=${PYTHON:-.venv/bin/python}
JOBS=${JOBS:-6}
export THREADS=${THREADS:-4}
FROZEN=f74c3a4

SMOKE=0; LIST=0
for a in "$@"; do
  case $a in
    --smoke) SMOKE=1 ;;
    --list) LIST=1 ;;
    *) echo "unknown argument: $a"; exit 2 ;;
  esac
done

if [ $SMOKE = 1 ]; then
  export OUT=outputs/task10_smoke
  export EXTRA="--quick --M 300 --mdn-epochs 10 --psi-epochs 10"
  SEEDS="0 1"
  SAVE="--save-model 10"
  ES_ARGS="--k 10 --n-hi 2000 --refine 2 --copies 4 --plugin-sizes 50,200 --J-list 1,3 --em-iters 30"
  M_LIST="300 400 500"
  E1200=20
  KKEEP="1,8"
else
  export OUT=outputs/task10
  export EXTRA=""
  SEEDS="0 1 2 3 4"
  SAVE="--save-model"
  ES_ARGS=""
  M_LIST="8000 2000 1000"
  E1200=1200
  KKEEP="1,8,64,200"
fi
export CACHE=$OUT/sim_cache
export STATUS=$OUT/status.txt
REDA=task10_reda_jobs.txt
mkdir -p "$OUT/logs" "$OUT/runs" "$OUT/jobs"
touch "$STATUS"

log() { echo "$(date '+%F %T')  $*" | tee -a "$OUT/progress.log"; }

# one job: name, kind, then arguments. kinds:
#   main      main.py with the simulation cache
#   nocache   main.py without it
#   es        error_sources.py, writes <dir>/error_sources.json
# output in runs/<name>/, log in logs/<name>.log, .done marker on success
run_one() {
  local name=$1 kind=$2; shift 2
  local dir=$OUT/runs/$name
  if [ -f "$dir/.done" ]; then
    echo "$name  skipped (done before)" >> "$STATUS"; return 0
  fi
  mkdir -p "$dir"
  local start=$(date +%s)
  local cmd
  case $kind in
    # extras first, so a job's own flags win (argparse keeps the last value)
    main)    cmd=("$PY" main.py $EXTRA "$@" --sim-cache "$CACHE" --outdir "$dir") ;;
    nocache) cmd=("$PY" main.py $EXTRA "$@" --outdir "$dir") ;;
    es)      cmd=("$PY" error_sources.py "$@" --out "$dir/error_sources.json") ;;
    *)       echo "$name  FAILED (unknown kind $kind)" >> "$STATUS"; return 0 ;;
  esac
  OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS "${cmd[@]}" \
    > "$OUT/logs/$name.log" 2>&1
  local rc=$?
  local mins=$(( ($(date +%s) - start) / 60 ))
  if [ $rc = 0 ]; then
    touch "$dir/.done"; echo "$name  ok  ${mins} min" >> "$STATUS"
  else
    echo "$name  FAILED (exit $rc)  ${mins} min" >> "$STATUS"
  fi
  return 0
}
export -f run_one

# run every line of a job file, $2 at a time
run_jobs() {
  local file=$1 n=$2
  # xargs -L joins a line ending in a blank to the next one
  sed -i 's/[[:space:]]*$//' "$file"
  log "start  $(basename "$file")  $(grep -c . "$file") jobs, $n at a time"
  xargs -P "$n" -L 1 bash -c 'run_one "$@"' _ < "$file"
  log "done   $(basename "$file")"
}

ALL7="kuramoto double_well multiplicative ou_gmm ou_uniform ou ou2d"
ONE_D="kuramoto double_well multiplicative ou_gmm ou_uniform ou"
ORDER="kuramoto double_well"
PARTICLE="kuramoto double_well multiplicative"
is_particle() { case " $PARTICLE " in *" $1 "*) return 0 ;; esac; return 1; }
main_job() {   # benchmark functional seed [extra flags]
  local b=$1 f=$2 s=$3; shift 3
  echo "${b}__${f}__s${s} main --benchmark $b --functional $f --seed $s --baselines full $*"
}

# stage 1: seed-0 moments, one at a time; cache misses, so they time the
# simulation (task 8) and fill the cache; models saved for error_sources.py
S1=$OUT/jobs/stage1_cost.txt
{
  for b in $ALL7; do
    if is_particle "$b"; then main_job "$b" moments 0 $SAVE
    else main_job "$b" moments 0; fi
  done
} > "$S1"

# stage 2: moments on the other seeds, longest first; fills the cache
S2=$OUT/jobs/stage2_fill.txt
{
  for b in $ALL7; do
    for s in $SEEDS; do [ "$s" = 0 ] || main_job "$b" moments "$s"; done
  done
} > "$S2"

# stage 3: everything that remains, longest first
S3=$OUT/jobs/stage3_rest.txt
{
  # task 5: the reference check on the new models, then its network runs
  for b in kuramoto double_well multiplicative; do
    echo "task5__check_${b} es --run $OUT/runs/${b}__moments__s0 $ES_ARGS"
  done
  for m in $M_LIST; do
    echo "task5__dw_M${m} nocache --benchmark double_well --M $m"
  done
  for n in 800 50; do
    echo "task5__dw_N${n} nocache --benchmark double_well --N $n"
  done
  # the 40 functional runs, all cache hits
  for b in $ORDER; do for s in $SEEDS; do main_job "$b" order "$s"; done; done
  for b in $ONE_D; do for s in $SEEDS; do main_job "$b" quantile "$s"; done; done
  echo "task5__ou2d_e1200 nocache --benchmark ou2d --mdn-epochs $E1200"
  # section 4.7: single observation and the K_keep sweep, five seeds
  for s in $SEEDS; do
    echo "kkeep__ou__s${s} main --benchmark ou --seed $s --sweep-kkeep $KKEEP --baselines none"
  done
  for s in $SEEDS; do
    echo "single__ou__s${s} main --benchmark ou --seed $s --single-particle --baselines emp"
  done
  # section 4.8: no network, confirms the published numbers
  echo "ito__multiplicative nocache --benchmark multiplicative --ito-strat-check"
  [ $SMOKE = 1 ] && echo "smoke__forced_failure main --benchmark no_such_benchmark"
} > "$S3"

# stage R: reda's runs, one per line in task10_reda_jobs.txt: name, then
# main.py arguments; comments and blank lines ignored
SR=$OUT/jobs/stageR_reda.txt
{
  [ -f "$REDA" ] && sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$REDA" \
    | awk '{ name = $1; $1 = ""; print "reda__" name " main" $0 }'
} > "$SR"

if [ $LIST = 1 ]; then
  for f in "$S1" "$S2" "$S3" "$SR"; do
    echo "$(basename "$f"): $(grep -c . "$f") jobs ($f)"
  done
  exit 0
fi

# stage 0: the numbers must come from the frozen code
GUARD="configs main.py benchmarks features.py mdn.py cylindrical.py evaluate.py sim_cache.py error_sources.py"
# shellcheck disable=SC2086
if ! git diff --quiet $FROZEN -- $GUARD; then
  echo "number-producing files differ from the frozen commit $FROZEN:"
  # shellcheck disable=SC2086
  git diff --stat $FROZEN -- $GUARD
  if [ $SMOKE = 0 ]; then echo "stopping"; exit 1; fi
  echo "smoke: carrying on"
fi

{ date; hostname; nproc; free -g 2>/dev/null; uptime; "$PY" --version;
  git rev-parse HEAD; git status --short; } > "$OUT/logs/environment.log" 2>&1

log "tests"
if "$PY" -m pytest tests -q > "$OUT/logs/tests.log" 2>&1; then
  echo "tests  ok" >> "$STATUS"
else
  echo "tests  FAILED (see logs/tests.log), carrying on" >> "$STATUS"
fi

run_jobs "$S1" 1
run_jobs "$S2" "$JOBS"
run_jobs "$S3" "$JOBS"
if [ -s "$SR" ]; then run_jobs "$SR" "$JOBS"
else log "stage R: no jobs in $REDA"; fi

log "tables"
"$PY" task10_tables.py "$OUT" > "$OUT/logs/tables.log" 2>&1 \
  || echo "tables  FAILED (see logs/tables.log)" >> "$STATUS"
tail -n 30 "$OUT/logs/tables.log"

# everything but the cache in one file to copy home
tar -czf "$OUT.tar.gz" --exclude="$(basename "$OUT")/sim_cache" \
  -C "$(dirname "$OUT")" "$(basename "$OUT")"
log "done -> $OUT.tar.gz"
grep -c '  ok' "$STATUS" | sed 's/^/jobs ok: /'
grep FAILED "$STATUS" || echo "no failures"
