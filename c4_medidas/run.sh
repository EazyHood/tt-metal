#!/bin/bash
# C4 (Claude cuenta 4): GDN decode, main vs fix on the same base, on the TT Console Wormhole N150.
# Builds two trees (~/tt-c4gdn-main at BASE, ~/tt-c4gdn-fix at FIX), then measures both with the board lock.
# Everything goes to ~/c4-gdn/. Start it with:  nohup bash ~/c4-gdn/run.sh > ~/c4-gdn/run.log 2>&1 &
set -o pipefail
BASE=4e533daf360f2c4d23e29b4f5160e0ca0d54d37d
FIX=d54fd7710ac6ac617608b16c0253c310e6ba1534
W=~/c4-gdn
mkdir -p $W
cd ~

unset VIRTUAL_ENV PYTHONPATH TT_METAL_RUNTIME_ROOT
export PYENV_VERSION=system LD_LIBRARY_PATH=/usr/local/lib:/usr/lib
# Shared instance: leave cores for the others' host timings (C8, 23-oct). C4_JOBS overrides.
export CMAKE_BUILD_PARALLEL_LEVEL=${C4_JOBS:-32}
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v python_env | paste -sd:)

echo "inicio $(date -u)"
if [ ! -d tt-c4gdn-fix ]; then
  git clone --filter=blob:none --no-checkout https://github.com/EazyHood/tt-metal.git tt-c4gdn-fix || exit 1
fi
cd ~/tt-c4gdn-fix
git fetch --filter=blob:none origin c4/gdn-decode-fp32-state c4/gdn-medida || exit 1
[ "$(git rev-parse origin/c4/gdn-decode-fp32-state)" = "$FIX" ] || { echo "ALTO: la rama del arreglo no esta en $FIX"; exit 2; }
git cat-file -e $BASE^{commit} || { echo "ALTO: falta la base $BASE"; exit 2; }
git show origin/c4/gdn-medida:c4_medidas/gdn_silicio.py > $W/gdn_silicio.py || exit 1
git checkout --detach $FIX || exit 1
[ -d ~/tt-c4gdn-main ] || git worktree add --detach ~/tt-c4gdn-main $BASE || exit 1

build() {  # $1 = tree, $2 = label
  cd $1 || return 1
  echo "== $2: submodulos $(date -u)"
  git submodule update --init --recursive > $W/sub-$2.log 2>&1 || { echo "ALTO: submodulos $2"; tail -20 $W/sub-$2.log; return 1; }
  echo "== $2: compilando $(date -u)"
  export TT_METAL_HOME=$1
  nice -n 19 ./build_metal.sh > $W/build-$2.log 2>&1 || { echo "ALTO: compilacion $2"; tail -40 $W/build-$2.log; return 3; }
  echo "== $2: python_env $(date -u)"
  PYTHON_ENV_DIR=$1/python_env ./create_venv.sh > $W/venv-$2.log 2>&1 || { echo "ALTO: venv $2"; tail -40 $W/venv-$2.log; return 4; }
  echo "== $2: listo $(date -u)"
}

medir() {  # $1 = tree, $2 = label
  cd $1 || return 1
  export TT_METAL_HOME=$1 PYTHONPATH=$1
  source $1/python_env/bin/activate
  mkdir -p $1/c4_medidas && cp $W/gdn_silicio.py $1/c4_medidas/
  echo "== $2: medida $(date -u)"
  flock ~/placa.lock timeout 3600 python c4_medidas/gdn_silicio.py $2 > $W/medida-$2.log 2>&1
  echo "medida $2 rc=$?"
  grep -E "^(label|CASE|  |FIN)" $W/medida-$2.log
  echo "== $2: pruebas de TT $(date -u)"
  T=tests/ttnn/unit_tests/operations/transformers
  # the new test comes with the fix; on main it shows what it catches
  cp ~/tt-c4gdn-fix/$T/test_fused_recurrent_gdn_long_memory.py $1/$T/ 2>/dev/null
  FILES=$(ls $T/test_fused_recurrent_gdn*.py)
  flock ~/placa.lock timeout 5400 python -m pytest -p no:cacheprovider -q -rfE $FILES > $W/pytest-$2.log 2>&1
  echo "pytest $2 rc=$?"
  tail -5 $W/pytest-$2.log
  deactivate
}

build ~/tt-c4gdn-fix fix || exit 1
build ~/tt-c4gdn-main main || exit 1
medir ~/tt-c4gdn-main main
medir ~/tt-c4gdn-fix fix
echo "FIN $(date -u)"
