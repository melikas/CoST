#!/bin/bash
# Every repository path a shell script uses must exist.
#
# run.sh checks a manifest of required files before it starts, and moving scripts/_style.py
# to tasks/style.py during a cleanup left that manifest pointing at the old path. A 24-task
# GPU array then died one second in, after building its environment, with
# `ERROR: missing from ...: scripts/_style.py`.
#
# That cleanup's verification ran the tests, pyflakes, every --help and `bash -n` on every
# script. None of those reads a filename out of a shell string.
#
# Comments are skipped: they discuss files that were deliberately deleted, and prose ends
# sentences with a period that is not part of the path.
set -u
cd "$(dirname "$0")/.." || exit 1
fail=0
DIRS='scripts|tasks|models|analysis|results|baselines|data_processing'
for sh in scripts/*.sh; do
  while IFS= read -r line; do
    case "${line#"${line%%[![:space:]]*}"}" in \#*) continue ;; esac
    for tok in $(printf '%s\n' "$line" | grep -oE "\b($DIRS)/[A-Za-z0-9_./-]+"); do
      tok="${tok%.}"; tok="${tok%,}"
      case "$tok" in */) continue ;; esac
      if [ ! -e "$tok" ]; then
        echo "  MISSING  $sh names '$tok', which does not exist"; fail=1
      fi
    done
  done < "$sh"
done
[ $fail -eq 0 ] && echo "PASS -- every repository path used by a shell script exists" || exit 1
