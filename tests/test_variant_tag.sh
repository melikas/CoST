#!/bin/bash
# The shell tag and the python tag must agree, or the RQ scripts are pointed at a directory
# nothing wrote -- which is how run 2074341 lost its entire RQ suite, non-fatally, while
# reporting success.
set -u
fail=0
check() {
  ABLATE="$1"; want="$2"
  SP_TAG=""; NW_TAG=""
  case " $ABLATE " in *" --season-pool "*)
    _rest="${ABLATE#*--season-pool }"; _sp="${_rest%% *}"
    [ -n "$_sp" ] && [ "$_sp" != "spec" ] && SP_TAG="_sp-${_sp}" ;; esac
  case " $ABLATE " in *" --noise-weight "*)
    _rest="${ABLATE#*--noise-weight }"; _nw="${_rest%% *}"
    [ -n "$_nw" ] && [ "$_nw" != "0" ] && [ "$_nw" != "0.0" ] && NW_TAG="_nw${_nw}" ;; esac
  got="${SP_TAG}${NW_TAG}"
  py=$(python - "$ABLATE" <<'PY'
import sys, re
a = sys.argv[1]
m = re.search(r"--season-pool\s+(\S+)", a)
sp = "" if not m or m.group(1) == "spec" else f"_sp-{m.group(1)}"
m = re.search(r"--noise-weight\s+(\S+)", a)
nw = "" if not m or float(m.group(1)) == 0 else f"_nw{float(m.group(1)):g}"
print(sp + nw)
PY
)
  if [ "$got" != "$want" ] || [ "$py" != "$want" ]; then
    echo "  FAIL  ABLATE='$ABLATE'  shell='$got'  python='$py'  want='$want'"; fail=1
  else
    echo "  ok    ABLATE='$ABLATE' -> '$got'"
  fi
}
check ""                                             ""
check "--noise-weight 0.3"                           "_nw0.3"
check "--noise-weight 0"                             ""
check "--season-pool spec_band"                      "_sp-spec_band"
check "--season-pool spec"                           ""
check "--season-pool spec_band --noise-weight 0.05"  "_sp-spec_band_nw0.05"
check "--drop-channels Steps --noise-weight 0.3"     "_nw0.3"
[ $fail -eq 0 ] && echo "PASS" || { echo "FAILED"; exit 1; }
