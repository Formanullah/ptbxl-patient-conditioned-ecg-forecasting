#!/bin/bash
# Phase 1.5 evidence: download-method benchmark (Linux). Run from an empty folder that
# contains SHA256SUMS.txt and the helper scripts par.ps1 / seq.ps1, with PROJECT set to the
# repository root, ./bin/aws (pip/uv-installed AWS CLI) and ./pwsh/pwsh (portable PowerShell).
# The subset list is the 1,974 files under records100/00000/.
[ -f subset.txt ] || grep " records100/00000/" SHA256SUMS.txt | awk '{print $2}' > subset.txt
# Phase 1.5 download benchmark: same 1,974-file subset, fresh folder per method,
# SHA-256 verification afterwards. Sequential methods are capped at 300 s.
set -u
B=$(cd "$(dirname "$0")" && pwd); P=${PROJECT:?}; PY="$P/.venv/bin/python"; DL="$P/scripts/download_ptbxl.py"
S3=https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/; PN=https://physionet.org/files/ptb-xl/1.0.3/
export AWS_CA_BUNDLE=/etc/pki/tls/cert.pem
sed "s|^|$S3|" $B/subset.txt > $B/urls_s3.txt
verify() { "$PY" "$DL" --root "$1" --include records100/00000/ --verify-only 2>/dev/null | python3 -c "import sys,json;t=sys.stdin.read();d=json.loads(t[t.index('{\n'):]);print(d['status'].get('ok',0))"; }
fresh() { rm -rf "$B/d_$1"; mkdir -p "$B/d_$1"; cp $B/SHA256SUMS.txt "$B/d_$1/"; echo "$B/d_$1"; }
timed() { local s=$(date +%s.%N); "$@" >/dev/null 2>&1; echo "$(echo "$(date +%s.%N)-$s" | bc)"; }
row() { printf "%-26s first=%7.1fs verified=%4s/1974 rerun=%s\n" "$1" "$2" "$3" "$4"; }

D=$(fresh py_s3); t=$(timed "$PY" "$DL" --root $D --include records100/00000/ --mirror $S3); r=$(timed "$PY" "$DL" --root $D --include records100/00000/ --mirror $S3); row python_s3_32threads $t $(verify $D) "${r}s"
D=$(fresh py_pn); t=$(timed "$PY" "$DL" --root $D --include records100/00000/ --mirror $PN); r=$(timed "$PY" "$DL" --root $D --include records100/00000/ --mirror $PN); row python_physionet_32thr $t $(verify $D) "${r}s"
D=$(fresh curl); awk -v d=$D -v s=$S3 '{printf "url = \"%s%s\"\noutput = \"%s/%s\"\n", s, $1, d, $1}' $B/subset.txt > $B/curl.cfg
t=$(timed curl -s -f -Z --parallel-max 32 --create-dirs -K $B/curl.cfg); r=$(timed curl -s -f -Z --parallel-max 32 --create-dirs -K $B/curl.cfg); row curl_parallel_s3_32 $t $(verify $D) "${r}s(re-downloads)"
D=$(fresh aws); A="$B/bin/aws s3 sync --no-sign-request --only-show-errors s3://physionet-open/ptb-xl/1.0.3/records100/00000/ $D/records100/00000/"
t=$(timed $A); r=$(timed $A); row aws_cli_s3_sync $t $(verify $D) "${r}s"
D=$(fresh pwsh_par); t=$(timed $B/pwsh/pwsh -NoProfile -File $B/par.ps1 -Dest $D -List $B/subset.txt -Base $S3); r=$(timed $B/pwsh/pwsh -NoProfile -File $B/par.ps1 -Dest $D -List $B/subset.txt -Base $S3); row pwsh7_parallel_s3_32 $t $(verify $D) "${r}s"
D=$(fresh zip); t=$(timed curl -s -f -L -o $B/ptbxl.zip https://physionet.org/content/ptb-xl/get-zip/1.0.3/); top=$(unzip -Z1 $B/ptbxl.zip | head -1 | cut -d/ -f1)
u=$(timed unzip -q -o $B/ptbxl.zip "$top/records100/00000/*" -d $D); mv $D/$top/records100 $D/; row zip_full_1.84GB $t $(verify $D) "unzip-subset=${u}s,size=$(du -h $B/ptbxl.zip|cut -f1)"
# Sequential methods: 300 s budget, report files/s.
for m in wget_list_s3 wget_recursive_physionet pwsh7_sequential_s3; do D=$(fresh $m)
  case $m in
    wget_list_s3) cmd=(wget -q -x -nH --cut-dirs=2 -N -i $B/urls_s3.txt -P $D);;
    wget_recursive_physionet) cmd=(wget -q -r -N -c -np -nH --cut-dirs=3 -R "index.html*" ${PN}records100/00000/ -P $D);;
    pwsh7_sequential_s3) cmd=($B/pwsh/pwsh -NoProfile -File $B/seq.ps1 -Dest $D -List $B/subset.txt -Base $S3);;
  esac
  s=$(date +%s.%N); timeout 300 "${cmd[@]}" >/dev/null 2>&1; e=$(echo "$(date +%s.%N)-$s" | bc); v=$(verify $D)
  printf "%-26s ran=%6.1fs verified=%4s rate=%.2f files/s -> est. %.0f s for 1,974 files, %.1f h for 43,602\n" $m $e $v $(echo "$v/$e" | bc -l) $(echo "1974*$e/$v" | bc -l) $(echo "43602*$e/$v/3600" | bc -l)
done
