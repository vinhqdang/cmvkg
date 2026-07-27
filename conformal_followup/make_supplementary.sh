#!/usr/bin/env bash
# Builds the anonymous supplementary-material archive for submission.
#
# Deliberately EXCLUDES the internal working logs (REVIEW_FINDINGS.md,
# CANONICAL_NUMBERS.md, NEW_ANALYSES.md, RESULTS.md, ALGORITHM.md,
# SEQUENTIAL.md). They carry revision history -- "an earlier draft claimed X",
# reviewer critique, retraction notes -- and a reviewer reading them would be
# reading the paper arguing with itself. Every number in them is reproducible
# from the scripts that ARE shipped, so nothing a reviewer needs is withheld.
#
# Four classes of de-anonymising leak were found when this was first built, and
# the scan below checks for all of them:
#   1. OAuth tokens / credentials in the GPU extraction scripts
#   2. author names, institution domains
#   3. code comments naming the authors' prior system as a familiar reference
#      (citing it in the paper is fine and third-person; a comment describing
#      its internals reads as ownership)
#   4. absolute paths containing the repository name, which alone identifies
#      the authors because it matches the cited prior work
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$SRC/../ccrc-supplementary.zip}"
S="$(mktemp -d)/ccrc-supplementary"
mkdir -p "$S"/{code,data,figures,data/qual_imgs}

cp "$SRC"/*.py                                              "$S/code/"
cp "$SRC"/raw_scores.json "$SRC"/owlv2_scores.json \
   "$SRC"/exp*.json "$SRC"/qual_picks.json                  "$S/data/"
cp "$SRC"/qual_imgs/*                                       "$S/data/qual_imgs/"
cp "$SRC"/*.png                                             "$S/figures/"
cp "$SRC"/SUPPLEMENTARY_README.md                           "$S/README.md"
rm -rf "$S/code/__pycache__"

# sanitize (idempotent)
sed -i "s|closer to CMVKG-Guard's L1|closer to a detector-based verification layer|" "$S/code/colab_exp5_owlv2.py"
sed -i "s|(proxy for CMVKG-Guard's UVS)|(proxy for a verification score)|"          "$S/code/conformal_sim.py"
grep -rl "/home/user" "$S" --include=*.py | while read -r f; do
  sed -i 's|"/home/user/[^"]*/manuscript_tmlr/\([^"]*\)"|os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "\1")|g' "$f"
  sed -i 's|/home/user/[a-z]*/conformal_followup/|./|g; s|/home/user/[a-z]*/|./|g' "$f"
done

# fail loudly rather than ship a leak
if grep -rilE "vinhqdang|buv\.edu|Quang|Phuong|uef\.edu|datla|cmvkg|/home/user|4/0AXEQx|ya29\.|client_secret|refresh_token" "$S" 2>/dev/null; then
  echo "ABORT: de-anonymising content found above" >&2; exit 1
fi
python3 - "$S" <<'PY'
import sys, glob
from PIL import Image
bad = []
for f in glob.glob(sys.argv[1] + "/**/*.png", recursive=True) + glob.glob(sys.argv[1] + "/**/*.jpg", recursive=True):
    for k, v in (Image.open(f).info or {}).items():
        if isinstance(v, str) and any(t in v.lower() for t in ("vinh", "buv", "quang", "phuong", "cmvkg")):
            bad.append((f, k))
raise SystemExit("ABORT: image metadata leak %s" % bad if bad else 0)
PY

rm -f "$OUT"
(cd "$(dirname "$S")" && zip -qr "$OUT" ccrc-supplementary -x '*__pycache__*' '*.DS_Store')
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
