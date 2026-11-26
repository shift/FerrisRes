#!/usr/bin/env bash
# Build, verify and publish a FerrisRes .deb from a git branch.
#
#   tools/deb_release.sh <branch> <version> [--dry-run]
#
# Example: tools/deb_release.sh fix/workgroup-2d-grid 0.2.8
#
# Steps, each of which has been done by hand and got wrong at least once:
#   1. a clean worktree at origin/<branch>
#   2. version bump in Cargo.toml and Cargo.lock, committed on a local branch
#   3. release build inside ubuntu:22.04 (glibc 2.35) — building on this host
#      produces a binary that Kaggle's 2.39 cannot... actually can, but the
#      packaging script refuses anything above 2.35, and 2.35 is the safe floor
#   4. verification: version, max GLIBC, size, sha256
#   5. upload as a new version of sectionme/ferrisres
#
# Exits non-zero on any verification failure, before publishing.
set -euo pipefail
BRANCH=${1:?branch}
VERSION=${2:?version}
DRY=${3:-}
REPO=/home/shift/code/ferrisres
WT=/tmp/opencode/wt-deb-$VERSION
OUT=/tmp/opencode/deb-out
CFG=${KAGGLE_CONFIG_DIR:-/tmp/opencode/kaggle-cfg}
export KAGGLE_CONFIG_DIR=$CFG

echo "== worktree $WT from origin/$BRANCH"
rm -rf "$WT"
git -C "$REPO" worktree add -f "$WT" "origin/$BRANCH" >/dev/null
cd "$WT"
git checkout -q -b "release/$VERSION"

echo "== version $VERSION"
python3 - "$VERSION" <<'PY'
import pathlib, re, sys
v = sys.argv[1]
p = pathlib.Path('Cargo.toml'); s = p.read_text()
s2 = re.sub(r'(?m)^version = "[0-9]+\.[0-9]+\.[0-9]+"$', f'version = "{v}"', s, count=1)
assert s2 != s, 'no root version line replaced'
p.write_text(s2)
q = pathlib.Path('Cargo.lock'); t = q.read_text()
t2 = re.sub(r'(name = "ferrisres"\nversion = ")[0-9.]+(")', rf'\g<1>{v}\g<2>', t, count=1)
assert t2 != t, 'no lock version line replaced'
q.write_text(t2)
PY
git add Cargo.toml Cargo.lock
git -c user.name=release -c user.email=release@local commit -q -m "chore: $VERSION"

if [ "$DRY" = "--dry-run" ]; then
  echo "== dry run: stopping before the container build"; exit 0
fi

echo "== release build in ubuntu:22.04 (this takes ~10 min)"
podman run --rm --network host -v "$WT:/src:Z" -w /src docker.io/library/ubuntu:22.04 bash -c '
  set -e
  apt-get update -qq >/dev/null
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl build-essential pkg-config \
    libssl-dev xz-utils dpkg-dev ca-certificates >/dev/null
  curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.100.0 --profile minimal >/dev/null 2>&1
  . $HOME/.cargo/env
  cargo build --release --no-default-features --features vulkan 2>&1 | tail -2
  ./packaging/build-deb.sh 2>&1 | tail -3
' 2>&1 | tail -8

DEB="$WT/target/deb/ferrisres_${VERSION}_amd64.deb"
test -f "$DEB" || { echo "FAIL: $DEB not produced"; exit 1; }

echo "== verify"
mkdir -p "$OUT"; cp "$DEB" "$OUT/"
SHA=$(sha256sum "$OUT/ferrisres_${VERSION}_amd64.deb" | cut -d' ' -f1)
DEBV=$(nix shell nixpkgs#dpkg --command dpkg-deb -f "$DEB" Version 2>/dev/null | tr -d '\r')
MAXGLIBC=$(objdump -T "$WT/target/release/ferrisres" 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -1)
SIZE=$(stat -c%s "$DEB")
echo "   package    $DEBV"
echo "   size       $SIZE"
echo "   max glibc  $MAXGLIBC (must be <= 2.35)"
echo "   sha256     $SHA"
[ "$DEBV" = "$VERSION" ] || { echo "FAIL: version $DEBV != $VERSION"; exit 1; }
case "$MAXGLIBC" in GLIBC_2.3[0-5]) ;; *) echo "FAIL: $MAXGLIBC exceeds 2.35"; exit 1;; esac

echo "== publish as a new version of sectionme/ferrisres"
DS=/tmp/opencode/ds-deb-$VERSION; rm -rf "$DS"; mkdir -p "$DS"
cp "$OUT/ferrisres_${VERSION}_amd64.deb" "$DS/"
printf '{"title": "ferrisres", "id": "sectionme/ferrisres", "licenses": [{"name": "other"}]}\n' > "$DS/dataset-metadata.json"
nix shell nixpkgs#python3Packages.kaggle --command kaggle datasets version -p "$DS" \
  -m "$VERSION: built from $BRANCH ($SHA)" 2>&1 | tail -2

echo "== done: $OUT/ferrisres_${VERSION}_amd64.deb  sha $SHA"
