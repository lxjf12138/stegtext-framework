#!/usr/bin/env bash
# Clone the upstream baseline repositories at the commits recorded in
# BASELINE_SOURCES.json into external_baselines/. Only Consistency is needed by
# the main comparison (Stepwise calls its unchanged candidate predicate); the
# others are used by audit tools.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$root/external_baselines"

fetch() {
    local name=$1 url=$2 commit=$3 dest="$root/external_baselines/$1"
    if [ ! -d "$dest/.git" ]; then
        GIT_LFS_SKIP_SMUDGE=1 git clone --quiet "$url" "$dest"
    fi
    git -C "$dest" -c advice.detachedHead=false checkout --quiet "$commit"
    echo "$name $(git -C "$dest" rev-parse HEAD)"
}

fetch Consistency https://github.com/ryehr/Consistency.git 962473d36905f2bd3e9125ae814ee0fd1963241c
if [ "${1:-}" = "--all" ]; then
    fetch SyncPool https://github.com/7-yaya/SyncPool.git a543ef23e18de870f9de63f81bffd4b6fc6414ed
    fetch Discop https://github.com/comydream/Discop.git 3c3a10099a242eae405b49cc4d09fba1abb148ad
    fetch iMEC https://github.com/schroederdewitt/perfectly-secure-steganography.git cc36ae0b585fae4c7fcf65f7ee4db43aac2edd28
    fetch RRC https://github.com/ryehr/RRC_steganography.git dae326259e4fca8bc4fcf460dafdbc0e88a0a71a
    fetch ARIMEC https://github.com/ssokota/mec.git e56ad6619498914fe289d91516febb1fa3be050f
fi
