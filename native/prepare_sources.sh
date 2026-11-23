#!/usr/bin/env bash
# Fetch the pinned operator sources outside the reproduction repository.
set -euo pipefail

if (( $# != 0 )); then
    echo "Usage: DS41_WORK=/path/to/work bash $0" >&2
    exit 2
fi
native_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$native_dir/source_revisions.sh"
src_root="${DS41_WORK:-$HOME/dsv41-mini-work}/src"
mkdir -p -- "$src_root"
stage=$(mktemp -d "$src_root/.prepare-sources.XXXXXX")
trap 'rm -rf -- "$stage"' EXIT

prepare_checkout() {
    local name=$1 revision=$2 checkout="$src_root/$1" head changes
    if [[ ! -e "$checkout" && ! -L "$checkout" ]]; then
        local url="https://gitcode.com/cann/$name.git"
        printf 'Fetching %s at %s\n' "$url" "$revision"
        git init --quiet "$stage/$name"
        git -C "$stage/$name" remote add origin "$url"
        git -C "$stage/$name" fetch --depth=1 origin "$revision"
        git -C "$stage/$name" checkout --quiet --detach "$revision"
        # A concurrent invocation may have already installed this checkout.
        mv -nT -- "$stage/$name" "$checkout" || [[ -e "$checkout" || -L "$checkout" ]]
    fi
    if [[ ! -e "$checkout/.git" ]] || ! head=$(git -C "$checkout" rev-parse --verify HEAD); then
        echo "Expected a Git checkout at $checkout ($revision). Existing path was left in place." >&2
        exit 1
    fi
    if [[ "$head" != "$revision" ]]; then
        echo "$checkout is at $head, expected $revision. Choose a fresh DS41_WORK or move the checkout." >&2
        exit 1
    fi
    changes=$(git -C "$checkout" status --porcelain --untracked-files=normal)
    if [[ -n "$changes" ]]; then
        echo "Local changes in $checkout (expected clean $revision). Use a fresh DS41_WORK or move the checkout." >&2
        exit 1
    fi
    printf 'Ready: %s (%s)\n' "$checkout" "$revision"
}

prepare_checkout ops-transformer "$OPS_TRANSFORMER_REV"
prepare_checkout cannbot-dsl "$CANNBOT_DSL_REV"
