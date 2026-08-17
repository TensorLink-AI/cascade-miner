#!/usr/bin/env bash
# One-command sync with the upstream cascade competition.
#
# The competition changes as decisions land in TensorLink-AI/cascade
# (`decisions/DEC-CA-*.md` + `chain.toml`), not as GitHub issues. This script
# automates the checklist at the bottom of notes/CONTRACT.md:
#
#   1. pull the read-only reference clone
#   2. reinstall it into the miner venv — scoring imports come from the
#      INSTALLED package, so a pull without reinstall still scores with the
#      old metric
#   3. surface what changed (chain.toml, decisions/, eval code) so the notes
#      can be updated where they went stale
#   4. regenerate notes/upstream-state.json + notes/UPSTREAM.md — the machine
#      half of that knowledge, which the tests pin notes/CONTRACT.md against
#   5. stamp the "Last synced" line in notes/CONTRACT.md
#   6. run the offline test suite
#
# Safe to run on a cron. `--check` only reports whether upstream moved
# (exit 1 if behind) and changes nothing, so it can drive an alert.
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

CASCADE_DIR="${CASCADE_DIR:-/root/cascade}"
VENV="${CASCADE_VENV:-$ROOT/.venv}"
VENV_PY="$VENV/bin/python"
EXTRAS="train,hippius,chain"
CHECK_ONLY=0

# Paths whose changes affect how we compete; anything else upstream is noise.
MINER_FACING=("chain.toml" "decisions/" "cascade/eval/" "cascade/shared/config.py" "docs/INTERFACE.md")

usage() {
    cat <<'EOF'
Usage: bash scripts/sync.sh [--check]

Sync the cascade reference clone and miner venv with upstream:
pull, reinstall into the venv, report miner-facing changes, regenerate
notes/upstream-state.json, stamp notes/CONTRACT.md, run tests.

  --check   fetch and report whether upstream moved (and whether the
            committed snapshot still matches it); change nothing.
            Exits 1 when behind (cron-friendly), 0 when current.
EOF
}

case "${1:-}" in
    "") ;;
    --check) CHECK_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
esac

step() { printf '\n=== %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
fail() { printf '    FAIL: %s\n' "$*" >&2; exit 1; }

[ -d "$CASCADE_DIR/.git" ] || fail "$CASCADE_DIR is not a git checkout; clone the read-only Cascade reference there or set CASCADE_DIR"

step "fetch upstream"
OLD="$(git -C "$CASCADE_DIR" rev-parse HEAD)"
git -C "$CASCADE_DIR" fetch --quiet origin || fail "git fetch failed"
UPSTREAM_REF="$(git -C "$CASCADE_DIR" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || true)"
[ -n "$UPSTREAM_REF" ] || UPSTREAM_REF="$(git -C "$CASCADE_DIR" symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null || echo origin/main)"
UPSTREAM="$(git -C "$CASCADE_DIR" rev-parse "$UPSTREAM_REF" 2>/dev/null)" \
    || fail "cannot resolve $UPSTREAM_REF in $CASCADE_DIR"

if [ "$OLD" = "$UPSTREAM" ]; then
    info "already current at ${OLD:0:7}"
    if [ "$CHECK_ONLY" = 1 ]; then
        # The clone has not moved, but the committed snapshot still has to
        # match it — a hand-edit or a half-finished sync shows up here.
        "${PYTHON:-python3}" "$ROOT/scripts/upstream_state.py" \
            --cascade-dir "$CASCADE_DIR" --check | sed 's/^/    /'
        [ "${PIPESTATUS[0]}" = 0 ] || exit 1
        exit 0
    fi
else
    BEHIND="$(git -C "$CASCADE_DIR" rev-list --count "$OLD..$UPSTREAM")"
    info "behind by $BEHIND commit(s): ${OLD:0:7} -> ${UPSTREAM:0:7}"
    if [ "$CHECK_ONLY" = 1 ]; then
        git -C "$CASCADE_DIR" log --oneline "$OLD..$UPSTREAM" | sed 's/^/    /'
        exit 1
    fi
    git -C "$CASCADE_DIR" merge --ff-only "$UPSTREAM" >/dev/null \
        || fail "fast-forward failed — the reference clone must never carry local edits"

    step "miner-facing changes ($OLD..${UPSTREAM:0:7})"
    CHANGED="$(git -C "$CASCADE_DIR" diff --name-only "$OLD" "$UPSTREAM" -- "${MINER_FACING[@]}")"
    if [ -n "$CHANGED" ]; then
        printf '%s\n' "$CHANGED" | sed 's/^/    /'
        info "REVIEW these against notes/CONTRACT.md — cadence, budgets, dedup and metric live here"
        git -C "$CASCADE_DIR" diff --stat "$OLD" "$UPSTREAM" -- chain.toml | sed 's/^/    /'
    else
        info "none — upstream commits do not touch contract, decisions, or eval code"
    fi
fi

NEW="$(git -C "$CASCADE_DIR" rev-parse --short HEAD)"

step "reinstall into venv"
if [ ! -x "$VENV_PY" ]; then
    fail "no venv at $VENV; run scripts/setup.sh first (or set CASCADE_VENV)"
fi
if ! command -v uv >/dev/null 2>&1; then
    fail "uv not on PATH; install it (https://astral.sh/uv) or reinstall manually"
fi
uv pip install --quiet --python "$VENV_PY" "$CASCADE_DIR[$EXTRAS]" \
    || fail "uv pip install '$CASCADE_DIR[$EXTRAS]' failed"
"$VENV_PY" -c 'import cascade' || fail "venv cannot import cascade after reinstall"
info "venv now runs cascade $NEW"

step "regenerate the upstream snapshot"
# Same generator the upstream-sync workflow runs, so the local box and CI can
# never disagree about what upstream says. Stdlib only — it does not need the
# venv, and it must keep working before a reinstall.
"${PYTHON:-python3}" "$ROOT/scripts/upstream_state.py" --cascade-dir "$CASCADE_DIR" \
    || fail "could not regenerate notes/upstream-state.json"

step "tests"
# BEFORE the stamp, deliberately. The suite carries the prose pins
# (tests/test_stale_references.py), so a pass is the evidence that
# notes/CONTRACT.md still agrees with the snapshot just regenerated — which is
# exactly what the stamp claims.
if (cd "$ROOT" && "$VENV_PY" -m unittest discover -s tests -q); then
    TESTS_PASSED=1
    info "test suite passed"
else
    TESTS_PASSED=0
    info "test suite FAILED"
fi

step "stamp notes/CONTRACT.md"
if [ "$TESTS_PASSED" = 0 ]; then
    info "NOT stamping — the stamp means 'prose reviewed at this revision', and"
    info "the prose pins above say notes/CONTRACT.md still contradicts cascade $NEW."
    info "Fold the reported changes in, then re-run this script to stamp."
    exit 1
fi
STAMP="Last synced: $(date -u +%F), cascade \`$NEW\`."
if grep -q '^Last synced:' "$ROOT/notes/CONTRACT.md"; then
    sed -i "s|^Last synced:.*|$STAMP|" "$ROOT/notes/CONTRACT.md"
    info "$STAMP"
else
    info "no 'Last synced:' line found in notes/CONTRACT.md — not stamping"
fi

step "done"
info "remember: numbers from before a metric change are not comparable —"
info "re-score saved per-window components with the reinstalled library"
info "(CLAUDE.md Rule 3) before comparing anything to a post-sync result."
