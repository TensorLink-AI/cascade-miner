#!/usr/bin/env bash
# One-command operator setup for cascade-miner.
#
# Every step is idempotent and never overwrites an existing venv, SSH key, or
# wallet. Steps that need money, wallet secrets, or the chain are reported for
# the operator rather than performed here: wallet creation is delegated to the
# operator-owned ops/create-next-hotkey wrapper, and registration is only ever
# printed as a next step.
#
#   bash scripts/setup.sh --cascade-dir /path/to/cascade
#
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

CASCADE_DIR="${CASCADE_DIR:-/root/cascade}"
VENV="${CASCADE_VENV:-$ROOT/.venv}"
PYTHON_VERSION="3.11"
EXTRAS="train,hippius,chain"
POOL_SNAPSHOT="${CASCADE_EVAL_POOL_SNAPSHOT:-latest}"
SSH_KEY="${LIUM_SSH_KEY:-$HOME/.ssh/id_ed25519}"
WALLET_NAME="${CASCADE_WALLET_NAME:-}"
WALLET_HOTKEY="${CASCADE_WALLET_HOTKEY:-default}"
WALLET_PATH="${BT_WALLET_PATH:-$HOME/.bittensor/wallets}"
CANDIDATE="generators/candidate"

WITH_WALLET=0
DRY_RUN=0
SKIP_VENV=0
SKIP_LIUM=0
SKIP_SSH_KEY=0
SKIP_POOL=0
SKIP_SEED=0
SKIP_VERIFY=0

READY=()
ACTIONS=()

usage() {
    cat <<'EOF'
Usage: bash scripts/setup.sh [options]

Prepares an operator host for cascade-miner:

  1. create the Python venv and install the local Cascade checkout
  2. install the Lium CLI (only needed to rent GPU pods)
  3. generate an SSH key for pod access if none exists
  4. report Bittensor wallet status (creation is delegated, see below)
  5. download the latest eval-pool snapshot
  6. seed the controller state file
  7. run `cascade verify` on the starter candidate
  8. print what is ready and what still needs operator action

Options:
  --cascade-dir PATH      read-only Cascade reference checkout
                          (default: $CASCADE_DIR or /root/cascade)
  --venv PATH             project venv to create (default: <repo>/.venv)
  --python VERSION        Python version for the venv (default: 3.11)
  --extras LIST           Cascade extras to install
                          (default: train,hippius,chain; torch is only needed
                          on GPU pods, so --extras hippius,chain saves disk)
  --eval-pool-snapshot S  latest (default), all, or a dated snapshot name
  --wallet-name NAME      wallet to look for in step 4
  --wallet-hotkey NAME    hotkey to look for in step 4 (default: default)
  --with-wallet           run the operator-owned create-hotkey wrapper
                          ($CASCADE_CREATE_HOTKEY_COMMAND, default
                          ops/create-next-hotkey). It refuses until an operator
                          implements it; this script never touches wallet
                          secrets, mnemonics, or the chain itself.
  --skip-venv, --skip-lium, --skip-ssh-key, --skip-pool, --skip-seed,
  --skip-verify           skip individual steps
  --dry-run               print what each step would run, change nothing
  -h, --help              show this help
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --cascade-dir) CASCADE_DIR="${2:?--cascade-dir needs a path}"; shift 2 ;;
        --venv) VENV="${2:?--venv needs a path}"; shift 2 ;;
        --python) PYTHON_VERSION="${2:?--python needs a version}"; shift 2 ;;
        --extras) EXTRAS="${2:?--extras needs a comma-separated list}"; shift 2 ;;
        --eval-pool-snapshot) POOL_SNAPSHOT="${2:?--eval-pool-snapshot needs a value}"; shift 2 ;;
        --wallet-name) WALLET_NAME="${2:?--wallet-name needs a name}"; shift 2 ;;
        --wallet-hotkey) WALLET_HOTKEY="${2:?--wallet-hotkey needs a name}"; shift 2 ;;
        --with-wallet) WITH_WALLET=1; shift ;;
        --skip-venv) SKIP_VENV=1; shift ;;
        --skip-lium) SKIP_LIUM=1; shift ;;
        --skip-ssh-key) SKIP_SSH_KEY=1; shift ;;
        --skip-pool) SKIP_POOL=1; shift ;;
        --skip-seed) SKIP_SEED=1; shift ;;
        --skip-verify) SKIP_VERIFY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

VENV_PY="$VENV/bin/python"

step() { printf '\n=== %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
# Under --dry-run nothing actually ran, so a "ready" line is a plan, not a fact.
ready() {
    if [ "$DRY_RUN" = 1 ]; then
        READY+=("(unverified, dry run) $1")
        info "would check: $1"
        return
    fi
    READY+=("$1")
    info "ok: $1"
}
action() { ACTIONS+=("$1"); info "needs action: $1"; }

# Run a command, or print it under --dry-run. Callers check the status.
attempt() {
    if [ "$DRY_RUN" = 1 ]; then
        info "would run: $*"
        return 0
    fi
    "$@"
}

# Same, but quiet on success and echoing the tail of the output on failure.
attempt_quiet() {
    if [ "$DRY_RUN" = 1 ]; then
        info "would run: $*"
        return 0
    fi
    local log status
    log="$(mktemp)"
    "$@" >"$log" 2>&1
    status=$?
    [ "$status" -eq 0 ] || tail -n 15 "$log" | sed 's/^/    | /'
    rm -f "$log"
    return "$status"
}

step "0/8 preflight"
info "repository:   $ROOT"
info "cascade dir:  $CASCADE_DIR"
info "venv:         $VENV"
[ "$DRY_RUN" = 1 ] && info "dry run: no changes will be made"
if [ -d "$CASCADE_DIR/.git" ]; then
    ready "Cascade reference checkout present ($CASCADE_DIR)"
else
    action "clone the read-only Cascade reference to $CASCADE_DIR (never edit it), or pass --cascade-dir"
fi
if [ -r /proc/meminfo ]; then
    MEM_MB="$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)"
    if [ -n "${MEM_MB:-}" ] && [ "$MEM_MB" -lt 4096 ]; then
        action "only ${MEM_MB} MB RAM available; \`cascade verify\` can be OOM-killed on large corpora"
    else
        info "available memory: ${MEM_MB:-unknown} MB"
    fi
fi
if [ -f "$ROOT/.env" ]; then
    ready ".env present"
else
    action "cp example.env .env and fill in HF_TOKEN, HIPPIUS_HUB_TOKEN, LIUM_API_KEY"
fi
[ -n "${HF_TOKEN:-}" ] || action "export HF_TOKEN before syncing the eval pool; anonymous Hub requests are rate-limited (429)"

step "1/8 Python venv and Cascade install"
if [ "$SKIP_VENV" = 1 ]; then
    info "skipped"
elif [ ! -d "$CASCADE_DIR" ] && [ "$DRY_RUN" = 0 ]; then
    action "cannot install Cascade: $CASCADE_DIR does not exist"
else
    if ! command -v uv >/dev/null 2>&1; then
        info "installing uv"
        attempt_quiet bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
        PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi
    if command -v uv >/dev/null 2>&1 || [ "$DRY_RUN" = 1 ]; then
        if [ -x "$VENV_PY" ]; then
            info "venv already exists"
        else
            attempt_quiet uv venv --python "$PYTHON_VERSION" "$VENV" \
                || action "uv venv failed for Python $PYTHON_VERSION"
        fi
        info "installing '$CASCADE_DIR[$EXTRAS]' (this takes a few minutes)"
        if attempt_quiet uv pip install --python "$VENV_PY" "$CASCADE_DIR[$EXTRAS]"; then
            if attempt_quiet "$VENV_PY" -c 'import cascade, huggingface_hub'; then
                ready "venv at $VENV imports cascade and huggingface_hub"
            else
                action "the venv cannot import cascade; re-run: uv pip install --python $VENV_PY '$CASCADE_DIR[$EXTRAS]'"
            fi
        else
            action "installing Cascade failed; see the output above"
        fi
    else
        action "install uv (https://astral.sh/uv) or create $VENV manually"
    fi
fi

step "2/8 Lium CLI"
if [ "$SKIP_LIUM" = 1 ]; then
    info "skipped"
elif command -v lium >/dev/null 2>&1; then
    ready "lium on PATH ($(command -v lium))"
else
    attempt_quiet bash -c 'curl -fsSL https://lium.io/install.sh | bash'
    if command -v lium >/dev/null 2>&1; then
        ready "lium installed ($(command -v lium))"
    else
        action "add the installer's bin directory (often ~/.lium/bin) to PATH, then check: lium --version"
    fi
fi
[ -n "${LIUM_API_KEY:-}" ] || action "export LIUM_API_KEY before any paid evaluation"

step "3/8 SSH key for pod access"
if [ "$SKIP_SSH_KEY" = 1 ]; then
    info "skipped"
elif [ -f "$SSH_KEY" ]; then
    ready "SSH key present ($SSH_KEY)"
else
    attempt mkdir -p "$(dirname "$SSH_KEY")"
    if attempt_quiet ssh-keygen -t ed25519 -N '' -C "cascade-miner" -f "$SSH_KEY"; then
        ready "generated SSH key $SSH_KEY"
        action "register ${SSH_KEY}.pub with Lium so rented pods accept it"
    else
        action "generate an SSH key at $SSH_KEY (ssh-keygen -t ed25519)"
    fi
fi

step "4/8 Bittensor wallet"
# Wallet secrets are deliberately out of scope for this script: creation,
# registration, and submission all run through operator-owned ops/ wrappers.
COLDKEY_DIR="$WALLET_PATH/${WALLET_NAME:-<wallet-name>}"
if [ -n "$WALLET_NAME" ] && [ -f "$COLDKEY_DIR/coldkeypub.txt" ]; then
    ready "coldkey $WALLET_NAME found under $WALLET_PATH"
    if [ -f "$COLDKEY_DIR/hotkeys/$WALLET_HOTKEY" ]; then
        ready "hotkey $WALLET_NAME/$WALLET_HOTKEY found"
    else
        action "create hotkey $WALLET_HOTKEY for wallet $WALLET_NAME with your operator wrapper"
    fi
elif [ -n "$WALLET_NAME" ]; then
    action "no coldkey $WALLET_NAME under $WALLET_PATH"
else
    action "pass --wallet-name to check a wallet, and keep wallet files outside this repository"
fi
if [ "$WITH_WALLET" = 1 ]; then
    CREATE_CMD="${CASCADE_CREATE_HOTKEY_COMMAND:-$ROOT/ops/create-next-hotkey}"
    info "delegating wallet creation to: $CREATE_CMD"
    if attempt bash -c "$CREATE_CMD"; then
        ready "operator create-hotkey wrapper completed"
    else
        action "$CREATE_CMD refused or failed; implement a reviewed, non-interactive wrapper before enabling create_hotkey"
    fi
fi
action "registration burns TAO and stays manual: btcli subnets register --netuid 91 --network finney (or your reviewed ops/register-next-hotkey)"

step "5/8 eval-pool snapshot ($POOL_SNAPSHOT)"
if [ "$SKIP_POOL" = 1 ]; then
    info "skipped"
elif [ ! -x "$VENV_PY" ] && [ "$DRY_RUN" = 0 ]; then
    action "eval-pool sync needs the venv from step 1"
else
    if attempt "$VENV_PY" -m miner.controller --sync-pool \
        --root "$ROOT" --eval-pool-snapshot "$POOL_SNAPSHOT"; then
        ready "eval pool mirrored into $ROOT/pools (snapshot: $POOL_SNAPSHOT)"
    else
        action "eval-pool sync failed; check HF_TOKEN and retry: $VENV_PY -m miner.controller --sync-pool"
    fi
fi

step "6/8 controller state"
if [ "$SKIP_SEED" = 1 ]; then
    info "skipped"
elif [ ! -x "$VENV_PY" ] && [ "$DRY_RUN" = 0 ]; then
    action "state seeding needs the venv from step 1"
else
    # The first cycle records the Cascade head, chain.toml, pool revision, and
    # latest receipt without invoking any improvement hook.
    if attempt_quiet "$VENV_PY" -m miner.controller --once \
        --root "$ROOT" --cascade-dir "$CASCADE_DIR" \
        --eval-pool-snapshot "$POOL_SNAPSHOT"; then
        ready "controller state seeded in runs/controller-state.json"
    else
        action "seed the controller state once the chain is reachable: $VENV_PY -m miner.controller --once"
    fi
fi

step "7/8 verify the starter candidate"
if [ "$SKIP_VERIFY" = 1 ]; then
    info "skipped"
elif [ ! -x "$VENV/bin/cascade" ] && [ "$DRY_RUN" = 0 ]; then
    action "cascade verify needs the venv from step 1"
elif [ ! -f "$ROOT/$CANDIDATE/generator.py" ] && [ "$DRY_RUN" = 0 ]; then
    action "no candidate at $CANDIDATE"
else
    if attempt_quiet "$VENV/bin/cascade" verify "$ROOT/$CANDIDATE" \
        --chain-toml "$CASCADE_DIR/chain.toml"; then
        ready "cascade verify passed for $CANDIDATE"
    else
        action "cascade verify failed for $CANDIDATE; on a small host this is usually memory"
    fi
fi

step "8/8 agent backend"
AGENT_FOUND=""
for candidate in claude codex hermes; do
    if command -v "$candidate" >/dev/null 2>&1; then
        AGENT_FOUND="$candidate"
        break
    fi
done
if [ -n "$AGENT_FOUND" ]; then
    ready "improvement agent CLI on PATH: $AGENT_FOUND (CASCADE_AGENT=auto picks it)"
elif [ -x /opt/hermes/bin/hermes ]; then
    action "hermes is installed at /opt/hermes/bin but not on PATH; export PATH=/opt/hermes/bin:\$PATH"
else
    action "install an agent CLI (claude, codex, hermes) or set CASCADE_AGENT=custom with CASCADE_AGENT_COMMAND"
fi

printf '\n=== summary\n'
printf '\nready (%d):\n' "${#READY[@]}"
for item in ${READY[@]+"${READY[@]}"}; do printf '  + %s\n' "$item"; done
printf '\nneeds operator action (%d):\n' "${#ACTIONS[@]}"
for item in ${ACTIONS[@]+"${ACTIONS[@]}"}; do printf '  - %s\n' "$item"; done
cat <<EOF

Next: read notes/CONTRACT.md and notes/METHOD.md, then run one controller poll

  set -a; source .env; set +a
  $VENV_PY -m miner.controller --once

Nothing above rents hardware, spends TAO, or submits on-chain. Paid evaluation
runs only through an approved gpu_evaluation action; submission stays one-shot
per hotkey and needs explicit human approval.
EOF
exit 0
