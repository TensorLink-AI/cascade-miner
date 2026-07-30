#!/bin/bash
# Paired GPU evaluation: rent pod → provision → train king + candidate → score → pull → stop
# Usage: bash scripts/run_paired_eval.sh
# Requires: .env sourced, lium on PATH, generators/king-control + generators/candidate set up
set -e

export CASCADE_DIR="${CASCADE_DIR:-/opt/data/cascade}"
export CASCADE_NETWORK="${CASCADE_NETWORK:-finney}"
export PATH="$HOME/.lium/bin:$PATH"

POD_NAME="cascade-eval-$(date +%s)"
REMOTE_ROOT="/root/cascade-miner"
REMOTE_PY="$REMOTE_ROOT/.venv/bin/python"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i $HOME/.ssh/id_ed25519"
SEEDS="${CASCADE_EVAL_SEEDS:-0,1,2}"
TRAIN_HOURS="${CASCADE_TRAIN_HOURS:-1}"
GPU="${CASCADE_GPU:-RTX4090}"
TTL_HOURS=5

echo "=== Renting pod: $POD_NAME (GPU: $GPU, TTL: ${TTL_HOURS}h) ==="
lium up --gpu "$GPU" -c 1 --name "$POD_NAME" --ttl "${TTL_HOURS}h" -y 2>&1 | tail -3

echo "=== Waiting for pod ==="
for i in $(seq 1 60); do
    POD_JSON=$(lium ps --format json 2>/dev/null) || true
    SSH_CMD=$(echo "$POD_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0]['ssh_cmd'])" 2>/dev/null) || true
    if [ -n "$SSH_CMD" ]; then
        PORT=$(echo "$POD_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0]['ports']['22'])")
        IP=$(echo "$POD_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0]['ip'])")
        echo "Pod ready: $IP:$PORT"
        break
    fi
    sleep 5
done

SSH="ssh $SSH_OPTS -p $PORT root@$IP"
SSH_TAR="ssh $SSH_OPTS -p $PORT root@$IP"

cleanup() {
    echo "=== Stopping pod ==="
    lium rm "$POD_NAME" 2>&1 | tail -1 || true
}
trap cleanup EXIT

echo "=== Provisioning pod ==="
$SSH "mkdir -p $REMOTE_ROOT/generators $REMOTE_ROOT/pools/snapshots $REMOTE_ROOT/scores $REMOTE_ROOT/ckpts"

echo "=== Pushing cascade reference ==="
tar -cf - --exclude=__pycache__ -C "$CASCADE_DIR" . | $SSH "mkdir -p /root/cascade && tar -xf - -C /root/cascade"

echo "=== Pushing generators ==="
for GEN in king-control candidate; do
    if [ -d "generators/$GEN" ]; then
        tar -cf - --exclude=__pycache__ -C "generators/$GEN" . | $SSH "mkdir -p $REMOTE_ROOT/generators/$GEN && tar -xf - -C $REMOTE_ROOT/generators/$GEN"
        echo "  pushed $GEN"
    fi
done

echo "=== Pushing miner code ==="
tar -cf - --exclude=__pycache__ -C miner/ . | $SSH "mkdir -p $REMOTE_ROOT/miner && tar -xf - -C $REMOTE_ROOT/miner"

echo "=== Pushing eval pool ==="
tar -cf - -C pools/snapshots . | $SSH "tar -xf - -C $REMOTE_ROOT/pools/snapshots"

echo "=== Installing deps on pod ==="
$SSH "curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; export PATH=/root/.local/bin:\$PATH; uv venv --python 3.11 $REMOTE_ROOT/.venv >/dev/null 2>&1; uv pip install --python $REMOTE_PY '/root/cascade[train]' pandas scikit-learn gpytorch networkx >/dev/null 2>&1" 2>&1
$SSH "$REMOTE_PY -c 'import torch; print(\"torch OK:\", torch.__version__)'" || { echo "DEPS FAILED"; exit 1; }

echo "=== Training: seeds=$SEEDS, train_hours=$TRAIN_HOURS ==="
IFS=',' read -ra SEED_LIST <<< "$SEEDS"
for SEED in "${SEED_LIST[@]}"; do
    for GEN in king-control candidate; do
        echo "--- [$GEN seed=$SEED] training ---"
        $SSH "cd $REMOTE_ROOT && nohup $REMOTE_PY -m miner.evaluate \
            $REMOTE_ROOT/generators/$GEN \
            --chain-toml /root/cascade/chain.toml \
            --pools-root $REMOTE_ROOT/pools/snapshots \
            --scores-root $REMOTE_ROOT/scores \
            --ckpt-root $REMOTE_ROOT/ckpts \
            --seed $SEED --train-hours $TRAIN_HOURS \
            > /tmp/${GEN}_seed${SEED}.log 2>&1 &" 2>&1

        # Poll for completion (check for TRAINED.json then summary.json)
        echo "  training in background, polling..."
        while true; do
            sleep 60
            DONE=$($SSH "test -f $REMOTE_ROOT/scores/${GEN}__seed${SEED}/summary.json && echo YES || echo NO" 2>/dev/null)
            if [ "$DONE" = "YES" ]; then
                echo "  [$GEN seed=$SEED] DONE"
                break
            fi
            # Check if process is still running
            ALIVE=$($SSH "pgrep -f 'miner.evaluate.*${GEN}.*--seed $SEED' > /dev/null && echo YES || echo NO" 2>/dev/null)
            if [ "$ALIVE" = "NO" ] && [ "$DONE" = "NO" ]; then
                echo "  [$GEN seed=$SEED] process ended without scores — checking log"
                $SSH "tail -10 /tmp/${GEN}_seed${SEED}.log" 2>&1
                # Maybe it finished training but eval crashed — check for TRAINED.json
                TRAINED=$($SSH "test -f $REMOTE_ROOT/ckpts/${GEN}__seed${SEED}/TRAINED.json && echo YES || echo NO" 2>/dev/null)
                if [ "$TRAINED" = "YES" ]; then
                    echo "  Training done, re-running eval phase..."
                    $SSH "cd $REMOTE_ROOT && $REMOTE_PY -m miner.evaluate \
                        $REMOTE_ROOT/generators/$GEN \
                        --chain-toml /root/cascade/chain.toml \
                        --pools-root $REMOTE_ROOT/pools/snapshots \
                        --scores-root $REMOTE_ROOT/scores \
                        --ckpt-root $REMOTE_ROOT/ckpts \
                        --seed $SEED --train-hours $TRAIN_HOURS" 2>&1 | tail -10
                    break
                else
                    echo "  [$GEN seed=$SEED] FAILED"
                    break
                fi
            fi
        done

        # Pull scores after each run
        $SSH "tar -cf - -C $REMOTE_ROOT/scores/ ." 2>/dev/null | tar -xf - -C scores/ 2>/dev/null || true
        echo "  pulled scores for $GEN seed=$SEED"
    done
done

echo "=== Pulling all scores ==="
$SSH "tar -cf - -C $REMOTE_ROOT/scores/ ." 2>/dev/null | tar -xf - -C scores/
echo "=== Score files: ==="
find scores -type f | sort
echo "=== COMPLETE ==="
