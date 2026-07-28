#!/bin/bash
# tpose-server 起動スクリプト。
#
# venv は既定で diffusers-server のものを共有する(DS_VENV で上書き可)。
# 理由: diffusers は git 版(0.40.0.dev0)を検証済みの状態で使う必要があり、
# 新しく pip install し直すと別コミットが入る可能性があるため
# (diffusers-server の CLAUDE.md 6番)。独立した venv を作った場合は
#   DS_VENV=/path/to/venv ./run.sh
# のように指定する。
#
# ポートは DS_TPOSE_PORT(既定 8610)。diffusers-server(8601)と同時起動できる
# ようにデフォルトをずらしてある。ただし **同じGPUを使うので同時に生成すると
# VRAM を食い合う**(このサーバの Edit は fp8-lightning で約35GB使う)。
set -eu

cd "$(dirname "$0")"

VENV="${DS_VENV:-/home/animede/diffusers-server/venv}"
PORT="${DS_TPOSE_PORT:-8610}"
HOST="${DS_TPOSE_HOST:-0.0.0.0}"

if [ ! -x "$VENV/bin/python" ]; then
    echo "エラー: venv が見つかりません: $VENV" >&2
    echo "       DS_VENV=/path/to/venv ./run.sh のように指定してください。" >&2
    exit 1
fi

echo "tpose-server: http://${HOST}:${PORT}  (venv=${VENV})"
exec "$VENV/bin/python" -m uvicorn app:app --host "$HOST" --port "$PORT"
