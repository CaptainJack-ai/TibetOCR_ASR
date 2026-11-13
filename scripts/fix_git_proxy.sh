#!/usr/bin/env bash
# Point git at the local HTTP proxy that macOS already uses.
#
# Why: git does NOT read the macOS system proxy. Browsers/VS Code do, so GitHub
# opens in the browser while `git push` dies with "Connection reset by peer".
#
#   ./scripts/fix_git_proxy.sh            # detect from scutil, test, then apply
#   ./scripts/fix_git_proxy.sh --port 7890
#   ./scripts/fix_git_proxy.sh --unset
set -euo pipefail

UNSET=0
PORT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --unset) UNSET=1; shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ "$UNSET" = "1" ]; then
  git config --global --unset http.proxy 2>/dev/null || true
  git config --global --unset https.proxy 2>/dev/null || true
  echo "已移除全局 git 代理配置"
  exit 0
fi

if [ -z "$PORT" ]; then
  PORT=$(scutil --proxy | awk '/HTTPPort/{print $3; exit}')
  HOST=$(scutil --proxy | awk '/HTTPProxy/{print $3; exit}')
  [ -z "${HOST:-}" ] && HOST=127.0.0.1
fi
HOST="${HOST:-127.0.0.1}"
PROXY="http://${HOST}:${PORT}"

if [ -z "$PORT" ]; then
  echo "未在系统设置里找到 HTTP 代理；请用 --port 指定，例如 --port 7890" >&2
  exit 1
fi

echo "检测到系统代理: ${PROXY}"
if ! curl -sS -m 12 -x "$PROXY" -o /dev/null -w "" https://github.com 2>/dev/null; then
  echo "该代理无法访问 github.com，请确认代理软件在运行" >&2
  exit 1
fi
echo "代理可用（github.com 200）"

git config --global http.proxy  "$PROXY"
git config --global https.proxy "$PROXY"
echo "已写入 ~/.gitconfig："
git config --global --get-regexp 'https?\.proxy'
echo
echo "验证： git ls-remote <repo>   （应能列出 refs；空输出说明远端仓库为空）"
