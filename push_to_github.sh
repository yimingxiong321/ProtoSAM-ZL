#!/bin/bash
# One-click push to GitHub via HTTPS + Personal Access Token.
# Usage: bash push_to_github.sh
set -eu

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_URL="https://github.com/yimingxiong321/ProtoSAM-ZL.git"
DEFAULT_USER="yimingxiong321"

# Prefer system git; fall back to PATH lookup.
if [ -x /usr/bin/git ]; then
    GIT=/usr/bin/git
elif command -v git >/dev/null 2>&1; then
    GIT="$(command -v git)"
else
    echo "错误: 找不到 git，请先 module load git 或安装 git。"
    exit 1
fi

cd "$REPO_DIR"

if ! "$GIT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "错误: 当前目录不是 git 仓库: $REPO_DIR"
    exit 1
fi

echo "=============================================="
echo " ProtoSAM-ZL → GitHub 一键 Push"
echo " 仓库: $REMOTE_URL"
echo "  git: $GIT"
echo "=============================================="

if ! "$GIT" diff --quiet 2>/dev/null || ! "$GIT" diff --cached --quiet 2>/dev/null; then
    echo
    echo "检测到未提交改动，是否先 commit？(y/N)"
    read -r DO_COMMIT
    DO_COMMIT_LC="$(echo "$DO_COMMIT" | tr '[:upper:]' '[:lower:]')"
    if [ "$DO_COMMIT_LC" = "y" ] || [ "$DO_COMMIT_LC" = "yes" ]; then
        "$GIT" add -A
        echo "请输入 commit message（回车使用默认）:"
        read -r MSG
        if [ -z "$MSG" ]; then
            MSG="Update ProtoSAM-ZL"
        fi
        "$GIT" commit -m "$MSG"
    else
        echo "已取消：请先 git commit 后再 push。"
        exit 1
    fi
fi

COMMITS=$("$GIT" rev-list --count HEAD 2>/dev/null || echo 0)
if [ "$COMMITS" -eq 0 ]; then
    echo "错误: 没有任何 commit，请先 git add && git commit。"
    exit 1
fi

echo
echo "GitHub 用户名（回车默认: $DEFAULT_USER）:"
read -r GH_USER
if [ -z "$GH_USER" ]; then
    GH_USER="$DEFAULT_USER"
fi

echo
echo "请输入 GitHub Personal Access Token（输入时不显示）:"
echo "  创建地址: https://github.com/settings/tokens  （勾选 repo 权限）"
read -rs GH_PAT
echo

if [ -z "$GH_PAT" ]; then
    echo "错误: PAT 不能为空。"
    exit 1
fi

# Ensure origin exists (compatible with older git).
if "$GIT" remote 2>/dev/null | grep -qx origin; then
    "$GIT" remote set-url origin "$REMOTE_URL"
else
    "$GIT" remote add origin "$REMOTE_URL"
fi

LOCAL_BRANCH="$("$GIT" rev-parse --abbrev-ref HEAD)"
PUSH_URL="https://${GH_USER}:${GH_PAT}@github.com/yimingxiong321/ProtoSAM-ZL.git"

echo
echo "正在 push 分支: ${LOCAL_BRANCH} ..."

PUSH_EC=1
PUSH_OUT=""

# Try push to same branch name first; fallback master -> main for new GitHub repos.
if GIT_TERMINAL_PROMPT=0 "$GIT" push "$PUSH_URL" "${LOCAL_BRANCH}:${LOCAL_BRANCH}" >/tmp/protosam_push.log 2>&1; then
    PUSH_EC=0
else
    PUSH_OUT="$(cat /tmp/protosam_push.log)"
    if [ "$LOCAL_BRANCH" = "master" ]; then
        echo "尝试 push 到远程 main 分支 ..."
        if GIT_TERMINAL_PROMPT=0 "$GIT" push "$PUSH_URL" "master:main" >/tmp/protosam_push.log 2>&1; then
            PUSH_EC=0
            "$GIT" branch -M main 2>/dev/null || true
            "$GIT" remote set-url origin "$REMOTE_URL"
            "$GIT" branch --set-upstream-to=origin/main main 2>/dev/null || true
        else
            PUSH_OUT="$(cat /tmp/protosam_push.log)"
        fi
    fi
fi
rm -f /tmp/protosam_push.log

# Scrub token from any error output
PUSH_OUT="$(echo "$PUSH_OUT" | sed "s/${GH_PAT}/***/g")"

if [ "$PUSH_EC" -eq 0 ]; then
    "$GIT" remote set-url origin "$REMOTE_URL"
    echo
    echo "Push 成功!"
    echo "  https://github.com/yimingxiong321/ProtoSAM-ZL"
else
    echo
    echo "Push 失败:"
    echo "$PUSH_OUT"
    echo
    echo "常见原因:"
    echo "  1. PAT 无 repo 权限或已过期"
    echo "  2. 远程已有 README，需先 pull 再 push"
    echo "  3. 网络无法访问 github.com"
    exit 1
fi

unset GH_PAT PUSH_URL
