#!/bin/bash

# 환경변수 로드
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

echo "🛑 [$(date)] 모든 스트리머 삭제 시도..."

# 프로세스가 있는지 먼저 확인하고 있을 때만 삭제
if sudo pm2 list | grep -q "pcb-streamer"; then
    sudo pm2 delete "/pcb-streamer-.*/"
    echo "✅ 삭제 완료"
else
    echo "ℹ️ 삭제할 프로세스가 없습니다. (이미 깨끗함)"
fi