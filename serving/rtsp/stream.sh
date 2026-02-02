#!/bin/bash

# 1. 기존에 돌아가던 스트리머가 있다면 삭제
sudo pm2 delete pcb-streamer || true

# 2. RTSP 송출 시작
sudo pm2 start "ffmpeg -re -stream_loop -1 -i /home/ubuntu/rtsp/1080p_optimized.mp4 -c copy -f rtsp rtsp://localhost:8554/pcb_stream" --name pcb-streamer

echo "⏳ 스트리머가 안정화될 때까지 3초간 대기합니다..."
sleep 3

# 3. 현재 pm2 프로세스 리스트 상태 확인
sudo pm2 status

# 4. 서버가 재부팅되어도 이 설정이 유지되도록 저장
sudo pm2 save

echo "🪽 RTSP 스트리밍 설정 및 저장이 완료되었습니다."
