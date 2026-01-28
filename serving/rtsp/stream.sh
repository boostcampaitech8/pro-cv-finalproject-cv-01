#!/bin/bash

# 1. 기존에 돌아가던 스트리머가 있다면 삭제
sudo pm2 delete pcb-streamer || true

# 2. RTSP 송출 시작
# UDP
sudo pm2 start "ffmpeg -re -stream_loop -1 -i /home/ubuntu/rtsp/1080p_optimized.mp4 -c copy -f rtsp rtsp://localhost:8554/pcb_stream" --name pcb-streamer

# TCP
# sudo pm2 start "ffmpeg -re -stream_loop -1 -rtsp_transport tcp -i /home/ubuntu/rtsp/1080p_optimized.mp4 -c copy -f rtsp rtsp://localhost:8554/pcb_stream" --name pcb-streamer --cwd /home/ubuntu/rtsp

# -c:v libx264 : H.264 인코딩 사용
# sudo pm2 start "ffmpeg -re -stream_loop -1 -i /home/ubuntu/rtsp/30fps_dark.mp4 -vf "scale=1280:720" -c:v libx264 -preset ultrafast -tune zerolatency -b:v 4000k -maxrate 4000k -bufsize 8000k -g 30 -rtsp_transport tcp -f rtsp rtsp://localhost:8554/pcb_stream" --name pcb-streamer --cwd /home/ubuntu/rtsp

# -preset ultrafast : 최고의 속도를 위한 설정
# -tune zerolatency : 최소 지연 시간을 위한 설정
# -b:v 4000k : 비트레이트 설정
# -maxrate 4000k : 최대 비트레이트 설정
# -bufsize 8000k : 버퍼 크기 설정
# -g 30 : GOP 크기 설정
# -rtsp_transport tcp : TCP 전송 사용
# -vf "scale=1280:720" : 해상도 조정

echo "⏳ 스트리머가 안정화될 때까지 3초간 대기합니다..."
sleep 3

# 3. 현재 pm2 프로세스 리스트 상태 확인
sudo pm2 status

# 4. 서버가 재부팅되어도 이 설정이 유지되도록 저장
sudo pm2 save

echo "🪽 RTSP 스트리밍 설정 및 저장이 완료되었습니다."
