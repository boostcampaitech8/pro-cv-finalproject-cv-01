#!/bin/bash

# start_stream.sh
# 기존 stream.sh의 "송출 시작" 부분만 떼어낸 스크립트입니다.

# (1) 카메라 수와 영상 소스는 필요에 따라 조절하세요.
# 비디오 파일 목록을 배열로 정의하면 각 채널마다 다른 영상을 송출할 수 있습니다.
VIDEO_SOURCES=(
    "/home/ubuntu/rtsp/health_optimized.mp4",
    "/home/ubuntu/rtsp/health_optimized.mp4",
    "/home/ubuntu/rtsp/health_optimized.mp4"
)
# 배열 길이를 카메라 수로 사용합니다.
NUM_CAMERAS=${#VIDEO_SOURCES[@]}

# 기존에 돌아가던 스트리머가 있다면 삭제(안해도 되지만 중복 방지용)
sudo pm2 delete pcb-streamer || true
sudo pm2 delete "/pcb-streamer-.*/" || true

# RTSP 송출 시작 (다중 채널)
for i in $(seq 1 $NUM_CAMERAS)
do
    src="${VIDEO_SOURCES[$((i-1))]}"
    echo "🚀 채널 $i 송출 시작 (소스: $src): rtsp://localhost:8554/pcb_stream_$i"
    sudo pm2 start "ffmpeg -re -stream_loop -1 -i $src -c copy -f rtsp rtsp://localhost:8554/pcb_stream_$i" --name pcb-streamer-$i
done

# 안정화를 위해 잠시 대기
echo "⏳ 스트리머들이 안정화될 때까지 3초간 대기합니다..."
sleep 3

# 프로세스 상태 확인
sudo pm2 status
sudo pm2 save

echo "🪽 총 $NUM_CAMERAS개 채널 RTSP 스트리밍 설정이 완료되었습니다."