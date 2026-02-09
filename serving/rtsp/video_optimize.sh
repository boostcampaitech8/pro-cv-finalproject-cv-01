# 1080p 원본을 RTSP 스트리밍에 최적화된 H.264로 변환하는 스크립트

INPUT="/home/ubuntu/rtsp/30fps_dark.mp4"
OUTPUT="/home/ubuntu/rtsp/1080p_optimized.mp4"

echo "🪽 인코딩을 시작합니다: $INPUT -> $OUTPUT"

ffmpeg -i "$INPUT" \
-c:v libx264 -preset ultrafast -crf 23 \
-b:v 5000k -maxrate 5000k -bufsize 10000k \
-pix_fmt yuv420p -movflags +faststart \
-g 30 -x264-params "repeat-headers=1:keyint=30" \
"$OUTPUT"

echo "🪽 인코딩이 완료되었습니다!"