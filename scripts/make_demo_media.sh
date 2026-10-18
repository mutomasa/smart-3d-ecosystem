#!/usr/bin/env bash
# Builds the README GIFs and the demo video from the Playwright recordings in docs/demo/raw.
#   1) BASE_URL=http://localhost:8010 npx tsx frontend/scripts/record-demo.ts docs/demo/raw
#   2) npx tsx frontend/scripts/generate-runs.ts && npx tsx frontend/scripts/record-dashboard.ts docs/demo/raw
#   3) ./scripts/make_demo_media.sh
set -euo pipefail
cd "$(dirname "$0")/../docs/demo"
RAW=raw
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf

# gif <input> <output> <start_s> <duration_s> <width> <fps> [extra filter before scale]
gif() {
  local in=$1 out=$2 ss=$3 t=$4 w=$5 fps=$6 pre=${7:-}
  local f="${pre:+$pre,}fps=${fps},scale=${w}:-1:flags=lanczos"
  ffmpeg -v error -y -ss "$ss" -t "$t" -i "$in" \
    -vf "${f},split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" \
    -loop 0 "$out"
}

label() {  # drawtext filter with a caption box at the bottom-left
  echo "drawtext=fontfile=${FONT}:text='$1':x=16:y=h-th-18:fontsize=26:fontcolor=white:box=1:boxcolor=0x000000AA:boxborderw=10"
}

gif $RAW/calm-school.webm calm-school.gif 2 8 720 12
gif $RAW/food-rush.webm food-rush.gif 4 9 720 12
gif $RAW/ml-policy.webm ml-policy.gif 5 9 720 12

# Same seed, same predators: Boids only (left) vs Rules (right)
ffmpeg -v error -y -ss 5 -t 10 -i $RAW/predator-boids.webm -ss 5 -t 10 -i $RAW/predator-rules.webm \
  -filter_complex "[0:v]$(label 'Boids only')[l];[1:v]$(label 'Rules (flee + forage)')[r];[l][r]hstack=inputs=2" \
  -c:v libvpx-vp9 -b:v 0 -crf 32 $RAW/predator-compare.webm

# README GIF: Boids only / Rules / ML side by side, with Japanese captions (predator-compare.ja.ass)
ffmpeg -v error -y -ss 5 -t 10 -i $RAW/predator-boids.webm -ss 5 -t 10 -i $RAW/predator-rules.webm -ss 5 -t 10 -i $RAW/predator-ml.webm \
  -filter_complex "[0:v][1:v][2:v]hstack=inputs=3,pad=iw:ih+220:0:0:color=0x030b1a,subtitles=predator-compare.ja.ass" \
  -c:v libvpx-vp9 -b:v 0 -crf 32 $RAW/predator-compare3.webm
gif $RAW/predator-compare3.webm predator-compare.gif 0 10 1200 10

# Dashboard: skip the loading screen
gif $RAW/dashboard.webm dashboard.gif 6 22 800 5

# Full demo video (H.264 MP4) with burned-in Japanese subtitles (subtitles.ja.ass)
encode() {  # encode <out> <ffmpeg input args...> -- <video filter>
  local out=$1; shift
  local args=(); while [[ $1 != -- ]]; do args+=("$1"); shift; done; shift
  ffmpeg -v error -y "${args[@]}" \
    -vf "scale=1280:800:force_original_aspect_ratio=decrease,pad=1280:800:(ow-iw)/2:(oh-ih)/2:color=0x030b1a,${1:+$1,}fps=30" \
    -c:v libx264 -pix_fmt yuv420p -crf 20 -an "$out"
}
clip() { encode "$4" -ss "$2" -t "$3" -i "$1" -- ""; }  # clip <in> <start> <dur> <out>
TMP=$(mktemp -d)
encode "$TMP/0.mp4" -ss 2 -t 8 -i $RAW/calm-school.webm -- "boxblur=6,eq=brightness=-0.28"  # intro card
clip $RAW/calm-school.webm 2 8 "$TMP/1.mp4"
clip $RAW/predator-compare.webm 0 10 "$TMP/2.mp4"
clip $RAW/food-rush.webm 4 9 "$TMP/3.mp4"
clip $RAW/ml-policy.webm 5 9 "$TMP/4.mp4"
clip $RAW/dashboard.webm 6 22 "$TMP/5.mp4"
encode "$TMP/6.mp4" -f lavfi -i color=c=0x030b1a:s=1280x800:r=30:d=8 -- ""  # results card
printf "file '%s'\n" "$TMP"/{0,1,2,3,4,5,6}.mp4 > "$TMP/list.txt"
ffmpeg -v error -y -f concat -safe 0 -i "$TMP/list.txt" -c copy "$TMP/plain.mp4"
ffmpeg -v error -y -i "$TMP/plain.mp4" -vf "subtitles=subtitles.ja.ass" \
  -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart smart-3d-ecosystem-demo.mp4
rm -rf "$TMP"
ls -lh *.gif *.mp4
