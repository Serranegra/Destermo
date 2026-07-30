#!/usr/bin/env bash
# Regenera todos os rasters da marca a partir dos SVGs.
# Requer: python3 -m pip install cairosvg  +  imagemagick
set -euo pipefail
cd "$(dirname "$0")"

python3 - << 'PY'
import cairosvg
jobs = [
    ("destermo-app-icon.svg", "icon-192.png", 192),
    ("destermo-app-icon.svg", "icon-512.png", 512),
    ("destermo-app-icon.svg", "apple-touch-icon.png", 180),
    ("favicon.svg", "favicon-16.png", 16),
    ("favicon.svg", "favicon-32.png", 32),
    ("favicon.svg", "favicon-48.png", 48),
    ("destermo-icon.svg", "destermo-icon-1024.png", 1024),
]
for src, out, size in jobs:
    cairosvg.svg2png(url=src, write_to=out, output_width=size, output_height=size)
    print("ok", out)
cairosvg.svg2png(url="og-banner.svg", write_to="og-banner.png", output_width=1200)
print("ok og-banner.png")
PY

convert favicon-16.png favicon-32.png favicon-48.png favicon.ico
echo "ok favicon.ico"
