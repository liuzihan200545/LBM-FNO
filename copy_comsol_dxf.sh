#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$project_dir/src/geometry/output/disks_periodic"
target_dir="/mnt/e/LBM-FNO-3D/COMSOL/resource"

if ! mountpoint -q /mnt/e; then
    echo "E: drive is not mounted at /mnt/e. Run this script in WSL." >&2
    exit 1
fi

shopt -s nullglob
dxf_files=("$source_dir"/*.dxf)
if ((${#dxf_files[@]} == 0)); then
    echo "No DXF files found in $source_dir" >&2
    echo "Generate them first: uv run --locked python src/geometry/disk_pack_periodic.py" >&2
    exit 1
fi

mkdir -p -- "$target_dir"
cp -v -- "${dxf_files[@]}" "$target_dir/"

for file in "${dxf_files[@]}"; do
    cmp -- "$file" "$target_dir/$(basename -- "$file")"
done
echo "Copied and verified ${#dxf_files[@]} DXF file(s) in $target_dir"
