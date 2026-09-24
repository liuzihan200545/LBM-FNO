#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_file="$project_dir/src/geometry/output/spheres_3d/PeriodicSphereCell.java"
target_dir="/mnt/e/LBM-FNO-3D/COMSOL/resource"
target_file="$target_dir/PeriodicSphereCell.java"
class_file="$target_dir/PeriodicSphereCell.class"
compiler="/mnt/d/COMSOL63/Multiphysics/bin/win64/comsolcompile.exe"

if [[ ! -f "$source_file" ]]; then
    echo "Missing $source_file" >&2
    echo "Generate it first: .venv/bin/python src/geometry/export_comsol_java.py" >&2
    exit 1
fi
if [[ ! -x "$compiler" ]]; then
    echo "COMSOL compiler not found: $compiler" >&2
    exit 1
fi
if [[ ! -d "$target_dir" ]]; then
    echo "COMSOL resource directory not found: $target_dir" >&2
    echo "Run this script in WSL with the D: and E: drives mounted." >&2
    exit 1
fi
if ! command -v wslpath >/dev/null 2>&1; then
    echo "wslpath is required; run this script inside WSL." >&2
    exit 1
fi

cp -- "$source_file" "$target_file"
cmp -- "$source_file" "$target_file"
echo "Compiling $(wslpath -w "$target_file") with COMSOL 6.3..."

stamp="$(mktemp "$target_dir/.comsolcompile.XXXXXXXX")"
trap 'rm -f -- "$stamp"' EXIT
cd -- "$target_dir"
"$compiler" "$(wslpath -w "$target_file")"

if [[ ! -s "$class_file" || ! "$class_file" -nt "$stamp" ]]; then
    echo "Compilation returned, but a new $class_file was not created." >&2
    exit 1
fi
echo "Ready: $(wslpath -w "$class_file")"
