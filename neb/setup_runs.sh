#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
NEB_DIR="$(pwd)"

(cd .. && python3 scripts/generate_lanthanide_structures.py)

cd "$NEB_DIR"

CONFIG_INI="config.ini"
LIGANDS=(dota dotah lutathera pluvicto)

for ligand in "${LIGANDS[@]}"; do
  for element_dir in "$ligand"/*/; do
    element_dir="${element_dir%/}"
    run_dir="$element_dir/run"

    mkdir -p "$run_dir"
    python3 xyz_to_con.py "$element_dir/$ligand.xyz" "$run_dir/reactant.con"
    python3 xyz_to_con.py "$element_dir/$ligand-TSAP.xyz" "$run_dir/product.con"
    cp "$CONFIG_INI" "$run_dir/config.ini"

    echo "set up $run_dir"
  done
done
