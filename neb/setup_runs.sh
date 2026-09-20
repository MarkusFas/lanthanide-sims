#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CONFIG_INI="config.ini"
LIGANDS=(dota dotah lutathera pluvicto)

for ligand in "${LIGANDS[@]}"; do
  for element_dir in "$ligand"/*/; do
    element_dir="${element_dir%/}"
    run_dir="$element_dir/run"

    mkdir -p "$run_dir"
    ase convert -f "$element_dir/$ligand.xyz" "$run_dir/reactant.con"
    ase convert -f "$element_dir/$ligand-TSAP.xyz" "$run_dir/product.con"
    cp "$CONFIG_INI" "$run_dir/config.ini"

    echo "set up $run_dir"
  done
done
