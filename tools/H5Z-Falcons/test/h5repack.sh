#!/bin/bash
set -euo pipefail

if [[ $# != 3 ]]
then
	echo "Usage: $0 [Falcons config file] [input_hdf5_file] [output_hdf5_file]"
	echo "Example: $0 falcons.conf testfloat_8_8_128.h5 testfloat_8_8_128_falcons.h5"
	exit 1
fi

configFile=$1
inputFile=$2
outputFile=$3
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"

repackArgs=$("${script_dir}/print_h5repack_args" -c "$configFile")
read -r -a repack_args <<< "$repackArgs"
echo "$repackArgs"

repack=h5repack-shared
if ! command -v "$repack" >/dev/null 2>&1; then
    repack=h5repack
fi
echo "Executing $repack with the following parameters: $repackArgs $inputFile $outputFile"
"$repack" "${repack_args[@]}" "$inputFile" "$outputFile"
