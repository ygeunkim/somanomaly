#!/usr/bin/env zsh

# sh files under script folder
cd "$(dirname "$0")"/.. || { echo "Failed cd"; exit 1; }

set -e
set -o pipefail

arg1=$1
arg2=$2
arg3=$3
arg4=$4

echo "Train data: $arg1"
echo "Test data: $arg2"
echo "True label: $arg3"
echo "Output file: $arg4"

python main.py -e $arg3 $arg1 $arg2 $arg4