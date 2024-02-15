#!/usr/bin/env zsh

# sh files under script folder
cd "$(dirname "$0")"/.. || { echo "Failed cd"; exit 1; }

set -e
set -o pipefail

arg1=$1
arg2=$2
arg3=$3
arg4=$4
arg5=$5

echo "Train data: $arg1"
echo "Test data: $arg2"
echo "True label: $arg3"
echo "Read col-range: $arg4"
echo "Output file: $arg5"

python detector.py -e $arg3 -c $arg4 $arg1 $arg2 .$arg5
