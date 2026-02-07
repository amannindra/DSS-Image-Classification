#!/bin/bash
set -e

BASE="/Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/all-7"
TAR="$BASE/model.tar.gz"
S3_PATH="s3://sagemaker-us-east-1-253490779227/all-7-2026-02-07-08-00-06-510/output/model.tar.gz"

python3 -c 'print("START")'

mkdir -p "$BASE"
python3 -c 'print("Making folders finished")'

aws s3 cp "$S3_PATH" "$TAR"
python3 -c 'print("S3 download finished")'

tar -tvf "$TAR"
python3 -c 'print("Tar listing finished")'

tar -xzf "$TAR" -C "$BASE"
python3 -c 'print("Extraction finished")'

rm -f "$TAR"
python3 -c 'print("All finished")'
