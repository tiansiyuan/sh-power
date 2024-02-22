#!/usr/bin/bash

mkdir -p ./model_store

echo "== Started model packaging =="
model_name="helmet_detection"
start_time="$(date -u +%s)"
torch-model-archiver \
  --model-name ${model_name} \
  --version 1.0 \
  --serialized-file ./helmet.torchscript.pt \
  --handler ./torchserve_handler.py \
  --export-path ./model_store \
  --extra-files ./index_to_name.json \
  --force
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "== Model packaging completed. Total $elapsed seconds elapsed =="