#!/bin/bash

gcloud ai-platform predict \
  --model "saved_model${@:1:1}" \
  --version "v${@:1:1}" \
  --json-instances prediction_input.json