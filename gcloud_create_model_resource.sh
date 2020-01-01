#!/bin/bash

gcloud ml-engine models create "saved_model${@:1:1}" \
	--regions 'us-east1' &

gcloud ml-engine versions create "v${@:1:1}" \
	--model "saved_model${@:1:1}" \
	--runtime-version 1.13 \
	--python-version 3.5 \
	--framework "tensorflow" \
  	--origin 'gs://diagnostics-unet-bucket/saved_models'