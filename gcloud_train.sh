#!/bin/bash

gcloud ml-engine jobs submit training "unet_job${@:1:1}" \
	--package-path './unet' \
	--module-name 'unet.unet_task' \
	--region 'us-east1' \
	--python-version 3.5 \
	--job-dir 'gs://diagnostics-unet-bucket/unet_jobs' 
	--stream-logs \
	-- \
    --train_image_path 'gs://diagnostics-unet-bucket/data/chest_xray/lung_segmentation/train/images/CXR_png/' \
    --train_mask_path  'gs://diagnostics-unet-bucket/data/chest_xray/lung_segmentation/train/images/masks/' &

gcloud ml-engine jobs stream-logs "unet_job${@:1:1}"