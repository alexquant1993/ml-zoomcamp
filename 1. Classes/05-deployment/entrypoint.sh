#!/bin/bash --login
# The --login ensures the bash configuration is loaded,

# Temporarily disable strict mode and activate conda:
set +euo pipefail
conda activate churn-model
# enable strict mode:
set -euo pipefail

# exec the final command:
gunicorn --bind=0.0.0.0:9696 predict:app