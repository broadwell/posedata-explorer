#!/usr/bin/env bash

if [ -f .env ]
then
  set -a;
  source .env;
  set +a;
fi

KERNEL_NAME=${KERNEL_NAME:-posedata-explorer}

# install/update virtualenv as needed
python -m pipenv install

# install/update an ipykernel based on this virtualenv
pipenv run python -m ipykernel install --user --name="$KERNEL_NAME"

# run a notebook server using the virtualenv kernel as the default kernel
pipenv run jupyter notebook --MultiKernelManager.default_kernel_name="$KERNEL_NAME" ${NO_BROWSER+--no-browser}
