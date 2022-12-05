# posedata-explorer

Notebooks and maybe scripts and other utilities for working with pose data extracted from videos


## Install and run

Assuming you have git, [pipenv](https://pipenv.pypa.io/en/latest/) and Python 3.10.x installed, just clone this repo into a folder and, from within that folder, run

> $ ./start_notebook_server.sh

to launch the notebook. All other relevant documentation is contained within the notebook.


## Configuration

Some configuration can be supplied by means of a `.env` file in the root of the project.  The following environment variables are supported:

* `DATA_FOLDER`: the directory the notebook will look in for videos and JSON files (defaults to the current working directory)
* `KERNEL_NAME`: the name of the kernel to use (defaults to `posedata-explorer` -- likely no need to change this)
* `NO_BROWSER`: if set (to any value) launches the notebook browser without automatically opening a browser tab

e.g. `.env` contents:

```
DATA_FOLDER=../data
KERNEL_NAME=posedata-explorer
NO_BROWSER=1 
```


## Development

Before committing any changes, run `git config --local include.path ../.gitconfig` to update the local git config for this repo.
