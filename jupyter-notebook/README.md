# Requirements

- [Core requirements](../DEPENDENCIES.md)
- Python 3.8+ for executing this notebook
- python3-tk package
- Everything besides ALUs dependencies are listed in `setup_aux_tools.sh` in https://github.com/cgi-estonia-space/ALUs-platform/tree/main/ubuntu/20_04/focal
- One can also utilize prebuilt docker container - `cgialus/alus-focal-jupyter:latest`

# Running instructions

It is best to run the notebook from its own python virtual environment. Environment can be created with
command  `python3 -m venv env` and it can be activated by the following command `source env/bin/activate`. In order to
deactivate the virtual environment use `deactivate`.

It may or may not be needed (depending on if one gets errors during installation of some requirement), but before installing `requirements.txt`, run:
- `pip install --upgrade setuptools`
- `pip install wheel`
- `pip install validators`

Install all the necessary requirements:

`pip install -r requirements.txt`

Launch Jupyter Notebook or Jupyter Lab:

`jupyter notebook` or `jupyter lab`

Execute all the cells in order providing all the necessary input.

# Running in prebuilt container

Container has pre-installed all ALUs runtime dependencies and Python tools for jupyter notebook. Also it has ALUs binaries packaged and accessible from `$PATH`.

- Get the container - `cgialus/alus-focal-jupyter:latest`
- Start container - `docker run -it --gpus all -p 8888:8888 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v <optional directory for input files from host>:<optional container destination directory> cgialus/alus-focal-jupyter:latest`
- `cd alus/jupyter-notebook`
- `jupyter notebook --ip 0.0.0.0 --port 8888  --allow-root --no-browser`
- Open the URL listed when server is started

When GUI does not work, try running `xhost +` before starting container

