# FeynGym

Reinforcement learning environment for integration-by-parts (IBP) reduction of Feynman integrals. FeynGym.jl contains the built-in RL environment covering the one-loop massive bubble integral, as used in [this study](https://arxiv.org/abs/2504.16045). For other integral families, the Julia package provides a generic linear solver for IBP reduction with feedback on the solve cost, which is suitable for black-box optimization of the total cost, but you need to generate the IBP equations externally, e.g. via the Python package pyfeyngym in this repository.

The subfolder `FeynGym.jl` is the Julia package, and the subfolder `PyFeynGym` is the Python interface package.

Additionally, `ppo_masking` provides another Python package, which is an implementation of proximal policy optimization (PPO) with support for action masking, forked from [CleanRL](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py).

## Installation
Clone the repository. Then create a new Python environment (I've only tested on Python 3.13). Run the shell commands
```
cd ppo_masking
pip install -e .
cd ..
cd pyfeyngym
pip install -e .
cd ..
```
Here I used the `-e` flag to allow you to make changes to the code without re-installing the packages.

Then expose the Julia package in the same repository to Python:
```
python install_julia_packages.py
```
The above command will install Julia itself, unless you already have an up-to-date installation from e.g. [juliaup](https://github.com/JuliaLang/juliaup). (Using pre-installed Julia on clusters is not recommended, as the version may be outdated.)

Optionally, check the installation works by running the example notebook `pyfeyngym/examples/IBPEnv_example.ipynb`.

## Examples
See `examples/pyfeyngym_multitarget.ipynb` for training using the PPO algorithm. This notebook also produces the IBP rollout illustrations under `figs/`. Each environment observation is a 9×9 image (when using the default size) with 4 channels. The actor network in PPO takes the observation as input and outputs a 9×9 image with 3 channels, which give the logits for the action probabilities (think of each action as a mouse click on one of the pixels, with either the left, right, or center mouse button, corresponding to the 3 channels). The critic network in PPO takes the same input but outputs a single number as the output, which estimates the expected future rewards.

See `examples/ppo_masking_example.ipynb` for a demonstration of the internal `ppo_masking` package.

See `examples/pyfeyngym_test_ibp_cost.ipynb` for a direct test of the pyfeyngym backend cost function used to score IBP equation orderings.

See `pyfeyngym/examples/IBPEnv_example.ipynb` for the functionality of the RL environment itself, without training.

See `pyfeyngym/examples/solve_eqs_finite_field.ipynb` for solving linear equations over finite fields.

# Use with Docker
Build the image:
```
docker build -t feyngym .
```
and start the container with NVIDIA GPU access enabled and port 8888 for the Jupyter server inside the container:
```
docker run --rm --device --gpus all -p 8888:8888 feyngym
```
This starts the Jupyter notebook server, and the web address for the server, starting with `http://127.0.0.1:8888/`, will be printed on the screen. Type the address into your web browser to start a Jupyter notebook session connected to the server in the container.
