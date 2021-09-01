# Code for the paper "Learning Language-Conditioned Robot Behavior from Offline Data and Crowd-Sourced Annotation"

This is code for the paper "Learning Language-Conditioned Robot Behavior from Offline Data and Crowd-Sourced Annotation" Suraj Nair, Eric Mitchell, Kevin Chen, Brian Ichter, Silvio Savarese, Chelsea Finn. 

The code contains the Mujoco simulation environments used for training the agent, as well as the data collection script used to collect data and annotations from the simulator. It also contains the training script to train the LORL language-conditioned reward function from either the simulated data (or pre-collected real robot data). It also contains code to load a pre-trained visual dynamics model, and use the dynamics model with the LORL reward for executing language-conditioned tasks in simulation.

## Installation

For installation please set up a virtual environment with conda/venv with python 3.6 and install requirements in `requirements.txt`.

For using the simulation environment, the key packages needed are `mujoco-py` and `metaworld`. For mujoco-py, be sure to have GPU rendering activated to make data collection and environment execution more efficient. For `metaworld` you will need an older [version](https://github.com/rlworkgroup/metaworld/tree/b016e6a25e485f1ffa8ccbf52df54ac204a81f31) of the codebase.

## Data Collection / Annotation

To run the data collection script run `python collect_data.py --savepath="data/<YOURPATH>/" --num_episodes=<NUM EPISODES>`

## Train Reward Function

## Visual Dynamics Model

## Planning with Visual Dynamics + LORL Reward

