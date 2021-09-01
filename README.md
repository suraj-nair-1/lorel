# Code for the paper "Learning Language-Conditioned Robot Behavior from Offline Data and Crowd-Sourced Annotation"

This is code for the paper "Learning Language-Conditioned Robot Behavior from Offline Data and Crowd-Sourced Annotation" Suraj Nair, Eric Mitchell, Kevin Chen, Brian Ichter, Silvio Savarese, Chelsea Finn. 

The code contains the Mujoco simulation environments used for training the agent, as well as the data collection script used to collect data and annotations from the simulator. It also contains the training script to train the LORL language-conditioned reward function from either the simulated data (or pre-collected real robot data). It also contains code to load a pre-trained visual dynamics model, and use the dynamics model with the LORL reward for executing language-conditioned tasks in simulation.

## Installation

For installation please set up a virtual environment with conda/venv with python 3.6 and install requirements in `requirements.txt`.

For using the simulation environment, the key packages needed are `mujoco-py` and `metaworld`. For mujoco-py, be sure to have GPU rendering activated to make data collection and environment execution more efficient. For `metaworld` you may need an older [version](https://github.com/rlworkgroup/metaworld/tree/b016e6a25e485f1ffa8ccbf52df54ac204a81f31) of the codebase.

## Data Collection / Annotation

To run the data collection script run
`python collect_data.py --savepath="data/<YOURPATH>/" --num_episodes=<NUM EPISODES>`

This should output an HDF5 file with the episodes and a CSV with the language labels to the save directory.

## Train Reward Function

Given the collected simulation data you can train the LORL reward with the following command. If you'd rather use the pre-collected dataset from the paper it can be found [here](https://drive.google.com/file/d/1pLnctqkOzyWZa1F1zTFqkNgUzSkUCtEv/view?usp=sharing).

`python train_lorl.py  --batchsize=32 --datapath=<PATH TO DATA FOLDER> --savedir=logs/ --expname=<EXPERIMENT NAME> --num_labeled=<HOW MANY EPISODES TO USE> --aug=1 --hidden_size=128 --trainsteps=400000 --fn=1 --robot=0 --langaug=1 --alpha=0`

You can also train the LORL reward on the pre-collected real robot dataset, consisting of a Franka Emika Panda operating over a desk, and crowdsourced language annotations of the episodes. This dataset can be found [here](https://drive.google.com/file/d/1r3lkVsGdLdxf3dr7HdhW_IPTR0wEDhEp/view?usp=sharing). If you do train the model on this data, you should simply change the command to:

`python train_lorl.py  --batchsize=32 --datapath=<PATH TO ROBOT DATA FOLDER> --savedir=logs/ --expname=<EXPERIMENT NAME> --num_labeled=<HOW MANY EPISODES TO USE> --aug=1 --hidden_size=128 --trainsteps=400000 --fn=1 --robot=1 --langaug=1 --alpha=0.25`

If you do use the real robot data, please cite the project from which the data was collected.
```
@inproceedings{
anonymous2021exampledriven,
title={Example-Driven Model-Based Reinforcement Learning for Solving Long-Horizon Visuomotor Tasks},
author={Anonymous},
booktitle={Submitted to 5th Annual Conference on Robot Learning },
year={2021},
url={https://openreview.net/forum?id=_daq0uh6yXr},
note={under review}
}
```

## Visual Dynamics Model

If you want to use the same visual planning pipeline as the paper, you will need to use the SV2P video prediction model [Babaeizadeh et al]. The code for this model is part of the `tensor2tensor` package, and you'll want to clone and install this [custom version of tensor2tensor](https://github.com/suraj-nair-1/tensor2tensor). This installation is tricky - it will automatically upgrade the tensorflow version, however SV2P still requires and old version. You will need to downgrade `tensorflow-gpu` and associated tensorflow packages `tensorflow-probability`, etc. to the ones listed in the `requirements.txt`. You'll also need to manually install `gast=0.2.2`. 

If the above is done correctly, you should be able to directly load the pre-trained SV2P data and SV2P model [here](https://drive.google.com/file/d/1hKbju9QSxYJbk5Ee3rnJif2VtWU-Vid-/view?usp=sharing). You can also re-train the SV2P dynamics model if desired using the `tensor2tensor` repo. 

## Planning with Visual Dynamics + LORL Reward

Finally, given the trained SV2P model and trained LORL reward, you can run CEM planning to complete language-conditioned tasks with the following command:

`python run_planning.py --batchsize=<NUMBER OF TRIALS> --savedir=logs/ --expname=<EXPERIMENT NAME> --reward_path=<PATH TO LORL REWARD CHECKPOINT> --cost="lorl" --instruction="turn faucet right" --verbose=0 --hidden_size=128 --cem_iters=3 --samples=200 --model_path=<PATH TO SV2P DIRECTORY>`

