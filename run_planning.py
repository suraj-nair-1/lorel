"""Planning Script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import copy
from absl import app
from absl import flags
import h5py
import gym
import lorl_env
from models import Discriminator
import numpy as np
from utils import save_im
import torch
import pandas as pd
from torch import nn, optim
import imageio
import wandb
import time
import cv2
from multiprocessing import Pool
import string
import tensorflow.compat.v1 as tf
from tensor2tensor.bin.t2t_decoder import create_hparams
from tensor2tensor.utils import registry


FLAGS = flags.FLAGS

flags.DEFINE_integer('batchsize', 32,
                     'Batch Size')
flags.DEFINE_integer('seed', 0,
                     'Seed')
flags.DEFINE_string('savedir', 'logs/',
                    'Where to save the model')
flags.DEFINE_string('datapath', 'data/jan_28_maze_4_30k/',
                    'Path to the HDF5 dataset')
flags.DEFINE_string('expname', 'planning',
                    'WandB Name')
flags.DEFINE_string('reward_path', 'logs/_maze_simple_reward_64_0.9_s0_lang1_/models_99000.pth',
                    'Path to reward model')
flags.DEFINE_string('cost', 'lorl', 'Cost Function to use during planning')
flags.DEFINE_string('model_path', None, 
                    'Path to dynamics model folder')
flags.DEFINE_string('instruction', 'close drawer',
                    'Command')
flags.DEFINE_string('newinstruction', None,
                    'Replacement instruction to use for a task')
flags.DEFINE_integer('verbose', 0,
                     'Verbose Planning Logging')
flags.DEFINE_integer('hidden_size', 128,
                     'Hidden size')
flags.DEFINE_integer('ph', 20,
                     'planning horizon')
flags.DEFINE_integer('cem_iters', 1,
                     'planning horizon')
flags.DEFINE_integer('samples', 200,
                     'planning horizon')
flags.DEFINE_string('wandb_entity', "surajn"
                    'Weights and Biases Entity')
flags.DEFINE_string('wandb_project', "lang",
                    'Weights and Biases Project')


def gt_reward(qpos, inital, instr):
  """Measure true task progress for different instructions"""
  if instr == "open drawer":
    dist = inital[14] - qpos[14]
    s = dist > 0.02
  elif instr == "close drawer":
    dist = qpos[14] - inital[14]
    s = dist > 0.02
  elif instr == "turn faucet right":
    dist = inital[13] - qpos[13]
    s = dist > np.pi / 10
  elif instr == "turn faucet left":
    dist = qpos[13] - inital[13]
    s = dist > np.pi / 10
  elif instr == "move black mug right":
    dist = inital[11] - qpos[11]
    s = dist > 0.02
  elif instr == "move white mug down":
    dist = qpos[10] - inital[10]
    s = dist > 0.02
  else:
    dist = 0
    s = 0
  return dist, s


def plan_learned(im, num_samples, model, ASIZE, cost="lorl", logdir = None, instruction = None, goalim=None, plansavedir=None, verbose=0, sv2p=None):
  """Plan actions to maximize reward using SV2P video prediction model"""
  ### Planning parameters
  CEM_ITERS = FLAGS.cem_iters
  REFIT = num_samples // 10
  mult=255.0
  mult2 = 1.0
  H = FLAGS.ph
  langs = [instruction]*num_samples
  
  ## Load models
  d = model
  forward_prediction_ops, forward_sess, forward_placeholders = sv2p
  
  im0 = torch.FloatTensor(im).cuda().unsqueeze(0).permute(0, 3, 1, 2) 
  img = torch.FloatTensor(goalim).cuda().unsqueeze(0).permute(0, 3, 1, 2) 
  for j in range(CEM_ITERS):
    rewards = []
    t0 = time.time()
    # Sample actions
    if j == 0:
      acts = np.random.uniform(-1, 1, (num_samples, H, ASIZE))
    else:
      acts = np.random.normal(mu, std, (num_samples, H, ASIZE))
    t1 = time.time()
    
    ### Feedforward actions through video prediction model
    forward_feed = {
        forward_placeholders['inputs']:
            np.repeat(np.expand_dims(np.expand_dims(im, 0), 0),
                      num_samples, axis=0) * 255.0,
        forward_placeholders['input_action']:
            acts[:, 0:1, :],
        forward_placeholders['targets']:
            np.zeros(forward_placeholders['targets'].shape),
        forward_placeholders['target_action']:
            acts
      }
    forward_predictions = forward_sess.run(forward_prediction_ops, forward_feed)
    final_states = torch.FloatTensor(forward_predictions[:, -1, :, :, :, 0]).cuda().permute(0, 3, 1, 2) / 255.0
    
    ### Compute reward for each trajectory
    if cost == "pixel":
      rewards = -((final_states - img.repeat(num_samples, 1, 1, 1))**2).mean((1, 2, 3)).cpu().detach().numpy()
    elif cost == "pixellpips":
      import lpips
      loss_fn_alex = lpips.LPIPS(net='alex').cuda()
      rimg = img.repeat(num_samples, 1, 1, 1)
      rimg = (rimg - 0.5) * 2
      final_states = (final_states - 0.5) * 2
      rewards = loss_fn_alex(final_states, rimg).mean((1, 2, 3)).cpu().detach().numpy()
    elif cost == "lorl":
      rewards = d(im0.repeat(num_samples, 1, 1, 1), final_states, langs[:]).cpu().detach().numpy()
    if verbose:
      ### Logs all sampled predictions and associated reward
      if not os.path.exists(f'{plansavedir}r/iter{j}/'):
        os.makedirs(f'{plansavedir}r/iter{j}/')
      for i in range(num_samples):
        imageio.mimsave(f'{plansavedir}r/iter{j}/_{str(rewards[i].round(4))}_{i}.gif', forward_predictions[i, :, :, :, :, 0].astype(np.uint8))
    ### Sort actions by reward and select best ones
    ids = np.argsort(rewards.reshape(-1))
    acts_sorted = acts[ids]
    acts_good = acts_sorted[-REFIT:]
    mu = acts_good.mean(0)
    std = acts_good.std(0)
    t3 = time.time()
  ### Return best action 
  best_ind = np.argmax(rewards)
  return (acts[best_ind])  


def main(argv):
  print(argv)
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
    
  torch.manual_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
    
  ### Setup Logging
  wandb.init(project=FLAGS.wandb_project, name=FLAGS.expname, reinit=True,
                    dir=tempfile.mkdtemp(), entity=FLAGS.wandb_entity)
  wandb.config.update(flags)
  batchsize = FLAGS.batchsize
  savedir = FLAGS.savedir + f"planning_{FLAGS.expname}_" + str(batchsize) + "_s" + \
      str(FLAGS.seed) + "/" + "_i" + str(FLAGS.instruction).replace(" ", "_") + "/" + "_ni" + str(FLAGS.newinstruction).replace(" ", "_") + "/" 
  savedir = savedir + "c" + str(FLAGS.cost) + "_m" + str(FLAGS.reward_path) + "/"
  if not os.path.exists(savedir):
    os.makedirs(savedir)
  
  ## Sampling/Env Parameters
  NUM_ACTS = FLAGS.samples 
  envname = "LorlEnv-v0"
  ASIZE = 5
  env = gym.make(envname)
  env.reset()

  
  ## If using LORL load model
  NUMTRIAL = batchsize
  hidden_size = FLAGS.hidden_size
  if FLAGS.cost == "lorl":
    d = Discriminator(hidden_size).cuda()
    model_dicts = torch.load(FLAGS.reward_path)
    d.load_state_dict(model_dicts['d'])
    d.requires_grad = False
    d.eval()
  else:
    d = None
  instr = FLAGS.instruction
  
  ## Load visual dynamics model
  homedir = FLAGS.model_path
  FLAGS.data_dir = homedir + '/data/'
  FLAGS.output_dir = homedir + '/out/'
  FLAGS.problem = 'lang_robot'
  FLAGS.hparams = 'video_num_input_frames=1,video_num_target_frames=20'
  FLAGS.hparams_set = 'next_frame_sv2p'
  FLAGS.model = 'next_frame_sv2p'
  # Create hparams
  hparams = create_hparams()
  hparams.video_num_input_frames = 1
  hparams.video_num_target_frames = 20

  # Params
  num_replicas = FLAGS.samples
  frame_shape = hparams.problem.frame_shape
  forward_graph = tf.Graph()
  with forward_graph.as_default():
    forward_sess = tf.Session()
    input_size = [num_replicas, hparams.video_num_input_frames]
    target_size = [num_replicas, hparams.video_num_target_frames]
    forward_placeholders = {
        'inputs':
            tf.placeholder(tf.float32, input_size + frame_shape),
        'input_action':
            tf.placeholder(tf.float32, input_size + [ASIZE]),
        'targets':
            tf.placeholder(tf.float32, target_size + frame_shape),
        'target_action':
            tf.placeholder(tf.float32, target_size + [ASIZE]),
    }
    # Creat model
    forward_model_cls = registry.model(FLAGS.model)
    forward_model = forward_model_cls(hparams, tf.estimator.ModeKeys.PREDICT)
    forward_prediction_ops, _ = forward_model(forward_placeholders)
    forward_saver = tf.train.Saver()
    forward_saver.restore(forward_sess,
                          homedir + '/out/model.ckpt-300000')
    print('LOADED SV2P!')
    sv2p_model = (forward_prediction_ops, forward_sess, forward_placeholders)

  
  all_dists = []
  all_s = []
  for i in range(NUMTRIAL):
    im = env.reset()
    
    ## Initialize state for different tasks
    if instr == "open drawer":
      env.sim.data.qpos[14] = 0 + np.random.uniform(-0.05, 0)
    elif instr == "close drawer":
      env.sim.data.qpos[14] = -0.1 + np.random.uniform(-0.05, 0.05)
    elif instr == "turn faucet right":
      env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
    elif instr == "turn faucet left":
      env.sim.data.qpos[13] = 0 + np.random.uniform(-np.pi/5, np.pi/5)
    elif instr == "move black mug right":
      env.sim.data.qpos[11] = -0.2 + np.random.uniform(-0.05, 0.05)
      env.sim.data.qpos[12] = 0.65 + np.random.uniform(-0.05, 0.05)
    elif instr == "move white mug down":
      env.sim.data.qpos[9] = -0.2 + np.random.uniform(-0.05, 0.05)
      env.sim.data.qpos[10] = 0.65 + np.random.uniform(-0.05, 0.05)

    if (instr == "move white mug down"):
      env._reset_hand(pos=[-0.1, 0.55, 0.1])
    elif (instr == "move black mug right"):
      env._reset_hand(pos=[-0.1, 0.55, 0.1])
    else:
      env._reset_hand(pos=[0, 0.45, 0.1])
    for _ in range(50):
      env.sim.step()

    reset_state = copy.deepcopy(env.sim.data.qpos[:])
    env.sim.data.qpos[:] = reset_state
    env.sim.data.qacc[:] = 0
    env.sim.data.qvel[:] = 0
    env.sim.step()

    ## Initialize goal image for initial state
    if instr == "open drawer":
      env.sim.data.qpos[14] = -0.15
    elif instr == "close drawer":
      env.sim.data.qpos[14] = 0.0
    elif instr == "turn faucet right":
      env.sim.data.qpos[13] -= np.pi/5
    elif instr == "turn faucet left":
      env.sim.data.qpos[13] += np.pi/5
    elif instr == "move black mug right":
      env.sim.data.qpos[11] -= 0.1
    elif instr == "move white mug down":
      env.sim.data.qpos[10] += 0.1
    env.sim.step()
    gim = env._get_obs()[:, :, :3]

    ## Reset inital state
    env.sim.data.qpos[:] = reset_state
    env.sim.data.qacc[:] = 0
    env.sim.data.qvel[:] = 0
    env.sim.step()

    initial_state = copy.deepcopy(env.sim.data.qpos[:])
    im = env._get_obs()[:, :, :3]
    initim = im
    save_im((initim*255.0).astype(np.uint8), savedir+f"initialim_{i}_{instr}.jpg")
    save_im((gim*255.0).astype(np.uint8), savedir+f"gim_ep_{i}_{instr}.jpg")
    episode_ims = [(im[:,:,:3]*255.0).astype(np.uint8)]
    
    ## If rephrasing instruction, use rephrased instruction
    if FLAGS.newinstruction is None:
      planinstr = instr
    else:
      planinstr = FLAGS.newinstruction
    
    ## Rollout random policy
    if FLAGS.cost == "random":
      for _ in range(FLAGS.ph):
        a = np.random.uniform(-1, 1, (ASIZE))
        im, r, done, _ = env.step(a)
        episode_ims.append((im[:,:,:3]*255.0).astype(np.uint8))
        ## Measure success, if successful finish episode
        dist, s = gt_reward(env.sim.data.qpos[:], initial_state, instr)
        if s:
          break
    else:
      s = 0
      plansavedir = savedir + str(i) + "/" 
      if not os.path.exists(plansavedir):
        os.makedirs(plansavedir)
      initim = env._get_obs()[:, :, :3]
      action = plan_learned(initim, NUM_ACTS, d, ASIZE, logdir=savedir, instruction=planinstr, goalim = gim, cost=FLAGS.cost, plansavedir=plansavedir, verbose=FLAGS.verbose, sv2p=sv2p_model)
      for a in action:
        im, r, done, _ = env.step(a)
        episode_ims.append((im[:,:,:3]*255.0).astype(np.uint8))
        ## Measure success, if successful finish episode
        dist, s = gt_reward(env.sim.data.qpos[:], initial_state, instr)
        if s:
          break
        
    print('-'*50, str(i))
    
    imageio.mimsave(f'{savedir}/episode_{str(i).zfill(4)}_dist_{dist}_{instr}_{FLAGS.cost}.gif', episode_ims)
    all_dists.append(dist)
    all_s.append(s)
  all_dists = np.array(all_dists)
  all_s = np.array(all_s)
  print(np.mean(all_dists))
  print(np.mean(all_s))
  logs = {}
  logs["Distance"] = np.mean(all_dists)
  logs["SR"] = np.mean(all_s)
  wandb.log(logs)

if __name__ == '__main__':
  app.run(main)
