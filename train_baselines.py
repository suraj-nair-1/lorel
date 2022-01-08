"""Training script for LOReL."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tempfile
from PIL import Image
from absl import app
from absl import flags
import h5py
from itertools import compress
import gym
from models import QFunc, Policy
import numpy as np
from utils import sample_batch_model
from utils import save_im
import torch
import torch.nn.functional as F
import random
from torchvision import transforms
import cv2
import pandas as pd
import copy
from torch import nn, optim
import imageio
import sklearn.metrics
import torchvision
import wandb
import time
import matplotlib.pyplot as plt


FLAGS = flags.FLAGS
flags.DEFINE_integer('batchsize', 32,
                     'Batch Size')
flags.DEFINE_integer('num_labeled', 30000,
                     'How many episodes of data to use')
flags.DEFINE_integer('trainsteps', 200000,
                     'Training Iterations')
flags.DEFINE_integer('seed', 0,
                     'Seed')
flags.DEFINE_string('datapath', 'data/jan_28_maze_4_30k/',
                    'Path to folder with the HDF5 dataset and labels')
flags.DEFINE_string('savedir', 'logs/',
                    'Where to save the model')
flags.DEFINE_string('expname', 'maze_simple',
                    'WandB Name')
flags.DEFINE_integer('finetune', 0,
                     'Finetune language embedding')
flags.DEFINE_float('alpha', 0.0,
                     'Alpha for noisy positives')
flags.DEFINE_integer('aug', 0,
                     'Data aug')
flags.DEFINE_integer('langaug', 0,
                     'Language emb aug')
flags.DEFINE_integer('hidden_size', 128,
                     'Hidden size')
flags.DEFINE_integer('sawyer', 0,
                     'Using sawyer data')
flags.DEFINE_integer('scratch', 0,
                     'LM from scratch (not pretrained)')
flags.DEFINE_float('lr', 0.00001,
                     'learning rate')
flags.DEFINE_string('wandb_entity', "surajn",
                    'Weights and Biases Entity')
flags.DEFINE_string('wandb_project', "lang",
                    'Weights and Biases Project')
flags.DEFINE_integer('rl', 0,
                     '1 for LCRL or 0 for LCBC')

def augment(data, augmentation, batchsize):
  """ Augments batch of images """
  if FLAGS.aug == 0:
    return data
  rand_idx = np.random.randint(0, batchsize, batchsize // 2)
  data[rand_idx] = augmentation(data[rand_idx])
  rand_idx = np.random.randint(0, batchsize, batchsize // 2)
  data[rand_idx] = augmentation(data[rand_idx])
  return data

def extract_ims(pos_pair):
  """ Extracts individual timestep images from episode batch """
  ims_0_batch = torch.FloatTensor(pos_pair[:, :, :, :, 0]).cuda().permute(0, 3, 1, 2)
  ims_s1_batch = torch.FloatTensor(pos_pair[:, :, :, :, 1]).cuda().permute(0, 3, 1, 2)
  ims_s2_batch = torch.FloatTensor(pos_pair[:, :, :, :, 2]).cuda().permute(0, 3, 1, 2)
  ims_g_batch = torch.FloatTensor(pos_pair[:, :, :, :, 3]).cuda().permute(0, 3, 1, 2)
  return ims_0_batch, ims_s1_batch, ims_s2_batch, ims_g_batch


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
    
  torch.manual_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
    
  ## Configure Weights and Biases Logging
  wandb.init(project=FLAGS.wandb_project, name=FLAGS.expname, reinit=True,
                    dir=tempfile.mkdtemp(), entity=FLAGS.wandb_entity)
  wandb.config.update(flags)
  batchsize = FLAGS.batchsize
  
  ## Logging Dir
  savedir = FLAGS.savedir + f"{FLAGS.expname}/bc_" + str(batchsize) + "_s" + str(FLAGS.seed) + "_n" + str(FLAGS.num_labeled) +  "_f" + str(FLAGS.finetune) + "_aug" + str(FLAGS.aug) + "_hs" + str(FLAGS.hidden_size) + "_rl" + str(FLAGS.rl) +"_sawyer" + str(FLAGS.sawyer) + '/'
  if not os.path.exists(savedir):
    os.makedirs(savedir)
    
  ## Load data
  num_labeled = FLAGS.num_labeled
  path = FLAGS.datapath + "data.hdf5"
  f = h5py.File(path, 'r')
  langs = pd.read_csv(FLAGS.datapath + "labels.csv")
  langs = langs["Text Description"].str.strip().to_numpy().reshape(-1)
  langs = langs[:num_labeled]
  nuq = len(np.unique(langs[:]))
  print("Unique Instructions", len(np.unique(langs[:])))
  image_data = []
  ct = 1000
  for dti in range(0, num_labeled, ct):
    image_data.append((f['sim']['ims'][dti:(dti+ct)]*255).astype(np.uint8))
  image_data = np.concatenate(image_data)
  actions = f['sim']['actions'][:num_labeled]
  NUMEP = image_data.shape[0]
  NUMSTEP = image_data.shape[1]
  H, W, C = image_data.shape[2:]
  
  ## Train/Test Splits
  num_train = int(0.9 * num_labeled)
  train_s = range(0,num_train)
  test_s = range(num_train, num_labeled)

  ## Init Model
  hidden_size = FLAGS.hidden_size
  ACT_SIZE = 5
  if FLAGS.rl:
    ## QFunction that takes initial state, current state, language, 
    ## action and predicts q val
    q = QFunc(hidden_size, ACT_SIZE, finetune=FLAGS.finetune).cuda()
    prms = list(q.parameters())
    target_q = copy.deepcopy(q)
    target_q.eval()
    NUM_ACTS = 100 ## Number of actions to sample when taking max
    gamma = 0.8 
  else:
    ## BC policy which takes current state, language and predicts action
    p = Policy(hidden_size, ACT_SIZE, finetune=FLAGS.finetune).cuda()
    prms = list(p.parameters())
  optimiser = optim.Adam(prms, lr=FLAGS.lr)
  
  ## Define Augmentations
  if FLAGS.aug:
    aug = torch.nn.Sequential(
        transforms.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.02),
        transforms.RandomAffine(20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
      ).cuda()
  else:
    aug = lambda a: a

  lss = []
  all_rews = []
  times = []
  for i in range(FLAGS.trainsteps):
    logs = {}
    ## Sample Batches
    t0 = time.time()
    train_pos_pair, train_pos_act, train_l_goals, _ = \
        sample_batch_model(batchsize, 
                           image_data, 
                           actions, 
                           langs,
                           sawyer=1,
                           robot=0, holdout=0, selection=train_s, alpha=FLAGS.alpha)

    t1 = time.time()
    if not FLAGS.rl:
      train_ims_0_batch, train_ims_s1_batch, train_ims_s2_batch, train_ims_g_batch = extract_ims(train_pos_pair)
      train_ims_s1_batch = augment(train_ims_s1_batch, aug, batchsize)
      train_pos_act = torch.FloatTensor(train_pos_act[:, :, 0]).cuda()

      ## Predicted action
      p_act = p(train_ims_s1_batch, train_l_goals)
      ## BC MSE Loss
      bc_loss = ((p_act - train_pos_act)**2).mean(-1)
      loss = (bc_loss).sum()
      logs["Train BC Loss"] = loss.mean().cpu().detach().numpy()
    else:
      ## Sample Batches
      train_ims_0_batch, train_ims_s1_batch, train_ims_s2_batch, train_ims_g_batch = extract_ims(train_pos_pair)
      train_pos_act = torch.FloatTensor(train_pos_act).cuda()
      train_ims_0_batch = augment(train_ims_0_batch, aug, batchsize)
      train_ims_s1_batch = augment(train_ims_s1_batch, aug, batchsize)
      train_ims_s2_batch = augment(train_ims_s2_batch, aug, batchsize)
      train_ims_g_batch = augment(train_ims_g_batch, aug, batchsize)

      ## Q value prediction on (s_t, a_t) and (s_T, a_T)
      negs = q(train_ims_0_batch, train_ims_s1_batch, train_l_goals, train_pos_act[:, :, 0])
      pos = q(train_ims_0_batch, train_ims_g_batch, train_l_goals, train_pos_act[:, :, 1])

      # Compute Q Target with Max_a Q(s_0, s_{t+1}, lang, a) with 100 sampled actions
      with torch.no_grad():
        ## Sample actions for the Max_a Q(s_0 s_{t+1}, lang, a)
        a_dist = torch.FloatTensor(batchsize, NUM_ACTS, ACT_SIZE).uniform_(-1, 1).cuda()
        
        ## Reshaping s_0, s_{t+1}, language instructions
        ims_s2_batch_t = train_ims_s2_batch.unsqueeze(1).repeat(1, NUM_ACTS, 1, 1, 1).reshape(batchsize*NUM_ACTS, 3, 64, 64)
        ims_0_batch_t = train_ims_0_batch.unsqueeze(1).repeat(1, NUM_ACTS, 1, 1, 1).reshape(batchsize*NUM_ACTS, 3, 64, 64)
        l_goals_t = np.expand_dims(train_l_goals, 1).repeat(NUM_ACTS, 1).reshape(batchsize*NUM_ACTS)
        a_dist_t = a_dist.reshape(batchsize*NUM_ACTS, ACT_SIZE)
        
        ## Target computation for (s_{t+1}, a_t)
        tgq_lng = target_q(ims_0_batch_t, ims_s2_batch_t, l_goals_t, a_dist_t).reshape(batchsize, NUM_ACTS, -1)
        ## Max over actions
        best_tgq_lng, _ = tgq_lng.max(1)
    
      ## Target value for (s_t, a_t)
      y_negs = gamma * best_tgq_lng
      ## Target for (s_T, a_T) is 1
      y_pos = torch.ones(y_negs.shape).cuda()
      y = torch.cat([y_negs,y_pos], 0)
      qs = torch.cat([negs,pos], 0)
      loss = ((y - qs)**2).mean() 
      logs["Negative Dist"] = wandb.Histogram(negs.squeeze().cpu().detach().numpy())
      logs["Positive Dist"] = wandb.Histogram(pos.squeeze().cpu().detach().numpy())
      logs["AV0"] = wandb.Histogram(tgq_lng.squeeze().cpu().detach().numpy()[0])
      logs["AV1"] = wandb.Histogram(tgq_lng.squeeze().cpu().detach().numpy()[1])
      logs["Train TD Loss"] = loss.mean().cpu().detach().numpy()
      
    t2 = time.time()
    
    optimiser.zero_grad()
    (loss).backward()
    optimiser.step()
    t3 = time.time()
    times.append([t1-t0, t2-t1, t3-t2])
    
    if FLAGS.rl and (i % 10 == 0):
      target_q.load_state_dict(q.state_dict())
      target_q.eval()

    
    ## Log losses
    if (i % 100 == 0):
      if not FLAGS.rl:
        test_pos_pair, test_pos_act, test_l_goals, test_smp_ind = sample_batch_model(batchsize, 
                           image_data, 
                           actions, 
                           langs,
                           sawyer=1,
                           robot=0, holdout=0, selection=test_s, alpha=FLAGS.alpha)
        test_ims_0_batch, test_ims_s1_batch, test_ims_s2_batch, test_ims_g_batch = extract_ims(test_pos_pair)

        test_ims_s1_batch = augment(test_ims_s1_batch, aug, batchsize)
        test_pos_act = torch.FloatTensor(test_pos_act[:, :, 0]).cuda()

        with torch.no_grad():
          p_act = p(test_ims_s1_batch, test_l_goals)
          t_bc_loss = ((p_act - test_pos_act)**2).mean(-1)
          t_loss = (t_bc_loss).sum()
        logs["Test BC Loss"] = t_loss.mean().cpu().detach().numpy()

      times = np.array(times)
      print(f"{i} - Sampling time {times[:, 0].mean()}, Forward Pass {times[:, 1].mean()}, Backward Pass {times[:, 2].mean()}")
      print(logs)
      times = []
      
    if (i % 10000 == 0):
      #### SAVING MODELS / IMAGES
      sdict = {}
      if FLAGS.rl:
        sdict['q'] = q.state_dict()
      else:
        sdict['p'] = p.state_dict()
      torch.save(sdict, 
                  os.path.join(savedir, 'models_%d.pth' % i))
    if (i % 100 == 0):
      wandb.log(logs, step = i)
        

if __name__ == '__main__':
  app.run(main)
