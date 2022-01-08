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
from models import Discriminator
import numpy as np
from utils import sample_batch_model
from utils import save_im
import torch
import torch.nn.functional as F
import random
from torchvision import transforms
import cv2
import pandas as pd
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
flags.DEFINE_integer('robot', 0,
                     'Using robot data')
flags.DEFINE_integer('fn', 0,
                     'Flipped Negatives')
flags.DEFINE_integer('filter', 0,
                     'Filter nothing episodes out of robot data.')
flags.DEFINE_integer('hidden_size', 128,
                     'Hidden size')
flags.DEFINE_integer('sawyer', 0,
                     'Using sawyer data')
flags.DEFINE_integer('scratch', 0,
                     'LM from scratch (not pretrained)')
flags.DEFINE_float('lr', 0.00001,
                     'learning rate')
flags.DEFINE_integer('holdout', 0,
                     'Holdout faucet left and black mug right tasks')
flags.DEFINE_string('resume', None,
                    'What ckpt to resume from')
flags.DEFINE_string('wandb_entity', "surajn",
                    'Weights and Biases Entity')
flags.DEFINE_string('wandb_project', "lang",
                    'Weights and Biases Project')

def augment(data, augmentation, batchsize):
  """ Augments batch of images """
  if FLAGS.aug == 0:
    return data
  if FLAGS.robot:
    for i in range(0, 12, 3):
      rand_idx = np.random.randint(0, batchsize, batchsize // 2)
      data[rand_idx, (i):(i+3), :, :] = augmentation(data[rand_idx, (i):(i+3), :, :])
      rand_idx = np.random.randint(0, batchsize, batchsize // 2)
      data[rand_idx, (i):(i+3), :, :] = augmentation(data[rand_idx, (i):(i+3), :, :])
  else:
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

def get_neg_ep(train_ims_0_batch, train_ims_g_batch, train_l_goals):
  """ Selects 'Negative Examples' for a batch of image/annotations """
  train_ims_0_batch_shuf = []
  train_ims_g_batch_shuf = []
  bs = train_ims_0_batch.shape[0]
  for ind in range(bs):
    notdone = 1
    ### Make sure sampled episode has different annotation
    while notdone:
      rand_idx = np.random.randint(0, bs)
      notdone = (train_l_goals[ind] == train_l_goals[rand_idx])
    train_ims_0_batch_shuf.append(train_ims_0_batch[rand_idx])
    train_ims_g_batch_shuf.append(train_ims_g_batch[rand_idx])
  train_ims_0_batch_shuf = torch.stack(train_ims_0_batch_shuf)
  train_ims_g_batch_shuf = torch.stack(train_ims_g_batch_shuf)
  return train_ims_0_batch_shuf, train_ims_g_batch_shuf

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
  savedir = FLAGS.savedir + f"{FLAGS.expname}/lorel_" + str(batchsize) + "_s" + str(FLAGS.seed) + "_n" + str(FLAGS.num_labeled) + "_f" + str(FLAGS.finetune) + "_alpha" + str(FLAGS.alpha) + "_aug" + str(FLAGS.aug) + "_filter" + str(FLAGS.filter)  +"_fn" + str(FLAGS.fn) + "_hs" + str(FLAGS.hidden_size) +"_robot" + str(FLAGS.robot) +"_langaug" + str(FLAGS.langaug) +"_holdout" + str(FLAGS.holdout) + '/'
  if not os.path.exists(savedir):
    os.makedirs(savedir)
    
  ## Load data
  num_labeled = FLAGS.num_labeled
  path = FLAGS.datapath + "data.hdf5"
  f = h5py.File(path, 'r')
  langs = pd.read_csv(FLAGS.datapath + "labels.csv")
  filtr = [True] * num_labeled
  if FLAGS.robot:
    langs1 = langs["Text Description 1"].str.strip().to_numpy().reshape(-1)
    langs2 = langs["Text Description 2"].str.strip().to_numpy().reshape(-1)
    langs1 = langs1[:num_labeled]
    langs2 = langs2[:num_labeled]
    langs1 = np.array(['' if x is np.isnan else x for x in langs1])
    langs2 = np.array(['' if x is np.isnan else x for x in langs2])
    if FLAGS.filter:
      filtr1 = np.array([int(("nothing" in l) or ("nan" in l) or ("wave" in l)) for l in langs1])
      filtr2 = np.array([int(("nothing" in l) or ("nan" in l) or ("wave" in l)) for l in langs2])
      filtr = (filtr1 + filtr2) == 0
      langs1 = langs1[filtr]
      langs2 = langs2[filtr]
    langs = (langs1, langs2)
  else:
    langs = langs["Text Description"].str.strip().to_numpy().reshape(-1)
    langs = langs[:num_labeled]
    nuq = len(np.unique(langs[:]))
    print("Unique Instructions", len(np.unique(langs[:])))
  image_data = []
  ct = 1000
  for dti in range(0, num_labeled, ct):
    image_data.append((f['sim']['ims'][dti:(dti+ct)]*255).astype(np.uint8))
  image_data = np.concatenate(image_data)[filtr]
  actions = f['sim']['actions'][:num_labeled][filtr]
  num_labeled = int(np.sum(filtr))
  NUMEP = image_data.shape[0]
  NUMSTEP = image_data.shape[1]
  H, W, C = image_data.shape[2:]
  
  num_train = int(0.9 * num_labeled)
  train_s = range(0,num_train)
  test_s = range(num_train, num_labeled)

  ## Init Model
  hidden_size = FLAGS.hidden_size
  d = Discriminator(hidden_size, 
                    finetune=FLAGS.finetune, 
                    robot=FLAGS.robot, 
                    langaug=FLAGS.langaug, 
                    scratch=FLAGS.scratch).cuda()
  if FLAGS.resume is not None:
    model_dicts = torch.load(FLAGS.resume)
    d.load_state_dict(model_dicts['d'])
  prms = list(d.parameters())
  optimiser = optim.Adam(prms, lr=FLAGS.lr)
  bce = nn.BCELoss(reduce=False)
  
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
    train_pos_pair, _, train_l_goals, _ = \
        sample_batch_model(batchsize, 
                           image_data, 
                           actions, 
                           langs,
                           sawyer=FLAGS.sawyer,
                           robot=FLAGS.robot, holdout=FLAGS.holdout, selection=train_s, alpha=FLAGS.alpha)
    train_ims_0_batch, train_ims_s1_batch, train_ims_s2_batch, train_ims_g_batch = extract_ims(train_pos_pair)
    train_ims_0_batch = augment(train_ims_0_batch, aug, batchsize)
    train_ims_g_batch = augment(train_ims_g_batch, aug, batchsize)
    
    ## Get Negative Examples
    train_ims_0_batch_shuf, train_ims_g_batch_shuf = get_neg_ep(train_ims_0_batch, train_ims_g_batch, train_l_goals)
    
    ## Forward Pass
    t1 = time.time()
    pos = d(train_ims_0_batch, train_ims_g_batch, train_l_goals)
    neg1 = d(train_ims_0_batch_shuf, train_ims_g_batch_shuf, train_l_goals)
    if FLAGS.fn:
      neg2 = d(train_ims_g_batch, train_ims_0_batch, train_l_goals)
      labels_pos = torch.ones(batchsize, 1).cuda()
      labels_neg = torch.zeros(batchsize*2 , 1).cuda()
      preds = torch.cat([pos, neg1, neg2], 0)
    else:
      labels_pos = torch.ones(batchsize, 1).cuda()
      labels_neg = torch.zeros(batchsize , 1).cuda()
      preds = torch.cat([pos, neg1], 0)
    labels = torch.cat([labels_pos, labels_neg], 0)
    loss = bce(preds, labels).mean()
    t2 = time.time()
    
    acc = (((1 * (preds > 0.5)) == labels) * 1.0).mean().cpu().detach().numpy()
    prec, rec, _, _ = sklearn.metrics.precision_recall_fscore_support(labels.cpu().detach().numpy(), 
                                     (1 * (preds > 0.5)).cpu().detach().numpy())
    ## Backward Pass
    optimiser.zero_grad()
    (loss).backward()
    optimiser.step()
    lss.append(loss.cpu().detach().numpy())
    t3 = time.time()
    times.append([t1-t0, t2-t1, t3-t2])
    
    ## Train Logging
    logs["Train BCE Loss"] = lss[-1]
    logs["Train Accuracy"] = acc
    logs["Train Precision"] = prec[1]
    logs["Train Recall"] = rec[1]
    
    
    ### SAVING MODELS / IMAGES
    if (i % 50000 == 0):
      savedir_now = savedir + f"{i}/"
      if not os.path.exists(savedir_now):
        os.makedirs(savedir_now)
      
      sdict = {}
      sdict['d'] = d.state_dict()
      torch.save(sdict, 
                  os.path.join(savedir, 'models_%d.pth' % i))
      
      if not FLAGS.robot:
        for b in np.random.randint(0, batchsize, (5,)):
          i0 = train_ims_0_batch[b].permute(1, 2, 0)
          ig = train_ims_g_batch[b].permute(1, 2, 0)
          im = (torch.cat([i0, ig], 0)).cpu().detach().numpy() * 255.0
          save_im(im.astype(np.uint8), 
                  savedir_now+f"pos_{train_l_goals[b]}_{pos[b][0].cpu().detach().numpy()}.png")

          i0 = train_ims_0_batch_shuf[b].permute(1, 2, 0)
          ig = train_ims_g_batch_shuf[b].permute(1, 2, 0)
          im = (torch.cat([i0, ig], 0)).cpu().detach().numpy() * 255.0
          save_im(im.astype(np.uint8), 
                  savedir_now+f"neg_{train_l_goals[b]}_{neg1[b][0].cpu().detach().numpy()}.png")

          if FLAGS.fn:
            i0 = train_ims_g_batch[b].permute(1, 2, 0)
            ig = train_ims_0_batch[b].permute(1, 2, 0)
            im = (torch.cat([i0, ig], 0)).cpu().detach().numpy() * 255.0
            save_im(im.astype(np.uint8), 
                    savedir_now+f"neg2_{train_l_goals[b]}_{neg2[b][0].cpu().detach().numpy()}.png")
    
    ## Test Data Eval
    if (i % 100 == 0):
      test_pos_pair, _, test_l_goals, _ = sample_batch_model(batchsize, 
                                                             image_data, actions, 
                                                             langs, alpha=0, 
                                                             sawyer=FLAGS.sawyer,
                                                             robot=FLAGS.robot, holdout=FLAGS.holdout, selection=test_s)
      test_ims_0_batch, test_ims_s1_batch, test_ims_s2_batch, test_ims_g_batch = extract_ims(test_pos_pair)

      with torch.no_grad():
        test_ims_0_batch_shuf, test_ims_g_batch_shuf = get_neg_ep(test_ims_0_batch, test_ims_g_batch, test_l_goals)
        pos = d(test_ims_0_batch, test_ims_g_batch, test_l_goals)
        neg = d(test_ims_0_batch_shuf, test_ims_g_batch_shuf, test_l_goals)
        if FLAGS.fn:
          neg2 = d(test_ims_g_batch, test_ims_0_batch, test_l_goals)
          labels_pos = torch.ones(batchsize, 1).cuda()
          labels_neg = torch.zeros(batchsize*2, 1).cuda()
          preds = torch.cat([pos, neg, neg2], 0)
        else:
          labels_pos = torch.ones(batchsize, 1).cuda()
          labels_neg = torch.zeros(batchsize, 1).cuda()
          preds = torch.cat([pos, neg], 0)
        labels = torch.cat([labels_pos, labels_neg], 0)
        loss = bce(preds, labels).mean()

        acc = (((1 * (preds > 0.5)) == labels) * 1.0).mean().cpu().detach().numpy()
        prec, rec, _, _ = sklearn.metrics.precision_recall_fscore_support(labels.cpu().detach().numpy(), 
                                     (1 * (preds > 0.5)).cpu().detach().numpy())
        
      ## Test Logging
      logs["Test BCE Loss"] = loss.cpu().detach().numpy()
      logs["Test Accuracy"] = acc
      logs["Test Precision"] = prec[1]
      logs["Test Recall"] = rec[1]
      
      times = np.array(times)
      print(f"{i} - Sampling+Processing time {times[:, 0].mean()}, Forward Pass {times[:, 1].mean()}, Backward Pass {times[:, 2].mean()}", flush=True)
      print(logs, flush=True)
      times = []
      
    ## Send to W&B
    if (i % 100 == 0):
      wandb.log(logs, step = i)
        

if __name__ == '__main__':
  app.run(main)
