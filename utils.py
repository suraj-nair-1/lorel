"""Utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torchvision import transforms
import cv2
import time
import numpy as np


def sample_batch_model(bs, ims, actions, desc=None, sawyer=0, robot=0, rl=0, holdout=0, selection=None, alpha=0):
  """Sample a batch"""
  ## Pick episodes
  index = np.random.choice(selection, bs, replace=True)
  if holdout:
    for i in range(0, bs):
      while True:
        mugcond = ("open" in desc[index[i]]) and ("drawer" in desc[index[i]])
        fcond = ("faucet" in desc[index[i]]) and ("left" in desc[index[i]])
        if mugcond or fcond:
          index[i] = np.random.choice(ims.shape[0], 1, replace=True)
        else:
          break
  ims = ims[index]
  
  ## Select language annotation for episode
  if robot:
    ## Robot episodes have two annotations, select one randomly
    desc1, desc2 = desc
    d1 = desc1[index]
    d2 = desc2[index]
    d = []
    for d_i in range(bs):
      if (np.random.uniform() < 0.5) or (d2[d_i] == "nan"):
        d.append(d1[d_i])
      else:
        d.append(d2[d_i])
    d = np.array(d)
  else:
    d = desc[index]
  tlen = ims.shape[1]
  actions = actions[index]
  
  ## Select s_0, s_t, s_{t+1}, s_T
  ts1 = np.random.randint(0, tlen - 1, bs)
  ts2 = ts1 + 1
  t0 = np.random.randint(0, max(1, int(alpha*tlen)), bs)
  tg = np.random.randint(min(tlen-1, int((1-alpha) * tlen)), tlen, bs)
  im_t0 = select_indices(ims, t0)
  im_ts1 = select_indices(ims, ts1)
  im_ts2 = select_indices(ims, ts2)
  im_tg = select_indices(ims, tg)
  pos_act = select_indices(actions, ts1)
  
  pos_pair_cat = np.stack([im_t0, im_ts1, im_ts2, im_tg], -1)
  return pos_pair_cat / 255., pos_act, d, index

def select_indices(tensor, indices):
  new_images = []
  for b in range(tensor.shape[0]):
    new_images.append(tensor[b, indices[b]])
  tensor = np.stack(new_images, 0)
  return tensor


def save_im(im, name):
  """Save an image."""
  im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
  cv2.imwrite(name, im.astype(np.uint8))


def read_im(path):
  """Read an image."""
  with open(path, 'rb') as fid:
    raw_im = np.asarray(bytearray(fid.read()), dtype=np.uint8)
    im = cv2.imdecode(raw_im, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # Opencv using BGR order
    im = cv2.resize(im, (64, 64), interpolation=cv2.INTER_LANCZOS4)
  return im
