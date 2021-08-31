"""Data generation script for LORL.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import gym
import lorl_env
import h5py
import numpy as np
import imageio
import time
import os
import pandas as pd
import random
FLAGS = flags.FLAGS
flags.DEFINE_string('savepath', 'data/generated_data/',
                    'Path to save the HDF5 dataset')
flags.DEFINE_integer('seed', 0,
                     'Seed')
flags.DEFINE_integer('num_episodes', 200,
                     'Number of episodes')

def get_motion_desc_sawyer(s0, st, color):
  """ Generate motion description for object
  based on initial and final XY position"""
  dl = st - s0
  descr = f"move {color} "
  dirs = []
  if dl[1] > 0.02:
    dirs.append("down ")
  elif dl[1] < -0.02:
    dirs.append("up ")
  if dl[0] > 0.02:
    dirs.append("left ")
  elif dl[0] < -0.02:
    dirs.append("right ")
  random.shuffle(dirs)
  if len(dirs) > 0:
    direction = "and ".join(dirs)
    descr += direction
  else:
    return []
  return [descr]


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  np.random.seed(FLAGS.seed)

  ### Data parameters
  forward_ep = FLAGS.num_episodes
  action_space = 5
  steps_per_ep = 20
  imsize = 64
  if not os.path.exists(FLAGS.savepath):
    os.makedirs(FLAGS.savepath)
  datapath = FLAGS.savepath + "data.hdf5"

  ### Dataset file
  f = h5py.File(datapath, 'w')
  sim_data = f.create_group('sim')
  sim_data.create_dataset('ims', (forward_ep, steps_per_ep, imsize, imsize, 3),
                          dtype='f')
  sim_data.create_dataset('actions', (forward_ep, steps_per_ep, action_space), dtype='f')
  dtst = h5py.special_dtype(vlen=str)
  sim_data.create_dataset('state', (forward_ep, steps_per_ep, 15), dtype='f')
  
  
  envname = "LorlEnv-v0"
  savedir = f"logs/datagen_labeled_{envname}_EP{forward_ep}_STEPS{steps_per_ep}"
  if not os.path.exists(savedir):
    os.makedirs(savedir)
  
  epnum = []
  final_ep_description = []
  env = gym.make(envname)
  ### Each episode
  for ep in range(forward_ep):
    t0 = time.time()
    im = env.reset()
    im = im[0]
    st0 = env.sim.data.qpos[:].copy()
    
    step = 0
    done = False
    images = [(255 * im[:,:,:3]).astype(np.uint8)] # For logging
    while not done:
      # Random action
      action = env.action_space.sample()
      st = env.sim.data.qpos[:].copy()
      
      f['sim']['ims'][ep, step] = im
      f['sim']['actions'][ep, step] = action
      f['sim']['state'][ep, step] = st
      
      step += 1
      im, r, done, _ = env.step(action)
      images.append((255 * im[:,:,:3]).astype(np.uint8)) # For logging
      
    ### Generating syntethic language description
    d_d, d_f =[], []
    d_w = get_motion_desc_sawyer(st0[9:11], st[9:11], "white mug") # White mug motion description
    d_b = get_motion_desc_sawyer(st0[11:13], st[11:13], "black mug") # Black mug motion description
    # Drawer motion description
    if st[14] - st0[14] > 0.02:
      d_d = ["close drawer "]
    elif st[14] - st0[14] < -0.02:
      d_d = ["open drawer "]
    # Faucet motion description
    if st[13] - st0[13] > np.pi/10:
      d_f = ["turn faucet left "]
    elif st[13] - st0[13] < -np.pi/10:
      d_f = ["turn faucet right "]
    all_d = d_w + d_b  + d_d + d_f
    ## Shuffle and combine motion descriptions to form episode description.
    random.shuffle(all_d)
    if len(all_d) > 0:
      final_d = "and ".join(all_d)
    else:
      final_d = "do nothing"
    
    imageio.mimsave(f'{savedir}/ep_{str(ep).zfill(8)}.gif', images)
      
    print(ep, final_d, time.time()- t0)
    epnum.append(ep)
    final_ep_description.append(final_d)

  ### Save data and labels
  f.flush()
  f.close()
  d = {"Episode Number": epnum, "Text Description": final_ep_description}
  df = pd.DataFrame(data=d)
  df.to_csv(FLAGS.savepath + "labels.csv", index=False)
  ### Summarize descriptions
  print(pd.Series(final_ep_description).value_counts())

if __name__ == '__main__':
  app.run(main)
