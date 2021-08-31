from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent


class LangEncoder(nn.Module):
  """Language Encoder Module (Distilbert)"""
  def __init__(self, nq = None, finetune = False, aug=False, scratch=False):
    super().__init__()
    self.finetune = finetune
    self.scratch = scratch # train from scratch vs load weights
    self.aug = aug
    self.device = "cuda"
    self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    if not self.scratch:
      self.model = AutoModel.from_pretrained("distilbert-base-uncased").to('cuda')
    else:
      self.model = AutoModel.from_config(config = AutoConfig.from_pretrained("distilbert-base-uncased")).to('cuda')
    self.lang_size = 768
      
  def forward(self, langs):
    try:
      langs = langs.tolist()
    except:
      pass
    
    if self.finetune:
      encoded_input = self.tokenizer(langs, return_tensors='pt', padding=True)
      input_ids = encoded_input['input_ids'].to(self.device)
      attention_mask = encoded_input['attention_mask'].to(self.device)
      lang_embedding = self.model(input_ids, attention_mask=attention_mask)[0][:, -1]
    else:
      with torch.no_grad():
        encoded_input = self.tokenizer(langs, return_tensors='pt', padding=True)
        input_ids = encoded_input['input_ids'].to(self.device)
        attention_mask = encoded_input['attention_mask'].to(self.device)
        lang_embedding = self.model(input_ids, attention_mask=attention_mask)[0][:, -1]
    if self.aug:
      lang_embedding +=  torch.distributions.Uniform(-0.1, 0.1).sample(lang_embedding.shape).cuda()
    return lang_embedding

class Discriminator(nn.Module):
  __constants__ = ['min_std_dev']

  def __init__(self, hidden_size, activation_function='relu', finetune = False, robot=False, langaug=False, scratch=False):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.sigm = nn.Sigmoid()
    self.dropout = nn.Dropout(0.2)
    self.robot = robot
    self.senc = LangEncoder(finetune=finetune, aug=langaug, scratch=scratch)
    lang_size = self.senc.lang_size
    
    if robot:
      self.enc = Encoder(hidden_size, ch = 24, robot=self.robot)
    else:
      self.enc = Encoder(hidden_size, ch = 6)
    self.fc1 = nn.Linear(lang_size + hidden_size , hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.fc4 = nn.Linear(hidden_size, 1)
    
  def forward(self, init_obs, goal_obs, lang_goal):
    lang_emb = self.senc(lang_goal)
    im_enc = self.enc(torch.cat([init_obs, goal_obs], dim=1))
    h = torch.cat([im_enc, lang_emb], dim=-1)
    h = self.dropout(self.act_fn(self.fc1(h)))
    h = self.dropout(self.act_fn(self.fc2(h)))
    h = self.dropout(self.act_fn(self.fc3(h)))
    h = self.sigm(self.fc4(h))
    return h
  

## Image Encoder
class Encoder(nn.Module):
  __constants__ = ['embedding_size']
  
  def __init__(self, hidden_size, activation_function='relu', ch=3, robot=False):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.softmax = nn.Softmax(dim=2)
    self.sigmoid = nn.Sigmoid()
    self.robot = robot
    if self.robot:
      g = 4
    else:
      g = 1
    self.conv1 = nn.Conv2d(ch, 32, 4, stride=2, padding=1, groups=g) #3
    self.conv1_2 = nn.Conv2d(32, 32, 4, stride=1, padding=1, groups=g)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1, groups=g)
    self.conv2_2 = nn.Conv2d(64, 64, 4, stride=1, padding=1, groups=g)
    self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1, groups=g)
    self.conv3_2 = nn.Conv2d(128, 128, 4, stride=1, padding=1, groups=g)
    self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=1, groups=g)
    self.conv4_2 = nn.Conv2d(256, 256, 4, stride=1, padding=1, groups=g)
    
    self.fc1 = nn.Linear(1024, 512)
    self.fc1_2 = nn.Linear(512, 512)
    self.fc1_3 = nn.Linear(512, 512)
    self.fc1_4 = nn.Linear(512, 512)
    self.fc2 = nn.Linear(512, hidden_size)

  def forward(self, observation):
    if self.robot:
      observation = torch.cat([
        observation[:, :3], observation[:,12:15], observation[:, 3:6], observation[:, 15:18],
        observation[:, 6:9], observation[:,18:21], observation[:, 9:12], observation[:, 21:],
        ], 1)
    hidden = self.act_fn(self.conv1(observation))
    hidden = self.act_fn(self.conv1_2(hidden))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv2_2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    hidden = self.act_fn(self.conv3_2(hidden))
    hidden = self.act_fn(self.conv4(hidden))
    hidden = self.act_fn(self.conv4_2(hidden))
    hidden = hidden.reshape(observation.shape[0], -1)
    
    hidden = self.act_fn(self.fc1(hidden))
    hidden = self.act_fn(self.fc1_2(hidden))
    hidden = self.act_fn(self.fc1_3(hidden))
    hidden = self.act_fn(self.fc1_4(hidden))
    hidden = self.fc2(hidden)
    return hidden
  
