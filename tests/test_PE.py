# %%
from src.PEUnit import LogicalPE, LogicalPEset
import pytest
import matplotlib.pyplot as plt
import matplotlib.markers
from mpl_toolkits.mplot3d import axes3d, art3d
import numpy as np
from typing import List
import matplotlib.animation as animation
from torchvision.models import alexnet
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

# %%

settings = dict(max_height=12,
                max_width=14,
                max_weight_spad=224,
                max_psum_spad=24,
                max_fmap_spad=12)


def load_data():
  model = alexnet(pretrained=False)
  # nn.Conv2d
  ifmap = torch.rand(1, 3, 224, 224)
  modelstat = summary(model, ifmap)
  conv2dlayers = [
      layer_info for layer_info in modelstat.summary_list if 'Conv2d' in layer_info.class_name]
  conv1 = conv2dlayers[0]
  weights = conv1.module.weight.detach().numpy()
  strides = conv1.module.stride
  out_shape = conv1.output_size
  padding = conv1.module.padding
  ifmap = F.pad(ifmap, padding + padding).numpy()
  return ifmap, weights, out_shape, strides

# %%


def test_mapping_one_conv2d():
  ifmap, weights, out_shape, strides = load_data()
  peset = LogicalPEset(None, **settings)
  one_conv2d_set = peset.from_one_conv2d(
      ifmap, weights, out_shape, 0, 0, 0, strides)
  one_conv2d_set.shape
  peset = one_conv2d_set[:, :7]
  ifmap_index = set([pe.ifmap_index for pe in peset.ravel()])

  len(ifmap_index) * 227 * 2 / 1024  # 单纯计算11x7 x2个非共用ifmap的是15.51kb. 肯定还有个16kb 和 2在哪里被抵消了
  return one_conv2d_set, ifmap, weights, strides

# %%


def test_mapping_conv2d() -> LogicalPEset:
  ifmap, weights, out_shape, strides = load_data()
  B = ifmap.shape[0]
  OC, IC, KH = weights.shape[0:3]
  OH = out_shape[2]
  peset = LogicalPEset(None, **settings)
  peset = peset.from_conv2d(ifmap, weights, out_shape, strides)
  peset[0, 1, 0, 0, 0]
  peset[0, 2, 0, 0, 0]
  # peset[0, 2, 0, 0, 0]
  # peset[0, 3, 0, 0, 0]
  # peset[0, 5, 0, 0, 0]
  # peset[0, 16, 0, 0, 0]
  return peset


# %%
def test_mapping_q_and_p() -> LogicalPEset:
  peset = test_mapping_conv2d()
  type(peset)
  newpeset = peset.spad_fusion(p=16, q=1, S=3)
  newpeset[0, 0, 0, 0, 0]
  print(newpeset.shape)
  type(newpeset)
  newpeset.__dict__
  return newpeset


# %%


def test_strip_mining() -> List[LogicalPEset]:
  # 如果OH大于pe array的width,那么就得截断
  peset = test_mapping_q_and_p()

  pesets = peset.strip_mining(7)
  return pesets
# %%


def test_mapping_t_and_r():
  pesets = test_strip_mining()
  tt = np.concatenate([pesets[0][:, :2, :, :, :], pesets[0][:, 2:, :, :, :]], -1)
  ss = set([tuple(t.ifmap_index) for t in tt[0, 0, 0].ravel()])
  # len(ss)
  35 * 228 * 16 / 1024
  # tt[0, 0, 0, 0, 0]
  # tt[0, 0, 0, 0, 7]
  pesets[0][0, 0, 0, 0, 0]
  t = pesets[0].spatial_fusion(t=2, r=1)
  t[0, 0, 0, 0, 0]
  new_pesets = [peset.spatial_fusion(t=2, r=1) for peset in pesets]
  print(new_pesets[0].shape)
  peset = new_pesets[0]
  return new_pesets

# %%


def test_peset_calc():
  pass
