# %% setup
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
from functools import reduce

# %%


def test_PE_show():
  fig = plt.figure()
  ax: axes3d.Axes3D = fig.add_subplot(projection='3d')

  n = 100

  PEarr = unit.PEArray(14, 12)
  height = np.arange(0, 100, 100 / 12)
  width = np.arange(0, 100, 100 / 14)
  axes = []
  for w in width:
    for h in height:
      for depth in range(PEarr.spad):
        axes.append(ax.scatter(w, depth / 3, h, marker='s'))
  axes: List[List[List[art3d.Patch3DCollection]]] = np.array(
      axes).reshape(len(width), len(height), PEarr.spad)
  # axes[0][0][0].
  # axes.shape
  # axes[0][0]
  anime = animation.FuncAnimation(fig, run, data_gen, interval=10, init_func=init)
  plt.show()


# %% test_logicalPE_mapping_one_conv2d
model = alexnet(pretrained=False)
# nn.Conv2d
ifmap = torch.rand(1, 3, 224, 224)
modelstat = summary(model, ifmap)
conv2dlayers = [
    layer_info for layer_info in modelstat.summary_list if 'Conv2d' in layer_info.class_name]
conv1 = conv2dlayers[0]

logicpe_set = LogicalPEset.from_one_conv2d(
    ifmap, conv1.module.weight, conv1.output_size, 0, 0, 0, conv1.module.stride)

# %% mapping one layer in to multi PE set

layer_set = LogicalPEset.from_conv2d(
    ifmap, conv1.module.weight,
    conv1.output_size, conv1.module.stride)

# layer_pes.shape
# %% NOTE striping mining
MAX_PEarray_width = 14
e = 7
if layer_set.shape[-1] > MAX_PEarray_width:
  splited_sets = np.array_split(layer_set, int(np.round(layer_set.shape[-1] / e)), -1)
  len(splited_sets)  # 55分7，前面都是7，最后是7和6一组。
  # 自动分组
  groups = []
  group = []
  width = 0
  for sets in splited_sets:
    if (width + sets.shape[-1]) <= MAX_PEarray_width:
      group.append(sets)
      width += sets.shape[-1]
    if width >= MAX_PEarray_width or (width + sets.shape[-1]) > MAX_PEarray_width:
      shape_list = [item.shape for item in group]
      all_same_shape = all([shape_list[0] == s for s in shape_list[1:]])
      if all_same_shape:  # 如果是相同的shape，stack在一起方便操作
        groups.append(np.stack(group, 3))
      else:
        groups.append(group.copy())
      group.clear()
      width = 0


# %% NOTE step 1 根据spad大小调整单个PE的内buf融合
Weight_Spad = 224
Psum_Spad = 24
Fmap_Spad = 12
# NOTE q different ic of p different oc.
# p 是oc psum reuse， q 是ic reuse
S = 3  # 最小的kenrel width是3，必须是3的倍数
p = 16
q = 1

group = groups[0]

for group in groups:
  w_spad = group[0, 0, 0, 0, 0, 0].weights_data.shape[0]
  assert(p * q * S < Weight_Spad and p < Psum_Spad and q * S < Fmap_Spad) # 不会超出Pe内部的内存限制
  # 先做ic的融合
  if q > 0:
    group[0]
    pass

# %% NOTE step 2 根据PE set的大小调整多个PE set融合。

# %%
ia = InfoArray(logicpes.sets.shape, dtype=LogicalPE, buffer=logicpes.sets, info='fuck')


# %%
ia[0][0]
ia.info

# %%

# %%
