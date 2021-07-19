from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import List, Tuple, Union, Dict
import torch
import torch.nn.functional as F


class PhysicsPE():
  def __init__(self, index: Tuple[int, int], spad_size=8, dtype: np.dtype = np.float32) -> None:
    self.spad: int = spad_size
    self.wbuf = np.zeros(spad_size, dtype)
    self.fbuf = np.zeros(spad_size, dtype)
    self.sbuf = np.zeros(1, dtype)
    self.noc = None
    self.clock = 0
    self.index = index

  def calc(self):
    self.sbuf[0] = np.sum(self.wbuf * self.fbuf)

  def clear(self):
    self.wbuf.fill(0)
    self.fbuf.fill(0)
    self.sbuf.fill(0)

  def __rand(self):
    self.wbuf = np.random.rand(*self.wbuf.shape).copy()
    self.fbuf = np.random.rand(*self.fbuf.shape).copy()
    self.sbuf = np.random.rand(*self.sbuf.shape).copy()

  def load_data(self):
    self.clock


class PhysicsPEArray():
  def __init__(self, width, height, spad_size=8, dtype: np.dtype = np.float32) -> None:
    self.spad = spad_size
    self.noc: List[List[PE]] = []
    for y in range(height):
      row = []
      for x in range(width):
        row.append(PhysicsPE((y, x), spad_size, dtype))
      self.noc.append(row)
    self.noc = np.array(self.noc)

  def __getitem__(self, *args):
    return self.noc.__getitem__(*args)

  def __setitem__(self, *args):
    return self.noc.__setitem__(*args)

  def calc(self):
    pass


class LogicalPE():
  def __init__(self, ifmap_index: List[int], ifmap_data: NDArray,
               weights_index: List[int], weights_data: NDArray,
               strides: int = (1, 1), result: NDArray = None) -> None:
    self.ifmap_index = ifmap_index
    self.ifmap_data = ifmap_data
    self.weights_index = weights_index
    self.weights_data = weights_data
    self.stride_h = strides[0]
    self.stride_w = strides[1]
    self.result: NDArray = result
    self.q = 1
    self.p = 1

  def __repr__(self) -> str:
    # def to_str_list(l):
    #   return [str(i) for i in l]
    # b, ic, ih = self.ifmap_index
    # oc, ic, wh = self.weights_index
    # oh = (ih - wh) // self.stride_h
    # NOTE 显示oh维护心智模型的一致性。
    # line_str = [f"{'weight':^12} {'ifmap':^12}"]
    # w_str = self.weights_index.__str__()
    # f_str = self.ifmap_index.__str__()
    # line_str.extend([f'{w:^12} {f:^12}' for w, f in zip(w_str.splitlines(), f_str.splitlines())])
    # return '\r' + '\n'.join(line_str)
    return f'<{self.weights_index},{self.ifmap_index}>'

  @property
  def p(self):
    return self._p

  @p.setter
  def p(self, value):
    self._p = value

  @property
  def q(self):
    return self._q

  @q.setter
  def q(self, value):
    self._q = value

  def conv1d(self) -> str:
    self.result = F.conv1d(torch.from_numpy(self.ifmap_data)[None, None, :], torch.from_numpy(
        self.weights_data)[None, None, :], stride=self.stride_w)[0, 0, :].numpy()

  def __eq__(self, o: object) -> bool:
    return (self.ifmap_index == o.ifmap_index) and (self.weights_index == o.weights_index)

  @classmethod
  def from_pack(cls, pes: List):
    assert pes.ndim == 1 and pes.shape[0] >= 1
    ifmap_index = np.stack([pe.ifmap_index for pe in pes])
    ifmap_data = np.stack([pe.ifmap_data for pe in pes])
    ifmap_data = ifmap_data.T.flatten()  # pack
    weights_index = np.stack([pe.weights_index for pe in pes])
    weights_data = np.stack([pe.weights_data for pe in pes])
    weights_data = weights_data.T.flatten()
    stride_h = list(set([pe.stride_h for pe in pes]))
    assert len(stride_h) == 1, "The packed pe must have same stirde"
    stride_w = list(set([pe.stride_w for pe in pes]))
    assert len(stride_h) == 1, "The packed pe must have same stirde"
    result = np.stack([pe.result for pe in pes])
    packed = cls(ifmap_index, ifmap_data, weights_index,
                 weights_data, stride_h + stride_w, result)
    packed.q = len(pes)
    return packed

  @classmethod
  def from_concat(cls, pes: List):
    assert pes.ndim == 1 and pes.shape[0] >= 1
    ifmap_index = np.concatenate([pe.ifmap_index for pe in pes])
    ifmap_data = np.concatenate([pe.ifmap_data for pe in pes])
    weights_index = np.concatenate([pe.weights_index for pe in pes])
    weights_data = np.concatenate([pe.weights_data for pe in pes])
    stride_h = list(set([pe.stride_h for pe in pes]))
    assert len(stride_h) == 1, "The packed pe must have same stirde"
    stride_w = list(set([pe.stride_w for pe in pes]))
    assert len(stride_h) == 1, "The packed pe must have same stirde"
    result = np.stack([pe.result for pe in pes])
    concated = cls(ifmap_index, ifmap_data, weights_index,
                   weights_data, stride_h + stride_w, result)
    concated.p = len(pes)
    return concated


class LogicalPEset(np.ndarray):
  settings: Dict[str, int] = None
  max_height: int = None
  max_width: int = None
  max_weight_spad: int = None
  max_psum_spad: int = None
  max_fmap_spad: int = None

  def __new__(subtype, data, **settings):
    if isinstance(data, LogicalPEset):
      return data
    if isinstance(data, np.ndarray):
      buffer = np.array(data, dtype=LogicalPE)
    else:
      buffer = np.array(data, dtype=LogicalPE, copy=True)
    obj = np.ndarray.__new__(subtype, buffer.shape, buffer.dtype, buffer)
    obj.settings = settings
    for k, v in settings.items():
      setattr(obj, k, v)
    return obj

  def __array_finalize__(self, obj):
    if obj is None:
      return

    if type(obj) is np.ndarray:
      raise NotImplementedError("Unsupport create object by view cast")

    if type(obj) is LogicalPEset:
      setattr(self, 'settings', getattr(obj, 'settings'))
      for k, v in getattr(obj, 'settings').items():
        setattr(self, k, v)

  @staticmethod
  def load_one_conv2d(ifmap: NDArray[np.float32], weight: NDArray[np.float32],
                      out_shape: List[int],
                      b: int, oc: int, ic: int,
                      strides: Tuple[int, int]) -> NDArray:
    if isinstance(ifmap, torch.Tensor):
      ifmap = ifmap.numpy()
    if isinstance(weight, torch.Tensor):
      weight = weight.detach().numpy()
    msets = []
    for wh in range(weight.shape[2]):
      sets = []
      for oh in range(out_shape[2]):
        ih = strides[0] * oh + wh
        sets.append(
            LogicalPE((b, ic, ih),
                      ifmap[b, ic, ih]
                      if ih < ifmap.shape[2]
                      else np.zeros(ifmap[b, ic, -1].shape),  # padding zero when in
                      (oc, ic, wh),
                      weight[oc, ic, wh],
                      strides))
      msets.append(sets)
    return np.array(msets, dtype=LogicalPE)

  def from_one_conv2d(self, ifmap: NDArray[np.float32], weight: NDArray[np.float32],
                      out_shape: List[int],
                      b: int, oc: int, ic: int,
                      strides: Tuple[int, int]) -> LogicalPEset:
    return LogicalPEset(LogicalPEset.load_one_conv2d(
        ifmap, weight, out_shape,
        b, oc, ic, strides),
        **self.settings)

  def from_conv2d(self, ifmap: NDArray[np.float32], weight: NDArray[np.float32], out_shape: List[int], strides: Union[int, Tuple[int, int]]) -> LogicalPEset:
    layer_pes = []
    for b in range(ifmap.shape[0]):
      for oc in range(weight.shape[0]):
        for ic in range(weight.shape[1]):
          layer_pes.append(LogicalPEset.load_one_conv2d(
              ifmap, weight, out_shape,
              b, oc, ic, strides))
    layer_pes = np.array(layer_pes, dtype=LogicalPE)
    layer_pes = layer_pes.reshape(ifmap.shape[0], weight.shape[0],
                                  weight.shape[1], *layer_pes.shape[1:])
    return LogicalPEset(layer_pes, **self.settings)

  def spad_ic_fusion(self, q: int) -> NDArray:
    # q个一组合并in channels，weights reuse，也就是in channel pack，先把ic移动到最后维度然后进行concat
    B, OC, IC, KH, OH = self.shape
    split_factor = int(IC / q)
    ic_splits = np.split(self, split_factor, axis=2)
    new_layer_set = []
    for ic_split in ic_splits:
      ic_split_q = np.moveaxis(ic_split, 2, -1)
      ic_split_q.shape
      pes = ic_split_q[0, 0, 0, 0]
      packed_pes = np.apply_along_axis(lambda pes: LogicalPE.from_pack(pes), -1, ic_split_q)
      new_layer_set.append(packed_pes)
    # 再合并到ic维度
    return LogicalPEset(np.stack(new_layer_set, 2), **self.settings)

  def spad_oc_fusion(self, p: int) -> NDArray:
    # p 合并多个out channels，此时多个psum不可融合，所以需要concat来做。
    B, OC, IC, KH, OH = self.shape
    split_factor = int(OC / p)
    oc_splits = np.split(self, split_factor, axis=1)
    new_layer_set = []
    for oc_split in oc_splits:
      oc_split_p = np.moveaxis(oc_split, 1, -1)
      # pes = oc_split_p[0, 0, 0, 0]
      concated_pes = np.apply_along_axis(lambda pes: LogicalPE.from_concat(pes), -1, oc_split_p)
      new_layer_set.append(concated_pes)
    # 再合并到oc维度
    return LogicalPEset(np.stack(new_layer_set, 1), **self.settings)

  def strip_mining(self, width: int) -> List[LogicalPEset]:
    B, OC, IC, KH, OH = self.shape
    if KH > self.max_height:
      raise ValueError(f"the eyeriss not support kernel size {KH} > {self.max_height}")
    if OH > self.max_width:
      split_factor = int(np.ceil(OH / width))
      splited_sets = np.array_split(self, split_factor, -1)
    return splited_sets

  def spatail_fusion_common(self, param: int, axis: int):
    shape = list(self.shape)
    B, OC, IC, KH, OH = shape
    splited_shape = shape.copy()
    splited_shape.insert(axis + 1, param)
    splited_shape[axis] = splited_shape[axis] // param
    # NOTE 虽然可以任意映射,但是必须满足w或者h放下concat
    if shape[axis] > param and ((OH * param <= self.max_width) or (KH * param <= self.max_height)):
      layer_set = self.reshape(*splited_shape)
      if OH * param <= self.max_width:
        layer_set = np.moveaxis(layer_set, axis, -1)
        merged_shape = list(layer_set.shape)
        merged_shape[-2] *= merged_shape[-1]
        merged_shape.pop(-1)
      elif KH * param <= self.max_height:
        layer_set = np.moveaxis(layer_set, axis, -2)
        merged_shape = list(layer_set.shape)
        merged_shape[-3] *= merged_shape[-2]
        merged_shape.pop(-2)
      layer_set = layer_set.reshape(*merged_shape)
    else:
      raise ValueError(f"the param = {param} can't merge on h = {KH} w = {OH}")
    return layer_set

  def spatail_ic_fusion(self, r: int):
    return self.spatail_fusion_common(r, axis=2)

  def spatail_oc_fusion(self, t: int):
    return self.spatail_fusion_common(t, axis=1)

  def spad_fusion(self, p=16, q=1, S=3) -> LogicalPEset:
    assert(p * q * S < self.max_weight_spad and p <
           self.max_psum_spad and q * S < self.max_fmap_spad)
    new_sets = self
    if q > 1:
      new_sets = new_sets.spad_ic_fusion(q)
    if p > 1:
      new_sets = new_sets.spad_oc_fusion(p)
    return new_sets

  def spatial_fusion(self, t=1, r=1) -> LogicalPEset:
    new_sets = self
    if t > 1:
      new_sets = new_sets.spatail_oc_fusion(t)
    if r > 1:
      new_sets = new_sets.spatail_ic_fusion(r)
    return new_sets
