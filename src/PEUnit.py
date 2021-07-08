import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import List, Tuple, Union
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
    line_str = [f"{'weight':^12} {'ifmap':^12}"]
    w_str = self.weights_index.__str__()
    f_str = self.ifmap_index.__str__()
    line_str.extend([f'{w:^12} {f:^12}' for w, f in zip(w_str.splitlines(), f_str.splitlines())])
    return '\r' + '\n'.join(line_str)

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

  def __new__(cls, ifmap: NDArray[np.float32], weight: NDArray[np.float32],
              out_shape: List[int], n: int, oc: int, ic: int,
              stride: Union[int, Tuple[int, int]]):
    # 根据当前类型构建一个array obj, 调用__array_finalize__设定新的方法
    dtype = LogicalPE
    buffer = None
    offset = 0
    strides = None
    order = None
    buffer = LogicalPEset.from_one_conv2d(ifmap, weight, out_shape, n, oc, ic, stride)
    obj = super().__new__(cls, buffer.shape, dtype,
                          buffer, offset, strides, order)
    obj.n = n
    obj.oc = oc
    obj.ic = ic
    return obj

  def __array_finalize__(self, obj):
    if obj is None:
      return
    self.info = getattr(obj, 'info', None)
    self.n = getattr(obj, 'n', None)
    self.oc = getattr(obj, 'oc', None)
    self.ic = getattr(obj, 'ic', None)

  @staticmethod
  def from_one_conv2d(ifmap: NDArray[np.float32], weight: NDArray[np.float32],
                      out_shape: List[int],
                      n: int, oc: int, ic: int,
                      strides: Tuple[int, int]) -> NDArray[LogicalPE]:
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
            LogicalPE((n, ic, ih),
                      ifmap[n, ic, ih]
                      if ih < ifmap.shape[2]
                      else np.zeros(ifmap[n, ic, -1].shape),  # padding zero when in
                      (oc, ic, wh),
                      weight[oc, ic, wh],
                      strides))
      msets.append(sets)
    return np.array(msets)

  @staticmethod
  def from_conv2d(ifmap: NDArray[np.float32], weight: NDArray[np.float32], out_shape: List[int], stride: Union[int, Tuple[int, int]]) -> NDArray[LogicalPE]:
    layer_pes = []
    for b in range(ifmap.shape[0]):
      for oc in range(weight.shape[0]):
        for ic in range(weight.shape[1]):
          layer_pes.append(LogicalPEset.from_one_conv2d(
              ifmap, weight, out_shape,
              b, oc, ic, stride))
    layer_pes = np.array(layer_pes, dtype=LogicalPE)
    layer_pes = layer_pes.reshape(
        ifmap.shape[0], weight.shape[0], weight.shape[1], *layer_pes.shape[1:])
    return layer_pes

  @staticmethod
  def spad_ic_fusion(layer_set: NDArray, q: int) -> NDArray:
    # q个一组合并in channels，weights reuse，也就是in channel pack，先把ic移动到最后维度然后进行concat
    B, OC, IC, KH, OH = layer_set.shape
    split_factor = int(IC / q)
    ic_splits = np.split(layer_set, split_factor, axis=2)
    new_layer_set = []
    for ic_split in ic_splits:
      ic_split_q = np.moveaxis(ic_split, 2, -1)
      ic_split_q.shape
      pes = ic_split_q[0, 0, 0, 0]
      packed_pes = np.apply_along_axis(lambda pes: LogicalPE.from_pack(pes), -1, ic_split_q)
      new_layer_set.append(packed_pes)
    return np.stack(new_layer_set, 2)  # 再合并到ic维度

  @staticmethod
  def spad_oc_fusion(layer_set: NDArray, q: int) -> NDArray:
    layer_set = LogicalPEset.spad_oc_fusion(layer_set, p)
    B, OC, IC, KH, OH = layer_set.shape
    split_factor = int(OC / p)
    oc_splits = np.split(layer_set, split_factor, axis=1)
    new_layer_set = []
    for oc_split in oc_splits:
      oc_split.shape
      oc_split_p = np.moveaxis(oc_split, 1, -1)
      pes = oc_split_p[0, 0, 0, 0]
      concated_pes = np.apply_along_axis(lambda pes: LogicalPE.from_concat(pes), -1, oc_split_p)
      new_layer_set.append(concated_pes)
    return np.stack(layer_set, 1)  # 再合并到oc维度

  @staticmethod
  def spad_fusion(layer_set: NDArray, p=16, q=1, S=3, Weight_Spad=224, Psum_Spad=24, Fmap_Spad=12):
    assert(p * q * S < Weight_Spad and p < Psum_Spad and q * S < Fmap_Spad)
    if q > 1:
      layer_set = LogicalPEset.spad_ic_fusion(layer_set, q)
    if p > 1:
      layer_set = LogicalPEset.spad_oc_fusion(layer_set, q)
    return layer_set