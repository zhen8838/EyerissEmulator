import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import List, Tuple, Union
import torch


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
  def __init__(self, ifmap_index: List[int], ifmap_data: NDArray, weights_index: List[int], weights_data: NDArray) -> None:
    self.ifmap_index = ifmap_index
    self.ifmap_data = ifmap_data
    self.weights_index = weights_index
    self.weights_data = weights_data

  def __repr__(self) -> str:
    def to_str_list(l):
      return [str(i) for i in l]
    return f'ifmap : [{",".join(to_str_list(self.ifmap_index))}]' + ' ' + f'weight : [{",".join(to_str_list(self.weights_index))}]'


class LogicalPEset(np.ndarray):

  def __new__(cls, ifmap: NDArray[np.float32], weight: NDArray[np.float32], out_shape: List[int], n: int, oc: int, ic: int, stride: Union[int, Tuple[int, int]]):
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
  def from_one_conv2d(ifmap: NDArray[np.float32], weight: NDArray[np.float32], out_shape: List[int], n: int, oc: int, ic: int, stride: Union[int, Tuple[int, int]]) -> NDArray[LogicalPE]:
    if isinstance(ifmap, torch.Tensor):
      ifmap = ifmap.numpy()
    if isinstance(weight, torch.Tensor):
      weight = weight.detach().numpy()
    msets = []
    for wh in range(weight.shape[2]):
      sets = []
      for oh in range(out_shape[2]):
        ih = stride * oh + wh if isinstance(stride, int) else stride[0] * oh + wh
        sets.append(
            LogicalPE((n, ic, ih),
                      ifmap[n, ic, ih]
                      if ih < ifmap.shape[2]
                      else np.ones(ifmap[n, ic, -1].shape),  # padding zero when in
                      (oc, ic, wh),
                      weight[oc, ic, wh]))
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
