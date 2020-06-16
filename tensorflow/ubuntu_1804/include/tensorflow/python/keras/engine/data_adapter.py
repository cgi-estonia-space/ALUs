# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Adapter module that convert different input data objects into tf.dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import contextlib
import functools
import itertools
import math
import random

import numpy as np
import six

from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.experimental.ops import distribute_options
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework.ops import composite_tensor
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

try:
  from scipy import sparse as scipy_sparse  # pylint: disable=g-import-not-at-top
except ImportError:
  scipy_sparse = None

try:
  import pandas as pd  # pylint: disable=g-import-not-at-top
except ImportError:
  pd = None

try:
  # In Python2 unicode is a scalar type
  scalar_types = (float, int, str, unicode)
except NameError:
  # In Python3 unicode is not present, it always uses string
  scalar_types = (float, int, str)


@six.add_metaclass(abc.ABCMeta)
class DataAdapter(object):
  """Base class for input data adapter.

  In TF 2.0, tf.data is the preferred API for user to feed in data. In order
  to simplify the training code path, all the input data object will be
  converted to `tf.data.Dataset` if possible.

  Note that since this class is mainly targeted for TF 2.0, it might have a lot
  of assumptions under the hood, eg eager context by default, distribution
  strategy, etc. In the meantime, some legacy feature support might be dropped,
  eg, Iterator from dataset API in v1, etc.

  The sample usage of this class is like:

  ```
  x = tf.data.Dataset.range(100)
  adapter_cls = [NumpyArrayDataAdapter, ..., DatasetAdapter]
  applicable_adapters = [cls for cls in adapter_cls if cls.can_handle(x)]
  if len(applicable_adapters) != 1:
    raise ValueError("Expect only one adapter class to handle the input")

  dataset = applicable_adapters[0](x).get_dataset()
  for data in dataset:
    # training
  ```
  """

  @staticmethod
  def can_handle(x, y=None):
    """Whether the current DataAdapter could handle the input x and y.

    Structure wise, x and y can be single object, or list of objects if there
    multiple input/output, or dictionary of objects when the intput/output are
    named.

    Args:
      x: input features.
      y: target labels. Note that y could be None in the case of prediction.

    Returns:
      boolean
    """
    raise NotImplementedError

  @abc.abstractmethod
  def __init__(self, x, y=None, **kwargs):
    """Create a DataAdapter based on data inputs.

    The caller must make sure to call `can_handle()` first before invoking this
    method. Provide unsupported data type will result into unexpected behavior.

    Args:
      x: input features.
      y: target labels. Note that y could be None in the case of prediction.
      **kwargs: Other keyword arguments for DataAdapter during the construction
        of the tf.dataset.Dataset. For example:
        - Numpy data might have `sample_weights` which will be used for
          weighting the loss function during training.
        - Numpy data might need to have `batch_size` parameter when constructing
          the dataset and iterator.
        - Certain input might need to be distribution strategy aware. When
          `distribution_strategy` is passed, the created dataset need to respect
          the strategy.
        DataAdapter might choose to ignore any keyword argument if it doesn't
        use it, or raise exception if any required argument is not provide.
    """
    if not self.can_handle(x, y):
      raise ValueError("{} Cannot handle input {}, {}".format(
          self.__class__, x, y))

  @abc.abstractmethod
  def get_dataset(self):
    """Get a dataset instance for the current DataAdapter.

    Note that the dataset returned does not repeat for epoch, so caller might
    need to create new iterator for the same dataset at the beginning of the
    epoch. This behavior might change in future.

    Returns:
      An tf.dataset.Dataset. Caller might use the dataset in different
      context, eg iter(dataset) in eager to get the value directly, or in graph
      mode, provide the iterator tensor to Keras model function.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def get_size(self):
    """Return the size (number of batches) for the dataset created.

    For certain type of the data input, the number of batches is known, eg for
    Numpy data, the size is same as (number_of_element / batch_size). Whereas
    for dataset or python generator, the size is unknown since it may or may not
    have a end state.

    Returns:
      int, the number of batches for the dataset, or None if it is unknown. The
      caller could use this to control the loop of training, show progress bar,
      or handle unexpected StopIteration error.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def batch_size(self):
    """Return the batch size of the dataset created.

    For certain type of the data input, the batch size is known, and even
    required, like numpy array. Where as for dataset, the batch is unknown
    unless we take a peek.

    Returns:
      int, the batch size of the dataset, or None if it is unknown.
    """
    raise NotImplementedError

  def representative_batch_size(self):
    """Return a representative size for batches in the dataset.

    This is not guaranteed to be the batch size for all batches in the
    dataset. It just needs to be a rough approximation for batch sizes in
    the dataset.

    Returns:
      int, a representative size for batches found in the dataset,
      or None if it is unknown.
    """
    return self.batch_size()

  @abc.abstractmethod
  def has_partial_batch(self):
    """Whether the dataset has partial batch at the end."""
    raise NotImplementedError

  @abc.abstractmethod
  def partial_batch_size(self):
    """The size of the final partial batch for dataset.

    Will return None if has_partial_batch is False or batch_size is None.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def should_recreate_iterator(self):
    """Returns whether a new iterator should be created every epoch."""
    raise NotImplementedError


class TensorLikeDataAdapter(DataAdapter):
  """Adapter that handles Tensor-like objects, e.g. EagerTensor and NumPy."""

  @staticmethod
  def can_handle(x, y=None):
    # TODO(kaftan): Check performance implications of using a flatten
    #  here for other types of inputs.
    flat_inputs = nest.flatten(x)
    if y is not None:
      flat_inputs += nest.flatten(y)

    tensor_types = (ops.Tensor, np.ndarray)
    if pd:
      tensor_types = (ops.Tensor, np.ndarray, pd.Series, pd.DataFrame)

    def _is_tensor(v):
      if isinstance(v, tensor_types):
        return True
      return False

    return all(_is_tensor(v) for v in flat_inputs)

  def __init__(self,
               x,
               y=None,
               sample_weights=None,
               sample_weight_modes=None,
               batch_size=None,
               epochs=1,
               steps=None,
               shuffle=False,
               **kwargs):
    super(TensorLikeDataAdapter, self).__init__(x, y, **kwargs)
    x = _process_numpy_inputs(x)
    y = _process_numpy_inputs(y)
    sample_weights = _process_numpy_inputs(sample_weights)
    sample_weight_modes = broadcast_sample_weight_modes(
        sample_weights, sample_weight_modes)

    # If sample_weights are not specified for an output use 1.0 as weights.
    (sample_weights, any_sample_weight, _
    ) = training_utils.handle_partial_sample_weights(
        y, sample_weights, sample_weight_modes, check_all_flat=True)

    if y is not None and any_sample_weight:
      inputs = (x, y, sample_weights)
    elif y is not None:
      # Sample weight is only needed for training, so if y is None, then
      # sample_weight is ignored.
      inputs = (x, y)
    else:
      inputs = (x,)

    num_samples = set(int(i.shape[0]) for i in nest.flatten(inputs))
    if len(num_samples) > 1:
      msg = "Data cardinality is ambiguous:\n"
      for label, data in zip(["x", "y", "sample_weight"], inputs):
        msg += "  {} sizes: {}\n".format(
            label, ", ".join(str(i.shape[0]) for i in nest.flatten(data)))
      msg += "Please provide data which shares the same first dimension."
      raise ValueError(msg)
    num_samples = num_samples.pop()

    # If batch_size is not passed but steps is, calculate from the input data.
    if steps and not batch_size:
      batch_size = int(math.ceil(num_samples / steps))

    if not batch_size:
      raise ValueError(
          "`batch_size` or `steps` is required for `Tensor` or `NumPy`"
          " input data.")

    self._size = int(math.ceil(num_samples / batch_size))
    self._batch_size = batch_size

    num_full_batches = int(num_samples // batch_size)
    self._partial_batch_size = num_samples % batch_size

    if isinstance(shuffle, str):
      shuffle = shuffle.lower()

    self._shuffle = shuffle
    # Vectorized version of shuffle.
    # This is a performance improvement over using `from_tensor_slices`.
    # The indices of the data are shuffled and batched, and these indices
    # are then zipped with the data and used to extract a batch of the data
    # at each step. The performance improvements here come from:
    # 1. vectorized batch using gather
    # 2. parallelized map
    # 3. pipelined permutation generation
    # 4. optimized permutation batching
    # 5. disabled static optimizations

    indices_dataset = dataset_ops.DatasetV2.range(1)
    if shuffle != "batch":
      indices_dataset = indices_dataset.repeat(epochs)

    def permutation(_):
      # It turns out to be more performant to make a new set of indices rather
      # than reusing the same range Tensor. (presumably because of buffer
      # forwarding.)
      indices = math_ops.range(num_samples, dtype=dtypes.int64)
      if shuffle and shuffle != "batch":
        indices = random_ops.random_shuffle(indices)
      return indices

    # We prefetch a single element. Computing large permutations can take quite
    # a while so we don't want to wait for prefetching over an epoch boundary to
    # trigger the next permutation. On the other hand, too many simultaneous
    # shuffles can contend on a hardware level and degrade all performance.
    indices_dataset = indices_dataset.map(permutation).prefetch(1)

    def slice_batch_indices(indices):
      """Convert a Tensor of indices into a dataset of batched indices.

      This step can be accomplished in several ways. The most natural is to
      slice the Tensor in a Dataset map. (With a condition on the upper index to
      handle the partial batch.) However it turns out that coercing the Tensor
      into a shape which is divisible by the batch size (and handling the last
      partial batch separately) allows for a much more favorable memory access
      pattern and improved performance.

      Args:
        indices: Tensor which determines the data order for an entire epoch.

      Returns:
        A Dataset of batched indices.
      """
      num_in_full_batch = num_full_batches * batch_size
      first_k_indices = array_ops.slice(indices, [0], [num_in_full_batch])
      first_k_indices = array_ops.reshape(
          first_k_indices, [num_full_batches, batch_size])

      flat_dataset = dataset_ops.DatasetV2.from_tensor_slices(first_k_indices)
      if self._partial_batch_size:
        index_remainder = dataset_ops.DatasetV2.from_tensors(array_ops.slice(
            indices, [num_in_full_batch], [self._partial_batch_size]))
        flat_dataset = flat_dataset.concatenate(index_remainder)

      if shuffle == "batch":
        # 1024 is a magic constant that has not been properly evaluated
        flat_dataset = flat_dataset.shuffle(1024).repeat(epochs)
      return flat_dataset

    indices_dataset = indices_dataset.flat_map(slice_batch_indices)

    dataset = self.slice_inputs(indices_dataset, inputs)

    if shuffle == "batch":
      def shuffle_batch(*batch):
        return nest.map_structure(random_ops.random_shuffle, batch)
      dataset = dataset.map(shuffle_batch)

    self._dataset = dataset

  def slice_inputs(self, indices_dataset, inputs):
    """Slice inputs into a Dataset of batches.

    Given a Dataset of batch indices and the unsliced inputs,
    this step slices the inputs in a parallelized fashion
    and produces a dataset of input batches.

    Args:
      indices_dataset: A Dataset of batched indices
      inputs: A python data structure that contains the inputs, targets,
        and possibly sample weights.

    Returns:
      A Dataset of input batches matching the batch indices.
    """
    dataset = dataset_ops.DatasetV2.zip((
        indices_dataset,
        dataset_ops.DatasetV2.from_tensors(inputs).repeat()
    ))

    def grab_batch(i, data):
      return nest.map_structure(lambda d: array_ops.gather(d, i, axis=0), data)

    dataset = dataset.map(
        grab_batch, num_parallel_calls=dataset_ops.AUTOTUNE)

    # Default optimizations are disabled to avoid the overhead of (unnecessary)
    # input pipeline graph serialization and deserialization
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    if self._shuffle:
      # See b/141490660 for more details.
      options.experimental_external_state_policy = (
          distribute_options.ExternalStatePolicy.IGNORE)
    dataset = dataset.with_options(options)
    return dataset

  def get_dataset(self):
    return self._dataset

  def get_size(self):
    return self._size

  def batch_size(self):
    return self._batch_size

  def has_partial_batch(self):
    return self._partial_batch_size > 0

  def partial_batch_size(self):
    return self._partial_batch_size or None

  def should_recreate_iterator(self):
    # An infinite dataset is always created here.
    return False


class GenericArrayLikeDataAdapter(TensorLikeDataAdapter):
  """Adapter that handles array-like data without forcing it into memory.

  As an example, this adapter handles `keras.utils.HDF5Matrix` which holds
  datasets that may be too big to fully fit into memory.

  Specifically, this adapter handles any Python class which implements:
  `__get_item__`, `__len__`, `shape`, and `dtype` with the same meanings
  as Numpy, but it ignores any case where all the inputs are Tensors or Numpy
  arrays (because that case is handled by the base TensorLikeDataAdapter).

  It ignores scipy sparse matrices and Composite Tensors because those are
  handled by the CompositeTensorDataAdapter.

  It also does not handle lists/tuples of scalars, because those are handled
  by the ListsOfScalarsDataAdapter.
  """

  @staticmethod
  def can_handle(x, y=None):
    flat_inputs = nest.flatten(x)
    if y is not None:
      flat_inputs += nest.flatten(y)

    def _is_array_like(v):
      """Return True if v is a Tensor, array, or is array-like."""
      return (
          hasattr(v, "__getitem__") and
          hasattr(v, "shape") and
          hasattr(v, "dtype") and
          hasattr(v, "__len__")
      )

    if (not TensorLikeDataAdapter.can_handle(x, y) and
        not CompositeTensorDataAdapter.can_handle(x, y)):
      return all(_is_array_like(v) for v in flat_inputs)
    else:
      return False

  def __init__(self, *args, **kwargs):
    logging.warn(
        "Keras is training/fitting/evaluating on array-like data. Keras may "
        "not be optimized for this format, so if your input data format is "
        "supported by TensorFlow I/O (https://github.com/tensorflow/io) we "
        "recommend using that to load a Dataset instead.")

    super(GenericArrayLikeDataAdapter, self).__init__(*args, **kwargs)

  def slice_inputs(self, indices_dataset, inputs):
    """Slice inputs into a Dataset of batches.

    Given a Dataset of batch indices and the unsliced inputs,
    this step slices the inputs in a parallelized fashion
    and produces a dataset of input batches.

    Args:
      indices_dataset: A Dataset of batched indices
      inputs: A python data structure that contains the inputs, targets,
        and possibly sample weights.

    Returns:
      A Dataset of input batches matching the batch indices.
    """
    flat_inputs = nest.flatten(inputs)
    def dynamic_shape_like(t):
      shape = list(t.shape)
      shape[0] = None
      return tuple(shape)

    flat_dtypes = [inp.dtype for inp in flat_inputs]
    contiguous = True
    if self._shuffle and self._shuffle != "batch":
      contiguous = False

    def grab_batch(indices):
      """Grab a batch of data from the inputs."""
      # This uses a py_function to avoid converting the array-like
      # into a Tensor before slicing it, because converting the array-like
      # to a Tensor may force it into memory..
      def py_method(ind):
        def slice_array(data):
          return training_utils.slice_arrays(data, ind.numpy(),
                                             contiguous=contiguous)
        return [slice_array(inp) for inp in flat_inputs]

      flat_out = script_ops.eager_py_func(py_method, [indices], flat_dtypes)
      for v, original_inp in zip(flat_out, flat_inputs):
        v.set_shape(dynamic_shape_like(original_inp))
      return nest.pack_sequence_as(inputs, flat_out)

    dataset = indices_dataset.map(
        grab_batch, num_parallel_calls=dataset_ops.AUTOTUNE)

    return dataset


class CompositeTensorDataAdapter(DataAdapter):
  """Adapter that handles composite tensor."""

  @staticmethod
  def can_handle(x, y=None):
    flat_inputs = nest.flatten(x)
    if y is not None:
      flat_inputs += nest.flatten(y)

    def _is_composite(v):
      # Dataset inherits from CompositeTensor but shouldn't be handled here.
      if (isinstance(v, composite_tensor.CompositeTensor) and
          not isinstance(v, dataset_ops.DatasetV2)):
        return True
      # Support Scipy sparse tensors if scipy is installed
      if scipy_sparse is not None and scipy_sparse.issparse(v):
        return True
      return False

    def _is_tensor_or_composite(v):
      if isinstance(v, (ops.Tensor, np.ndarray)):
        return True
      return _is_composite(v)

    return (any(_is_composite(v) for v in flat_inputs) and
            all(_is_tensor_or_composite(v) for v in flat_inputs))

  def __init__(self,
               x,
               y=None,
               sample_weights=None,
               sample_weight_modes=None,
               batch_size=None,
               steps=None,
               shuffle=False,
               **kwargs):
    super(CompositeTensorDataAdapter, self).__init__(x, y, **kwargs)
    x = _process_numpy_inputs(x)
    y = _process_numpy_inputs(y)
    sample_weights = _process_numpy_inputs(sample_weights)
    sample_weight_modes = broadcast_sample_weight_modes(
        sample_weights, sample_weight_modes)

    # If sample_weights are not specified for an output use 1.0 as weights.
    (sample_weights, any_sample_weight, _
    ) = training_utils.handle_partial_sample_weights(
        y, sample_weights, sample_weight_modes, check_all_flat=True)

    if y is not None and any_sample_weight:
      inputs = (x, y, sample_weights)
    elif y is not None:
      # Sample weight is only needed for training, so if y is None, then
      # sample_weight is ignored.
      inputs = (x, y)
    else:
      inputs = (x,)

    dataset = dataset_ops.DatasetV2.from_tensor_slices(inputs)
    num_samples = int(nest.flatten(x)[0].shape[0])
    if shuffle:
      dataset = dataset.shuffle(num_samples)

    # If batch_size is not passed but steps is, calculate from the input data.
    if steps and not batch_size:
      batch_size = int(math.ceil(num_samples/steps))

    if not batch_size:
      raise ValueError(
          "`batch_size` or `steps` is required for `Tensor` or `NumPy`"
          " input data.")

    dataset = dataset.batch(batch_size)
    self._size = int(math.ceil(num_samples / batch_size))
    self._batch_size = batch_size
    self._has_partial_batch = (self._size != (num_samples // batch_size))

    self._partial_batch_size = None
    if self._has_partial_batch:
      self._partial_batch_size = (
          num_samples - (self._size - 1) * self._batch_size)

    self._dataset = dataset

  def get_dataset(self):
    return self._dataset

  def get_size(self):
    return self._size

  def batch_size(self):
    return self._batch_size

  def has_partial_batch(self):
    return self._has_partial_batch

  def partial_batch_size(self):
    return self._partial_batch_size

  def should_recreate_iterator(self):
    return True


class ListsOfScalarsDataAdapter(DataAdapter):
  """Adapter that handles lists of scalars and lists of lists of scalars."""

  @staticmethod
  def can_handle(x, y=None):
    handles_x = ListsOfScalarsDataAdapter._is_list_of_scalars(x)
    handles_y = True
    if y is not None:
      handles_y = ListsOfScalarsDataAdapter._is_list_of_scalars(y)
    return handles_x and handles_y

  @staticmethod
  def _is_list_of_scalars(inp):
    if isinstance(inp, scalar_types):
      return True
    if isinstance(inp, (list, tuple)):
      return ListsOfScalarsDataAdapter._is_list_of_scalars(inp[0])
    return False

  def __init__(self,
               x,
               y=None,
               sample_weights=None,
               sample_weight_modes=None,
               batch_size=None,
               shuffle=False,
               standardize_function=None,
               **kwargs):
    super(ListsOfScalarsDataAdapter, self).__init__(x, y, **kwargs)
    x = np.asarray(x)
    if y is not None:
      y = np.asarray(y)
    if sample_weights is not None:
      sample_weights = np.asarray(sample_weights)
    sample_weight_modes = broadcast_sample_weight_modes(
        sample_weights, sample_weight_modes)

    if standardize_function is not None:
      x, y, sample_weights = standardize_function(
          x=x, y=y, sample_weight=sample_weights)

    self._internal_adapter = TensorLikeDataAdapter(
        x,
        y=y,
        sample_weights=sample_weights,
        sample_weight_modes=sample_weight_modes,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs)

  def get_dataset(self):
    return self._internal_adapter.get_dataset()

  def get_size(self):
    return self._internal_adapter.get_size()

  def batch_size(self):
    return self._internal_adapter.batch_size()

  def has_partial_batch(self):
    return self._internal_adapter.has_partial_batch()

  def partial_batch_size(self):
    return self._internal_adapter.partial_batch_size()

  def should_recreate_iterator(self):
    return True


class DatasetAdapter(DataAdapter):
  """Adapter that handles `tf.data.Dataset`."""

  @staticmethod
  def can_handle(x, y=None):
    return isinstance(x, (dataset_ops.DatasetV1, dataset_ops.DatasetV2))

  def __init__(self,
               x,
               y=None,
               sample_weights=None,
               steps=None,
               standardize_function=None,
               **kwargs):
    super(DatasetAdapter, self).__init__(x, y, **kwargs)
    if not is_none_or_empty(y):
      raise ValueError("`y` argument is not supported when using "
                       "dataset as input.")
    if not is_none_or_empty(sample_weights):
      raise ValueError("`sample_weight` argument is not supported when using "
                       "dataset as input.")

    if standardize_function is not None:
      x = standardize_function(x)

    # Note that the dataset instance is immutable, its fine to reusing the user
    # provided dataset.
    self._dataset = x

    # The user-provided steps.
    self._user_steps = steps

  def get_dataset(self):
    return self._dataset

  def get_size(self):
    # The size of dataset is unknown, unless its fully consumed.
    return None

  def batch_size(self):
    return None

  def has_partial_batch(self):
    return False

  def partial_batch_size(self):
    return None

  def should_recreate_iterator(self):
    # If user doesn't supply `steps`, or if they supply `steps` that
    # exactly equals the size of the `Dataset`, create a new iterator
    # each epoch.
    return (self._user_steps is None or
            cardinality.cardinality(self._dataset).numpy() == self._user_steps)


class GeneratorDataAdapter(DataAdapter):
  """Adapter that handles python generators and iterators."""

  @staticmethod
  def can_handle(x, y=None):
    return ((hasattr(x, "__next__") or hasattr(x, "next"))
            and hasattr(x, "__iter__")
            and not isinstance(x, data_utils.Sequence))

  def __init__(self, x, y=None, sample_weights=None, standardize_function=None,
               workers=1, use_multiprocessing=False, max_queue_size=10,
               **kwargs):
    # Generators should never shuffle as exhausting the generator in order to
    # shuffle the batches is inefficient.
    kwargs.pop("shuffle", None)

    if not is_none_or_empty(y):
      raise ValueError("`y` argument is not supported when using "
                       "python generator as input.")
    if not is_none_or_empty(sample_weights):
      raise ValueError("`sample_weight` argument is not supported when using "
                       "python generator as input.")
    super(GeneratorDataAdapter, self).__init__(x, y, **kwargs)

    # Since we have to know the dtype of the python generator when we build the
    # dataset, we have to look at a batch to infer the structure.
    peek, x = self._peek_and_restore(x)
    assert_not_namedtuple(peek)

    (peek, wrap_in_tuple, elements_to_keep, partial_sample_weight,
     sample_weight_modes, nested_shape, nested_dtypes
    ) = self._canonicalize_peek(peek, kwargs.get("sample_weight_modes"))

    # Note that dataset API takes a callable that creates a generator object,
    # rather than generator itself, which is why we define a function here.
    generator_fn = self._make_callable(x, workers, use_multiprocessing,
                                       max_queue_size)

    generator_fn = self._make_bridging_callable(
        generator_fn, wrap_in_tuple, peek, elements_to_keep,
        partial_sample_weight, sample_weight_modes)

    dataset = dataset_ops.DatasetV2.from_generator(
        generator_fn, nested_dtypes, output_shapes=nested_shape)

    if standardize_function is not None:
      dataset = standardize_function(dataset)

    if workers == 1 and not use_multiprocessing:
      dataset = dataset.prefetch(1)

    self._dataset = dataset

  def _canonicalize_peek(self, peek, sample_weight_modes):
    """Map the peeked batch into a regular form.

    This function serves two purposes. First, it determines if per-batch
    transformations are needed. Second, it extracts the structure to be used
    by Dataset.from_generator.

    Args:
      peek: The first batch of the user's data
      sample_weight_modes: Optional structure indicating how to handle sample
        weights. If it is a string, it will be mapped to match the target
        structure.

    Returns:
      An updated peek and various inspection results.
    """
    wrap_in_tuple = False
    if not isinstance(peek, tuple):
      peek, wrap_in_tuple = (peek,), True

    if len(peek) not in (1, 2, 3):
      raise ValueError(
          "Output of generator should be a tuple of 1 or 2 or 3 elements: "
          "(input,) or (input, target) or (input, target, sample_weights). "
          "Received {}".format(peek))

    x_peek, y_peek, sample_weights_peek = list(peek) + [None] * (3 - len(peek))

    any_sample_weight, partial_sample_weight = False, False
    sample_weight_modes = broadcast_sample_weight_modes(
        sample_weights_peek if sample_weights_peek is not None else y_peek,
        sample_weight_modes)

    if len(peek) == 3:
      (sample_weights_peek, any_sample_weight, partial_sample_weight
      ) = training_utils.handle_partial_sample_weights(
          y_peek, sample_weights_peek, sample_weight_modes, check_all_flat=True)
      peek = (x_peek, y_peek, sample_weights_peek)

    # Users often return None for fields which are not used. For instance:
    # (x, y, None) to indicate no sample weights.
    if len(peek) >= 2 and y_peek is None:
      if any_sample_weight:
        raise ValueError("Found sample weights but no targets\n{}".format(peek))
      elements_to_keep = 1
    elif len(peek) == 3 and not any_sample_weight:
      elements_to_keep = 2
    else:
      elements_to_keep = len(peek)

    def dynamic_shape_like(t):
      return tuple(None for _ in t.shape)

    def convert_for_inspection(t):
      if getattr(t, "shape", None) and getattr(t, "dtype", None):
        return t
      return np.array(t, dtype=backend.floatx())

    canonicalized_peek = nest._list_to_tuple(  # pylint: disable=protected-access
        nest.map_structure(convert_for_inspection, peek[:elements_to_keep]))
    nested_dtypes = nest.map_structure(lambda t: t.dtype, canonicalized_peek)
    nested_shape = nest.map_structure(dynamic_shape_like, canonicalized_peek)

    try:
      self._first_batch_size = int(nest.flatten(canonicalized_peek)[0].shape[0])
    except IndexError:
      raise IndexError("Could not infer batch size from: {}".format(peek))

    return (peek, wrap_in_tuple, elements_to_keep, partial_sample_weight,
            sample_weight_modes, nested_shape, nested_dtypes)

  @staticmethod
  def _peek_and_restore(x):
    peek = next(x)
    return peek, itertools.chain([peek], x)

  def _make_callable(self, x, workers, use_multiprocessing, max_queue_size):
    """Create a callable, and possibly include an Enqueuer."""
    if workers > 1 or (workers > 0 and use_multiprocessing):
      if use_multiprocessing:
        logging.warning(
            UserWarning("Using a generator with `use_multiprocessing=True` "
                        "and multiple workers may duplicate your data. "
                        "Please consider using the `tf.data.Dataset`."))
      def generator_fn():
        enqueuer = data_utils.GeneratorEnqueuer(
            x, use_multiprocessing=use_multiprocessing)
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        return enqueuer.get()
    else:
      generator_fn = lambda: x
    return generator_fn

  @staticmethod
  def _make_bridging_callable(
      generator_fn, wrap_in_tuple, peek, elements_to_keep,
      partial_sample_weight, sample_weight_modes):
    """Optional compatibility layer between user's data and Dataset."""
    must_prune_nones = (elements_to_keep != len(peek))
    try:
      nest.assert_same_structure(peek, nest._list_to_tuple(peek))  # pylint: disable=protected-access
      must_extract_lists = False
    except TypeError:
      must_extract_lists = True

    # No additional transformations are needed.
    if not (wrap_in_tuple or must_extract_lists or must_prune_nones or
            partial_sample_weight):
      return generator_fn

    def wrapped_generator():
      """Remove Nones and lists before invoking Dataset.from_generator."""
      for batch in generator_fn():
        if wrap_in_tuple:
          batch = (batch,)

        if must_extract_lists:
          batch = nest._list_to_tuple(batch)  # pylint: disable=protected-access

        if must_prune_nones:
          batch = batch[:elements_to_keep]

        if partial_sample_weight:
          sample_weights, _, _ = training_utils.handle_partial_sample_weights(
              batch[1], batch[2], sample_weight_modes, check_all_flat=False)
          batch = batch[:2] + (sample_weights,)

        yield batch

    return wrapped_generator

  def get_dataset(self):
    return self._dataset

  def get_size(self):
    return None

  def batch_size(self):
    return None

  def representative_batch_size(self):
    return self._first_batch_size

  def has_partial_batch(self):
    return False

  def partial_batch_size(self):
    return

  def should_recreate_iterator(self):
    return False


class KerasSequenceAdapter(GeneratorDataAdapter):
  """Adapter that handles `keras.utils.Sequence`."""

  @staticmethod
  def can_handle(x, y=None):
    return isinstance(x, data_utils.Sequence)

  def __init__(self, x, y=None, sample_weights=None, standardize_function=None,
               shuffle=False, workers=1, use_multiprocessing=False,
               max_queue_size=10, **kwargs):
    if not is_none_or_empty(y):
      raise ValueError("`y` argument is not supported when using "
                       "`keras.utils.Sequence` as input.")
    if not is_none_or_empty(sample_weights):
      raise ValueError("`sample_weight` argument is not supported when using "
                       "`keras.utils.Sequence` as input.")
    self._size = len(x)
    self._shuffle_sequence = shuffle
    super(KerasSequenceAdapter, self).__init__(
        x,
        standardize_function=standardize_function,
        shuffle=False,  # Shuffle is handed in the _make_callable override.
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=max_queue_size,
        **kwargs)

  @staticmethod
  def _peek_and_restore(x):
    return x[0], x

  def _make_callable(self, x, workers, use_multiprocessing, max_queue_size):
    if workers > 1 or (workers > 0 and use_multiprocessing):
      def generator_fn():
        enqueuer = data_utils.OrderedEnqueuer(
            x, use_multiprocessing=use_multiprocessing,
            shuffle=self._shuffle_sequence)
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        return enqueuer.get()
    else:
      def generator_fn():
        order = range(len(x))
        if self._shuffle_sequence:
          # Match the shuffle convention in OrderedEnqueuer.
          order = list(order)
          random.shuffle(order)

        for i in order:
          yield x[i]

    return generator_fn

  def get_size(self):
    return self._size

  def should_recreate_iterator(self):
    return True


ALL_ADAPTER_CLS = [
    ListsOfScalarsDataAdapter, TensorLikeDataAdapter,
    GenericArrayLikeDataAdapter, DatasetAdapter,
    GeneratorDataAdapter, KerasSequenceAdapter, CompositeTensorDataAdapter,
]


def select_data_adapter(x, y):
  """Selects a data adapter than can handle a given x and y."""
  adapter_cls = [cls for cls in ALL_ADAPTER_CLS if cls.can_handle(x, y)]
  if not adapter_cls:
    # TODO(scottzhu): This should be a less implementation-specific error.
    raise ValueError(
        "Failed to find data adapter that can handle "
        "input: {}, {}".format(
            _type_name(x), _type_name(y)))
  elif len(adapter_cls) > 1:
    raise RuntimeError(
        "Data adapters should be mutually exclusive for "
        "handling inputs. Found multiple adapters {} to handle "
        "input: {}, {}".format(
            adapter_cls, _type_name(x), _type_name(y)))
  return adapter_cls[0]


def _type_name(x):
  """Generates a description of the type of an object."""
  if isinstance(x, dict):
    key_types = set(_type_name(key) for key in x.keys())
    val_types = set(_type_name(key) for key in x.values())
    return "({} containing {} keys and {} values)".format(
        type(x), key_types, val_types)
  if isinstance(x, (list, tuple)):
    types = set(_type_name(val) for val in x)
    return "({} containing values of types {})".format(
        type(x), types)
  return str(type(x))


def _process_numpy_inputs(inputs):
  """Process numpy array inputs.

  For numpy inputs, it is possible to be single numpy array, or list/dict of
  them. They could also be preprocessed by other lib to match with the order
  of position for the model. The result here should be something that can be
  used to build dataset.

  Args:
    inputs: single or list/tuple/dict of numpy array.
  Returns:
    numpy arrays can be used to build dataset.
  """
  if is_none_or_empty(inputs):
    return None
  flat_inputs = nest.flatten(inputs)
  if len(flat_inputs) == 1:
    return flat_inputs[0]

  def _convert_non_tensor(x):
    # Don't call `ops.convert_to_tensor_v2` on all `inputs` because
    # `SparseTensors` can't be converted to `Tensor`.
    if isinstance(x, np.ndarray):
      return ops.convert_to_tensor_v2(x)
    return x

  inputs = nest.map_structure(_convert_non_tensor, inputs)
  # For more complicated structure, we only convert the out most list to tuple
  # since dataset will stack the list, but treat elements in the tuple as
  # individual element.
  return training_utils.list_to_tuple(inputs)


def is_none_or_empty(inputs):
  # util method to check if the input is a None or a empty list.
  # the python "not" check will raise an error like below if the input is a
  # numpy array
  # "The truth value of an array with more than one element is ambiguous.
  # Use a.any() or a.all()"
  return inputs is None or not nest.flatten(inputs)


def broadcast_sample_weight_modes(target_structure, sample_weight_modes):
  """Match sample_weight_modes structure with output structure."""
  if target_structure is None or not nest.flatten(target_structure):
    return sample_weight_modes

  if isinstance(sample_weight_modes, str):
    if isinstance(target_structure, dict):
      return {key: sample_weight_modes for key in target_structure.keys()}
    return [sample_weight_modes for _ in target_structure]

  if sample_weight_modes:
    try:
      nest.assert_same_structure(
          training_utils.list_to_tuple(target_structure),
          training_utils.list_to_tuple(sample_weight_modes))
    except (ValueError, TypeError):
      target_str = str(nest.map_structure(lambda _: "...", target_structure))
      mode_str = str(nest.map_structure(lambda _: "...", sample_weight_modes))

      # Attempt to coerce sample_weight_modes to the target structure. This
      # implicitly depends on the fact that Model flattens outputs for its
      # internal representation.
      try:
        sample_weight_modes = nest.pack_sequence_as(
            target_structure, nest.flatten(sample_weight_modes))
        logging.warning(
            "sample_weight modes were coerced from\n  {}\n    to  \n  {}"
            .format(target_str, mode_str))
      except (ValueError, TypeError):
        raise ValueError(
            "Unable to match target structure and sample_weight_modes "
            "structure:\n  {}\n    to  \n  {}".format(target_str, mode_str))

  return sample_weight_modes


def assert_not_namedtuple(x):
  if (isinstance(x, tuple) and
      # TODO(b/144192902): Use a namedtuple checking utility.
      hasattr(x, "_fields") and
      isinstance(x._fields, collections.Sequence) and
      all(isinstance(f, six.string_types) for f in x._fields)):
    raise ValueError(
        "Received namedtuple ({}) with fields `{}` as input. namedtuples "
        "cannot, in general, be unambiguously resolved into `x`, `y`, "
        "and `sample_weight`. For this reason Keras has elected not to "
        "support them. If you would like the value to be unpacked, "
        "please explicitly convert it to a tuple before passing it to "
        "Keras.".format(x.__class__, x._fields))


class DataHandler(object):
  """Handles iterating over epoch-level `tf.data.Iterator` objects."""

  # TODO(omalleyt): Handle `validation_split` with separate utility.
  # TODO(omalleyt): Handle `validation_data` batch size when `x` is a gen.
  def __init__(self,
               x,
               y=None,
               sample_weight=None,
               batch_size=None,
               steps_per_epoch=None,
               initial_epoch=0,
               epochs=1,
               shuffle=False,
               class_weight=None,
               max_queue_size=10,
               workers=1,
               use_multiprocessing=False):

    self._initial_epoch = initial_epoch
    self._epochs = epochs
    self._insufficient_data = False

    train_adapter_cls = select_data_adapter(x, y)
    self._train_adapter = train_adapter_cls(
        x,
        y,
        batch_size=batch_size,
        steps=steps_per_epoch,
        epochs=epochs,
        sample_weights=sample_weight,
        shuffle=shuffle,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        distribution_strategy=ds_context.get_strategy())

    strategy = ds_context.get_strategy()
    dataset = self._train_adapter.get_dataset()
    if class_weight:
      dataset = dataset.map(_make_class_weight_map_fn(class_weight))
    self._train_dataset = strategy.experimental_distribute_dataset(dataset)
    self._steps_per_epoch = self._infer_steps(steps_per_epoch)

  def enumerate_epochs(self):
    """Yields `(epoch, tf.data.Iterator)`."""
    data_iterator = iter(self._train_dataset)
    for epoch in range(self._initial_epoch, self._epochs):
      if self._insufficient_data:  # Set by `catch_stop_iteration`.
        break
      if self._train_adapter.should_recreate_iterator():
        data_iterator = iter(self._train_dataset)
      yield epoch, data_iterator

  @contextlib.contextmanager
  def catch_stop_iteration(self):
    """Catches errors when an iterator runs out of data."""
    try:
      yield
    except (StopIteration, errors.OutOfRangeError):
      if (self._train_adapter.get_size() is None and
          self._steps_per_epoch is None and self._current_step > 0):
        # The input passed by the user ran out of batches.
        # Now we know the cardinality of the input(dataset or generator).
        self._steps_per_epoch = self._current_step
      else:
        self._insufficient_data = True
        total_epochs = self._epochs - self._initial_epoch
        logging.warning(
            "Your input ran out of data; interrupting training. "
            "Make sure that your dataset or generator can generate at "
            "least `steps_per_epoch * epochs` batches (in this case, "
            "{} batches). You may need to use the repeat() function "
            "when building your dataset.".format(total_epochs *
                                                 self._steps_per_epoch))

  def steps(self):
    """Yields steps for the current epoch."""
    self._current_step = 0
    # `self._steps_per_epoch` can be changed by `catch_stop_iteration`.
    while (self._steps_per_epoch is None or
           self._current_step < self._steps_per_epoch):
      if self._insufficient_data:  # Set by `catch_stop_iteration`.
        break
      yield self._current_step
      self._current_step += 1

  def _infer_steps(self, steps):
    """Infers steps_per_epoch needed to loop through a dataset."""
    if steps is not None:
      return steps

    adapter_steps = self._train_adapter.get_size()
    if adapter_steps is not None:
      return adapter_steps

    dataset = self._train_dataset
    if (ds_context.get_strategy().extended._in_multi_worker_mode() and  # pylint: disable=protected-access
        (dataset.options().experimental_distribute.auto_shard_policy !=
         distribute_options.AutoShardPolicy.OFF)):
      # If the dataset would be auto-sharded, we should not infer a local
      # steps_per_epoch due to the possible inbalanced sharding between workers.
      return None

    size = cardinality.cardinality(dataset)
    if size == cardinality.INFINITE and steps is None:
      raise ValueError("When passing an infinitely repeating dataset, you "
                       "must specify how many steps to draw.")
    if size >= 0:
      return size
    return None


def _make_class_weight_map_fn(class_weight):
  """Applies class weighting to a `Dataset`.

  The `Dataset` is assumed to be in format `(x, y)` or `(x, y, sw)`, where
  `y` must be a single `Tensor`.

  Arguments:
    class_weight: A map where the keys are integer class ids and values are
      the class weights, e.g. `{0: 0.2, 1: 0.6, 2: 0.3}`

  Returns:
    A function that can be used with `tf.data.Dataset.map` to apply class
    weighting.
  """
  class_ids = list(sorted(class_weight.keys()))
  expected_class_ids = list(range(len(class_ids)))
  if class_ids != expected_class_ids:
    error_msg = (
        "Expected `class_weight` to be a dict with keys from 0 to one less "
        "than the number of classes, found {}").format(class_weight)
    raise ValueError(error_msg)

  class_weight_tensor = ops.convert_to_tensor_v2(
      [class_weight[c] for c in class_ids])

  def _class_weights_map_fn(*data):
    """Convert `class_weight` to `sample_weight`."""
    if len(data) == 2:
      x, y = data
      sw = None
    else:
      x, y, sw = data

    if nest.is_sequence(y):
      raise ValueError(
          "`class_weight` is only supported for `Model`s with a single output.")

    cw = array_ops.gather_v2(class_weight_tensor, y)
    if sw is not None:
      cw = math_ops.cast(cw, sw.dtype)
      if len(cw.shape.as_list()) > len(sw.shape.as_list()):
        cw = array_ops.squeeze(cw)
      # `class_weight` and `sample_weight` are multiplicative.
      sw = sw * cw
    else:
      sw = cw

    return x, y, sw

  return _class_weights_map_fn


def train_validation_split(arrays, validation_split, shuffle=True):
  """Split arrays into random train and validation subsets.

  Arguments:
    arrays: Tensors to split. Allowed inputs are arbitrarily nested structures
      of Tensors and NumPy arrays.
    validation_split: Float between 0 and 1. The proportion of the dataset to
      include in the validation split. The rest of the dataset will be included
      in the training split.
    shuffle: Bool. Whether to shuffle the data before performing a split. If
      `False`, the last `validation_split` fraction of that training data will
      become the validation split.

  Returns:
    `(train_arrays, validation_arrays)`
  """

  def _can_split(t):
    tensor_types = (ops.Tensor, np.ndarray)
    if pd:
      tensor_types = (ops.Tensor, np.ndarray, pd.Series, pd.DataFrame)
    return isinstance(t, tensor_types) or t is None

  flat_arrays = nest.flatten(arrays)
  if not all(_can_split(t) for t in flat_arrays):
    raise ValueError(
        "`validation_split` is only supported for Tensors or NumPy "
        "arrays, found: {}".format(arrays))

  if all(t is None for t in flat_arrays):
    return arrays, arrays

  first_non_none = None
  for t in flat_arrays:
    if t is not None:
      first_non_none = t
      break

  # Assumes all arrays have the same batch shape or are `None`.
  batch_dim = int(first_non_none.shape[0])
  indices = ops.convert_to_tensor_v2(range(batch_dim))
  if shuffle:
    indices = random_ops.random_shuffle(indices)
  split_at = int(math.floor(batch_dim * (1. - validation_split)))
  train_indices = indices[:split_at]
  val_indices = indices[split_at:]

  def _split(t, indices):
    if t is None:
      return t
    t = ops.convert_to_tensor_v2(t)
    return array_ops.gather_v2(t, indices)

  train_arrays = nest.map_structure(
      functools.partial(_split, indices=train_indices), arrays)
  val_arrays = nest.map_structure(
      functools.partial(_split, indices=val_indices), arrays)

  return train_arrays, val_arrays
