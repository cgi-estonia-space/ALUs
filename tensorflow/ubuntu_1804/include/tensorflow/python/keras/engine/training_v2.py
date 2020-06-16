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
"""Training related logic for Keras model in TF 2.0 context.

Note that all the code under this module is under active development, please DO
NOT use it unless you are really sure what you are doing.
"""

# pylint: disable=protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.framework import errors
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.distribute import distributed_training_utils as dist_utils
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_v2_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import traceme
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib


# The list of DataAdapter that support validation_split, only numpy and data
# tensor support validation_split for now.
_ADAPTER_FOR_VALIDATION_SPLIT = [data_adapter.TensorLikeDataAdapter,
                                 data_adapter.GenericArrayLikeDataAdapter]

# The list of DataAdapter that support model._standardize_user_data. Currently
# keras.sequence/python generator will cause error when calling
# model._standardize_user_data, this should be updated in future cl, eg, the
# dataset/generate/sequence input will be peeked and processed by
# model._standardize_user_data()
_ADAPTER_FOR_STANDARDIZE_USER_DATA = [
    data_adapter.TensorLikeDataAdapter,
    data_adapter.GenericArrayLikeDataAdapter,
    data_adapter.CompositeTensorDataAdapter
]


def run_one_epoch(model,
                  iterator,
                  execution_function,
                  dataset_size=None,
                  batch_size=None,
                  strategy=None,
                  steps_per_epoch=None,
                  num_samples=None,
                  mode=ModeKeys.TRAIN,
                  training_context=None,
                  total_epochs=None):
  """Run the execution function with the data from iterator.

  Given the dataset iterator and execution function, get the data from iterator
  and call it with the execution function to get the result (metric/loss).
  It will run for steps_per_epoch or until to the iterator is fully consumed.

  Args:
    model: The keras model to run.
    iterator: the dataset iterator to fetch the data.
    execution_function: a tf.function that can be called with data.
    dataset_size: the size of iterator, None when unknown.
    batch_size: The size of the current batch.
    strategy: the distribution strategy instance from the model.
    steps_per_epoch: the number of steps to run for the epoch.
    num_samples: the number of samples for the whole epoch if known. This can be
      used to calculate the final partial batch, and scale the loss.
    mode: the mode for the current epoch.
    training_context: the context that contains callbacks and progress bar.
    total_epochs: the total number of epochs that will be run.
      Used when throw error when the iterator unexpectedly
      reaches its end.
  Returns:
    The loss and metric value from the model.
  """
  # Only use the sample to count if there is a partial batch at the end.
  use_steps = num_samples is None

  if mode == ModeKeys.PREDICT:
    aggregator = training_utils.OutputsAggregator(
        use_steps=use_steps,
        steps=steps_per_epoch,
        num_samples=num_samples,
        batch_size=batch_size)
  else:
    aggregator = training_utils.MetricsAggregator(
        use_steps=use_steps, steps=steps_per_epoch, num_samples=num_samples)
  callbacks = training_context.callbacks
  progbar = training_context.progbar

  if callbacks.model.stop_training:
    return

  target_steps = steps_per_epoch or np.inf
  step = 0

  while step < target_steps:
    if use_steps:
      current_batch_size = 1
    elif step < target_steps - 1:
      current_batch_size = batch_size
    else:
      current_batch_size = num_samples - step * batch_size
    with training_context.on_batch(
        step=step, mode=mode, size=current_batch_size) as batch_logs:
      try:
        batch_outs = execution_function(iterator)
      except (StopIteration, errors.OutOfRangeError):
        # TODO(kaftan): File bug about tf function and errors.OutOfRangeError?
        # Are there any other C++ errors tf function should recapture?
        # The only acceptable case here is that the input has a unknown
        # length, and configured to fully consume it.
        if (dataset_size is None
            and steps_per_epoch is None
            and step > 0):
          # The input passed by the user ran out of batches.
          # Now we know the cardinality of the input(dataset or generator).
          steps_per_epoch = step
          aggregator.steps = steps_per_epoch
          if mode == ModeKeys.TRAIN:
            progbar.params['steps'] = steps_per_epoch
            progbar.progbar.target = steps_per_epoch
        else:
          callbacks.model.stop_training = True
          logging.warning(
              'Your input ran out of data; interrupting training. '
              'Make sure that your dataset or generator can generate at '
              'least `steps_per_epoch * epochs` batches (in this case, '
              '{} batches). You may need to use the repeat() function '
              'when building your dataset.'.format(
                  total_epochs * steps_per_epoch))
        # In either case, break out the loop for training batch.
        # Also note the training_context that data inputs are exhausted, so all
        # the post batch hooks can be skipped.
        batch_logs['data_exhausted'] = True
        break

      if mode != ModeKeys.PREDICT:
        data_batch_size = batch_outs['batch_size']
        batch_outs = (batch_outs['total_loss'] + batch_outs['output_losses']
                      + batch_outs['metrics'])
        if current_batch_size != data_batch_size:
          batch_logs['size'] = data_batch_size
          current_batch_size = data_batch_size
      else:
        batch_outs = training_v2_utils._aggregate_predict_results(
            strategy, batch_outs, model)

      if step == 0:
        aggregator.create(batch_outs)

      if use_steps:
        aggregator.aggregate(batch_outs)
      else:
        aggregator.aggregate(
            batch_outs,
            batch_start=step * batch_size,
            batch_end=step * batch_size + current_batch_size)
      cbks.make_logs(model, batch_logs, batch_outs, mode)
      step += 1

    if callbacks.model.stop_training:
      break

  # End of an epoch.
  aggregator.finalize()
  return aggregator.results


class Loop(training_utils.TrainingLoop):
  """The training loop for the TF 2.0.

  This class has some existing assumption for runtime, eg eager by default,
  have distribution strategy, etc.
  """

  def fit(
      self, model, x=None, y=None, batch_size=None, epochs=1, verbose=1,
      callbacks=None, validation_split=0., validation_data=None, shuffle=True,
      class_weight=None, sample_weight=None, initial_epoch=0,
      steps_per_epoch=None, validation_steps=None, validation_freq=1,
      max_queue_size=10, workers=1, use_multiprocessing=False, **kwargs):
    batch_size = model._validate_or_infer_batch_size(
        batch_size, steps_per_epoch, x)

    strategy = model.distribute_strategy
    batch_size, steps_per_epoch = dist_utils.process_batch_and_step_size(
        strategy,
        x,
        batch_size,
        steps_per_epoch,
        ModeKeys.TRAIN,
        validation_split=validation_split)
    dist_utils.validate_callbacks(input_callbacks=callbacks,
                                  optimizer=model.optimizer)
    # Enter tf.distribute.Strategy scope.
    with strategy.scope():
      training_data_adapter, validation_adapter = _process_training_inputs(
          model,
          x,
          y,
          batch_size=batch_size,
          epochs=epochs,
          sample_weights=sample_weight,
          class_weights=class_weight,
          validation_split=validation_split,
          steps_per_epoch=steps_per_epoch,
          shuffle=shuffle,
          validation_data=validation_data,
          validation_steps=validation_steps,
          distribution_strategy=strategy,
          max_queue_size=max_queue_size,
          workers=workers,
          use_multiprocessing=use_multiprocessing)

      total_samples = _get_total_number_of_samples(training_data_adapter)
      use_sample = total_samples is not None
      do_validation = (validation_adapter is not None)

      recreate_training_iterator = (
          training_data_adapter.should_recreate_iterator())
      if not steps_per_epoch:
        # TODO(b/139762795): Add step inference for when steps is None to
        # prevent end of sequence warning message.
        steps_per_epoch = training_data_adapter.get_size()

      # tf.print('{} on {} steps.'.format(ModeKeys.TRAIN, steps_per_epoch))
      training_context = TrainingContext()

      training_dataset = training_data_adapter.get_dataset()
      # Raise an error if steps_per_epoch isn't specified but the dataset
      # is infinite.
      # TODO(scottzhu): This check should probably happen in the adapter
      inferred_steps = training_utils.infer_steps_for_dataset(
          model,
          training_dataset,
          steps_per_epoch,
          steps_name='steps_per_epoch',
          epochs=0)

      steps_per_epoch = (
          inferred_steps if steps_per_epoch is None else steps_per_epoch)

      training_dataset = strategy.experimental_distribute_dataset(
          training_dataset)

      training_function = training_v2_utils._get_or_make_execution_function(
          model, ModeKeys.TRAIN)

      training_data_iter = None
      if do_validation:
        validation_dataset = validation_adapter.get_dataset()
        if not validation_steps:
          # Raise an error if validation_steps isn't specified but the
          # validation dataset is infinite.
          validation_steps = (
              validation_adapter.get_size() or
              training_utils.infer_steps_for_dataset(
                  model,
                  validation_dataset,
                  validation_steps,
                  steps_name='validation_steps'))
        eval_function = training_v2_utils._get_or_make_execution_function(
            model, ModeKeys.TEST)
        eval_data_iter = None
        validation_dataset = strategy.experimental_distribute_dataset(
            validation_dataset)
        val_total_samples = _get_total_number_of_samples(validation_adapter)
      else:
        val_total_samples = None

      if verbose and (total_samples or steps_per_epoch):
        _print_train_info(total_samples, steps_per_epoch, val_total_samples,
                          validation_steps)

      training_callbacks = cbks.configure_callbacks(
          callbacks,
          model,
          do_validation=do_validation,
          batch_size=batch_size,
          epochs=epochs,
          steps_per_epoch=steps_per_epoch,
          samples=total_samples or steps_per_epoch,
          count_mode='samples' if use_sample else 'steps',
          verbose=0,  # Handle ProgBarLogger separately in this loop.
          mode=ModeKeys.TRAIN)

      with training_context.on_start(model, training_callbacks, use_sample,
                                     verbose, ModeKeys.TRAIN):

        initial_epoch = model._maybe_load_initial_epoch_from_ckpt(
            initial_epoch, ModeKeys.TRAIN)

        for epoch in range(initial_epoch, epochs):
          if training_context.callbacks.model.stop_training:
            break

          # Training
          with training_context.on_epoch(epoch, ModeKeys.TRAIN) as epoch_logs:
            model.reset_metrics()
            if training_data_iter is None or recreate_training_iterator:
              if training_data_iter is not None and ds_context.has_strategy():
                # TODO(kaftan): remove this when MultiDeviceIterator is a
                ## compositetensor (unless this is more efficient)
                training_data_iter._initializer  # pylint: disable=pointless-statement
              else:
                training_data_iter = iter(training_dataset)

            training_result = run_one_epoch(
                model,
                training_data_iter,
                training_function,
                dataset_size=training_data_adapter.get_size(),
                batch_size=training_data_adapter.batch_size(),
                strategy=strategy,
                steps_per_epoch=steps_per_epoch,
                num_samples=total_samples,
                mode=ModeKeys.TRAIN,
                training_context=training_context,
                total_epochs=epochs)
            cbks.make_logs(model, epoch_logs, training_result, ModeKeys.TRAIN)

            # In the case of steps_per_epoch = None, the final cardinality will
            # be determined when the inputs are fully consumed (eg dataset or
            # generator). Update the steps_per_epoch to the new value.
            if (steps_per_epoch is None
                and training_context.progbar.progbar.target is not None):
              steps_per_epoch = training_context.progbar.progbar.target

            # Evaluation
            if (do_validation and
                training_utils.should_run_validation(validation_freq, epoch) and
                not training_callbacks.model.stop_training):
              if eval_data_iter is not None and ds_context.has_strategy():
                # TODO(kaftan): remove this when MultiDeviceIterator is a
                ## compositetensor (unless this is more efficient)
                eval_data_iter._initializer  # pylint: disable=pointless-statement
              else:
                eval_data_iter = iter(validation_dataset)

              validation_callbacks = cbks.configure_callbacks(
                  training_callbacks,
                  model,
                  batch_size=batch_size,
                  epochs=1,
                  steps_per_epoch=validation_steps,
                  samples=val_total_samples or validation_steps,
                  count_mode='samples' if use_sample else 'steps',
                  verbose=0,  # Handle ProgBarLogger separately in this loop.
                  mode=ModeKeys.TEST)

              eval_context = TrainingContext()
              with eval_context.on_start(
                  model,
                  validation_callbacks,
                  use_sample,
                  verbose=0,
                  mode=ModeKeys.TEST):
                with eval_context.on_epoch(epoch, ModeKeys.TEST):
                  model.reset_metrics()
                  eval_result = run_one_epoch(
                      model,
                      eval_data_iter,
                      eval_function,
                      dataset_size=validation_adapter.get_size(),
                      batch_size=validation_adapter.batch_size(),
                      strategy=strategy,
                      steps_per_epoch=validation_steps,
                      num_samples=val_total_samples,
                      mode=ModeKeys.TEST,
                      training_context=eval_context,
                      total_epochs=1)
                  cbks.make_logs(model, epoch_logs, eval_result, ModeKeys.TEST,
                                 prefix='val_')

    return model.history

  def _model_iteration(
      self, model, mode, x=None, y=None, batch_size=None, verbose=1,
      sample_weight=None, steps=None, callbacks=None, max_queue_size=10,
      workers=1, use_multiprocessing=False, **kwargs):

    batch_size = model._validate_or_infer_batch_size(
        batch_size, steps, x)
    strategy = model.distribute_strategy
    batch_size, steps = dist_utils.process_batch_and_step_size(
        strategy, x, batch_size, steps, mode)
    dist_utils.validate_callbacks(input_callbacks=callbacks,
                                  optimizer=model.optimizer)
    # Enter tf.distribute.Strategy scope.
    with strategy.scope():
      adapter = _process_inputs(
          model,
          mode,
          x,
          y,
          batch_size=batch_size,
          sample_weights=sample_weight,
          steps=steps,
          distribution_strategy=strategy,
          max_queue_size=max_queue_size,
          workers=workers,
          use_multiprocessing=use_multiprocessing)
      total_samples = _get_total_number_of_samples(adapter)
      use_sample = total_samples is not None
      dataset = adapter.get_dataset()

      if not steps:
        # Raise an error if `steps` isn't specified but the dataset
        # is infinite.
        steps = adapter.get_size() or training_utils.infer_steps_for_dataset(
            model, dataset, steps, steps_name='steps')

      # tf.print('{} on {} steps.'.format(ModeKeys.TRAIN, steps_per_epoch))
      training_context = TrainingContext()
      if training_v2_utils._should_add_batch_index_to_element(strategy, mode):
        dataset = training_v2_utils._add_batch_index_to_element(dataset)
      dataset = strategy.experimental_distribute_dataset(dataset)

      execution_function = training_v2_utils._get_or_make_execution_function(
          model, mode)

      data_iterator = iter(dataset)

      callbacks = cbks.configure_callbacks(
          callbacks,
          model,
          do_validation=False,
          batch_size=batch_size,
          epochs=1,
          steps_per_epoch=steps,
          samples=total_samples,
          count_mode='samples' if use_sample else 'steps',
          verbose=0,  # Handle ProgBarLogger separately in this loop.
          mode=mode)

      with training_context.on_start(
          model, callbacks, use_sample, verbose, mode):
        with training_context.on_epoch(0, mode) as epoch_logs:
          model.reset_metrics()
          result = run_one_epoch(
              model,
              data_iterator,
              execution_function,
              dataset_size=adapter.get_size(),
              batch_size=adapter.batch_size(),
              strategy=strategy,
              steps_per_epoch=steps,
              num_samples=total_samples,
              mode=mode,
              training_context=training_context,
              total_epochs=1)
          cbks.make_logs(model, epoch_logs, result, mode)

    if len(result) == 1:
      result = result[0]
    return result

  def evaluate(
      self, model, x=None, y=None, batch_size=None, verbose=1,
      sample_weight=None, steps=None, callbacks=None, max_queue_size=10,
      workers=1, use_multiprocessing=False, **kwargs):
    return self._model_iteration(
        model, ModeKeys.TEST, x=x, y=y, batch_size=batch_size, verbose=verbose,
        sample_weight=sample_weight, steps=steps, callbacks=callbacks,
        max_queue_size=max_queue_size, workers=workers,
        use_multiprocessing=use_multiprocessing, **kwargs)

  def predict(self, model, x, batch_size=None, verbose=0, steps=None,
              callbacks=None, max_queue_size=10, workers=1,
              use_multiprocessing=False, **kwargs):
    return self._model_iteration(
        model, ModeKeys.PREDICT, x=x, batch_size=batch_size, verbose=verbose,
        steps=steps, callbacks=callbacks, max_queue_size=max_queue_size,
        workers=workers, use_multiprocessing=use_multiprocessing, **kwargs)


def _process_training_inputs(model,
                             x,
                             y,
                             batch_size=None,
                             epochs=1,
                             sample_weights=None,
                             class_weights=None,
                             steps_per_epoch=None,
                             validation_split=0.,
                             validation_data=None,
                             validation_steps=None,
                             shuffle=True,
                             distribution_strategy=None,
                             max_queue_size=10,
                             workers=1,
                             use_multiprocessing=False):
  """Process the data input for fit() with respect to validation_split."""
  if validation_split and 0. < validation_split < 1. and validation_data:
    raise ValueError('validation_data and validation_split cannot be used '
                     'at same time.')

  adapter_cls = data_adapter.select_data_adapter(x, y)

  # Handle validation_split, we want to split the data and get the training
  # section before we give it to data adapter.
  if validation_split and 0. < validation_split < 1.:
    if adapter_cls not in _ADAPTER_FOR_VALIDATION_SPLIT:
      raise ValueError(
          '`validation_split` argument is not supported when '
          'data adapter is {}. Received: x={}, validation_split={}'.format(
              adapter_cls, x, validation_split))
    # Retrieve the training section from x and y, and then construct dataset
    # from it.
    x, y, sample_weights = model._standardize_user_data(
        x,
        y,
        sample_weight=sample_weights,
        class_weight=class_weights,
        batch_size=batch_size,
        check_steps=False,
        steps=steps_per_epoch)
    (x, y, sample_weights,
     val_x, val_y,
     val_sample_weights) = training_utils.split_training_and_validation_data(
         x, y, sample_weights, validation_split)

    sample_weight_modes = [
        e.sample_weight_mode for e in model._training_endpoints
    ]
    train_adapter = adapter_cls(
        x,
        y,
        batch_size=batch_size,
        steps=steps_per_epoch,
        epochs=epochs,
        sample_weights=sample_weights,
        sample_weight_modes=sample_weight_modes,
        shuffle=shuffle,
        distribution_strategy=distribution_strategy)

    val_adapter = adapter_cls(
        val_x,
        val_y,
        steps=validation_steps,
        sample_weights=val_sample_weights,
        sample_weight_modes=sample_weight_modes,
        batch_size=batch_size,
        distribution_strategy=distribution_strategy)
  else:
    train_adapter = _process_inputs(
        model,
        ModeKeys.TRAIN,
        x,
        y,
        sample_weights=sample_weights,
        batch_size=batch_size,
        steps=steps_per_epoch,
        epochs=epochs,
        class_weights=class_weights,
        shuffle=shuffle,
        distribution_strategy=distribution_strategy,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing)
    val_adapter = None
    if validation_data:
      (val_x, val_y,
       val_sample_weights) = training_utils.unpack_validation_data(
           validation_data, raise_if_ambiguous=False)
      # For eval data, we use a representative batch size of the
      # training data if batch_size was unknown.
      # This is useful for generator/sequence training data input with numpy
      # validation data input.
      if not batch_size:
        batch_size = train_adapter.representative_batch_size()
      val_adapter = _process_inputs(
          model,
          ModeKeys.TEST,
          val_x,
          val_y,
          steps=validation_steps,
          sample_weights=val_sample_weights,
          batch_size=batch_size,
          class_weights=class_weights,
          distribution_strategy=distribution_strategy)
    elif validation_steps:
      raise ValueError('`validation_steps` should not be specified if '
                       '`validation_data` is None.')
  return train_adapter, val_adapter


def _process_inputs(model,
                    mode,
                    x,
                    y,
                    batch_size=None,
                    epochs=1,
                    sample_weights=None,
                    class_weights=None,
                    shuffle=False,
                    steps=None,
                    distribution_strategy=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False):
  """Process the inputs for fit/eval/predict()."""
  adapter_cls = data_adapter.select_data_adapter(x, y)
  standardize = functools.partial(
      model._standardize_user_data,
      class_weight=class_weights,
      batch_size=batch_size,
      check_steps=False,
      steps=steps)
  if adapter_cls in _ADAPTER_FOR_STANDARDIZE_USER_DATA:
    standardize_function = None
    x, y, sample_weights = standardize(
        x, y, sample_weight=sample_weights)
  elif adapter_cls is data_adapter.ListsOfScalarsDataAdapter:
    standardize_function = standardize
  else:
    def standardize_function(dataset):
      """Data adapters can standardize when appropriate."""
      # First we call _standardize_user_data with the dataset since that has
      # enough structure to build the model.
      if not model._is_compiled:
        # We don't actually care about the values of these attributes, but they
        # are only created in compile and are accessed in _standardize_user_data
        model._training_endpoints = getattr(model, '_training_endpoints', [])
        model.sample_weight_mode = getattr(model, 'sample_weight_mode', None)

      standardize(dataset, extract_tensors_from_dataset=False)

      # Then we map using only the tensor standardization portion.
      def map_fn(x, y=None, sample_weights=None):
        """Tensor manipulation portion of standardization for Dataset.map."""
        if (y is None and sample_weights is None):
          # namedtuples are forbidden because it is ambiguous if they should be
          # unpacked. If y or sample_weights is present then `x` was not the
          # top level structure, and the correct behavior is unambiguous.
          data_adapter.assert_not_namedtuple(x)

        standardized = model._standardize_tensors(
            x, y, sample_weights,
            run_eagerly=False,
            dict_inputs=isinstance(x, dict),
            is_dataset=False,
            class_weight=class_weights,
            batch_size=None)
        x, y, sample_weights = nest._list_to_tuple(standardized)
        if y is None:
          return (x,)
        if sample_weights is None:
          return x, y
        return x, y, sample_weights
      return dataset.map(map_fn, num_parallel_calls=dataset_ops.AUTOTUNE)

  if mode == ModeKeys.PREDICT:
    sample_weight_modes = None
  else:
    sample_weight_modes = [
        e.sample_weight_mode for e in model._training_endpoints
    ] or model.sample_weight_mode

  adapter = adapter_cls(
      x,
      y,
      standardize_function=standardize_function,
      batch_size=batch_size,
      epochs=epochs,
      steps=steps,
      sample_weights=sample_weights,
      sample_weight_modes=sample_weight_modes,
      shuffle=shuffle,
      distribution_strategy=distribution_strategy,
      max_queue_size=max_queue_size,
      workers=workers,
      use_multiprocessing=use_multiprocessing)

  return adapter


def _get_total_number_of_samples(adapter):
  if not adapter.get_size() or not adapter.batch_size():
    return None
  total_sample = adapter.get_size() * adapter.batch_size()
  if adapter.has_partial_batch():
    total_sample -= (adapter.batch_size() - adapter.partial_batch_size())
  return total_sample


def _print_train_info(total_samples, steps, val_total_samples, val_steps):
  increment = 'samples' if total_samples else 'steps'
  conjunction = 'on' if total_samples else 'for'
  msg = 'Train {} {} {}'.format(conjunction, total_samples or steps, increment)
  if val_total_samples or val_steps:
    increment = 'samples' if val_total_samples else 'steps'
    conjunction = 'on' if val_total_samples else 'for'
    msg += ', validate {} {} {}'.format(conjunction, val_total_samples or
                                        val_steps, increment)
  print(msg)


class TrainingContext(object):
  """Utility object that wrap around callbacks and progress bars."""

  @tf_contextlib.contextmanager
  def on_start(self, model, callbacks=None, use_samples=False, verbose=0,
               mode=ModeKeys.TRAIN):
    """Provide a scope for the whole training process."""
    # TODO(omalleyt): Handle ProgBar as part of Callbacks once hooks are ready.
    progbar = training_utils.get_progbar(
        model, 'samples' if use_samples else 'steps')
    progbar.params = callbacks.params
    progbar.params['verbose'] = verbose
    callbacks.model.stop_training = False
    callbacks._call_begin_hook(mode)
    progbar.on_train_begin()

    # Cache those two instance so that it can be used in other functions.
    self.callbacks = callbacks
    self.progbar = progbar

    try:
      yield
      model._successful_loop_finish = True
    finally:
      # End of all epochs
      self.callbacks._call_end_hook(mode)

  @tf_contextlib.contextmanager
  def on_epoch(self, epoch=0, mode=ModeKeys.TRAIN):
    """Provide a scope for running one epoch."""
    epoch_logs = {}
    if mode == ModeKeys.TRAIN:
      self.callbacks.on_epoch_begin(epoch, epoch_logs)
    self.progbar.on_epoch_begin(epoch, epoch_logs)
    try:
      yield epoch_logs
    finally:
      if mode == ModeKeys.TRAIN:
        # Epochs only apply to `fit`.
        self.callbacks.on_epoch_end(epoch, epoch_logs)
      self.progbar.on_epoch_end(epoch, epoch_logs)

  @tf_contextlib.contextmanager
  def on_batch(self, step=0, mode=ModeKeys.TRAIN, size=1):
    """Provide a scope for running one batch."""
    with traceme.TraceMe(
        'TraceContext', graph_type=mode, step_num=step, batch_size=size):
      batch_logs = {'batch': step, 'size': size}
      self.callbacks._call_batch_hook(
          mode, 'begin', step, batch_logs)
      self.progbar.on_batch_begin(step, batch_logs)
      try:
        yield batch_logs
      finally:
        if not batch_logs.pop('data_exhausted', False):
          self.callbacks._call_batch_hook(
              mode, 'end', step, batch_logs)
          self.progbar.on_batch_end(step, batch_logs)
