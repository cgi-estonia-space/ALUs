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
"""Utilities that help manage directory path in distributed settings.

In multi-worker training, the need to write a file to distributed file
location often requires only one copy done by one worker despite many workers
that are involved in training. The option to only perform saving by chief is
not feasible for a couple of reasons: 1) Chief and workers may each contain
a client that runs the same piece of code and it's preferred not to make
any distinction between the code run by chief and other workers, and 2)
saving of model or model's related information may require SyncOnRead
variables to be read, which needs the cooperation of all workers to perform
all-reduce.

This set of utility is used so that only one copy is written to the needed
directory, by supplying a temporary write directory path for workers that don't
need to save, and removing the temporary directory once file writing is done.

Example usage:
```
# Before using a directory to write file to.
self.log_write_dir = write_dirpath(self.log_dir, get_distribution_strategy())
# Now `self.log_write_dir` can be safely used to write file to.

...

# After the file is written to the directory.
remove_temp_dirpath(self.log_dir, get_distribution_strategy())

```

Experimental. API is subject to change.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.lib.io import file_io


def _get_base_dirpath(strategy):
  task_id = strategy.extended._task_id  # pylint: disable=protected-access
  return 'workertemp_' + str(task_id)


def _is_temp_dir(dirpath, strategy):
  return dirpath.endswith(_get_base_dirpath(strategy))


def _get_temp_dir(dirpath, strategy):
  if _is_temp_dir(dirpath, strategy):
    return dirpath
  return os.path.join(dirpath, _get_base_dirpath(strategy))


def write_dirpath(dirpath, strategy=None):
  """Returns the write path that should be used to save file distributedly."""
  if strategy is None:
    # Infer strategy from `distribution_strategy_context` if not given.
    strategy = distribution_strategy_context.get_strategy()
  if strategy is None:
    # If strategy is still not available, this is not in distributed training.
    # Fallback to original dirpath.
    return dirpath
  if not strategy.extended._in_multi_worker_mode():  # pylint: disable=protected-access
    return dirpath
  if multi_worker_util.is_chief():
    return dirpath
  # If this worker is not chief and hence should not save file, save it to a
  # temporary directory to be removed later.
  return _get_temp_dir(dirpath, strategy)


def remove_temp_dirpath(dirpath, strategy=None):
  """Removes the temp path after writing is finished."""
  if strategy is None:
    # Infer strategy from `distribution_strategy_context` if not given.
    strategy = distribution_strategy_context.get_strategy()
  if strategy is None:
    # If strategy is still not available, this is not in distributed training.
    # Fallback to no-op.
    return
  if strategy.extended._in_multi_worker_mode():  # pylint: disable=protected-access
    if not multi_worker_util.is_chief():
      # If this worker is not chief and hence should not save file, remove
      # the temporary directory.
      file_io.delete_recursively(_get_temp_dir(dirpath, strategy))
