# lint as: python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""A tool to generate api_docs for TensorFlow2.

```
python generate2.py --output_dir=/tmp/out
```

Requires a local installation of `tensorflow_docs`:

```
pip install git+https://github.com/tensorflow/docs
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path
import textwrap

from absl import app
from absl import flags

import tensorflow as tf

from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import doc_generator_visitor
from tensorflow_docs.api_generator import generate_lib

import tensorboard
import tensorflow_estimator
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_export
from tensorflow.python.util import tf_inspect

# `tf` has an `__all__` that doesn't list important things like `keras`.
# The doc generator recognizes `__all__` as the list of public symbols.
# So patch `tf.__all__` to list everything.
tf.__all__ = [item_name for item_name, value in tf_inspect.getmembers(tf)]

# tf_export generated two copies of the module objects.
# This will just list compat.v2 as an alias for tf. Close enough, let's not
# duplicate all the module skeleton files.
tf.compat.v2 = tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "code_url_prefix",
    "/code/stable/tensorflow",
    "A url to prepend to code paths when creating links to defining code")

flags.DEFINE_string("output_dir", "/tmp/out",
                    "A directory, where the docs will be output to.")

flags.DEFINE_bool("search_hints", True,
                  "Include meta-data search hints at the top of each file.")

flags.DEFINE_string(
    "site_path", "", "The prefix ({site-path}/api_docs/python/...) used in the "
    "`_toc.yaml` and `_redirects.yaml` files")

_PRIVATE_MAP = {
    "tf": ["python", "core", "compiler", "examples", "tools", "contrib"],
    # There's some aliasing between the compats and v1/2s, so it's easier to
    # block by name and location than by deleting, or hiding objects.
    "tf.compat.v1.compat": ["v1", "v2"],
    "tf.compat.v2.compat": ["v1", "v2"]
}

tf.__doc__ = """
  ## TensorFlow

  ```
  pip install tensorflow
  ```
  """


def generate_raw_ops_doc():
  """Generates docs for `tf.raw_ops`."""

  warning = textwrap.dedent("""\n
    Note: `tf.raw_ops` provides direct/low level access to all TensorFlow ops.
    See [the RFC](https://github.com/tensorflow/community/blob/master/rfcs/20181225-tf-raw-ops.md)
    for details. Unless you are library writer, you likely do not need to use
    these ops directly.""")

  table_header = textwrap.dedent("""

      | Op Name | Has Gradient |
      |---------|:------------:|""")

  parts = [warning, table_header]

  for op_name in sorted(dir(tf.raw_ops)):
    try:
      ops._gradient_registry.lookup(op_name)  # pylint: disable=protected-access
      has_gradient = "\N{HEAVY CHECK MARK}\N{VARIATION SELECTOR-16}"
    except LookupError:
      has_gradient = "\N{CROSS MARK}"
    link = (
        '<a id={op_name} href="{FLAGS.site_path}/api_docs/python/tf/raw_ops">'
        '{op_name}</a>').format(op_name=op_name, FLAGS=FLAGS)
    parts.append(
        "| {link} | {has_gradient} |".format(link=link,
                                             has_gradient=has_gradient))

  return "\n".join(parts)


# The doc generator isn't aware of tf_export.
# So prefix the score tuples with -1 when this is the canonical name, +1
# otherwise. The generator chooses the name with the lowest score.
class TfExportAwareVisitor(doc_generator_visitor.DocGeneratorVisitor):
  """A `tf_export`, `keras_export` and `estimator_export` aware doc_visitor."""

  def _score_name(self, name):
    all_exports = [tf_export.TENSORFLOW_API_NAME, tf_export.ESTIMATOR_API_NAME]

    for api_name in all_exports:
      canonical = tf_export.get_canonical_name_for_symbol(
          self._index[name], api_name=api_name)
      if canonical is not None:
        break

    canonical_score = 1
    if canonical is not None and name == "tf." + canonical:
      canonical_score = -1

    scores = super()._score_name(name)
    return (canonical_score,) + scores


def _hide_layer_and_module_methods():
  """Hide methods and properties defined in the base classes of keras layers."""
  # __dict__ only sees attributes defined in *this* class, not on parent classes
  module_contents = list(tf.Module.__dict__.items())
  layer_contents = list(tf.keras.layers.Layer.__dict__.items())

  for name, obj in module_contents + layer_contents:
    if name == "__init__":
      continue

    if isinstance(obj, property):
      obj = obj.fget

    if isinstance(obj, (staticmethod, classmethod)):
      obj = obj.__func__

    try:
      doc_controls.do_not_doc_in_subclasses(obj)
    except AttributeError:
      pass


def build_docs(output_dir, code_url_prefix, search_hints=True):
  """Build api docs for tensorflow v2.

  Args:
    output_dir: A string path, where to put the files.
    code_url_prefix: prefix for "Defined in" links.
    search_hints: Bool. Include meta-data search hints at the top of each file.
  """
  # The custom page will be used for raw_ops.md not the one generated above.
  doc_controls.set_custom_page_content(tf.raw_ops, generate_raw_ops_doc())

  _hide_layer_and_module_methods()

  try:
    doc_controls.do_not_generate_docs(tf.tools)
  except AttributeError:
    pass

  try:
    doc_controls.do_not_generate_docs(tf.compat.v1.pywrap_tensorflow)
  except AttributeError:
    pass

  try:
    doc_controls.do_not_generate_docs(tf.pywrap_tensorflow)
  except AttributeError:
    pass

  try:
    doc_controls.do_not_generate_docs(tf.flags)
  except AttributeError:
    pass

  base_dir = path.normpath(path.join(tf.__file__, "../.."))

  base_dirs = (
      path.join(base_dir, "tensorflow_core"),
      # External packages base directories
      path.dirname(tensorboard.__file__),
      path.dirname(tensorflow_estimator.__file__),
  )

  code_url_prefixes = (
      code_url_prefix,
      # External packages source repositories,
      "https://github.com/tensorflow/tensorboard/tree/master/tensorboard",
      "https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator",
  )

  doc_generator = generate_lib.DocGenerator(
      root_title="TensorFlow 2",
      py_modules=[("tf", tf)],
      base_dir=base_dirs,
      search_hints=search_hints,
      code_url_prefix=code_url_prefixes,
      site_path=FLAGS.site_path,
      visitor_cls=TfExportAwareVisitor,
      private_map=_PRIVATE_MAP)

  doc_generator.build(output_dir)


def main(argv):
  del argv
  build_docs(
      output_dir=FLAGS.output_dir,
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints)


if __name__ == "__main__":
  app.run(main)
