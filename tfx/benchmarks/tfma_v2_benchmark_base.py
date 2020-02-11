# Copyright 2020 Google LLC. All Rights Reserved.
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
"""TFMA v2 benchmark."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

# Standard Imports

from absl import flags
import apache_beam as beam
from apache_beam.runners.portability import fn_api_runner

import tensorflow.compat.v2 as tf
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis import model_util as tfma_model_util
from tensorflow_model_analysis import types as tfma_types
from tensorflow_model_analysis.evaluators import metrics_and_plots_evaluator_v2
from tensorflow_model_analysis.extractors import input_extractor
from tensorflow_model_analysis.extractors import predict_extractor_v2
from tensorflow_model_analysis.metrics import metric_specs

import tfx
from tensorflow.python.platform import test  # pylint: disable=g-direct-tensorflow-import

FLAGS = flags.FLAGS

# Maximum number of examples to read from the dataset.
# TFMA is much slower than TFT, so we may have to read a smaller subset of the
# dataset.
MAX_NUM_EXAMPLES = 100000


# TODO(b/147827582): Also add "TF-level" Keras benchmarks for how TFMAv2
# gets predictions / computes metrics.
class TFMAV2BenchmarkBase(test.Benchmark):
  """TFMA benchmark."""

  def __init__(self, dataset):
    super(TFMAV2BenchmarkBase, self).__init__()
    self._dataset = dataset

    self._eval_config = tfma.config.EvalConfig(
        model_specs=[tfma.config.ModelSpec(label_key="tips")],
        metrics_specs=metric_specs.example_count_specs())
    self._model_loader = tfma_types.ModelLoader(
        tags=[tf.saved_model.SERVING],
        construct_fn=tfma_model_util.model_construct_fn(
            eval_saved_model_path=self._dataset.trained_saved_model_path(),
            tags=[tf.saved_model.SERVING]))

  def report_benchmark(self, **kwargs):
    if "extras" not in kwargs:
      kwargs["extras"] = {}
    # Note that the GIT_COMMIT_ID is not included in the packages themselves:
    # it must be injected by an external script.
    kwargs["extras"]["commit_tfx"] = getattr(tfx, "GIT_COMMIT_ID",
                                             tfx.__version__)
    kwargs["extras"]["commit_tfma"] = getattr(tfma, "GIT_COMMIT_ID",
                                              tfma.__version__)
    super(TFMAV2BenchmarkBase, self).report_benchmark(**kwargs)

  def benchmarkMiniPipeline(self):
    """Benchmark a "mini" version of TFMA - predict, slice and compute metrics.

    Runs a "mini" version of TFMA in a Beam pipeline. Records the wall time
    taken for the whole pipeline.
    """
    pipeline = beam.Pipeline(runner=fn_api_runner.FnApiRunner())
    raw_data = (
        pipeline
        | "Examples" >> beam.Create(
            self._dataset.read_raw_dataset(
                deserialize=False, limit=MAX_NUM_EXAMPLES))
        | "InputsToExtracts" >> tfma.InputsToExtracts())

    eval_shared_model = tfma_types.EvalSharedModel(
        model_path=self._dataset.trained_saved_model_path(),
        model_loader=self._model_loader)

    _ = (
        raw_data
        | "InputExtractor" >>
        input_extractor.InputExtractor(eval_config=self._eval_config).ptransform
        | "V2PredictExtractor" >> predict_extractor_v2.PredictExtractor(
            eval_config=self._eval_config,
            eval_shared_model=eval_shared_model).ptransform
        | "SliceKeyExtractor" >> tfma.extractors.SliceKeyExtractor().ptransform
        | "V2ComputeMetricsAndPlots" >>
        metrics_and_plots_evaluator_v2.MetricsAndPlotsEvaluator(
            eval_config=self._eval_config,
            eval_shared_model=eval_shared_model).ptransform)

    start = time.time()
    result = pipeline.run()
    result.wait_until_finish()
    end = time.time()
    delta = end - start

    self.report_benchmark(
        iters=1,
        wall_time=delta,
        extras={
            "num_examples": self._dataset.num_examples(limit=MAX_NUM_EXAMPLES)
        })
