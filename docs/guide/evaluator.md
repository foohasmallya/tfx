# The Evaluator Component

The Evaluator component performs deep analysis on the training
results of your models. This analysis helps you understand how your model performs on
different subsets of your dataset.

*   Consumes: An `EvalSavedModel` from a [Trainer](trainer.md) component
*   Emits: Analysis results (Which are written to [TensorFlow Metadata](mlmd.md))

## Evaluator and TensorFlow Model Analysis

Evaluator leverages the [TensorFlow Model Analysis](tfma.md) library to perform
the analysis. TFMA, in turn, uses [Apache Beam](beam.md) for scalable processing.

## Using the Evaluator Component

A Evaluator component is very easy to deploy and requires little
customization.
Typical code looks like this:

```python
from tfx import components
import tensorflow_model_analysis as tfma

...

# For TFMA evaluation
taxi_eval_spec = [
    tfma.SingleSliceSpec(),
    tfma.SingleSliceSpec(columns=['trip_start_hour'])
]

model_analyzer = components.Evaluator(
      examples=examples_gen.outputs['examples'],
      feature_slicing_spec=taxi_eval_spec,
      model_exports=trainer.outputs['model']
      )
```
