# The ModelValidator Component

The ModelValidator component helps you validate your exported models.
ModelValidator ensures that they are good enough to be pushed to production.

ModelValidator compares new models against a baseline (Such as the currently serving
model) to determine if they're good enough relative to the baseline. It does so by
evaluating both models on a dataset (e.g. Holdout data, or a golden data set) and computing
their performance on metrics (e.g. AUC, loss, etc.) If the new model's metrics meet user-specified
criteria relative to the baseline model (e.g. AUC is not lower), the model is "blessed"
(marked as good), indicating to the [Pusher](pusher.md) that it is ok to push the model
to production.

*   Consumes: A schema from a SchemaGen component, and statistics from a
    StatisticsGen component.
*   Output: Validation results are written to [TensorFlow Metadata](mlmd.md).

## Using the ModelValidator Component

An ModelValidator component is very easy to deploy and
requires little customization. Typical code looks like this:

```python
from tfx import components
import tensorflow_model_analysis as tfma

...

# For model validation
taxi_mv_spec = [tfma.SingleSliceSpec()]

model_validator = components.ModelValidator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'])
```
