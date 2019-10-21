# The ExampleValidator Component

The ExampleValidator component identifies different classes of anomalies in training and serving
data. For example, it
can:

1.  Perform validity checks by comparing data statistics against a schema that
    codifies expectations of the developer
1.  Detect training-serving skew by comparing training and serving
    data
1.  Detect data drift by looking at a series of data

The ExampleValidator pipeline component identifies any anomalies in the example data
by comparing data statistics computed by the StatisticsGen pipeline component against a
schema. The inferred schema codifies properties which the input data is expected to
satisfy, and can be modified by the developer.

* Consumes: A schema from a SchemaGen component and statistics from a StatisticsGen
component.
* Emits: Validation results

## ExampleValidator and TensorFlow Data Validation

ExampleValidator makes extensive use of the [TensorFlow Data Validation](tfdv.md) library
for validating your input data.

## Using the ExampleValidator Component

An ExampleValidator pipeline component is typically very easy to deploy and
requires little customization. Typical code looks like this:

```python
from tfx import components

...

validated_stats = components.ExampleValidator(
      statistics=compute_eval_stats.outputs['statistics'],
      schema=infer_schema.outputs['schema']
      )
```
