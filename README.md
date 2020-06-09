# MACE
Model Agnostic Concept based Explanations

This code is the reference implementation of the methods described in our paper Model Agnostic Concept based Explanations.

The codebase currently supports VGG and ResNet50 architectures. Extending support is fairly easy and straightforward. An example of the working of our approach can be seen in `mace.py`

## Usage
To use our approach for your image classifier. Simply create a custom model that feeds the outputs of the last convolution layer of the classifier into our mace unit. 

```python
class InterpretClassification(Model):
  
  def __init__(self, *args, **kwargs):
    self.base_model = your_model
    self.mace_unit = InterpretCNN(*args, **kwargs)
    ....
    
  def call(self, inputs):
    features = self.base_model.get_features()
    outputs = self.mace_unit(features)
    ....
    return outputs
    
````
.
