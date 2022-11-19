# TorchScript: migration from python host to cpp host

## Research part in python

### Define the neural net

```python
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 1)
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain("tanh"))
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain("tanh"))
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return x
```

### Define miscellaneous settings

```python
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
loss_func = nn.MSELoss(reduction="sum")
```



### Training the neural net

```python
x = x[..., None].to(device)
yn = yn[..., None].to(device)
train = False
if train:
  model.train()
  while True:
## Extract the graph of the model
      ypred = model(x)
      loss = loss_func(ypred, yn)
      optimizer.zero_grad()
      loss.backward()
      print(loss)
      optimizer.step()
      if loss.item() < 2.5e-1:
          break
  torch.save(model.state_dict(), r'train.pkl')
```

## Extract the graph of the model

If the neural network has a static graph, pytorch just needs to pass a input
through the net to know the graph. Otherwise, we use functionalities provided by
pytorch to handle dynamic graphes (containing if else, while, ...)

```python
script = torch.jit.trace(model, (x)) # for static graphs
script = torch.jit.script(model) # for dynamic graphs
```

### Save the graph to disk

```python
script.save("model_graph.pt") # save graph to disk so we can access this graph from libtorch cpp API
```

## Migrate to production for deployment

### Access model from CPP: example

```cpp
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    /**
    * Load the model
    */
    module = torch::jit::load(argv[1]);
    /**
    * Construct the inputs
    */
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::linspace(-0.25, 0.75, 10).reshape({-1, 1}));
    std::cout << "output:" << inputs[0] << std::endl;
    /**
    * Forward pass through the model
    */
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << "output:" << output << std::endl;
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

}
```


## Output

### Python

```
input: tensor([[-0.2500],
        [-0.1389],
        [-0.0278],
        [ 0.0833],
        [ 0.1944],
        [ 0.3056],
        [ 0.4167],
        [ 0.5278],
        [ 0.6389],
        [ 0.7500]])
output: tensor([[3.2891],
        [3.1674],
        [3.0473],
        [2.9433],
        [2.8749],
        [2.8628],
        [2.9192],
        [3.0372],
        [3.1870],
        [3.3278]], grad_fn=<AddmmBackward0>)
```
### Cpp

```
output:-0.2500
-0.1389
-0.0278
 0.0833
 0.1944
 0.3056
 0.4167
 0.5278
 0.6389
 0.7500
[ CPUFloatType{10,1} ]
output: 3.2891
 3.1674
 3.0473
 2.9433
 2.8749
 2.8628
 2.9192
 3.0372
 3.1870
 3.3278
[ CPUFloatType{10,1} ]
```
