# C++ Autograd Engine

This is a single-file reverse-mode automatic differentiation engine written in C++. It builds a dynamic computation graph at runtime to perform backpropagation. It includes the autograd core, neural network primitives (MLP), and an SGD optimizer.

## Technical Implementation

The engine manages the computation graph using `std::shared_ptr` for all nodes. This approach avoids manual memory management while preventing dangling pointers that typically occur in dynamic graphs due to temporary objects. By inheriting from `std::enable_shared_from_this`, each node is able to safely capture a self-reference within its backward pass lambda, ensuring the node persists until the entire gradient accumulation is finished. 
Nodes are instantiated exclusively through the static `Value::make()` method.

## Usage

### 1. Dynamic Graph and Backpropagation

Use `V(x)` to wrap doubles into graph nodes. The engine overloads arithmetic operators, allowing for natural mathematical syntax.

```cpp
SP a = V(3.0);
SP b = V(4.0);

// Construct graph
SP f = (a + b) * V(2.0); 

// Run backprop
f->backward(); 

std::cout << "df/da: " << a->grad << "\n"; // Outputs 2.0
```

Supported Operations: `+`, `-`, `*`, `/`, `pow(exp)`, `tanh()`, `relu()`.

### 2. Neural Network Modules

- `Neuron`: Implements a dot product of inputs and weights plus bias, with an optional non-linearity.
- `Layer`: A collection of neurons operating on the same input vector.
- `MLP`: A stacked architecture of layers forming a multi-layer perceptron.

### 3. Training

The `SGD` optimizer is used to update parameters based on accumulated gradients.

```cpp
SGD optim(model.params(), 0.05);

// Standard loop
optim.zero_grad();
loss->backward();
optim.step();
```

## Compilation

The engine is contained entirely in `main.cpp`. No external dependencies are required.

```bash
g++ -O3 -std=c++11 main.cpp -o autograd
./autograd
```