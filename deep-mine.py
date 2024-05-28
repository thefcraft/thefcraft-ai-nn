# import torch.nn as nn
import numpy as np

# GLOBAL VARIABLES
DEFAULT = None
# GPU = 'cuda'
CPU = 'cpu'
np.random.seed(42)

class AutoGrad:
    def __init__(self, function:callable, delta:int=1e-9) -> None:
        self.function = function
        self.delta = delta
    def __call__(self, x):
        return (self.function(x + self.delta) - self.function(x))/(self.delta)

class Tensor:
    def __init__(self, array:np.ndarray, _children=(), _op='', label=''):
        self.shape = array.shape
        self.array = array
        self.grad = np.zeros(self.shape, dtype=np.float32)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    def __repr__(self) -> str:
        # return f"Value(shape={self.array.shape}, grad={self.grad.shape})"
        return self.array.__repr__()
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other if isinstance(other, np.ndarray) else other*np.ones_like(self.array))
        out = Tensor(self.array + other.array, (self, other), '+')
        def _backward():
            self.grad += np.ones_like(self.array) * out.grad
            if other.shape != out.shape:
                other.grad += (np.ones_like(other.array) * out.grad).sum(axis=0)
            else:
                other.grad += np.ones_like(other.array) * out.grad
            # self.grad += 1.0 * out.grad
            # other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    def __radd__(self, other):
        return self + other
    def __neg__(self): return self * (-1)
    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return other + (-self)
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other if isinstance(other, np.ndarray) else  other*np.ones_like(self.array))
        out = Tensor(self.array * other.array, (self, other), '*')
        def _backward():
            self.grad += other.array * out.grad
            other.grad += self.array * out.grad
        out._backward = _backward
        return out
    def __rmul__(self, other): return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supports int/float powers for now"
        out = Tensor(self.array ** other, (self, ), f'**{other}')
        def _backward():
            self.grad += other * (self.array ** (other-1)) * out.grad
        out._backward = _backward
        return out
    
    def dot(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        assert len(other.shape) == 1, "support only one dimension use mm instead"
        ...
    def mm(self, other):
        assert len(other.shape) == 2, "not support higher dimensions then matrix"
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.array.dot(other.array), (self, other), 'mm')
        def _backward():
            # self.grad += np.tile(other.array.sum(axis=-1), (self.shape[-2], 1))# * out.grad
            self.grad += np.dot(out.grad, other.array.T)
            # other.grad += np.tile(self.array.sum(axis=-2), (other.shape[-1], 1)).T #* out.grad
            other.grad = np.dot(self.array.T, out.grad)
        out._backward = _backward
        return out
    # def matmul
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited: 
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = np.ones(self.shape)
        for node in reversed(topo):
            node._backward()
    def exp(self):
        x = self.array
        out = Tensor(np.exp(x), (self, ), 'exp')
        def _backward():
            self.grad += out.array * out.grad
        out._backward = _backward
        return out
    def mean(self, axis=0):
        x = self.array
        out = Tensor(x.mean(axis=axis), (self, ), 'mean')
        def _backward():
            self.grad += out.grad/self.shape[0]
        out._backward = _backward
        return out
    def sum(self, axis=0):
        x = self.array
        out = Tensor(x.sum(axis=axis), (self, ), 'sum')
        def _backward():
            self.grad += out.grad
        out._backward = _backward
        return out
    def step(self, lr): 
        lr = np.array([lr], dtype=np.float32)
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited: 
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        for node in reversed(topo):
            node.array -= lr * node.grad
    def zero_grad(self): 
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited: 
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        for node in reversed(topo):
            node.grad *= 0
    def item(self):
        assert len(self.shape) == 1 and self.shape[0] == 1
        return self.array[0]
    
class Loss: ...
class MSELoss:
    def __init__(self):
        ...
    def __call__(self, y:Tensor, y_real:Tensor, axis=0):
        return ((y - y_real)**2).mean(axis=axis)

class Optimizer: ...
class Adam: ...
class SGD: ... # StochasticGradDescent

class Activation: ...
class ReLU(Activation): ...

class Sigmoid(Activation): 
    def __init__(self): 
        self.grad = None
    def __call__(self, x:Tensor):
        return self.forward(x)
    def forward(self, x:Tensor):
        self.grad = self.backward(x)
        # return 1/(1+np.exp(-x))
        return (Tensor(np.ones_like(x.array))+(-x).exp())**(-1)
    
    def backward(self, x):
        ...
        # return x * (1-x)

class Module:
    def __init__(self): ...
    def __call__(self): ...

class Linear:
    def __init__(self, in_features, out_features, bias=True, device=DEFAULT, dtype=DEFAULT):
        self.dtype = dtype if dtype is not None else np.float32
        self.device = device if device is not None else CPU
        
        self.W = Tensor(np.random.uniform(-1, 1, (in_features, out_features)).astype(self.dtype)) # no need to transpose the tensors as i am initialized here with Transpose shape
        # self.W = np.random.random((in_features, out_features)).astype(self.dtype)
        self.B = Tensor(np.zeros(out_features, dtype=self.dtype))
    def __call__(self, X:Tensor):
        return self.forward(X)
    def forward(self, x:Tensor):
        return x.mm(self.W) + self.B
        # return np.dot(x, self.W) + self.B
    def backward(self, x):
        ...



class TestModel:
    def __init__(self) -> None:
        self.fc1 = Linear(30, 10)
        self.fn1 = Sigmoid()
        self.fc2 = Linear(10, 1)
        self.fn2 = Sigmoid()
    def __call__(self, X):
        return self.forward(X)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fn1(x)
        x = self.fc2(x)
        # return x
        return self.fn2(x)
    def backward(self, loss): 
        print("backward...")
        output_delta = self.fn2.grad * loss
        
        
        


# X = Tensor(np.random.randint(0, 10, (16, 2)).astype(np.float32))
# y_real = Tensor(np.random.randint(0, 10, (16, 1)).astype(np.float32))

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load real data (Breast Cancer Wisconsin dataset)
data = load_breast_cancer()
X, y_real = data.data.astype(np.float32), data.target.astype(np.float32)
y_real = y_real.reshape(-1, 1)

# Step 2: Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_real, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = Tensor(X_train), Tensor(X_test), Tensor(y_train), Tensor(y_test)

model = TestModel()
criterion = MSELoss()
for epoch in range(5000):
    y = model(X_train)    
    loss = criterion(y, y_train)
    if epoch % 100 == 0:
        print(f"EPOCH: {epoch} | LOSS: {loss.item():.4f}")
    loss.backward()
    loss.step(0.003)
    loss.zero_grad()

    
# Step 5: Evaluate model
outputs = model(X_test)
preds = outputs.array

accuracy = (np.round(preds) == y_test.array).mean()
print(f'Accuracy on test data: {accuracy:.4f}')





























# def fn(x):
#     return -7*(x**2)+9

# def realdfn(x):
#     return -7*(x)*2

# dfn = AutoGrad(fn)

# print(fn(3))
# print(dfn(3))
# print(realdfn(3))









# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn

# # Generate dataset
# x_values = np.random.uniform(-5, 5, 100)  # Random x values between -5 and 5
# y_values = fn(x_values)  # Corresponding y values

# # Normalize input data to the range [-1, 1]
# x_values_normalized = 2 * (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values)) - 1
# denormalize_min = np.min(y_values)
# denormalize_max = np.max(y_values) 
# y_values_normalized = 2 * (y_values - denormalize_min) / (denormalize_max - denormalize_min) - 1

# # Convert to PyTorch tensors
# x_values_tensor = torch.tensor(x_values_normalized, dtype=torch.float32).reshape(-1, 1)
# y_values_tensor = torch.tensor(y_values_normalized, dtype=torch.float32).reshape(-1, 1)

# # Define the neural network model
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(1, 2)  # First hidden layer
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(2, 1)  # Second hidden layer

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu1(x)
#         x = self.fc2(x)
#         return x
# # Create model, loss function, and optimizer
# model = Net()
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# # Training loop
# num_epochs = 10000
# for epoch in range(num_epochs):
#     # Forward pass
#     outputs = model(x_values_tensor)
#     loss = criterion(outputs, y_values_tensor)

#     # Backward and optimize
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if (epoch+1) % 100 == 0:
#         print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# # Plot the original function and the model predictions
# with torch.no_grad():
#     predicted = model(x_values_tensor).numpy().reshape(-1)
#     predicted = ((predicted + 1) * (denormalize_max - denormalize_min) / 2) + denormalize_min

# plt.scatter(x_values, y_values, color='red', label='Original data')
# plt.scatter(x_values, predicted, color='green', label='Model predictions')
# plt.title('Model approximation of the function')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)
# plt.show()