# Implementation of different ML models
This repository consists of several Implementations of machine learning algorithms without using any external libraries such as scikit-learn, keras, pytorch, etc.

## Requirements
- Python 3.7.4
- numpy 1.17.2
- scipy 1.3.1

## Currently implemented algorithms
- Logistic Regression
- Decision Tree
- Random Forest
- Fully Connected Neural Network

(as of 10/17/19)

I will try to implement other algorithms in the near future.

## Files
### 1. Non-neural networks
Implemented in simple_model.py. All algorithms are implemented as classes.
#### 1.1 Logistic Regression
Logistic Regression using stochastic gradient descent. Call LogisticRegression() to make an instance. Some parameters are
- lr_rate: learning rate
- lr_decay: learning rate decay
- penalty: loss penalization method (l1, l2, or none)
- etc.

Examples:
```python
hyperparams = {
  "lr_rate": 0.01,
  "penalty": "l2"
}
logreg = LogisticRegression(lr_rate=hyperparams["lr_rate"], penalty=hyperparams["penalty"])
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
```

#### 1.2 Decision Tree
Decision Tree implementation. Tree size as hyperparameter.

Examples:
```python
clf = DecisionTree(tree_size=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

#### 1.3 Random Forest
Random forest tree classifier. Mainly based on decision tree implementation above.

### 2. Neural networks
#### 2.1 solver.py
This is the solver class to solve any neural networks implemented in the classifiers folder. You will first need to make an instance of your model and then pass it to the Solver.

#### 2.2 optim.py
Has the functions for gradient descent. Implementations so far are,
- sgd: stochastic gradient descent
- rmsprop: RMSprop
- adam: Adam

#### 2.3 layers.py
Defining what each layer in the network takes in and outputs. Layers include affine layer, ReLu layer, and the softmax loss layer.

#### 2.4 network.py (in classfiers folder)
Implementation of the fully-connected network. Using the layers defined in layers.py, it will initialize a neural network of any number of hidden layers and size.

Examples:
```python
data = {
  "X_train": X_train,
  "y_train": y_train,
  "X_test": X_test,
  "y_test": y_test
}

hyperparams = {'lr_decay': 0.97,
               'num_epochs': 50,
               'batch_size': 200,
               'learning_rate': 1e-3
              }

hidden_dim = [128, 64] # this should be a list of units for each hiddent layer
model = FullyConnectedNet(input_dim=784, hidden_dim=hidden_dim, num_classes=10)
solver = Solver(model, lr_rate=hyperparams["learning_rate"], batch_size=hyperparams["batch_size"],
                num_epochs=hyperparams["num_epochs"], lr_decay=hyperparams["lr_decay"])

solver.train(X_train, y_train)
```

#### 2.5 gradient_check.py and test_grad.py
Some test files to check if all layers are defined correctly.
