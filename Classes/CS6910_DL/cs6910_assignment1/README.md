# CS6910 Deep Learning - CS23E001 Shuvrajeet Das

Assignment 1:
The goal of this assignment is twofold:

- Implement and use gradient descent (and its variants) with backpropagation for a classification task
- Get familiar with wandb which is a cool tool for running and keeping track of a large number of experiments

## Python versions (including libraries):

```
Python - 3.11.6
Numpy - 1.26.3
Matplotlib - 3.8.2
Wandb - 0.16.2
```

## Setup:

```bash
python3 -m pip install numpy
python3 -m pip install matplotlib
python3 -m pip install wandb
```

## Project Structure:

The project contains the implementtaion of various algorithms for running a Deep Neural Network.

## Features:

### Activation Functions:

- [x] relu (Rectified Linear Unit Activation) $$f(x) = \max(0, x)$$
- [x] sigmoid (Sigmoid Activation) $$f(x) = \frac{1}{1 + e^{-x}} $$
- [x] tanh (Hyperbolic Tangent Activation) $$f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$
- [x] selu (Scaled Exponential Linear Unit Activation) $$f(x) = \text{scale} \times (\max(0, x) + \min(0, \alpha \times (\exp(x) - 1))) $$, where $\alpha \approx 1.6733$ and $\text{scale} \approx 1.0507$
- [x] gelu (Gaussian Error Linear Unit Activation) $$f(x) = x \times \Phi(x) $$, where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution
- [x] leaky_relu (Leaky Rectified Linear Unit Activation) $$f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{otherwise} \end{cases} $$, where $\alpha$ is a small positive constant
- [x] elu (Exponential Linear Unit Activation) $$f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha \times (e^x - 1) & \text{otherwise} \end{cases} $$, where $\alpha$ is a small positive constant
- [x] swish (Swish Activation) $$f(x) = x \times \sigma(\beta \times x) $$, where $\sigma(x)$ is the sigmoid function and $\beta$ is a constant
- [x] softplus (Softplus Activation) $$f(x) = \ln(1 + e^x) $$
- [x] mish (Mish Activation) $$f(x) = x \times \tanh(\ln(1 + e^x)) $$
- [x] softmax (Softmax Activation) $$f(x*i) = \frac{e^{x_i}}{\sum*{j=1}^{N} e^{x_j}} $$ for $ i = 1, 2, \ldots, N $

### Optimizers:

- [x] sgd (Stochastic Gradient Descent)
      $$ w_{t+1} = w_t - \eta \nabla L(w_t) $$

- [x] momentum*gd (Momentum Gradient Descent)
      $$ v_{t+1} = \gamma v_t + \eta \nabla L(w_t) $$
	    $$ w_{t+1} = w_t - v_{t+1} $$

- [x] nesterov*gd (Nesterov Accelerated Gradient)
      $$ v_{t+1} = \gamma v_t + \eta \nabla L(w_t - \gamma v_t) $$
      $$ w_{t+1} = w_t - v_{t+1} $$

- [x] adagrad (Adaptive Gradient Algorithm)
      $$ G_{t+1} = G_t + (\nabla L(w_t))^2 $$
      $$ w_{t+1} = w_t - \frac{\eta}{\sqrt{G_{t+1}} + \epsilon} \nabla L(w_t) $$

- [x] rmsprop (Root Mean Square Propagation)
      $$ G_{t+1} = \beta G_t + (1 - \beta) (\nabla L(w_t))^2 $$
      $$ w_{t+1} = w_t - \frac{\eta}{\sqrt{G_{t+1}} + \epsilon} \nabla L(w_t) $$

- [x] adadelta (Adaptive Delta)
      $$ G_{t+1} = \beta G_t + (1 - \beta) (\nabla L(w_t))^2 $$
      $$ \Delta w_{t+1} = - \frac{\sqrt{\Delta w_t + \epsilon}}{\sqrt{G_{t+1}} + \epsilon} \nabla L(w_t) $$
      $$ w_{t+1} = w_t + \Delta w_{t+1} $$

- [x] adam (Adaptive Moment Estimation)
      $$ m_{t+1} = \beta_1 m_t + (1 - \beta_1) \nabla L(w_t) $$
      $$ v_{t+1} = \beta_2 v_t + (1 - \beta_2) (\nabla L(w_t))^2 $$
      $$ \hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}} $$
      $$ \hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}} $$
      $$ w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \hat{m}\_{t+1} $$

- [x] adamax (Adam with Infinity Norm)
      $$ m_{t+1} = \beta_1 m_t + (1 - \beta_1) \nabla L(w_t) $$
      $$ u_{t+1} = \max(\beta_2 u_t, |\nabla L(w_t)|) $$
      $$ w_{t+1} = w_t - \frac{\eta}{u_{t+1}} m\_{t+1} $$

- [x] nadam (Nesterov Adam)
      $$ m_{t+1} = \beta_1 m_t + (1 - \beta_1) \nabla L(w_t) $$
      $$ v_{t+1} = \beta_2 v_t + (1 - \beta_2) (\nabla L(w_t))^2 $$
      $$ \hat{m}_{t+1} = \frac{\beta_1 m_{t+1} + (1 - \beta_1) \nabla L(w_t)}{1 - \beta_1^{t+1}} $$
      $$ \hat{v}_{t+1} = \frac{\beta_2 v_{t+1}}{1 - \beta_2^{t+1}} $$
      $$ w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \hat{m}\_{t+1} $$

### Weight Initializers:

- [x] random (Random Normal)
- [x] xavier_normal (Xavier Normal Initialization)
- [x] xavier_uniform (Xavier Uniform Initialization)
- [x] he_normal (He Normal Initialization)
- [x] he_uniform (He Uniform Initialization)

## Report:

The complete report, including experiment results and analysis, has been generated using Wandb. You can access it here.

## GitHub Repository:

The code for this project is available on GitHub. You can find it here.

Note: Please adhere to academic integrity policies and refrain from collaborating or discussing the assignment with other students.
