#!/usr/bin/env python
# coding: utf-8

# # Homework 4: Neural Networks for Recognition, Detection, and Tracking
# 
# #### **For each question please refer to the handout for more details.**
# 
# Programming questions begin at **Q2**. **Remember to run all cells** and save the notebook to your local machine as a pdf for gradescope submission.
# 

# # Collaborators
# **List your collaborators for all questions here**:
# 
# 
# ---

# # Q1 Theory
# 

# ## Q1.1 (3 points)
# 
# 
# Softmax is defined as below, for each index $i$ in a vector $x \in \mathbb{R}^d$.
# $$ softmax(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} $$
# Prove that softmax is invariant to translation, that is
# $$ softmax(x) = softmax(x + c) \quad \forall c \in \mathbb{R}.$$    
# Often we use $c = -\max x_i$. Why is that a good idea? (Tip: consider the range of values that numerator will have with $c=0$ and $c = -\max x_i$)

# ---
# 
# ### Proof of Translation Invariance
# 
# The softmax function for a vector $ x \in \mathbb{R}^d $ is defined as:
# $$
# \text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j} e^{x_j}}.
# $$
# 
# Let $ c \in \mathbb{R} $ be a scalar, and consider $ x + c $, which represents adding $ c $ to every component of $ x $. The softmax of $ x + c $ is:
# 
# $$
# \text{softmax}(x + c)_i = \frac{e^{(x_i + c)}}{\sum_{j} e^{(x_j + c)}}.
# $$
# 
# The numerator becomes:
# $$
# e^{(x_i + c)} = e^{x_i} e^c,
# $$
# and the denominator becomes:
# $$
# \sum_{j} e^{(x_j + c)} = \sum_{j} e^{x_j} e^c = e^c \sum_{j} e^{x_j}.
# $$
# 
# Substituting back into the softmax formula:
# $$
# \text{softmax}(x + c)_i = \frac{e^{x_i} e^c}{e^c \sum_{j} e^{x_j}} = \frac{e^{x_i}}{\sum_{j} e^{x_j}} = \text{softmax}(x)_i.
# $$
# 
# Thus, the softmax function is invariant to translation:
# $$
# \text{softmax}(x) = \text{softmax}(x + c), \quad \forall c \in \mathbb{R}.
# $$
# 
# ---
# 
# ### Why Choose $ c = -\max x_i $?
# 
# When computing the softmax, we often choose $ c = -\max x_i $. This ensures numerical stability in computation. Here's why:
# 
# 1. **Numerical Stability**: 
#    - Without adjusting $ x $, the values $ e^{x_i} $ can grow extremely large for large $ x_i $, leading to potential overflow in floating-point operations.
#    - By setting $ c = -\max x_i $, the largest value of $ x_i + c $ becomes 0. Hence, the range of $ x_i + c $ is shifted such that the largest term is $ e^0 = 1 $, and all other terms are less than or equal to 1. This prevents overflow and ensures better numerical precision.
# 
# 2. **Reducing Magnitude of Terms**:
#    - When $ c = 0 $, the numerator $ e^{x_i} $ and the denominator $ \sum_j e^{x_j} $ can have large values, making it harder for computers to handle.
#    - With $ c = -\max x_i $, the numerator and denominator are rescaled to smaller, manageable values while maintaining the same relative proportions.
#    - Another option instead of setting $ c = -\max x_i $ is to set $ c = \sum_i x_i/n $ which could perform the task of preventing the blow up of $e^x$
# 
# ---

# ## Q1.2
# 
# Softmax can be written as a three-step process, with $s_i = e^{x_i}$, $S = \sum s_i$ and $softmax(x)_i = \frac{1}{S} s_i$.

# ### Q1.2.1 (1 point)
# 
# As $x \in \mathbb{R}^d$, what are the properties of $softmax(x)$, namely what is the range of each element? What is the sum over all elements?

# ### Properties of $ \text{softmax}(x) $
# 
# Given the three-step process for softmax:
# 1. $ s_i = e^{x_i} $,
# 2. $ S = \sum_{i=1}^d s_i = \sum_{i=1}^d e^{x_i} $,
# 3. $ \text{softmax}(x)_i = \frac{s_i}{S} = \frac{e^{x_i}}{\sum_{j=1}^d e^{x_j}} $,
# 
# we can derive the following properties of $ \text{softmax}(x) $:
# 
# #### 1. Range of Each Element
# Each element of $ \text{softmax}(x) $, denoted as $ \text{softmax}(x)_i $, satisfies:
# $$
# \text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^d e^{x_j}}.
# $$
# Since $ e^{x_i} > 0 $ for all $ x_i $, and the denominator $ \sum_{j=1}^d e^{x_j} > 0 $, it follows that:
# $$
# 0 < \text{softmax}(x)_i < 1, \quad \forall i.
# $$
# 
# #### 2. Sum Over All Elements
# The sum of all elements in $ \text{softmax}(x) $ is:
# $$
# \sum_{i=1}^d \text{softmax}(x)_i = \sum_{i=1}^d \frac{e^{x_i}}{\sum_{j=1}^d e^{x_j}}.
# $$
# Simplifying:
# $$
# \sum_{i=1}^d \text{softmax}(x)_i = \frac{\sum_{i=1}^d e^{x_i}}{\sum_{j=1}^d e^{x_j}} = 1.
# $$
# 

# ### Q1.2.2 (1 point)
# 
# One could say that $\textit{"softmax takes an arbitrary real valued vector $x$ and turns it into a $\rule{3cm}{0.1mm}$''}$.

# ---
# 
# Discrete Distribution
# 
# ---

# ### Q1.2.3 (1 point)
# 
# Now explain the role of each step in the multi-step process.

# ---
# 
# ### Role of Each Step in the Multi-Step Softmax Process
# 
# The softmax function can be computed in three steps:
# 
# 1. **Exponentiation: $s_i = e^{x_i}$**
#    - **Purpose**: This step transforms the input vector $x$ into the exponential space, ensuring that all values are positive.
#    - **Effect**: Larger values of $x_i$ result in exponentially larger values of $s_i$, which magnifies differences between elements of $x$. This step ensures that larger input values have a proportionally higher contribution to the final probabilities.
# 
# 2. **Summation: $S = \sum_{i=1}^d s_i$**
#    - **Purpose**: This computes the total sum of the exponentiated values. It acts as a normalization factor in the next step.
#    - **Effect**: The summation ensures that the final probabilities (computed in the next step) will sum to 1. This is crucial for interpreting the output of the softmax function as a probability distribution.
# 
# 3. **Normalization: $\text{softmax}(x)_i = \frac{s_i}{S}$**
#    - **Purpose**: Dividing each $s_i$ by $S$ converts the exponentiated values into probabilities.
#    - **Effect**: Each value is scaled to lie between 0 and 1, and the total sum of the elements is exactly 1. This makes the output interpretable as a categorical probability distribution.
# 
# 
# ---

# ## Q1.3 (3 points)
# 
# Show that multi-layer neural networks without a non-linear activation function are equivalent to linear regression.
# 
# 
# 

# ---
# 
# ### Showing that Multi-Layer Neural Networks Without Non-Linear Activation Are Equivalent to Linear Regression
# 
# A multi-layer neural network consists of layers that transform input data $ x $ into output predictions. Each layer typically performs a linear transformation followed by a non-linear activation. Without the non-linear activation, the network reduces to a single linear transformation.
# 
# #### Multi-Layer Neural Network Without Non-Linearity
# 
# Suppose we have a neural network with $ L $ layers, each represented as:
# $$
# h^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}, \quad l = 1, 2, \dots, L,
# $$
# where:
# - $ W^{(l)} $ and $ b^{(l)} $ are the weights and biases of the $ l $-th layer,
# - $ h^{(l-1)} $ is the input to the $ l $-th layer,
# - $ h^{(l)} $ is the output of the $ l $-th layer,
# - For the first layer, $ h^{(0)} = x $, the input to the network.
# 
# If there are no non-linear activation functions, the output of the $ L $-th layer is:
# $$
# y = W^{(L)} (W^{(L-1)} \dots (W^{(1)} x + b^{(1)}) + \dots + b^{(L-1)}) + b^{(L)}.
# $$
# 
# #### Simplifying the Multi-Layer Transformation
# 
# Let us simplify the nested linear transformations:
# 1. Expand the nested form:
#    $$
#    y = W^{(L)} W^{(L-1)} \dots W^{(1)} x + W^{(L)} W^{(L-1)} \dots b^{(1)} + \dots + b^{(L)}.
#    $$
# 2. Combine all the weights $ W^{(l)} $ into a single effective weight matrix $ W_{\text{eff}} = W^{(L)} W^{(L-1)} \dots W^{(1)} $.
# 3. Combine all the biases into a single effective bias vector $ b_{\text{eff}} $:
#    $$
#    b_{\text{eff}} = W^{(L)} W^{(L-1)} \dots b^{(1)} + \dots + b^{(L)}.
#    $$
# 
# Thus, the output simplifies to:
# $$
# y = W_{\text{eff}} x + b_{\text{eff}}.
# $$
# 
# 
# ---

# ## Q1.4 (3 points)
# 
# Given the sigmoid activation function $\sigma(x) = \frac{1}{1+e^{-x}}$, derive the gradient of the sigmoid function and show that it can be written as a function of $\sigma(x)$ (without having access to $x$ directly).

# ---
# ### Deriving the Gradient of the Sigmoid Function
# 
# The sigmoid activation function is defined as:
# $$
# \sigma(x) = \frac{1}{1 + e^{-x}}.
# $$
# 
# 
# The derivative is given as
# $$
# \frac{d}{dx} \sigma(x) = \frac{e^{-x}}{(1 + e^{-x})^2}.
# $$
# 
# Expressing  the Derivative in Terms of $ \sigma(x) $
# 
# From the definition of $ \sigma(x) $:
# $$
# \sigma(x) = \frac{1}{1 + e^{-x}}.
# $$
# 
# Rearranging:
# $$
# 1 + e^{-x} = \frac{1}{\sigma(x)}.
# $$
# 
# Thus:
# $$
# e^{-x} = \frac{1}{\sigma(x)} - 1 = \frac{1 - \sigma(x)}{\sigma(x)}.
# $$
# 
# Substitute this into the derivative:
# $$
# \frac{d}{dx} \sigma(x) = \frac{e^{-x}}{(1 + e^{-x})^2} = \frac{\frac{1 - \sigma(x)}{\sigma(x)}}{\left(\frac{1}{\sigma(x)}\right)^2}.
# $$
# 
# Simplify:
# $$
# \frac{d}{dx} \sigma(x) = \sigma(x)(1 - \sigma(x)).
# $$
# 
# #### Final Gradient Expression
# 
# The gradient of the sigmoid function can be written as:
# $$
# \frac{d}{dx} \sigma(x) = \sigma(x)(1 - \sigma(x)).
# $$
# 
# 
# ---

# ## Q1.5 (12 points)
# 
# Given $y = Wx + b$ (or $y_i = \sum_{j=1}^d  x_{j} W_{ij} + b_i$), and the gradient of some loss $J$ (a scalar) with respect to $y$, show how to get the gradients $\frac{\partial J}{\partial W}$, $\frac{\partial J}{\partial x}$ and $\frac{\partial J}{\partial b}$. Be sure to do the derivatives with scalars and re-form the matrix form afterwards. Here are some notional suggestions.
# $$ x \in \mathbb{R}^{d \times 1} \quad y \in \mathbb{R}^{k \times 1} \quad W \in \mathbb{R}^{k \times d} \quad b \in \mathbb{R}^{k \times 1} \quad \frac{\partial J}{\partial y} = \delta \in \mathbb{R}^{k \times 1} $$

# ---
# 
# ### Computing Gradients for $y = Wx + b$
# 
# We are given:
# - $y_i = \sum_{j=1}^d x_j W_{ij} + b_i$,
# - $J$: scalar loss function,
# - $\delta = \frac{\partial J}{\partial y} \in \mathbb{R}^{k \times 1}$.
# 
# We need to compute the gradients:
# 1. $\frac{\partial J}{\partial W}$
# 2. $\frac{\partial J}{\partial x}$
# 3. $\frac{\partial J}{\partial b}$.
# 
# 
# ### 1. Gradient of $J$ with Respect to $W$
# 
# Using the chain rule:
# $$
# \frac{\partial J}{\partial W_{ij}} = \frac{\partial J}{\partial y_i} \frac{\partial y_i}{\partial W_{ij}}.
# $$
# 
# From $y_i = \sum_{j=1}^d x_j W_{ij} + b_i$, we see that:
# $$
# \frac{\partial y_i}{\partial W_{ij}} = x_j.
# $$
# 
# Thus:
# $$
# \frac{\partial J}{\partial W_{ij}} = \delta_i x_j,
# $$
# where $\delta_i = \frac{\partial J}{\partial y_i}$.
# 
# In matrix form, this becomes:
# $$
# \frac{\partial J}{\partial W} = \delta x^T,
# $$
# where $\delta \in \mathbb{R}^{k \times 1}$ and $x^T \in \mathbb{R}^{1 \times d}$, so $\frac{\partial J}{\partial W} \in \mathbb{R}^{k \times d}$.
# 
# 
# ### 2. Gradient of $J$ with Respect to $x$
# 
# Using the chain rule:
# $$
# \frac{\partial J}{\partial x_j} = \sum_{i=1}^k \frac{\partial J}{\partial y_i} \frac{\partial y_i}{\partial x_j}.
# $$
# 
# From $y_i = \sum_{j=1}^d x_j W_{ij} + b_i$, we see that:
# $$
# \frac{\partial y_i}{\partial x_j} = W_{ij}.
# $$
# 
# Thus:
# $$
# \frac{\partial J}{\partial x_j} = \sum_{i=1}^k \delta_i W_{ij}.
# $$
# 
# In matrix form, this becomes:
# $$
# \frac{\partial J}{\partial x} = W^T \delta,
# $$
# where $W^T \in \mathbb{R}^{d \times k}$ and $\delta \in \mathbb{R}^{k \times 1}$, so $\frac{\partial J}{\partial x} \in \mathbb{R}^{d \times 1}$.
# 
# 
# ### 3. Gradient of $J$ with Respect to $b$
# 
# Using the chain rule:
# $$
# \frac{\partial J}{\partial b_i} = \frac{\partial J}{\partial y_i} \frac{\partial y_i}{\partial b_i}.
# $$
# 
# From $y_i = \sum_{j=1}^d x_j W_{ij} + b_i$, we see that:
# $$
# \frac{\partial y_i}{\partial b_i} = 1.
# $$
# 
# Thus:
# $$
# \frac{\partial J}{\partial b_i} = \delta_i.
# $$
# 
# In matrix form, this becomes:
# $$
# \frac{\partial J}{\partial b} = \delta,
# $$
# where $\delta \in \mathbb{R}^{k \times 1}$.
# 
# ---

# ## Q1.6
# When the neural network applies the elementwise activation function (such as sigmoid), the gradient of the activation function scales the backpropogation update. This is directly from the chain rule, $\frac{d}{d x} f(g(x)) = f'(g(x)) g'(x)$.

# ### Q1.6.1 (1 point)
# 
# Consider the sigmoid activation function for deep neural networks. Why might it lead to a "vanishing gradient" problem if it is used for many layers (consider plotting the gradient you derived in Q1.4)?

# ---
# ### The Vanishing Gradient Problem in Sigmoid Activation Function
# 
# #### Recap of the Sigmoid Gradient
# 
# From **Q1.4**, the gradient of the sigmoid activation function is:
# $$
# \sigma'(x) = \sigma(x)(1 - \sigma(x)).
# $$
# 
# The sigmoid function itself is defined as:
# $$
# \sigma(x) = \frac{1}{1 + e^{-x}}.
# $$
# 
# #### Behavior of the Sigmoid Gradient
# 
# 1. **Range of the Sigmoid Function**:
#    - $\sigma(x)$ outputs values in the range $(0, 1)$.
#    - As $x \to \infty$, $\sigma(x) \to 1$, and as $x \to -\infty$, $\sigma(x) \to 0$.
# 
# 2. **Gradient Magnitude**:
#    - The gradient $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ is maximized when $\sigma(x) = 0.5$, which happens when $x = 0$.
#    - The maximum gradient is:
#      $$
#      \sigma'(0) = 0.5(1 - 0.5) = 0.25.
#      $$
#    - For large positive or negative values of $x$, $\sigma'(x)$ approaches $0$ because $\sigma(x)$ approaches $0$ or $1$.
# 
# #### Impact on Backpropagation in Deep Networks
# 
# 1. **Scaling Effect**:
#    - The chain rule for backpropagation involves multiplying gradients layer by layer. At each layer with a sigmoid activation function, the gradient is scaled by $\sigma'(x)$.
#    - Since $\sigma'(x) \in (0, 0.25]$, the gradient is always less than or equal to $0.25^L$. 
# 
# 2. **Vanishing Gradient Across Many Layers**:
#    - In deep networks, gradients are propagated backward through many layers. If each layer scales the gradient by a factor less than $0.25$, the overall gradient diminishes exponentially as it propagates to earlier layers.
#    - This can cause the weights in earlier layers to receive extremely small updates, effectively "freezing" their learning.
# 
# 
# ---

# ### Q1.6.2 (1 point)
# Often it is replaced with $\tanh(x) = \frac{1-e^{-2x}}{1+e^{-2x}}$. What are the output ranges of both $\tanh$ and sigmoid? Why might we prefer $\tanh$?

# ---
# 
# ### Comparing $\tanh(x)$ and $\sigma(x)$
# 
# #### 1. Output Ranges
# 
# - **Sigmoid Function**:
#   $$ 
#   \sigma(x) = \frac{1}{1 + e^{-x}}
#   $$
#   - **Range**: $(0, 1)$.
#   - Outputs are always positive, centered around $0.5$ for small inputs.
# 
# - **Tanh Function**:
#   $$
#   \tanh(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}
#   $$
#   - **Range**: $(-1, 1)$.
#   - Outputs are symmetric about $0$.
# 
# #### 2. Why Prefer $\tanh$?
# 
# **Key Advantages of $\tanh$**:
# 1. **Centered Around Zero**:
#    - The $\tanh(x)$ function outputs values in the range $(-1, 1)$, making it zero-centered.
#    - This symmetry helps gradients to propagate more effectively during backpropagation because positive and negative inputs to a layer do not always result in positive activations (as is the case with sigmoid).
# 
# 2. **Larger Gradient for Mid-Range Inputs**:
#    - The gradient of $\tanh(x)$ is:
#      $$
#      \tanh'(x) = 1 - \tanh^2(x).
#      $$
#    - At $x = 0$, the gradient is maximized: $\tanh'(0) = 1$.
#    - Compared to sigmoid, $\tanh$ generally has larger gradients for the same input values, reducing the vanishing gradient issue.
# 
# 3. **Better for Deep Networks**:
#    - The zero-centered nature and larger gradient range make $\tanh$ preferable for deep networks, as it ensures faster convergence and reduces the risk of gradients vanishing in early layers.
# 
# #### Summary
# 
# - **Output Ranges**:
#   - Sigmoid: $(0, 1)$.
#   - Tanh: $(-1, 1)$.
# - **Preference for Tanh**:
#   - Zero-centered outputs help gradients propagate better.
#   - Larger gradients reduce the risk of vanishing gradients in deep networks.
# 
# ---

# ### Q1.6.3 (1 point)
# Why does $\tanh(x)$ have less of a vanishing gradient problem? (plotting the gradients helps! for reference: $\tanh'(x) = 1 - \tanh(x)^2$)

# ---
# 
# 
# Why Does $\tanh(x)$ Have Less of a Vanishing Gradient Problem?
# 
# The gradient of $\tanh(x)$ is given by:
# $$
# \tanh'(x) = 1 - \tanh^2(x).
# $$
# 
# Gradient Behavior of $\tanh(x)$
# 
# 1. **Maximum Gradient**:
#    - The gradient of $\tanh(x)$ is maximized at $x = 0$:
#      $$
#      \tanh'(0) = 1 - \tanh^2(0) = 1.
#      $$
# 
# 2. **Decay of the Gradient**:
#    - For large positive or negative $x$, $\tanh(x)$ saturates at $1$ or $-1$, making $\tanh'(x) \to 0$. However, the decay to zero is slower compared to the sigmoid function.
# 
# 3. **Comparison to Sigmoid**:
#    - For the sigmoid function, $\sigma'(x) = \sigma(x)(1 - \sigma(x))$, the gradient is always in the range $(0, 0.25]$, with a maximum at $\sigma(x) = 0.5$.
#    - For $\tanh(x)$, $\tanh'(x)$ is in the range $(0, 1]$, allowing larger gradients in the mid-range of input values.
# 
# 
# ---

# ### Q1.6.4 (1 point)
# $\tanh$ is a scaled and shifted version of the sigmoid. Show how $\tanh(x)$ can be written in terms of $\sigma(x)$.

# ---
# 
# ### $\tanh(x)$ as a Scaled and Shifted Version of $\sigma(x)$
# 
# The sigmoid function is defined as:
# $$
# \sigma(x) = \frac{1}{1 + e^{-x}}.
# $$
# 
# The $\tanh(x)$ function is defined as:
# $$
# \tanh(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}.
# $$
# 
# #### Step 1: Rewrite $\tanh(x)$ in Terms of $\sigma(x)$
# 
# Let us express $e^{-2x}$ in terms of $e^{-x}$. Note that:
# $$
# e^{-2x} = (e^{-x})^2.
# $$
# 
# Using the definition of $\sigma(x)$, we know:
# $$
# \sigma(x) = \frac{1}{1 + e^{-x}} \quad \text{so} \quad 1 - \sigma(x) = \frac{e^{-x}}{1 + e^{-x}}.
# $$
# 
# Thus:
# $$
# e^{-x} = \frac{1 - \sigma(x)}{\sigma(x)}.
# $$
# 
# Now substitute $e^{-x}$ into $\tanh(x)$:
# $$
# \tanh(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}.
# $$
# 
# Substitute $e^{-2x} = \left( \frac{1 - \sigma(x)}{\sigma(x)} \right)^2$:
# $$
# \tanh(x) = \frac{1 - \left( \frac{1 - \sigma(x)}{\sigma(x)} \right)^2}{1 + \left( \frac{1 - \sigma(x)}{\sigma(x)} \right)^2}.
# $$
# 
# #### Step 2: Simplify
# 
# Let $\sigma(x) = s$ for brevity. Then $1 - \sigma(x) = 1 - s$. Substituting:
# $$
# \tanh(x) = \frac{1 - \left( \frac{1 - s}{s} \right)^2}{1 + \left( \frac{1 - s}{s} \right)^2}.
# $$
# 
# Simplify the numerator and denominator. After simplifications, the result is:
# $$
# \tanh(x) = 2\sigma(2x) - 1.
# $$
# 
# #### Final Expression
# 
# The $\tanh(x)$ function can be written in terms of $\sigma(x)$ as:
# $$
# \tanh(x) = 2\sigma(2x) - 1.
# $$
# 
# 
# ---
# 

# # Q2 Implement a Fully Connected Network
# 
# Run the following code to import the modules you'll need. When implementing the functions in Q2, make sure you run the test code (provided after Q2.3) along the way to check if your implemented functions work as expected.

# In[1]:


import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid1 import ImageGrid

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation


# ## Q2.1 Network Initialization

# ### Q2.1.1 (3 points)
# 
# Why is it not a good idea to initialize a network with all zeros? If you imagine that every layer has weights and biases, what can a zero-initialized network output be after training?

# ---
# 
# ### Why Zero Initialization is a Bad Idea
# 
# #### 1. Symmetry Problem
# - If all weights are initialized to zero, all neurons in a layer produce identical outputs.
# - During backpropagation, all weights receive identical gradients and update the same way, preventing the network from learning complex patterns.
# 
# #### 2. Gradient Flow
# - Gradients for all neurons in a layer are identical due to the symmetric initialization, reinforcing the problem and halting effective learning.
# 
# ### Output of a Zero-Initialized Network After Training
# 
# If a network is initialized with all weights and biases set to zero:
# 
# 1. **Constant Output**:
#    - All neurons in a layer will produce identical outputs due to symmetric inputs and weights.
#    - The output of the network will likely be constant or a simple function that does not depend on the input.
# 
# 2. **Failure to Learn**:
#    - Backpropagation updates will be identical for all weights in a layer, resulting in no differentiation between neurons.
#    - The network cannot learn meaningful patterns and will likely output the same value for all inputs after training.
# 
# #### Conclusion
# A zero-initialized network will produce a constant or restricted output after training, failing to generalize or solve the given task effectively.
# 
# ---

# ### Q2.1.2 (3 points)
# 
# Implement the initialize_weights() function to initialize the weights for a single layer with Xavier initialization, where $Var[w] = \frac{2}{n_{in}+ n_{out}} $ where $n$ is the dimensionality of the vectors and you use a uniform distribution to sample random numbers (see eq 16 in [Glorot et al]).

# In[2]:


############################## Q 2.1.2 ##############################
def initialize_weights(in_size,out_size,params,name=''):
    """
    we will do XW + b, with the size of the input data array X being [number of examples, in_size]
    the weights W should be initialized as a 2D array
    the bias vector b should be initialized as a 1D array, not a 2D array with a singleton dimension
    the output of this layer should be in size [number of examples, out_size]
    """
    W, b = None, None

    ##########################
    # Xavier initialization variance
    var = 2 / (in_size + out_size)
    
    # Initialize weights from a uniform distribution
    limit = np.sqrt(var)
    W = np.random.uniform(-limit, limit, (in_size, out_size))
    
    # Initialize biases to zero
    b = np.zeros(out_size)
    ##########################


    params['W' + name] = W
    params['b' + name] = b


# ### Q2.1.3 (2 points)
# 
# Why do we scale the initialization depending on layer size (see Fig 6 in the [Glorot et al])?

# ---
# ### Why Scale Initialization Based on Layer Size?
# 
# #### Purpose of Scaling Initialization
# 
# - **Maintain Variance of Activations**: Scaling the initialization depending on layer size helps maintain the variance of the activations across layers.
# - **Prevent Vanishing or Exploding Activations**: Without appropriate scaling, activations can vanish (become too small) or explode (become too large) as they propagate through layers, hindering effective learning.
# 
# #### Xavier Initialization Scaling
# 
# - **Variance Calculation**: Xavier initialization sets the variance of the weights as:
#   $$
#   \text{Var}[w] = \frac{2}{n_{\text{in}} + n_{\text{out}}}
#   $$
#   where $ n_{\text{in}} $ and $ n_{\text{out}} $ are the number of input and output units of a layer, respectively.
#   
# - **Uniform Distribution Limits**: Weights are sampled from a uniform distribution within:
#   $$
#   w \sim \text{Uniform}\left( -\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}} \right)
#   $$
#   ensuring that the variance of activations remains consistent.
# 
# #### Explanation Using Figure 6 from Glorot et al.
# 
# - **Activation Variance Across Layers**: Figure 6 demonstrates how different initialization methods affect the variance of activations across layers.
# - **Unscaled Initialization**: Without scaling, the variance can increase or decrease exponentially with depth, leading to activation saturation or loss of signal.
# - **Scaled Initialization**: By scaling the weights based on layer size, Xavier initialization keeps the activation variance stable across layers, facilitating better learning.
# 
# #### Summary
# 
# - **Scaling Depends on Layer Size**: The scaling accounts for the number of input and output connections, balancing the flow of data through the network.
# - **Improved Training Dynamics**: Properly scaled initialization leads to more efficient training, faster convergence, and improved performance.
# ---

# ## Q2.2 Forward Propagation

# ### Q2.2.1 (4 points)
# 
# Implement the sigmoid() function, which computes the elementwise sigmoid activation of entries in an input array. Then implement the forward() function which computes forward propagation for a single layer, namely $y = \sigma(X W + b)$.

# In[3]:


############################## Q 2.2.1 ##############################
def sigmoid(x):
    """
    Implement an elementwise sigmoid activation function on the input x,
    where x is a numpy array of size [number of examples, number of output dimensions].
    """
    res = 1 / (1 + np.exp(-x))  # Compute the sigmoid activation function
    return res


# In[4]:


############################## Q 2.2.1 ##############################
def forward(X, params, name='', activation=sigmoid):
    """
    Do a forward pass for a single layer that computes the output: activation(XW + b)

    Keyword arguments:
    X -- input numpy array of size [number of examples, number of input dimensions]
    params -- a dictionary containing parameters, as how you initialized in Q 2.1.2
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    # compute the output values before and after the activation function
    pre_act = np.dot(X, params['W' + name]) + params['b' + name]  # Linear transformation
    post_act = activation(pre_act)  # Apply activation function

    # store the pre-activation and post-activation values
    # these will be important in backpropagation
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act


# ### Q2.2.2 (3 points)
# 
# Implement the softmax() function. Be sure to use the numerical stability trick you derived in Q1.1 softmax.

# In[5]:


############################## Q 2.2.2 ##############################
def softmax(x):
    """
    x is a numpy array of size [number of examples, number of classes].
    Softmax should be done for each row.
    """
    # Subtract the max value in each row for numerical stability
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    
    # Compute the exponentials of the shifted values
    exp_x = np.exp(x_shifted)
    
    # Compute the softmax for each row
    res = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    return res


# ### Q2.2.3 (3 points)
# 
# Implement the compute_loss_and_acc() function to compute the accuracy given a set of labels, along with the scalar loss across the data. The loss function generally used for classification is the cross-entropy loss.
# 
# $$L_{f}(\mathbf{D}) = - \sum_{(x, y)\in \mathbf{D}}y \cdot \log(f(x))$$
# 
# Here $\mathbf{D}$ is the full training dataset of $N$ data samples $x$ (which are $D \times 1$ vectors, $D$ is the dimensionality of data) and labels $y$ (which are $C\times 1$ one-hot vectors, $C$ is the number of classes), and $f:\mathbb{R}^D\to[0,1]^C$ is the classifier which outputs the probabilities for the classes.
# The $\log$ is the natural $\log$.

# In[6]:


import numpy as np

############################## Q 2.2.3 ##############################
def compute_loss_and_acc(y, probs):
    """
    Compute total loss and accuracy.

    Keyword arguments:
    y -- the labels, which is a numpy array of size [number of examples, number of classes]
    probs -- the probabilities output by the classifier, i.e. f(x), which is a numpy array of size [number of examples, number of classes]
    """
    # Compute cross-entropy loss
    # Avoid log(0) by adding a small epsilon for numerical stability
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1 - epsilon)
    loss = -np.sum(y * np.log(probs))

    # Compute accuracy
    predicted_labels = np.argmax(probs, axis=1)
    true_labels = np.argmax(y, axis=1)
    acc = np.mean(predicted_labels == true_labels)

    return loss, acc


# ## Q2.3 Backwards Propagation

# ### Q2.3 (7 points)
# 
# Implement the backwards() function to compute backpropagation for a single layer, given the original weights, the appropriate intermediate results, and the gradient with respect to the loss. You should return the gradient with respect to the inputs (grad_X) so that it can be used in the backpropagation for the previous layer. As a size check, your gradients should have the same dimensions as the original objects.
# 

# In[7]:


############################## Q 2.3 ##############################
def sigmoid_deriv(post_act):
    """
    We give this to you, because you proved it in Q1.4
    it's a function of the post-activation values (post_act).
    """
    res = post_act * (1.0 - post_act)
    return res

def backwards(delta, params, name='', activation_deriv=sigmoid_deriv):
    """
    Do a backpropagation pass for a single layer.

    Keyword arguments:
    delta -- gradients of the loss with respect to the outputs (errors to back propagate), in [number of examples, number of output dimensions]
    params -- a dictionary containing parameters, as how you initialized in Q 2.1.2
    name -- name of the layer
    activation_deriv -- the derivative of the activation function
    """
    grad_X, grad_W, grad_b = None, None, None
    # Retrieve the parameters and cached intermediate results
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # Chain rule: Compute the gradient through the activation function
    delta = delta * activation_deriv(post_act)

    # Compute gradients with respect to weights, biases, and inputs
    grad_W = np.dot(X.T, delta)  # Gradient with respect to W
    grad_b = np.sum(delta, axis=0)  # Gradient with respect to b
    grad_X = np.dot(delta, W.T)  # Gradient with respect to X

    # Store the gradients in the params dictionary
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b

    return grad_X


# Make sure you run below test code along the way to check if your implemented functions work as expected.

# In[8]:


def linear(x):
    # Define a linear activation, which can be used to construct a "no activation" layer
    return x

def linear_deriv(post_act):
    return np.ones_like(post_act)


# In[102]:


# test code
# generate some fake data
# feel free to plot it in 2D, what do you think these 4 classes are?
g0 = np.random.multivariate_normal([3.6,40],[[0.05,0],[0,10]],10)
g1 = np.random.multivariate_normal([3.9,10],[[0.01,0],[0,5]],10)
g2 = np.random.multivariate_normal([3.4,30],[[0.25,0],[0,5]],10)
g3 = np.random.multivariate_normal([2.0,10],[[0.5,0],[0,10]],10)
x = np.vstack([g0,g1,g2,g3])

# we will do XW + B in the forward pass
# this implies that the data X is in [number of examples, number of input dimensions]

# create labels
y_idx = np.array([0 for _ in range(10)] + [1 for _ in range(10)] + [2 for _ in range(10)] + [3 for _ in range(10)])
# turn to one-hot encoding, this implies that the labels y is in [number of examples, number of classes]
y = np.zeros((y_idx.shape[0],y_idx.max()+1))
y[np.arange(y_idx.shape[0]),y_idx] = 1
print("data shape: {} labels shape: {}".format(x.shape, y.shape))

# parameters in a dictionary
params = {}

# Q 2.1.2
# we will build a two-layer neural network
# first, initialize the weights and biases for the two layers
# the first layer, in_size = 2 (the dimension of the input data), out_size = 25 (number of neurons)
initialize_weights(2,25,params,'layer1')
# the output layer, in_size = 25 (number of neurons), out_size = 4 (number of classes)
initialize_weights(25,4,params,'output')
assert(params['Wlayer1'].shape == (2,25))
assert(params['blayer1'].shape == (25,))
assert(params['Woutput'].shape == (25,4))
assert(params['boutput'].shape == (4,))

# with Xavier initialization
# expect the means close to 0, variances in range [0.05 to 0.12]
print("Q 2.1.2: {}, {:.2f}".format(params['blayer1'].mean(),params['Wlayer1'].std()**2))
print("Q 2.1.2: {}, {:.2f}".format(params['boutput'].mean(),params['Woutput'].std()**2))

# Q 2.2.1
# implement sigmoid
# there might be an overflow warning due to exp(1000)
test = sigmoid(np.array([-1000,1000]))
print('Q 2.2.1: sigmoid outputs should be zero and one\t',test.min(),test.max())
# a forward pass on the first layer, with sigmoid activation
h1 = forward(x,params,'layer1',sigmoid)
assert(h1.shape == (40, 25))

# Q 2.2.2
# implement softmax
# a forward pass on the second layer (the output layer), with softmax so that the outputs are class probabilities
probs = forward(h1,params,'output',softmax)
# make sure you understand these values!
# should be positive, 1 (or very close to 1), 1 (or very close to 1)
print('Q 2.2.2:',probs.min(),min(probs.sum(1)),max(probs.sum(1)))
assert(probs.shape == (40,4))

# Q 2.2.3
# implement compute_loss_and_acc
loss, acc = compute_loss_and_acc(y, probs)
# should be around -np.log(0.25)*40 [~55] or higher, and 0.25
# if it is not, check softmax!
print("Q 2.2.3 loss: {}, acc:{:.2f}".format(loss,acc))

# Q 2.3
# here we cheat for you, you can use it in the training loop in Q2.4
# the derivative of cross-entropy(softmax(x)) is probs - 1[correct actions]
delta1 = probs - y

# backpropagation for the output layer
# we already did derivative through softmax when computing delta1 as above
# so we pass in a linear_deriv, which is just a vector of ones to make this a no-op
delta2 = backwards(delta1,params,'output',linear_deriv)
# backpropagation for the first layer
backwards(delta2,params,'layer1',sigmoid_deriv)

# the sizes of W and b should match the sizes of their gradients
for k,v in sorted(list(params.items())):
    if 'grad' in k:
        name = k.split('_')[1]
        # print the size of the gradient and the size of the parameter, the two sizes should be the same
        print('Q 2.3',name,v.shape, params[name].shape)


# ## Q2.4 Training Loop: Stochastic Gradient Descent

# ### Q2.4 (5 points)
# Implement the get_random_batches() function that takes the entire dataset (x and y) as input and splits it into random batches. Write a training loop that iterates over the batches, does forward and backward propagation, and applies a gradient update. The provided code samples batch only once, but it is also common to sample new random batches at each epoch. You may optionally try both strategies and note any difference in performance.

# In[107]:


############################## Q 2.4 ##############################
def get_random_batches(x, y, batch_size):
    """
    Split x (data) and y (labels) into random batches.
    Return a list of [(batch1_x, batch1_y), ...].
    """
    batches = []

    # Shuffle the data
    num_examples = x.shape[0]
    indices = np.arange(num_examples)
    np.random.shuffle(indices)

    x_shuffled = x[indices]
    y_shuffled = y[indices]
    print(x_shuffled.shape, y_shuffled.shape)
    # Split into batches
    for i in range(0, num_examples, batch_size):
        batch_x = x_shuffled[i:i + batch_size]
        batch_y = y_shuffled[i:i + batch_size]
        batches.append((batch_x, batch_y))

    return batches


# In[108]:


# Q 2.4
batches = get_random_batches(x,y,5)
batch_num = len(batches)
# print batch sizes
print([_[0].shape for _ in batches])
print(batch_num)


# In[12]:


############################## Q 2.4 ##############################
# WRITE A TRAINING LOOP HERE
max_iters = 200
learning_rate = 1e-3
# with default settings, you should get loss <= 35 and accuracy >= 75%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    num_batches = len(batches)

    for xb, yb in batches:
        ##########################
        ##### Forward Pass #####
        ##########################
        # Forward pass for the first layer
        h1 = forward(xb, params, name='layer1', activation=sigmoid)

        # Forward pass for the output layer
        preds = forward(h1, params, name='output', activation=softmax)

        ##########################
        ##### Compute Loss #####
        ##########################
        # Compute loss and accuracy
        loss, acc = compute_loss_and_acc(yb, preds)
        total_loss += loss
        total_acc += acc

        ##########################
        ##### Backward Pass #####
        ##########################
        # Backpropagation for the output layer
        delta1 = preds - yb
        delta2 = backwards(delta1, params, name='output', activation_deriv=linear_deriv)

        # Backpropagation for the first layer
        backwards(delta2, params, name='layer1', activation_deriv=sigmoid_deriv)

        ##########################
        ##### Gradient Update #####
        ##########################
        # Update weights and biases
        for key in params.keys():
            if key.startswith('grad_'):
                param_key = key[5:]  # Remove "grad_" prefix
                params[param_key] -= learning_rate * params[key]

    # Calculate average accuracy
    avg_acc = total_acc / num_batches

    # Print progress
    if itr % 100 == 0:
        print("itr: {:03d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss, avg_acc))


# # Q3 Training Models
# 
# Run below code to download and put the unzipped data in '/content/data' folder.
# 
# We have provided you three data .mat files to use for this section.
# The training data in nist36_train.mat contains samples for each of the 26 upper-case letters of the alphabet and the 10 digits. This is the set you should use for training your network.
# The cross-validation set in nist36_valid.mat contains samples from each class, and should be used in the training loop to see how the network is performing on data that it is not training on. This will help to spot overfitting.
# Finally, the test data in nist36_test.mat contains testing data, and should be used for the final evaluation of your best model to see how well it will generalize to new unseen data.

# In[13]:


if not os.path.exists('./content/data'):
  os.mkdir('./content/data')
  get_ipython().system('wget http://www.cs.cmu.edu/~lkeselma/16720a_data/data.zip -O /content/data/data.zip')
  get_ipython().system('unzip "./content/data/data.zip" -d "/content/data"')
  os.system("rm /content/data/data.zip")


# In[14]:


get_ipython().system('ls /content/data')


# ## Q3.1 (5 points)
# 
# Train a network from scratch. Use a single hidden layer with 64 hidden units, and train for at least 50 epochs. The script will generate two plots:
#     
# (1) the accuracy on both the training and validation set over the epochs, and
#     
# (2) the cross-entropy loss averaged over the data.
#     
# Tune the batch size and learning rate for accuracy on the validation set of at least 75\%. Hint: Use fixed random seeds to improve reproducibility.

# In[ ]:


train_data = scipy.io.loadmat('/Users/shrinivas/workspace/16-720/HW4/content/data/nist36_train.mat')
valid_data = scipy.io.loadmat('/Users/shrinivas/workspace/16-720/HW4/content/data/nist36_valid.mat')
test_data = scipy.io.loadmat('/Users/shrinivas/workspace/16-720/HW4/content/data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

if True: # view the data
    for crop, y in zip(train_x, train_y):
        print(train_x.shape)
        plt.imshow(crop.reshape(32,32).T, cmap="Greys")
        plt.show()
        break


# In[58]:


print(y)


# In[17]:


############################## Q 3.1 ##############################
max_iters = 200
# Pick a batch size and learning rate
batch_size = 32
learning_rate = 0.01
hidden_size = 64

batches = get_random_batches(train_x, train_y, batch_size)
batch_num = len(batches)

params = {}

# Initialize layers
initialize_weights(train_x.shape[1], hidden_size, params, "layer1")
initialize_weights(hidden_size, train_y.shape[1], params, "output")
layer1_W_initial = np.copy(params["Wlayer1"])  # Copy for Q3.3

train_loss = []
valid_loss = []
train_acc = []
valid_acc = []

for itr in range(max_iters):
    # Record training and validation loss and accuracy for plotting
    h1 = forward(train_x, params, 'layer1', sigmoid)
    probs = forward(h1, params, 'output', softmax)
    loss, acc = compute_loss_and_acc(train_y, probs)
    train_loss.append(loss / train_x.shape[0])
    train_acc.append(acc)

    h1 = forward(valid_x, params, 'layer1', sigmoid)
    probs = forward(h1, params, 'output', softmax)
    loss, acc = compute_loss_and_acc(valid_y, probs)
    valid_loss.append(loss / valid_x.shape[0])
    valid_acc.append(acc)

    total_loss = 0
    total_acc = 0
    for xb, yb in batches:
        # Forward pass
        h1 = forward(xb, params, 'layer1', sigmoid)
        probs = forward(h1, params, 'output', softmax)

        # Compute loss and accuracy
        batch_loss, batch_acc = compute_loss_and_acc(yb, probs)
        total_loss += batch_loss
        total_acc += batch_acc

        # Backward pass
        delta1 = probs - yb  # Gradient of cross-entropy loss
        delta2 = backwards(delta1, params, name='output', activation_deriv=linear_deriv)
        backwards(delta2, params, name='layer1', activation_deriv=sigmoid_deriv)

        # Apply gradient updates
        for key in params.keys():
            if key.startswith('grad_'):
                param_key = key[5:]  # Remove "grad_" prefix
                params[param_key] -= learning_rate * params[key]

    avg_acc = total_acc / batch_num

    if itr % 2 == 0:
        print("itr: {:02d}   loss: {:.2f}   acc: {:.2f}".format(itr, total_loss / train_x.shape[0], avg_acc))

# Record final training and validation accuracy and loss
h1 = forward(train_x, params, 'layer1', sigmoid)
probs = forward(h1, params, 'output', softmax)
loss, acc = compute_loss_and_acc(train_y, probs)
train_loss.append(loss / train_x.shape[0])
train_acc.append(acc)

h1 = forward(valid_x, params, 'layer1', sigmoid)
probs = forward(h1, params, 'output', softmax)
loss, acc = compute_loss_and_acc(valid_y, probs)
valid_loss.append(loss / valid_x.shape[0])
valid_acc.append(acc)

# Report validation accuracy; aim for 75%
print('Validation accuracy: ', valid_acc[-1])

# Compute and report test accuracy
h1 = forward(test_x, params, 'layer1', sigmoid)
test_probs = forward(h1, params, 'output', softmax)
_, test_acc = compute_loss_and_acc(test_y, test_probs)
print('Test accuracy: ', test_acc)


# In[ ]:


# save the final network
import pickle

saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('/Users/shrinivas/workspace/16-720/HW4/content/q3_weights.pickle', 'wb') as handle:
  pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ## Q3.2 (3 points)
# 
# The provided code will visualize the first layer weights as 64 32x32 images, both immediately after initialization and after full training. Generate both visualizations. Comment on the learned weights and compare them to the initialized weights. Do you notice any patterns?

# In[20]:


############################## Q 3.2 ##############################
# visualize weights
fig = plt.figure(figsize=(8,8))
plt.title("Layer 1 weights after initialization")
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.05)
for i, ax in enumerate(grid):
    ax.imshow(layer1_W_initial[:,i].reshape((32, 32)).T)
    ax.set_axis_off()
plt.show()

v = np.max(np.abs(params['Wlayer1']))
fig = plt.figure(figsize=(8,8))
plt.title("Layer 1 weights after training")
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.05)
for i, ax in enumerate(grid):
    ax.imshow(params['Wlayer1'][:,i].reshape((32, 32)).T, vmin=-v, vmax=v)
    ax.set_axis_off()
plt.show()


# ---
# 
# After comparing the two plots of Layer 1 weights, we observe distinct differences before and after training. **Before training**, the weights appear as random noise with no discernible structure. This randomness is expected due to standard initialization techniques, such as Xavier or He initialization. All neurons display visually similar patterns since the weights are initialized without any prior knowledge of the data, and there is no bias toward any specific feature. **After training**, the weights show clear patterns and structures, indicating that the network has learned meaningful features from the data. These patterns suggest that the neurons are specializing in detecting distinct features, such as edges, textures, or shapes. The weights also exhibit stronger contrast and smoother transitions, reflecting the emergence of structured feature detectors. This progression highlights effective learning within the neural network. If any weights still appear random after training, it could indicate issues such as insufficient training, dead neurons, or poor optimization. Overall, the comparison demonstrates the transformation of weights from random initialization to meaningful feature representations after training.
# 
# ---

# ## Q3.3 (3 points)
# 
# Use the code in Q3.1 to train and generate accuracy and loss plots for each of these three networks:
# 
# (1) one with $10$ times your tuned learning rate,
#     
# (2) one with one-tenth your tuned learning rate, and
# 
# (3) one with your tuned learning rate.
#     
# Include total of six plots (two will be the same from Q3.1). Comment on how the learning rates affect the training, and report the final accuracy of the best network on the test set. Hint: Use fixed random seeds to improve reproducibility.

# In[21]:


############################## Q 3.3 ##############################
##########################
############################## Q 3.3 ##############################
# Define different learning rates
tuned_learning_rate = 0.01  # Replace this with the tuned learning rate from Q3.1
learning_rates = {
    "High LR (10x)": tuned_learning_rate * 10,
    "Low LR (0.1x)": tuned_learning_rate * 0.1,
    "Tuned LR": tuned_learning_rate
}

# Results storage
results = {}

# Fixed random seed for reproducibility
np.random.seed(42)

# Train and evaluate the model for each learning rate
for name, lr in learning_rates.items():
    print(f"Training with {name}: Learning Rate = {lr}")

    # Reinitialize parameters
    params = {}
    initialize_weights(train_x.shape[1], hidden_size, params, "layer1")
    initialize_weights(hidden_size, train_y.shape[1], params, "output")

    # Storage for losses and accuracies
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []

    # Training loop
    for itr in range(max_iters):
        # Record training and validation loss and accuracy for plotting
        h1 = forward(train_x, params, 'layer1', sigmoid)
        probs = forward(h1, params, 'output', softmax)
        loss, acc = compute_loss_and_acc(train_y, probs)
        train_loss.append(loss / train_x.shape[0])
        train_acc.append(acc)

        h1 = forward(valid_x, params, 'layer1', sigmoid)
        probs = forward(h1, params, 'output', softmax)
        loss, acc = compute_loss_and_acc(valid_y, probs)
        valid_loss.append(loss / valid_x.shape[0])
        valid_acc.append(acc)

        # Training with batches
        total_loss = 0
        total_acc = 0
        for xb, yb in get_random_batches(train_x, train_y, batch_size):
            # Forward pass
            h1 = forward(xb, params, 'layer1', sigmoid)
            probs = forward(h1, params, 'output', softmax)

            # Compute loss and accuracy
            batch_loss, batch_acc = compute_loss_and_acc(yb, probs)
            total_loss += batch_loss
            total_acc += batch_acc

            # Backward pass
            delta1 = probs - yb  # Gradient of cross-entropy loss
            delta2 = backwards(delta1, params, name='output', activation_deriv=linear_deriv)
            backwards(delta2, params, name='layer1', activation_deriv=sigmoid_deriv)

            # Apply gradient updates
            for key in params.keys():
                if key.startswith('grad_'):
                    param_key = key[5:]  # Remove "grad_" prefix
                    params[param_key] -= lr * params[key]

        if itr % 2 == 0:
            print(f"itr: {itr:02d}   loss: {total_loss / train_x.shape[0]:.2f}   acc: {total_acc / batch_num:.2f}")

    # Record results for plotting
    results[name] = {
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "train_acc": train_acc,
        "valid_acc": valid_acc
    }

    # Compute final test accuracy
    h1 = forward(test_x, params, 'layer1', sigmoid)
    test_probs = forward(h1, params, 'output', softmax)
    _, test_acc = compute_loss_and_acc(test_y, test_probs)
    results[name]["test_acc"] = test_acc
    print(f"{name} Test Accuracy: {test_acc:.2f}")

############################## Plotting ##############################

for name, result in results.items():
    epochs = range(1, max_iters + 1)

    # Plot training and validation loss
    plt.figure()
    plt.plot(epochs, result["train_loss"], label="Training Loss")
    plt.plot(epochs, result["valid_loss"], label="Validation Loss")
    plt.title(f"Loss vs Epochs ({name})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot training and validation accuracy
    plt.figure()
    plt.plot(epochs, result["train_acc"], label="Training Accuracy")
    plt.plot(epochs, result["valid_acc"], label="Validation Accuracy")
    plt.title(f"Accuracy vs Epochs ({name})")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

############################## Analysis ##############################

# Compare test accuracy
for name, result in results.items():
    print(f"{name}: Final Test Accuracy = {result['test_acc']:.2f}")


##########################


# ---
# 
# Key comments:
# 
# High learning rate (10x) may lead to oscillations or instability in loss and poor convergence.
# 
# Low learning rate (0.1x) results in slower convergence and potentially underfitting.
# 
# Tuned learning rate should achieve the best balance between speed and stability, yielding the highest accuracy.
# 
# 
# ---

# ## Q3.4 (3 points)
# 
# Compute and visualize the confusion matrix of the test data for your best model. Comment on the top few pairs of classes that are most commonly confused.

# In[22]:


############################## Q 3.4 ##############################
confusion_matrix = np.zeros((train_y.shape[1], train_y.shape[1]))

# Compute confusion matrix
##########################
##### Your code here #####

# Forward pass on test set
h1 = forward(test_x, params, 'layer1', sigmoid)
probs = forward(h1, params, 'output', softmax)

# Convert predictions to one-hot encoding
predicted_labels = np.argmax(probs, axis=1)
true_labels = np.argmax(test_y, axis=1)

# Populate confusion matrix
for t, p in zip(true_labels, predicted_labels):
    confusion_matrix[t, p] += 1

##########################

# Visualize confusion matrix
import string
plt.imshow(confusion_matrix, interpolation='nearest', cmap='viridis')
plt.grid()
plt.xticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]), rotation=45)
plt.yticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()
plt.title("Confusion Matrix")
plt.show()


# ---
# 0(zero) and O(Oh) are confused most frequently since they basically look the same.
# 
# Another confusion is between Y and V due to the downward pointing vertex common to both.
# 
# Z and 2 are also confused since 2 is the cursive version of Z
# 
# ---

# # Q4 Object Detection and Tracking

# ## **Initialization**
# 
# Run the following code, which imports the modules you'll need and defines helper functions you may need to use later in your implementations.

# In[23]:


import cv2
import numpy as np
import torchvision
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import torch
import argparse
from PIL import Image
from matplotlib import pyplot as plt
import os
import glob

# Utility functions

coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Create different colors for each class.
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def draw_boxes(boxes, labels, image):
    """
    Draws the bounding box around a detected object, also with labels
    """
    image = image.copy()
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # Line width
    tf = max(lw - 1, 1) # Font thickness.
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            img=image,
            pt1=(int(box[0]), int(box[1])),
            pt2=(int(box[2]), int(box[3])),
            color=color[::-1],
            thickness=lw
        )
        cv2.putText(
            img=image,
            text=coco_names[labels[i]],
            org=(int(box[0]), int(box[1]-5)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=lw / 3,
            color=color[::-1],
            thickness=tf,
            lineType=cv2.LINE_AA
        )
    return image


def draw_single_track(all_frames, track, track_idx):
    """
    Visualize a track
    """
    image_vis_list = []
    start_frame = track['start_frame']
    num_frames_in_track = len(track['bboxes'])
    print('Visualizing track {} with {} frames, starting from frame {}'.format(track_idx, num_frames_in_track, start_frame))

    for track_frame_num in range(num_frames_in_track):
        frame_num = start_frame + track_frame_num
        image, _, _, _ = all_frames[frame_num]
        bbox = track['bboxes'][track_frame_num]
        image_viz = image.copy()

        # print('Frame: {}, Bbox: {}'.format(frame_num, bbox))
        cv2.rectangle(image_viz, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 4)

        xcentroid, ycentroid = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
        text = "ID {}".format(track_idx)

        cv2.putText(image_viz, text, (xcentroid - 10, ycentroid - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.circle(image_viz, (xcentroid, ycentroid), 6, (0, 255, 0), -1)

        image_vis_list.append(image_viz)

    return image_vis_list


def draw_multi_tracks(all_frames, tracks):
    """
    Visualize multiple tracks
    """
    # Mapping from frame number to a list of (bbox, track_idx) tuples
    viz_per_frame = {}

    # Image visualization list
    image_vis_list = []

    # Track idx to color (each track idx has a color)
    track_to_color = {}

    # Loop through the tracks and got the proper info
    for track_idx, track in enumerate(tracks):
        start_frame = track['start_frame']
        num_frames_in_track = len(track['bboxes'])
        print('Visualizing track {} with {} frames, starting from frame {}'.format(track_idx, num_frames_in_track, start_frame))

        for track_frame_num in range(num_frames_in_track):
            frame_num = start_frame + track_frame_num
            bbox = track['bboxes'][track_frame_num]

            # Ensure bbox is valid
            if len(bbox) != 4:
                raise ValueError(f"Invalid bbox format: {bbox}. Expected a 4-element sequence.")

            if frame_num not in viz_per_frame:
                viz_per_frame[frame_num] = []
            viz_per_frame[frame_num].append((bbox, track_idx))

    # Loop through the frames and draw the boxes
    for frame_num, (image, bboxes, confidences, class_ids) in enumerate(all_frames):
        image_viz = image.copy()

        if frame_num not in viz_per_frame:
            continue

        for bbox, track_idx in viz_per_frame[frame_num]:
            if track_idx not in track_to_color:
                track_to_color[track_idx] = np.random.randint(0, 255, size=3)

            color = track_to_color[track_idx]
            color = (int(color[0]), int(color[1]), int(color[2]))

            # Ensure bbox is integers before drawing
            bbox = [int(coord) for coord in bbox]
            cv2.rectangle(image_viz, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 4)

            xcentroid, ycentroid = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
            text = "ID {}".format(track_idx)

            cv2.putText(image_viz, text, (xcentroid - 15, ycentroid - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
            cv2.circle(image_viz, (xcentroid, ycentroid), 10, color, -1)

        image_vis_list.append(image_viz)

    return image_vis_list


# ## **Set up data**
# 

# In[24]:


if not os.path.exists('car_frames_simple.zip'):
  get_ipython().system('wget https://www.andrew.cmu.edu/user/kvuong/car_frames_simple.zip -O car_frames_simple.zip')
  get_ipython().system('unzip -qq "car_frames_simple.zip"')
  print("downloaded and unzipped data")


# ## **Problem 4.1**: Object Detection with Faster-RCNN
# 

# In[25]:


def get_model(device):
    """
    Load the pretrained model + inference transform
    """
    # Load the model
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    # Load the model onto the computation device
    model = model.eval().to(device)
    # inference transform
    preprocess = weights.transforms()

    return model, preprocess


def predict(image, model, device, detection_threshold):
    """
    Predicts bounding boxes, scores, and class labels for objects detected in an image.
    Only returns detections with confidence above the specified threshold.

    Args:
        image (torch.Tensor): The input image tensor.
        model (torchvision.models.detection.FasterRCNN): The object detection model.
        device (torch.device): The device to perform computations on.
        detection_threshold (float): Confidence threshold for filtering detections.

    Returns:
        boxes (numpy.ndarray): Bounding boxes of detected objects above the confidence threshold. Shape (N, 4),
            where N is the number of detections. Bbox format: (x1, y1, x2, y2)
        scores (numpy.ndarray): Confidence scores for the detected objects. Shape (N,)
        labels (numpy.ndarray): Class labels for the detected objects. Shape (N,)
    """

    # Move input image to the specified device (GPU/CPU)
    image = image.to(device)

    # Add a batch dimension to the image tensor
    image = image.unsqueeze(0)

    # Run the forward pass
    with torch.no_grad():
        outputs = model(image)

    # Extract the scores, bounding boxes, and labels
    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()

    # Apply the detection threshold
    keep = scores >= detection_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    return boxes, scores, labels


def run_detector(image_path, model, preprocess, det_threshold=0.9):
    """
    Runs the object detector on a given image and retrieves bounding boxes, confidence scores,
    and class labels for detected objects.

    Args:
        image_path (str): Path to the image file to detect objects in.
        model (torchvision.models.detection.FasterRCNN): The object detection model.
        preprocess (callable): Preprocessing function for the image.
        det_threshold (float): Confidence threshold for detections.

    Returns:
        image_np (numpy.ndarray): Original image in numpy array format (for visualization later)
        bboxes (numpy.ndarray): Bounding boxes of detected objects.
        confidences (numpy.ndarray): Confidence scores for the detected objects.
        class_ids (numpy.ndarray): Class labels for the detected objects.
    """
    # Read image to tensor (0-255 uint8)
    image_torch = read_image(image_path)
    image_np = image_torch.permute(1, 2, 0).numpy()

    # Read image to tensor (0-255 uint8)
    image_torch = read_image(image_path)
    image_np = image_torch.permute(1, 2, 0).numpy()  # Convert for visualization

    # Apply preprocessing
    image_processed = preprocess(image_torch)

    # Run the predict function
    bboxes, confidences, class_ids = predict(image_processed, model, device, det_threshold)

    return image_np, bboxes, confidences, class_ids


# In[26]:


# Define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cpu':
  print('!!! WARNING: USING CPU ONLY, THIS WILL BE VERY SLOW !!!')

# First, load the model and preprocessor
model, preprocess = get_model(device)


# In[27]:


# TODO: either use wget or manually upload the image to temporary storage (please don't use the same image as the example in the pdf)
get_ipython().system('wget "/path/to/your/image/to/download/here" -O example.png')
image_path = "/Users/shrinivas/workspace/16-720/HW4/testimages/nyc.png"

# Run the detector on the image
output_det = run_detector(image_path, model, preprocess, det_threshold=0.9)
image, bboxes, confidences, class_ids = output_det
image_with_boxes = draw_boxes(bboxes, class_ids, image)
plt.imshow(image_with_boxes)
plt.axis('off')
plt.tight_layout()
plt.show()


# In[28]:


# TODO: run object detector on every image inside the data folder
image_folder = "./car_frames_simple"
image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))

output_detections = []
for image_path in image_paths:
    output_det = run_detector(image_path, model, preprocess, det_threshold=0.9)
    output_detections.append(output_det)

# Visualize a few images (first and last image for example)
indices = [0, len(output_detections) - 1]
for idx in indices:
    image, bboxes, confidences, class_ids = output_detections[idx]
    image_with_boxes = draw_boxes(bboxes, class_ids, image)
    plt.imshow(image_with_boxes)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ## **Problem 4.2**: Multi-object-tracking with IOU-based tracker
# 

# ### **Compute IOU**

# In[29]:


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.

    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.

    Returns:
        float: intersection-over-union of bbox1, bbox2
    """

    # Calculate the coordinates for the intersection rectangle
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Check if there is an overlap
    if x1 >= x2 or y1 >= y2:
        return 0.0  # No overlap

    # Calculate the area of the intersection rectangle
    intersection_area = (x2 - x1) * (y2 - y1)

    # Calculate the area of both bounding boxes
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Calculate the union area
    union_area = bbox1_area + bbox2_area - intersection_area

    # Return the IoU value
    return intersection_area / union_area


# ### **IOU-based Tracker**

# In[30]:


def track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min):
    tracks_active = []
    tracks_finished = []
    next_track_id = 0  # Counter for unique track IDs

    for frame_num, detections_frame in enumerate(detections):
        # Filter detections based on sigma_l
        dets = [det for det in detections_frame if det['score'] >= sigma_l]

        updated_tracks = []
        for track in tracks_active:
            track_updated = False

            # Find the best matching detection
            best_match = None
            best_iou = 0
            for det in dets:
                current_iou = iou(track['bboxes'][-1], det['bbox'])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_match = det

            # Extend the track if IoU is above sigma_iou
            if (best_match is not None) and best_iou >= sigma_iou:
                track['bboxes'].append(best_match['bbox'])
                track['max_score'] = max(track['max_score'], best_match['score'])
                dets.remove(best_match)
                track_updated = True

            # If the track was updated, keep it active
            if track_updated:
                updated_tracks.append(track)
            else:
                # Finish track if not updated and it meets the criteria
                if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min:
                    tracks_finished.append(track)

        # Create new tracks for unmatched detections
        for det in dets:
            new_track = {
                'id': next_track_id,  # Assign a unique ID to the new track
                'bboxes': [det['bbox']],
                'max_score': det['score'],
                'start_frame': frame_num
            }
            next_track_id += 1
            updated_tracks.append(new_track)

        # Update the active tracks
        tracks_active = updated_tracks

    # Finish remaining active tracks
    tracks_finished += [track for track in tracks_active
                        if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min]

    return tracks_finished


# ### **Run Tracker**

# In[31]:


def run_tracker(frames, sigma_l=0.1, sigma_h=0.9, sigma_iou=0.9, t_min=2):
    # Track objects in the video
    detections = []
    for frame_num, (image, bboxes, confidences, class_ids) in enumerate(frames):
        dets = []
        for bbox, confidence, class_id in zip(bboxes, confidences, class_ids):
            dets.append({'bbox': (bbox[0], bbox[1], bbox[2], bbox[3]),
                         'score': confidence,
                         'class': class_id})
        detections.append(dets)

    print('Running tracker...')
    tracks = track_iou(detections, sigma_l=sigma_l, sigma_h=sigma_h, sigma_iou=sigma_iou, t_min=t_min)
    print('Tracker finished!')
    return tracks


# In[32]:


# TODO: From the detections, run the tracker to obtain a list of tracks
# get the detection for images in car_frames_simple
output_detections = []
for image_path in image_paths:
    output_det = run_detector(image_path, model, preprocess, det_threshold=0.9)
    output_detections.append(output_det)


# In[33]:


sigma_l, sigma_h, sigma_iou, t_min = 0.4, 0.7, 0.3, 2
output_tracks = run_tracker(output_detections, sigma_l=sigma_l, sigma_h=sigma_h, sigma_iou=sigma_iou, t_min=t_min)
# Visualize the tracks
image_vis_list = draw_multi_tracks(output_detections, output_tracks)
print(len(image_vis_list))
print(indices)

# TODO: Visualize a few images (here we show first, middle, and last image for example)
indices = [0, len(output_detections) // 2, len(output_detections) - 1]
for idx in range(len(image_vis_list)):
    plt.imshow(image_vis_list[idx])
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# # Q5 (Extra Credit) Extract Text from Images
# 
# Run below code to download and put the unzipped data in '/content/images' folder. We have provided you with 01_list.jpg, 02_letters.jpg, 03_haiku.jpg and 04_deep.jpg to test your implementation on.

# In[34]:


if not os.path.exists('/Users/shrinivas/workspace/16-720/HW4/content/images'):
  os.mkdir('/Users/shrinivas/workspace/16-720/HW4/content/images')
  get_ipython().system('wget http://www.cs.cmu.edu/~lkeselma/16720a_data/images.zip -O /content/images/images.zip')
  get_ipython().system('unzip "/Users/shrinivas/workspace/16-720/HW4/content/images/images.zip" -d "/content/images"')
  os.system("rm /Users/shrinivas/workspace/16-720/HW4/content/images/images.zip")


# In[35]:


ls /content/images


# ## Q5.1 (Extra Credit) (4 points)
# 
# The method outlined above is pretty simplistic, and while it works for the given text samples, it makes several assumptions. What are two big assumptions that the sample method makes?

# ---
# 
# ### Two Big Assumptions in the Sample Method
# 
# 1. **Assumption about Character Segmentation:**
#    - **Assumption:** The method assumes that each character is distinct and well-separated from neighboring characters, enabling clear segmentation using connected components analysis.
#    - **Issue:** In real-world scenarios, characters may overlap or be connected (e.g., cursive handwriting or certain fonts), which can lead to incorrect bounding box generation or merging of multiple characters into a single component.
# 
# 2. **Assumption about Uniform Line and Text Alignment:**
#    - **Assumption:** The method assumes that the text is aligned in straight lines and characters within a line are evenly spaced for grouping and sorting.
#    - **Issue:** Text in images might be rotated, curved, or follow a non-standard alignment (e.g., slanted handwriting or artistic fonts), making it challenging to correctly group characters into lines or determine their order.
# 
# These assumptions simplify the task but may limit accuracy and robustness for more complex or real-world text images.
# 
# ---
# 

# ## Q5.2 (Extra Credit) (10 points)
# 
# Implement the findLetters() function to find letters in the image. Given an RGB image, this function should return bounding boxes for all of the located handwritten characters in the image, as well as a binary black-and-white version of the image im. Each row of the matrix should contain [y1,x1,y2,x2], the positions of the top-left and bottom-right corners of the box. The black-and-white image should be between 0.0 to 1.0, with the characters in white and the background in black (consistent with the images in nist36). Hint: Since we read text left to right, top to bottom, we can use this to cluster the coordinates.

# In[128]:


def findLetters(image):
    """
    Takes a color image and returns a list of bounding boxes and a black-and-white image.
    Each bounding box is represented as [y1, x1, y2, x2], the positions of the
    top-left and bottom-right corners of the box.
    The black-and-white image is between 0.0 to 1.0, with the characters in white
    and the background in black.
    """
    import numpy as np
    from skimage import color, filters, morphology, measure

    bboxes = []
    bw = None

    # Convert the RGB image to grayscale
    gray = color.rgb2gray(image)

    # Apply Otsu's threshold to binarize the image
    thresh = filters.threshold_otsu(gray)
    binary = gray < thresh  # Invert the image so that characters are white

    # Remove small objects (noise) from the binary image
    cleaned = morphology.remove_small_objects(binary, min_size=50)

    # Fill small holes inside the foreground objects
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=50)

    # Perform morphological closing to connect disconnected components
    selem = morphology.rectangle(3, 3)
    cleaned = morphology.closing(cleaned, selem)

    # apply a Gaussian filter to the image
    bw = filters.gaussian(cleaned, sigma=1)

    # Label connected components
    labels = measure.label(cleaned, connectivity=2)

    # Obtain the properties of the labeled regions
    regions = measure.regionprops(labels)

    # Loop through each region and extract bounding boxes
    for region in regions:
        # Skip small regions that may not be characters
        if region.area < 100:
            continue
        # Extract the bounding box coordinates
        minr, minc, maxr, maxc = region.bbox
        bboxes.append([minr, minc, maxr, maxc])

    # Convert the cleaned binary image to float type
    bw = cleaned.astype(float)

    return bboxes, bw


# ## Q5.3 (Extra Credit) (3 points)
# 
# Using the provided code below, visualize all of the located boxes on top of the binary image to show the accuracy of your findLetters() function. Include all the provided sample images with the boxes.

# In[129]:


############################## Q 5.3 ##############################
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for imgno, img in enumerate(sorted(os.listdir('/Users/shrinivas/workspace/16-720/HW4/content/images'))):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('/Users/shrinivas/workspace/16-720/HW4/content/images',img)))
    bboxes, bw = findLetters(im1)

    print('\n' + img)
    plt.imshow(1-bw, cmap="Greys") # reverse the colors of the characters and the background for better visualization
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()


# ## Q5.4 (Extra Credit) (8 points)
# 
# You will now load the image, find the character locations, classify each one with the network you trained in Q3.1, and return the text contained in the image. Be sure you try to make your detected images look like the images from the training set. Visualize them and act accordingly. If you find that your classifier performs poorly, consider dilation under skimage morphology to make the letters thicker.
# 
# Your solution is correct if you can correctly detect most of the letters and classify approximately 70\% of the letters in each of the sample images.
# 
# Run your code on all the provided sample images in '/content/images'. Show the extracted text. It is fine if your code ignores spaces, but if so, please provide a written answer with manually added spaces.

# In[132]:


import os
import numpy as np
import skimage
from skimage import io, img_as_float
from skimage.transform import resize
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Ensure that the necessary functions are defined: forward, sigmoid, softmax
# Also, ensure that the trained parameters are available from your Q3.1 implementation.

# Load the trained network parameters
import pickle
import string
letters = np.array([char for char in string.ascii_uppercase[:26]] + [str(digit) for digit in range(10)])
params = pickle.load(open('./content/q3_weights.pickle', 'rb'))

for imgno, img in enumerate(sorted(os.listdir('./content/images'))):
    # Read the image and convert it to float
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('./content/images', img)))
    # Find letters in the image
    bboxes, bw = findLetters(im1)
    print('\nProcessing image:', img)

    # Convert bounding boxes to a NumPy array
    bboxes = np.array(bboxes)

    # Find the y-centroids of each bounding box to cluster them into lines
    bbox_centroids = np.mean(bboxes[:, [0, 2]], axis=1).reshape(-1, 1)

    # Cluster bounding boxes into lines based on their vertical positions
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=20, metric='euclidean', linkage='single')
    clustering.fit(bbox_centroids)
    labels = clustering.labels_

    # Group bounding boxes by lines
    lines = []
    for label in np.unique(labels):
        line_indices = np.where(labels == label)[0]
        line_bboxes = bboxes[line_indices]
        # Sort the bounding boxes within the line from left to right
        sorted_indices = np.argsort(line_bboxes[:, 1])  # Sort by x1 coordinate (column)
        sorted_line_bboxes = line_bboxes[sorted_indices]
        lines.append(sorted_line_bboxes)

    # Sort the lines from top to bottom based on the average y-coordinate
    lines = sorted(lines, key=lambda x: np.mean(x[:, [0, 2]]))

    # Process each line
    for line in lines:
        line_text = ''  # Initialize an empty string to store the predicted text of the line
        # Process each character in the line
        for bbox in line:
            minr, minc, maxr, maxc = bbox.astype(int)
            # Crop the character from the binary image
            char_img = bw[minr:maxr, minc:maxc]
            # Make the image square by padding
            h, w = char_img.shape
            pad_size = max(h, w)
            pad_h = (pad_size - h) // 2
            pad_w = (pad_size - w) // 2
            char_img_padded = np.pad(char_img, ((pad_h, pad_size - h - pad_h),
                                                (pad_w, pad_size - w - pad_w)),
                                     'constant', constant_values=0.0)
            # Resize to 32x32 pixels to match the training data
            char_img_resized = resize(char_img_padded, (32, 32))

            # apply a gaussain blur
            char_img_resized = skimage.filters.gaussian(char_img_resized, sigma=0.5)
            # Transpose the image as required by the dataset
            char_img_resized = char_img_resized.T
            # Flatten the image into a vector
            char_img_flat = char_img_resized.flatten()
            # Reshape for the network input
            X = char_img_flat.reshape(1, -1)

            # Forward pass through the network
            h1 = forward(X, params, 'layer1', sigmoid)
            probs = forward(h1, params, 'output', softmax)
            # Get predicted label
            predicted_label = np.argmax(probs, axis=1)[0]
            # Map label to corresponding character
            predicted_char = letters[predicted_label]

            # Plot the character image
            plt.imshow(char_img_padded, cmap='gray')
            plt.title(f'Predicted Character: {predicted_char}')
            plt.axis('off')
            plt.show()

            # Print the predicted character
            print('Predicted Character:', predicted_char)
            # Append the character to the line text
            line_text += predicted_char
        # After processing all characters in the line, print the recognized text
        print('Recognized Text:', line_text)


# ---
# 
# YOUR ANSWER HERE... (if your code ignores spaces)
# 
# ---
