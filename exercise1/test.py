# %% [markdown]
# # Exercise 1: Maximum-likelihood fitting
# 
# In this exercise we will implement a maximum-likelihood fit using the common example of fitting a Gaussian curve to some normally-distributed data.
# 
# The main task will be to implement a gradient-descent algorithm.
# 

# %% [markdown]
# ## Parameters
# 
# The objective of fitting a PDF to data is to find the values of the PDF's parameters that maximise the likelihood, thus giving the best description of a dataset (i.e. the best fit).
# 
# While it's possible to store parameters as simple `float`s, we ask that you implement a class that stores the parameter value and a range that the value cannot stray outside during the fit. This will be especially helpful while testing and debugging your minimiser

# %% [markdown]
# ## PDFs and likelihoods
# 
# A Gaussian PDF is defined as:
# $$
# f(x|\mu,\sigma) = \frac{1}{\sigma \sqrt{2 \pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
# $$
# 
# The likelihood is the product of the PDF evaluated on all points $\{x_i\}$ in a dataset (assuming all points are independent):
# 
# $$
# \mathcal{L}(\mu,\sigma) = \prod_i f(x_i|\mu,\sigma)
# $$
# 
# Rather than maximise the likelihood, it is easier computationally to minimise the negative log-likelihood:
# $$
# -\log(\mathcal{L}(\mu,\sigma)) = -\sum_i\log(f(x_i|\mu,\sigma))
# $$

# %% [markdown]
# ## Minimisation via gradient descent
# 
# Gradient descent is a method of finding the local minimum of a differentiable function by following the local gradient.
# 
# Note that the local minimum may not be the same as the global minimum. There are strategies to avoid a local-but-not-global minimum, such as performing a fit multiple times with random starting values.
# 
# Say we have a multivariate function $L(\vec{\theta})$, where $\vec{\theta} = [\theta_1, \theta_2, ..., \theta_n].$ _**NB**: in this exercise, $L(\vec{\theta})=-\log(\mathcal{L}(\mu,\sigma))$, the negative log-likelihood._
# 
# The gradient of the function is $$\nabla_\theta L(\vec{\theta}) = \left[\frac{\partial L}{\partial \theta_1}, \frac{\partial L}{\partial \theta_2}, ..., \frac{\partial L}{\partial \theta_n}\right].$$
# 
# Starting from some initial position $\vec{\theta}_i$, one can descend the gradient towards the minimum by subtracting an amount proportional to the local gradient: $$\vec{\theta}_{t} = \vec{\theta}_{t-1} - \eta_t \nabla_\theta L(\vec{\theta}_{t-1}),$$ where $\eta_t$ is the "step size". This can be repeated until $\vec{\theta}$ converges on the values that minimise $L(\vec{\theta})$ (i.e. the best-fit values $\vec{\hat{\theta}}$).
# 
# The exact criterion for achieving convergence is up to you to decide and implement.
# A good starting point is to use the relative change $$\left|\frac{L(\vec{\theta}_{i})-L(\vec{\theta}_{i-1})}{L(\vec{\theta}_{i})+L(\vec{\theta}_{i-1})}\right|$$ and stop when it goes below some threshold.
# 
# Choosing an appropriate step size $\eta_t$ is crucial for an optimal balance between speed and precision, and many methods are available for doing this. Note that it does not need to be a fixed size and can be adjusted at each iteration.
# 
# ### Batch vs stochastic
# 
# In batch gradient descent, the parameters are updated using the likelihood calculated over the full dataset.
# 
# In stochastic gradient descent, the parameters are updated for each datapoint (using $\nabla_\theta L(\vec{\theta}, x_i)$), or a sub-sample of the full dataset (mini-batch).
# 
# *See the lecture notes for more.*
# 
# ### Momentum
# 
# Useful particularly in stochastic/mini-batch gradient descent is the idea of 'momentum'. Where the parameters are updated using: $$\vec{\theta}_t = \vec{\theta}_{t-1} - \vec{v}_t,$$ where $$\vec{v}_t = \gamma \vec{v}_{t-1} + \eta_t \nabla_\theta L(\vec{\theta}_{t-1}, x),$$ where $\gamma$ is the momentum parameter $0 < \gamma < 1$ (typically around 0.9).
# 
# ### Nesterov's accelerated gradient descent
# 
# Using momentum we can 'look ahead' to where the next update will be approximately, without calculating a new gradient: $$\vec{\theta}_t \approx \vec{\theta}_{t-1} - \gamma \vec{v}_{t-1}$$
# 
# We can instead use that position when calculating the new gradient, so $\vec{v}_t$ becomes: $$\vec{v}_t = \gamma \vec{v}_{t-1} + \eta_t \nabla_\theta L(\vec{\theta}_{t-1} - \gamma \vec{v}_{t-1}, x).$$

# %% [markdown]
# ## Implementing your own gradient-descent minimiser
# 
# Here you should implement:
# 1. A parameter class that holds the value and allowed range of a parameter
#   - When setting a value outside the allowed range, force the value to equal the nearest boundary
# 2. A Gaussian PDF class or function using parameters that control its mean and standard deviation
# 3. A gradient-descent minimiser which iteratively:
#   - Calcualtes the likelihood gradient at the current values of the fit parameters
#   - Updates the parameters following the gradient
#   - Saves the likelihood in a list (for plotting later)
#   - Stops the fit if it has converged (or if a maximum number of iterations have been reached)

# %%
# Implement the parameter class, the Guassian PDF and gradient-descent minimisation here
# You can create extra cells if you wish

import numpy as np
from scipy.stats import norm
import copy

class Theta:
    value = None
    upperBound = None
    lowerBound = None

    def __init__(self, lowerBound, upperBound):
        if upperBound < lowerBound:
            print("upper boundary lower than lower boundary")
        elif upperBound == lowerBound:
            print("upper boundary equal to lower boundary")
        else:
            self.upperBound = upperBound
            self.lowerBound = lowerBound

    def update(self, newValue):
        self.value = np.clip(newValue, self.lowerBound, self.upperBound)

param = Theta(-1,1)
param.update(0.5)
print(param.value)

sigma = Theta(-1,1)
mean = Theta(-20,20)
sigma.update(1)
mean.update(0)

def normalPDF(x, params): #params is [mean,sigma]
    # print(norm(params[0].value,params[1].value).pdf(x), x, params[0].value, params[1].value)
    return norm(params[0].value,params[1].value).pdf(x)

def logLikely(data, params, function):
    sum = 0
    tmp = np.empty(len(data))
    for i, x in enumerate(data):
        tmp[i]  = function(x,params)
    #print(tmp, "tmp", np.log(tmp))
    sum = np.sum(np.log(tmp))
    return -sum

#likelyhood function with (dataarray, parameterarray, function)
#theta is deepcopy
def gradient(theta, likelyhood, data, directionIndex, eps, function):
    theta[directionIndex].update(theta[directionIndex].value + eps/2)
    pos = likelyhood(data,theta,function)
    # print("grad", pos, directionIndex, theta[0].value)
    theta[directionIndex].update(theta[directionIndex].value - eps)
    neg = likelyhood(data,theta,function)
    # print("grad", pos, neg, directionIndex, theta[0].value)
    grad = (pos - neg)/eps
    return grad


def gradientStep(theta, learningRate, likelyhood, data, eps, function):
    tmp = copy.deepcopy(theta)
    for i,par in enumerate(theta):
        par.update(par.value - learningRate*gradient(copy.deepcopy(tmp),likelyhood,data,i,eps,function))
    return theta

def momentumGradientStep(theta, learningRate, likelyhood, data, eps, function, gamma, v):
    tmp = copy.deepcopy(theta)
    newV = np.empty(2)
    for i,vi in enumerate(v):
        #print( (gamma*vi + learningRate*gradient(copy.deepcopy(tmp),likelyhood,data,i,eps,function)), i, type(vi))
        newV[i] = (gamma*vi + learningRate*gradient(copy.deepcopy(tmp),likelyhood,data,i,eps,function))
    for i,par in enumerate(theta):
        par.update(par.value - newV[i])
        #print(par.value, newV[i])
    return newV

def nesterovStep(theta, learningRate, likelyhood, data, eps, function, gamma, v):
    tmp = copy.deepcopy(theta)
    newV = np.empty(2)
    for i,par in enumerate(tmp):
        par.update(par.value - gamma * v[i])
    for i,vi in enumerate(v):
        # print(learningRate*gradient(copy.deepcopy(tmp),likelyhood,data,i,eps,function), i)
        newV[i] = (gamma*vi + learningRate*gradient(copy.deepcopy(tmp),likelyhood,data,i,eps,function))
    for i,par in enumerate(theta):
        par.update(par.value - newV[i])
    return newV

def gradientDecend(theta, learningRate, likelyhood, data, Gradeps, function, maxStep, stopEps):
    LtMinusOne = 0
    Lt = 0
    Step = 0 
    likelyhoodHistory = np.empty(0)
    while Step < maxStep:
        LtMinusOne = likelyhood(data, theta, function)
        gradientStep(theta, learningRate, likelyhood, data, Gradeps, function)
        Lt = likelyhood(data, theta, function)
        likelyhoodHistory= np.append(likelyhoodHistory,Lt)
        Step += 1
        if (np.abs((LtMinusOne - Lt)))/(np.abs((LtMinusOne + Lt))) < stopEps:
            print(Step)
            return (theta, likelyhoodHistory)
    return (theta, likelyhoodHistory)

def stochasticGradientDecent(theta, learningRate, likelyhood, data, Gradeps, function, maxStep, stopEps,batchSize):
    LtMinusOne = 0
    Lt = 0
    likelyhoodHistory = np.empty(0)
    Step = 0 
    while Step < maxStep:
        LtMinusOne = likelyhood(data, theta, function)
        gradientStep(theta, learningRate, likelyhood, np.random.choice(data,size=batchSize), Gradeps, function)
        Lt = likelyhood(data, theta, function)
        likelyhoodHistory= np.append(likelyhoodHistory,Lt)
        Step += 1
        if (np.abs((LtMinusOne - Lt)))/(np.abs((LtMinusOne + Lt))) < stopEps:
            print(Step)
            return (theta, likelyhoodHistory)
    return (theta, likelyhoodHistory)

def momentumGradientDecent(theta, learningRate, likelyhood, data, Gradeps, function, maxStep, stopEps, batchSize, gamma):
    LtMinusOne = 0
    Lt = 0
    likelyhoodHistory = np.empty(0)
    Step = 0 
    v = np.asarray([0,0])
    while Step < maxStep:
        LtMinusOne = likelyhood(data, theta, function)
        v = momentumGradientStep(theta, learningRate, likelyhood, np.random.choice(data,size=batchSize), Gradeps, function, gamma, v)
        Lt = likelyhood(data, theta, function)
        likelyhoodHistory= np.append(likelyhoodHistory,Lt)
        Step += 1
        if (np.abs((LtMinusOne - Lt)))/(np.abs((LtMinusOne + Lt))) < stopEps:
            print(Step)
            return (theta, likelyhoodHistory)
    return (theta, likelyhoodHistory)

def nesterovGradientDecent(theta, learningRate, likelyhood, data, Gradeps, function, maxStep, stopEps, batchSize, gamma):
    LtMinusOne = 0
    Lt = 0
    Step = 0 
    likelyhoodHistory = np.empty(0)
    v = np.asarray([0,0])
    while Step < maxStep:
        LtMinusOne = likelyhood(data, theta, function)
        v = nesterovStep(theta, learningRate, likelyhood, np.random.choice(data,size=batchSize), Gradeps, function, gamma, v)
        Lt = likelyhood(data, theta, function)
        likelyhoodHistory= np.append(likelyhoodHistory,Lt)
        Step += 1
        if (np.abs((LtMinusOne - Lt)))/(np.abs((LtMinusOne + Lt))) < stopEps:
            print(Step)
            return (theta, likelyhoodHistory)
    print("max steps")
    return (theta, likelyhoodHistory)

sigma = Theta(-1,1)
mean = Theta(-20,20)
sigma.update(1)
mean.update(20)
theta = np.asarray([mean,sigma])
data = np.asarray([0,1,2])

print(gradient(copy.deepcopy(theta),logLikely,data,0,0.001,normalPDF))
print(gradient(copy.deepcopy(theta),logLikely,data,1,0.001,normalPDF))

# %% [markdown]
# ## Fitting data
# 
# The data sample provided with this sheet consists of Gaussian-distributed measurements of the $B^0$ meson mass, in units of MeV. The standard deviation of the distribution is dominated by the resolution of the detector used to make the measurement.
# 
# Your task is to obtain best-fit values for the mass of the meson (i.e. the mean, $\mu$) and the detector resolution (i.e. the standard deviation $\sigma$) using your minimiser and Gaussian PDFs implemented above.
# 
# In the cell below, the data is loaded into a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) and plotted as a [Matplotlib histogram](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html). This should give you an idea of which starting values and ranges to set for the parameters.

# %%
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("test_data.csv", names=["mass"])

num_bins = 50
_, bins, _ = plt.hist(df["mass"], num_bins)
bin_width = (bins[-1] - bins[0])/num_bins
plt.xlabel("Mass [MeV]")
plt.ylabel(f"Events / ({bin_width:.1f} MeV)")
plt.show()

print(len(df["mass"]))

# %%
# Use this cell to perform a fit to the data

params = []
likelyhoodHistory = []

sigma = Theta(0,300)
mean = Theta(5000,5600)
sigma.update(400)
mean.update(5250)
theta = np.asarray([mean,sigma])
likelyhoodHistory.append(gradientDecend(theta, 5, logLikely, df["mass"], 0.001, normalPDF, 20, 0.0002)[1])
print(theta[0].value, theta[1].value)
params.append(theta)

# %%
sigma = Theta(0,300)
mean = Theta(5000,5600)
sigma.update(400)
mean.update(5250)
theta = np.asarray([mean,sigma])
likelyhoodHistory.append(stochasticGradientDecent(theta, 5, logLikely, df["mass"], 0.001, normalPDF, 20, 0.00001, 350)[1])
print(theta[0].value, theta[1].value)
params.append(theta)


# %%
sigma = Theta(0,300)
mean = Theta(5000,5600)
sigma.update(400)
mean.update(5250)
theta = np.asarray([mean,sigma])
likelyhoodHistory.append(momentumGradientDecent(theta, 10, logLikely, df["mass"], 0.001, normalPDF, 30, 0.000001,350, 0.9)[1])
print(theta[0].value, theta[1].value)
params.append(theta)


# %%
sigma = Theta(0,300)
mean = Theta(5000,5600)
sigma.update(400)
mean.update(5250)
theta = np.asarray([mean,sigma])
likelyhoodHistory.append(nesterovGradientDecent(theta, 10, logLikely, df["mass"], 0.001, normalPDF, 30, 0.000001,350, 0.9)[1])
print(theta[0].value, theta[1].value)
params.append(theta)


# %% [markdown]
# ## Plotting the results
# 
# 1. Plot the PDF with best-fit values overlaid on to a histogram of the data.
#   - _**Hint**: to achieve the same normalisation for the data and the PDF, you may choose to use `density=True` in the arguments to `pyplot.hist`_
# 1. Plot a 'likelihood trace' (i.e. the 'history' of the likelihood at each iteration) for the following cases:
#   - Several choices of fixed step size for batch gradient descent
#   - The effect of different methods of gradient-descent (6 in total)
#     - batch
#     - mini-batch
#     - stochastic
#     - all of the above using the Nesterov technique

# %%
num_bins = 50
_, bins, _ = plt.hist(df["mass"], num_bins, density=True)
bin_width = (bins[-1] - bins[0])/num_bins
plt.xlabel("Mass [MeV]")
plt.ylabel(f"Events / ({bin_width:.1f} MeV)")

x = np.linspace(5000,5600,1000)
labels = ["GD","SGD","momentumGD","Nesterov"]
for i, theta in enumerate(params):
    fx = [normalPDF(xi,theta) for xi in x]
    plt.plot(x,fx, label = labels[i])
plt.legend()
plt.show()

# %%
for i,Li in enumerate(likelyhoodHistory):
    plt.plot(Li, label = labels[i])

plt.xlabel("steps")
plt.ylabel(f"LogLikelyhood")
plt.legend()
plt.show()


