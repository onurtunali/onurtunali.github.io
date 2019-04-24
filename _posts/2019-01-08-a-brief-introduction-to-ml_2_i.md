---
layout: post
title: A Brief Introduction to Machine Learning for Engineers 2 Part I
categories: ML
date:   2019-01-08 22:54:40 +0300
excerpt: Notes on the 2 Chapter of "A Brief Introduction to Machine Learning for Engineers".
---

* content
{:toc}



# 2. A "Gentle" Introduction Through Linear Regression

Three main learning frameworks namely frequentist, bayesian and minimum description length (MDL) are studied in this chapter. It is is not possible to make inference or generalization without assumptions. Set of all assumptions made by learning algorithm is called **inductive bias**. How to predict according to frequentist and bayesian. MDL considers parsimonious description of data.

## 2.2 Inference

**Inference problem setup:** predict random variable $${\rm t}$$, observation $${\rm x} = x$$ known and $$P({\rm t}, {\rm x})$$ denotes joint probability distribution.

To define an optimal inference, a loss function is established to curb the deviation between $$t$$ and $$\hat{t}$$ (predicted). An important loss function family is as follows:

$$ \ell_{q}~(t,\hat{t}) =  \vert  t - \hat{t}  \vert ^{q} ~~~~ (2.1)$$

Once a loss function is determined, optimal $$\hat{t}(x)$$ is characterized by minimizing $$\ell$$.

**Generalization Loss:** This loss measures the performance of our predictor for the whole population.

$$ L_{p}~(\hat{t}) = E_{(x,t)\sim P(x,t)}~[\ell_{q}~(t,\hat{t})]~~~~ (2.2)$$

The solution to this problem is given by optimal predictor:


$$ t^{*}(x) = \underset{\hat{t}}{argmin} ~ E_{t \vert x \sim P(t \vert x)}~[\ell_{q}~(t,\hat{t})  \vert  x]~~~~ (2.3)$$

This is expression is found using law of iterated expectation: $$ E_{(x,t)\sim P(x,t)}~[ Y ]= E_{ x \sim P(x)}~[E_{(t  \vert  x )\sim P(t \vert x)}~[ Y  \vert  x]]$$ and if we plug $$\ell_{q}~(t,\hat{t})$$ instead of Y, we obtain (2.3) which is called optimum predictor or Bayes prediction (not used in the book because it leads to confusion).

**Theorem 1:** *If chosen loss function is quadratic loss $$\ell_{2} =  \vert t-\hat{t}(x) \vert ^{2}$$, then the best predictor is $$ t^{*}(x) = E_{(t  \vert  x )\sim P(t \vert x)}~[ t \vert x]$$ which is the mean of the conditional probability $$P(t \vert x)$$.*

Let's show this for the discrete case:

$$ E_{(t  \vert  x )\sim P(t \vert x)}~[ \ell_{2}~(t,\hat{t}(x)) \vert x] = \sum_{t}^{} (t-\hat{t}(x))^{2} P(t \vert x) $$

expanding the quadratic form

$$ = \sum_{t}^{} t^{2}(x) P(t \vert x) - 2 \hat{t}(x) \underset{\text{equal to expected value t \vert x}}{\sum_{t}^{} t P(t \vert x)} + \hat{t}^{2}(x) \underset{\text{equal to 1, prob. axioms}}{\sum_{t}^{} P(t \vert x)} $$

and then taking the derivative w.r.t. $$\hat{t}(x)$$ and setting it to 0

$$ \frac{d E_{(t  \vert  x )\sim P(t \vert x)}~[ \ell_{2}~(t,\hat{t}(x)) \vert x]}{d\hat{t}(x)} =  0 - 2 \hat{t}(x)  E_{(t  \vert  x )\sim P(t \vert x)}~[ t \vert x] + 2 \hat{t}(x) = 0 $$

$$ \hat{t}(x) = E_{(t  \vert  x )\sim P(t \vert x)}~[ t \vert x] ~ \text{ which is the optimum predictor so } \hat{t}(x) = t^{*}(x)$$

Main goal is to come up with a predictor $$ \hat{t} $$ yielding a performance $$L_{p}~(\hat{t})$$ as close as possible to $$L_{p}~(t^{*}(x))$$. Nevertheless one should note that $$p(x,t)$$ is generally not known and this formulation of the problem applies to frequentist approach.

## 2.3 Frequentist framework:
This framework assumes that parameters exist and are constant. Also data points are drawn from an unknown underlying distribution such as $$P(x,t)$$ with independent identically distributed (i.i.d. assumption):

$$ ({\rm t_{n}}, {\rm x_{n}}) \underset{i.i.d.}{\sim} P(x,t), i=1...N$$



Since $$P(x,t)$$ is unknown two different approaches are :

- **Separate learning and inference:** Learn $$P_{D}~(t \vert x)$$ an approximation to $$P(t \vert x)$$ from data set $$D$$ and then plug this to (2.3).

$$ \hat{t}_{D}~(x) = \underset{\hat{t}}{argmin}~E_{t \vert x \sim P_{D}~(t \vert x)}~[\ell_{q}~(t,\hat{t})  \vert  x]~~~~ (2.6)$$

- **Direct inference:** Empirical Risk Minimization (ERM) technique. Learn directly an approximation $$\hat{t}~(x)$$ of the optimal decision rule by minimizing an empirical estimate

$$ \hat{t}_{D} = \underset{\hat{t}}{argmin} L_{D}~(\hat{t})~ (2.7)$$

Where empirical risk or empirical loss $$L_{D}~(\hat{t})$$ defined as:

$$ L_{D}~(\hat{t}) = \frac{1}{N} \sum_{n=1}^{N} \ell_{q}~(t_{n},\hat{t}(x_{n}) $$

First approach is more flexible. If $$P_{D}~(t \vert x)$$ is a good approximation, then any loss function can be used (of course producing different performances). Second approach is tied to the specific loss function.


**Linear regression example (page 20):**

*For x and t $$P(t,x) = P(t \vert x) P(x)$$ is not known but $$P(t \vert x) \sim \mathcal{N}~(\sin{(2\pi x)}, 0.1)$$ and $$x \sim \mathcal{U}~(0,1)$$*

Since $${\rm x}$$ has a uniform pdf and $${\rm t} \vert {\rm x}$$ has a Gaussian pdf

$$f(x) = 1, ~ f(t \vert x)=\frac{1}{\sqrt{2\pi}0.1} \exp \left(-\frac{1}{2}  \left(\frac{t - \sin{2\pi x}}{0.1}\right)^2 \right) $$

So for a given $$ {\rm x}=x_{0}$$, our best bet for the prediction of $$t$$ is setting $$\hat{t}~(x_{0}) = \sin{2\pi x_{0}}$$ which is the mean of the pdf. However using (2.3) and $$\ell_{2}$$ loss function, we can derive $$\hat{t}(x)$$ analytically as follows:

$$ E_{t \vert x \sim P(t \vert x)}~[\ell_{q}~(t,\hat{t})  \vert  x] = E_{t \vert x \sim P(t \vert x)}~[(t-\hat{t})^{2}  \vert  x]$$

$$ = \int (t-\hat{t})^{2} ~ \frac{1}{\sqrt{2\pi}~0.1} \exp \left(-\frac{1}{2}  \left(\frac{t - \sin{2\pi x}}{0.1} \right)^2 \right) ~ dt $$

$$ I = \frac{1}{\sqrt{2\pi}~0.1} \exp \left(-\frac{1}{2}  \left(\frac{t - \sin{2\pi x}}{0.1} \right)^2 \right) $$

$$ E_{t \vert x \sim P(t \vert x)}~[(t-\hat{t})^{2}  \vert  x] = \int t^2 I dt - 2 \hat{t} \int t I dt + \hat{t}^2 \int I dt $$

$$ \int t I dt$$ is equal to mean which is $$\sin{2\pi x}$$ and $$ \int I dt = 1$$, finally expression becomes

$$ E_{t \vert x \sim P(t \vert x)}~[(t-\hat{t})^{2}  \vert  x] = \int t^2 I dt - 2 \hat{t} \sin{2\pi x} + \hat{t}^2 $$

If we take the derivative of the final expression with respect to $$\hat{t}$$ and set to 0, we can find $$\hat{t}$$ minimizing the $$ E_{t \vert x \sim P(t \vert x)}~[(t-\hat{t})^{2}  \vert  x] $$:

$$ \frac{d E_{t \vert x \sim P(t \vert x)}~[(t-\hat{t})^{2}  \vert  x]}{d \hat{t}} = 0 - 2 \sin{2\pi x} + 2 \hat{t} = 0$$

$$ \hat{t} = \sin{2\pi x} $$

Finally, if evaluate $$ E_{t \vert x \sim P(t \vert x)}~[(t-\sin{2\pi x})^{2}  \vert  x]$$ with plugging in our prediction function, expression coincides with the definition of variance. So naturally, our generalization loss is the variance of $$P(t \vert x)$$ which is 0.1.


Summary of two solution approaches to the inference problem:

| Method Components | Direct Inference via ERM | Separate Learning ( $$P(t \vert x)$$ ) and Inference |
| ---- |----| --- |
| Dataset | $$ D = \{ (x_{1},t_{1}), ..., (x_{N},t_{N}) \}$$ |$$ D = \{ (x_{1},t_{1}), ..., (x_{N},t_{N}) \}$$
|Loss function |  $$ \ell_{q}~(t,\hat{t}) $$ | $$ \ell_{q}~(t,\hat{t}) $$ |
|Probability Model |  - | $$ {\rm t} \vert {\rm x} \sim P(t \vert x; \theta)$$ |
| Empirical Loss | $$ \mathcal{L}_{D}~(\hat{t}) = \frac{1}{N} \sum_{n=1}^{N} \ell_{q}~(t_{n},~\hat{t}(x_{n}))$$| -|
|Learning process | by minimizing $$ \mathcal{L}_{D}~(\hat{t})$$ | by maximizing model probability |


### Discriminative vs generative probabilistic models

**Discriminative model:** To learn the approximation of $$P_{D}~(t \vert x)$$, we start with a family of parametric distributions which is called hypothesis class. Then using a guiding criteria, predictive function is chosen by evaluation of a metric. In this setting $$\hat{t}$$ is predictive function and $$P(t \vert x;\theta)$$ is predictive distribution. Let's model $$\hat{t}$$ as a polynomial function with adding Gaussian noise

$$ \hat{t} = \mathcal{M}(x,w) = \sum_{j=0}^{M} w_{j} x^{j} = w^{T} \phi(x) $$
$$ \phi(x) = [1 ~ x ~ x^{2} ~ ... ~ x^{M}]^{T} $$

Since mean of $$P(t \vert x)$$ corresponds to the highest value of pdf in Gaussian distribution

$$ t \vert x \sim \mathcal{N}(\mathcal{M}(x,w), \beta^{-1}) $$

In here, $$\beta$$ is called precision $$ {1}/{\beta} = \sigma^{2}$$, so higher precision means lower variance.

**Generative model:** Instead of modeling predictive function, we can directly model the joint distribution $$P(x,t \vert \theta)$$ which requires stronger assumptions. Once we have learned the distribution from data, we can calculate our optimum predictor (2.6) for any loss function. Nevertheless, if starting assumptions mismatch with ground truth or real underlying distribution, bias issue would be more grave.

### Model order and model parameters

During the linear regression example two set of parameters are chosen: Degree of polynomial $$M$$ which is model order or hyperparameter that adjusts the complexity of hypothesis class, and second set is model parameters $$P(t\vert x;\theta),~ \theta = (w,\beta)$$. It should be noted that we have used a semicolon for $$ \theta $$ which indicates that it is not a random variable and a model parameter. This distinction is required due to frequentist interpretation and will be clear when we study bayesian framework.   Hypreparameters are learned through validation. Model parameters are determined through learning process.

### Maximum likelihood (ML) learning

Assuming $$M$$ is fixed, ML chooses the $$\theta$$ which maximizes the probability of observing $$D$$. We need to write the problem of $$D$$ in terms of assumed discriminative model.

$$ P(t_{D}  \vert  X_{D}, w, \beta)  = \prod_{j=1}^{N} P(t_{j}  \vert  x_{j}; w, \beta) ~~ ~ \text{ data is i.i.d so product of individual pdf's is justified} $$

Using Gaussian distribution, we need to maximize the following likelihood function

$$ = \prod_{j=1}^{N} \mathcal{N}(t_{n}  \vert  \mathcal{M}(x,w), \beta^{-1}) $$

Since it is easier to work with logarithms (in this case $$\ln$$)

$$ \ln P(t_{D}  \vert  X_{D}, w, \beta) = \ln ~ \left( \prod_{j=1}^{N} \mathcal{N}(t_{n}  \vert  \mathcal{M}(x,w), \beta^{-1}) \right)  $$

$$ \ln P(t_{D}  \vert  X_{D}, w, \beta) = \ln ~ \left( (\frac{\beta}{\sqrt{2\pi}})^{N} \exp \left(-\frac{1}{2}  \frac{\sum_{n=1}^{N}~(t_{n} - \hat{t})^{2}}{\beta^{-1}} \right) \right)  $$

$$ = \frac{N}{2} \ln( \frac{\beta}{2\pi} ) -\frac{\beta}{2}  \sum_{n=1}^{N}~(t_{n} - \hat{t}(x_{n})^{2} $$

Finally, multiplying the expression above with -2 and dividing by $$N$$ (algebraic operations with constants doesn't affect an optimization problem), we can turn this problem into a minimization:

$$ - \ln P(t_{D}  \vert  X_{D}, w, \beta) = \underset{w, \beta}{min}- \ln( \frac{\beta}{2\pi} ) + \frac{\beta}{N}\sum_{n=1}^{N}~(t_{n} - \hat{t}(x_{n})^{2} ~~~~ (2.15)$$

This expression is called negative log likelihood. By only focusing on w and discarding $$\beta$$

$$ \mathcal{L_{D}} (w) = \underset{w}{min} \frac{1}{N}\sum_{n=1}^{N}~(t_{n} - \hat{t}(x_{n})^{2} ~~~~ (2.16) $$

Let's remember that we define $$ \hat{t} = \mathcal{M}(x,w) = w^{T} \phi(x)$$, so we try to find w minimizing $$\mathcal{L_{D}} (w)$$ which is called training loss and the same as empirical loss used in ERM if $$\ell_{q} = \ell_{2}$$.


The ERM problem can be solved in closed form if we define

$$ t_{D} \in \mathbb{R}^{N} \text{ as a column vector consisting of all t's in } D~ \text{ and } t_{D} = \begin{bmatrix} t_{1} \\ .\\.\\.\\ t_{N} \end{bmatrix}$$

$$ X_{D} \in \mathbb{R}^{NxM} ~ \text{ and } ~ X_{D} = \begin{bmatrix} \phi(x_{1}) \\ .\\.\\.\\ \phi(x_{N}) \end{bmatrix} = \begin{bmatrix} \phi_{0}~(x_{1}) & ... & \phi_{M}~(x_{1}) \\ .\\.\\.\\ \phi_{0}~(x_{N}) & ... & \phi_{M}~(x_{N}) \end{bmatrix} $$

and use these in $$\mathcal{L_{D}}~(w)$$

$$ \mathcal{L_{D}}~(w) = \frac{1}{N}  \vert  \vert t_{D} - X_{D}~w  \vert  \vert ^{2} = \frac{1}{N} (t_{D} - X_{D}~w)^{T} (t_{D} - X_{D}~w) $$

$$ = \frac{1}{N}~(t_{D}^{T}~t_{D} - t_{D}^{T}~X_{D}~w - w^{T}~X_{D}^{T}~t_{D} + w^{T}~X_{D}^{T}~X_{D}~ w) $$

$$ = \frac{1}{N}~(t_{D}^{T}~t_{D} - \underset{\text{scalar so transpose is the same}}{(t_{D}^{T}~X_{D}~w)^{T}} - w^{T}~X_{D}^{T}~t_{D} + w^{T}~X_{D}^{T}~X_{D}~ w) $$

$$ = \frac{1}{N}~(t_{D}^{T}~t_{D} - 2w^{T}~X_{D}^{T}~t_{D} + w^{T}~X_{D}^{T}~X_{D}~ w) $$

Since our starting expression is convex and has a global extremum, if we take the derivative of the expression above with respect to w and set it to 0:

$$ \frac{\partial \mathcal{L_{D}}~(w)}{\partial w} = 0 - 2 X_{D}^{T}~t_{D} +  (X_{D}^{T}~X_{D} + (X_{D}^{T}~X_{D})^{T})~w = 0$$

For a compact treatment (meaning just a list of identities and formulas) of vector and matrix differentiation please check <a href="http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3274/pdf/imm3274.pdf" target="blank"> Matrix Cookbook</a>

$$ \text{Since } X_{D}^{T}~X_{D} \text{ is symmetric } = 0 - 2 X_{D}^{T}~t_{D} +  2 X_{D}^{T}~X_{D}~w = 0$$

$$ X_{D}^{T}~X_{D}~w = X_{D}^{T}~t_{D}$$

$$ (X_{D}^{T}~X_{D})^{-1} (X_{D}^{T}~X_{D})~w = (X_{D}^{T}~X_{D})^{-1} X_{D}^{T}~t_{D}$$

$$ w_{ML} = (X_{D}^{T}~X_{D})^{-1} X_{D}^{T}~t_{D} $$

The expression $$(X_{D}^{T}~X_{D})^{-1} X_{D}^{T} $$ is called Moore-Penrose pseudo inverse and denoted with  $$ X^{\dagger}$$.

After finding the $$w_{ML}$$, we can plug this value into the negative log likelihood (2.15) and try to find $$\beta$$ by differentiating w.r.t. $$\beta$$ and setting it 0

$$ \frac{d (-\ln P(t_{D}  \vert  X_{D}, w, \beta))}{d \beta} = - \frac{1}{\beta}  + \frac{1}{N}\sum_{n=1}^{N}~(t_{n} - \hat{t}(x_{n})^{2} = 0 $$

$$ \frac{1}{\beta_{ML}} = \frac{1}{N}\sum_{n=1}^{N}~(t_{n} - \hat{t}(x_{n})^{2} = \mathcal{L}_{D} (w_{ML}) $$

### Overfitting and underfitting

Choosing right model complexity (model order $$M$$) for optimum result. In previous example $$\phi(x) = [1 ~ x ~ x^{2} ~ ... ~ x^{M}]$$ and $$ w = [w_{0} ~ ... ~ w{M}]$$, therefore setting any $$w_{i} = 0$$ decreases the model complexity. When $$M = 1$$ model is not rich enough to capture the data so underfits and $$M=9$$ model is too complex so overfitting occurs in other words training loss $$\mathcal{L}_{D}~(w_{ML})$$ is small, however generalization loss $$\mathcal{L}_{P}~(w_{ML})$$ is large.

When dataset is large comparing to the model parameters, $$\mathcal{L}_{D}~(w_{ML})$$ provides an accurate measurement for $$\mathcal{L}_{P}~(w_{ML})$$. If we define $$w^{*}$$ as follows:

$$w^{*} = \underset{w}{argmin} ~ \mathcal{L}_{P}~(w) ~~~~ (2.21) $$

assuming N is large enough, $$w_{ML}$$ ~ tends to $$w^{*}$$. To put it differently, when $$N \to \infty ~~ \mathcal{L}_{D}~(w_{ML}) \approx \mathcal{L}_{P}~(w_{ML}) \approx \mathcal{L}_{P}~(w^{*})$$

- N is small, overfitting causing estimation error dominates bias caused by small model order
- N is large, harder to overfit, so estimation error is dominated by bias depending on the choice of model order

We can decompose bias and estimation error as follows and let's remember that $$t^{*}$$ (2.3) is the optimum predictor:

$$ L_{p}~(w_{ML}) = L_{p}~(t^{*}) + \underset{\text{Bias}}{\left[L_{p}~(w^{*}) - L_{p}~(t^{*}) \right]} + \underset{\text{Estimation error}}{\left[L_{p}~(w_{ML}) - L_{p}~(w^{*}) \right]} ~~~~ (2.22) $$

- Increasing N decreases estimation error which means $$ L_{p}~(w_{ML}) \approx L_{p}~(w^{*})$$. To put is differently model reacher its potential.

- It is clear from (2.22) that increasing N has no effect on bias. Because this is inherent to the model type and model order.

### Validation and testing

Dataset given for a machine learning task is divided into three:
- **Training set:** Used during learning process for determining model parameters.
- **Validation set:** Used to determine hyperparameters of algorithm, so we don't overfit.
- **Test set:** Used to evaluate overall performance of algorithm after model and hyperparameters are finalized.

Since $$L_{p}~(t^{*})$$ is not possible to calculate due to the fact that $$P(x,t)$$ is unknown, we utilize test set to calculate an approximation to $$L_{p}~(t^{*})$$ by testing values never used before.

### Maximum likelihood `code practice`

- N is the sample size
- $$L_{D}~(\hat{t})$$ is training loss
- $$L_{P}~(\hat{t})$$ is generalization loss which is approximated by validation using Root Mean Squared Metric

Observations:

- Notice the difference between $$L_{D}~(\hat{t})$$ and $$L_{p}~(\hat{t})$$ for M = 9 and `N=15`. This is a clear sign of overfitting.

- By increasing `N=1000`, it is easy to see that optimum model order $$M$$ = 5.

- Even though increasing N helps finding optimum $$M$$, we are still sampling from a constrained domain meaning $$0 \leq x \leq 1$$ which doesn't capture the periodic nature of true distribution ($$\sin 2 \pi x$$).

- By increasing `domain_range = 2`, we immediately see that our previous optimum $$M$$ = 5 doesn't yield the best result anymore. $$M$$ = 7 is the new optimum model order. One interpretation of this phenomenon is that true distribution has more 0 values in the new range, therefore higher degree polynomial is a better fit.

- Takeaway lesson is that in real life situation our sample size is generally constant, so jumping to conclusions just because we are getting good results from validation and test doesn't always mean our model will perform competently in real life.



```python
%matplotlib inline

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, linalg


N = 15  # sample size
M = 9  # model order
domain_range = 1 # range of x values

def t_true(x):
    return np.sin(2 * np.pi * x)


def fi(x, M):
    return np.array([x**i for i in range(0, M + 1)])


def data_generator(N, test_ratio=0.3):
    np.random.seed(42)  # To obtain the same result
    train_ratio = 1 - test_ratio
    limit_index = math.floor(N * train_ratio)
    nonoutlier_ratio = math.ceil(N * 0.7)

    X = np.linspace(0, domain_range, N, endpoint=False)
    nonoutlier_index = np.random.choice(X.shape[0], nonoutlier_ratio, replace=False)
    noise = np.array(stats.norm.rvs(loc=0, scale=0.35, size=N))
    noise[nonoutlier_index] = 0
    t = t_true(X) + noise

    # Randomization of dataset
    index = np.random.choice(X.shape[0], X.shape[0], replace=False)
    X = X[index]
    t = t[index]

    X_test = X[limit_index: None]
    X_train = X[: limit_index]

    t_test = t[limit_index: None]
    t_train = t[: limit_index]

    print(f" Training size: {X_train.shape[0]} and test size: {X_test.shape[0]}")
    return X_train, t_train, X_test, t_test



def design_matrix(X, m):
    Xd = np.array([fi(x, m) for x in X])
    return Xd


def fit_ML(X, t, m):
    Xd = design_matrix(X, m)
    Xpenrose = np.matmul(linalg.inv(np.matmul(Xd.T, Xd)), Xd.T)
    w = np.matmul(Xpenrose, t)
    return w


def predic(w, X_test, m):
    Xd = design_matrix(X_test, m)
    t_predic = np.matmul(Xd, w)
    return t_predic


def rms_calculator(t1, t2):
    """ L_P(t): Generalization loss"""

    result = np.sqrt(np.sum((t1 - t2)**2) / len(t1))
    result = round(result, 3)
    return result

def training_loss(t1,t2):
    """ L_D(t): Training loss"""
    result = np.sum((t1 - t2)**2) / len(t1)
    result = round(result, 3)
    return result


X_train, t_train, X_test, t_test = data_generator(N, test_ratio=0.2)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), dpi=120)

for m in [1,5,7,9]:

    w = fit_ML(X_train, t_train, m)
    t_predic = predic(w, X_test, m)
    RMS = rms_calculator(t_test, t_predic)
    L_dt = training_loss(t_train, predic(w, X_train, m) )

    x = np.linspace(0, domain_range, 50)
    y = predic(w, x, m)
    ax1.plot(x, y, label=f"m = {m}, " + r"$$L_{D}(\hat{t}) = $$" + f" {L_dt}")

    ax2.scatter(X_test, t_predic, label=f"m = {m}, " + r" $$L_{P}(\hat{t}) = $$" + f" {RMS}")
    ax2.set_title(f"Test Values vs Predicted Values\n(Validation)", fontweight="bold")


ax1.set_title(f"Predictor Functions\n (Sample Size = {N})", fontweight="bold")
ax1.set_ylim(-3, 3)
ax1.set_xlabel("x")
ax1.set_ylabel("t")
ax1.scatter(X_train, t_train, label=f"True values", marker="o", facecolor="none", edgecolor="k")
ax1.legend(fontsize=9)

ax2.scatter(X_test, t_test, label="True values", marker="o", facecolor="none", edgecolor="k")
ax2.legend(fontsize=9)
ax2.set_ylim(-3, 3)
ax2.set_xlabel("x")
ax2.set_ylabel("t")
plt.tight_layout()
plt.savefig('ml_results_1.png')
plt.show()
plt.show()

```

     Training size: 12 and test size: 3



![](/img/abitmlfe/ml_results.png)


---

| Previous post: [Chapter 1]({{ site.url }}/ml/2019/01/01/a-brief-introduction-to-ml_1.html) |  Next post: [Chapter 2: Part II]({{ site.url }}/ml/2019/01/15/a-brief-introduction-to-ml_2_ii.html) |
