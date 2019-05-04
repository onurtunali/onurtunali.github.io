---
layout: post
title: Empirical Rademacher Complexity and Its Implications to Deep Learning
categories: ML
date:   2019-02-01 22:54:40 +0300
excerpt: In machine learning, Rademacher complexity is used to measure the capacity of a hypothesis class from which utilized algorithm chooses its predictors and probably first proposed in [1].
---

* content
{:toc}

[Jupyter notebook]({{base.url}}/jupyter_notebooks/rademacher_complexity.ipynb)

In machine learning, Rademacher complexity is used to measure the capacity of a hypothesis class from which utilized algorithm chooses its predictors and probably first proposed in [[1]](#references). In computational learning theory, it is possible to bound generalization loss by way of using data dependent rademacher complexity under certain assumptions. This is helpful in terms of studying  and deciding the *learnability* of a problem. However, when it comes to deep learning recent findings somewhat challenge the status quo of previous claim [[2]](#references) which will be covered in later sections. Before diving into formal treatment, an informal example and explanation is given.

Let's start with a classification problem. Chosen machine learning algorithm selects its predictor function from a hypothesis class $$ h \in \mathcal{H}$$ such that $$h : \mathcal{X} \to \mathbb{R}$$. Since the task in question is classification, $$h : \mathcal{X} \to \{-1,+1\}$$. So provided a training data set $$\mathcal{D} = \{x_{1},..,x_{N}\}$$, our ideal predictor should output $$ h(x) = -1$$ if input $$x$$ is not a member of class and $$h(x) = +1$$ if it is. Because we are unaware of underlying distribution, at first an $$ \mathcal{H}$$ consists of $$h$$'s producing $$1$$ for every instance of $$x$$ is chosen.

Since size of training data is $$N$$, total number of data instance assignments is $$2^{N}$$. Therefore, our preferred $$\mathcal{H}$$ would yield a correct classification at the rate of $$ 1 / 2^{N}$$ since it classifies every instance as $$+1$$. Considering this example, Rademacher complexity formally measures the capacity of $$\mathcal{H}$$ such that how many of these assignments (of course in reality these assignments follow a distribution) can be classified correctly by $$\mathcal{H}$$.

**Disclaimer:** Combinatorics proof is rather strained and verbose. Nevertheless, the merit of combinatorics proof underlies its explanation of how an overly complicated hypothesis class is able to memorize (shatter in VC terminology) all instances of given data set.


## Formal definition

Given a class of real-valued functions $$ h \in \mathcal{H}$$ (hypothesis class)  defined on $$\mathcal{X}$$, a data sample of size $$n$$, $$\mathcal{D} = \{x_{1},..,x_{n}\}$$ distributed with respect to $$P$$ over $$\mathcal{X}$$ and finally binary values of $$ \sigma = \{\sigma_{1},...,\sigma_{n}\}$$ drawn independently from rademacher distribution which is $$ \sigma_{i} \sim \mathcal{Bern(1/2)}, P(\sigma_{i} = 1) = P(\sigma_{i} = -1) = 1/2 $$. After giving the basic concepts, we can define the following expression as the correlation:

$$ \sup_{h \in H} \left( \frac{1}{N} \sum_{i=1}^{n}~ \sigma_{i}~ h_{i} \right). \tag{1} $$

In this expression $$h_{i} = h(x_{i})$$ and expected value of correlation (1) with respect to $$\sigma$$

$$ \widehat{Rad_{n}}~(\mathcal{H}) = E_{\sigma}~\left[\sup_{h \in H} \left( \frac{1}{N} \sum_{i=1}^{n}~ \sigma_{i}~h_{i} \right) \right]. \tag{2} $$

is defined as *empirical rademacher complexity*. It will be shown that

$$ 0 \leq \widehat{Rad_{n}}~(\mathcal{H}) \leq 1 \tag{3} $$

Finally, the rademacher complexity of the class $$\mathcal{H}$$ is defined as the expectation
of $$ \widehat{Rad_{n}}~(\mathcal{H}) $$ over samples $$\mathcal{D}$$ of size n drawn according to distribution of $$P$$

$$ Rad_{n}~(\mathcal{H}) = E_{P} ~ [ \widehat{Rad_{n}}~(\mathcal{H})] \tag{4} $$


Two important points are in $$ \widehat{Rad_{n}}~(\mathcal{H}) $$ given data sample is constant, but $$\sigma$$ values are changing. In $$ Rad_{n}~(\mathcal{H})$$, both $$\sigma$$ and data sample is changing. Following proofs are constructed regarding a binary classifier.

## Combinatorics proof

**Claim:**  $$ 0 \leq \widehat{Rad_{n}}~(\mathcal{H}) \leq 1 $$

**Lower bound:** Firstly, we start with lower bound  and show that it is 0. Let's decide on an $$ \mathcal{H} $$ having a single $$h$$ in other words $$ \vert \mathcal{H} \vert = 1 $$ . Therefore, correlation (1) doesn't need supremum since there is only one $$h(\mathcal{D}) = \{h_{1}, ... , h_{N}\}$$. So equation (2) becomes

$$ \widehat{Rad_{n}}~(\mathcal{H}) = E_{\sigma}~\left[ \left( \frac{1}{N} \sum_{i=1}^{n}~ \sigma_{i}~h_{i} \right) \right] \tag{5} $$

In total there are $$2^{N}$$ different $$\sigma$$'s and the probability of any $$\sigma^{(j)}$$ is $$P(\sigma^{(j)}) = 1 / 2^{N}$$ due to the fact that $$ \sigma = \{\sigma_{1},...,\sigma_{n}\} $$ and $$\sigma_{i} \sim \mathcal{Bern(1/2)}$$. If we expand the expected value expression

$$ \widehat{Rad_{n}}~(\mathcal{H}) = \left( \frac{1}{N} \sum_{i=1}^{n}~ \sigma_{i}^{(1)}~h_{i} \right) P(\sigma^{(1)}) + ... + \left( \frac{1}{N} \sum_{i=1}^{n}~ \sigma_{i}^{(2^{N})}~h_{i} \right) P(\sigma^{(2^{N})}) $$

$$ \widehat{Rad_{n}}~(\mathcal{H}) = \left( \frac{1}{N} \sum_{i=1}^{n}~ \sigma_{i}^{(1)}~h_{i} \right) \frac{1}{2^{N}} + ... + \left( \frac{1}{N} \sum_{i=1}^{n}~ \sigma_{i}^{(2^{N})}~h_{i} \right) \frac{1}{2^{N}} $$

$$ \widehat{Rad_{n}}~(\mathcal{H}) = \frac{1}{2^{N}}~ \frac{1}{N} \left[ \left(  \sum_{i=1}^{n}~ \sigma_{i}^{(1)}~h_{i} \right)  + ... + \left(  \sum_{i=1}^{n}~ \sigma_{i}^{(2^{N})}~h_{i} \right) \right] \tag{6} $$

Now, we need to determine a way to calculate $$\sum_{i=1}^{n}~ \sigma_{i}^{(j)}~h_{i}$$ expressions. As the result of $$\vert \mathcal{H} \vert = 1$$, correlation $$\sum_{i=1}^{n}~ \sigma_{i}^{(j)}~h_{i}$$ changes according to the number of mismatches between $$h_{i}$$ and $$\sigma_{i}^{(j)}$$. At most it is $$N$$ and at least it is $$-N$$. So we can generalize the value of $$\sum_{i=1}^{n}~ \sigma_{i}^{(j)}~h_{i}$$ in terms of mismatches between given $$\sigma^{(j)}$$ and $$h$$. If there are $$k$$ mismatches, total sum would be $$(N-k) \times +1 + k \times -1 = N - 2k$$ and there would be $$\binom{N}{k}$$ different cases of $$k$$ mismatches. Remembering the fact that $$\sigma$$'s contain all possible binary distributions, we can rewrite (6) as the sum of $$0$$ mismatches through $$N$$ mismatches.

$$ = \frac{1}{2^{N}}~ \frac{1}{N} \left[ \binom{N}{0} (N - 2.0) ~ + \binom{N}{1} (N - 2.1) ~ + ... + \binom{N}{N-1)} (N - 2(N-1)) ~ + \binom{N}{N} (N - 2.N) \right] $$

$$  = \frac{1}{2^{N}}~ \frac{1}{N} \left[ \binom{N}{0} N ~ + \binom{N}{1} (N - 2) ~ + ... + \binom{N}{N-1)} (-N+2) ~ + \binom{N}{N} (-N) \right] $$

Since $$\binom{N}{k} = \binom{N}{ N-k ~}$$, previous expression is modified into

$$ \widehat{Rad_{n}}~(\mathcal{H}) = \frac{1}{2^{N}}~ \frac{1}{N} \left[ \binom{N}{0} N ~ + \binom{N}{1} (N - 2) ~ + ... - \binom{N}{1} (N-2) ~ - \binom{N}{0} N \right] = 0 $$

This result holds for both when $$N$$ is odd and even.

**Upper bound:** Secondly, we start with upper bound  and show that it is 1. This part of the proof is straightforward. Let's decide on an $$ \vert \mathcal{H} \vert = 2^{N}$$ meaning for a given $$\mathcal{D} = \{x_{1},..,x_{n}\}$$ every possible class assignment can be correctly classified by one of the $$h$$ in $$\mathcal{H}$$. Therefore, correlation (1.1) is 1 because there is always an $$h$$ with no mismatches making $$ 1/N ~\sum_{i=1}^{n}~ \sigma_{i}~h_{i} = (1/N) N $$, so supremum is 1. Finally, (5) becomes

$$ \widehat{Rad_{n}}~(\mathcal{H}) = \frac{1}{2^{N}} \left[ \underbrace{1 + ... + 1}_{2^{N}} \right]$$

$$ \widehat{Rad_{n}}~(\mathcal{H}) = 1 $$

$$ 0 \leq \widehat{Rad_{n}}~(\mathcal{H}) \leq 1 ~~~~ \Box  $$

## $$ \widehat{Rad_{n}}~(\mathcal{H}) $$ `code practice`

Following code calculates $$ \widehat{Rad_{n}}~(\mathcal{H}) $$ of a given hypothesis class with certain capacity. It doesn't require any external library, so it is possible to run the code with standard python.


```python
import itertools

n = 6  # Sample size Global variable.
test_capacity = 1  # Capacity of hypothesis class H.

WHOLE_CAPACITY_TRIAL = True  # If True, every capacity is calculated.
trial_range = [2**x for x in range(n + 1)]


class HypothesisClass:

    """ capacity should be between 1 and 2^n """

    def __init__(self, capacity):

        self.capacity = capacity
        if self.capacity > 2**n:
            raise ValueError("Capacity is greater than 2^n")
        self.whole_class = itertools.islice(itertools.product([-1, 1], repeat=n), self.capacity)

    def __iter__(self):
        return self.whole_class

    def __next__(self):
        return list(next(self.whole_class))


def rademacher_distirbution(n):
    distribution = itertools.product([-1, 1], repeat=n)
    return distribution


def supremum_finder(sigma_sample, capacity):
    H = HypothesisClass(capacity)
    supremum = -1 * float("inf")

    def correlation(sigma_sample, hypothesis_instance):
        result = sum([sigma_sample[i] * hypothesis_instance[i] for i in range(n)])
        return result

    for hypothesis_instance in H:
        hypothesis_instance = list(hypothesis_instance)
        result = correlation(sigma_sample, hypothesis_instance)
        if result > supremum:
            supremum = result

    return supremum


def emprical_rademacher_complexity(capacity):

    result = 0
    sigma_distribution = rademacher_distirbution(n)
    print(f"Capacity of H is {capacity}")

    for index, sigma_sample in enumerate(sigma_distribution):
        sigma_sample = list(sigma_sample)
       # print(f"Sigma {index} is {sigma_sample}")
        current_supremum = supremum_finder(sigma_sample, capacity)
        result += current_supremum

    return round(result / (n * 2**n), 2)


def main():

    complexity_results = []

    for capacity in trial_range:
        complexity_results.append(emprical_rademacher_complexity(capacity))

    return complexity_results


if __name__ == "__main__":

    if WHOLE_CAPACITY_TRIAL:

        print(f"\n \t Emprical Rademacher complexity of H with capacities \
        {trial_range}:\n \t Rad(H) = {main()}")

    else:
        rademacher_complexity_of_H = emprical_rademacher_complexity(test_capacity)

        print(f"\n \t Emprical Rademacher complexity of H with capacity \
        {test_capacity}:\n \t Rad(H) = {rademacher_complexity_of_H}")

```

    Capacity of H is 1
    Capacity of H is 2
    Capacity of H is 4
    Capacity of H is 8
    Capacity of H is 16
    Capacity of H is 32
    Capacity of H is 64

     	 Emprical Rademacher complexity of H with capacities [1, 2, 4, 8, 16, 32, 64]:
     	 Rad(H) = [0.0, 0.17, 0.33, 0.5, 0.67, 0.83, 1.0]


## Implications to deep learning

In conventional machine learning approaches, results of computational learning theory states that we can bound generalization loss $$\mathcal{L}_{p}~(h)$$ with using empirical loss $$\mathcal{L}_{D}~(h)$$ (for further detail check [a brief introduction to ML series](https://onurtunali.github.io/ml/2019/01/08/a-brief-introduction-to-ml_2_i.html)), empirical rademacher complexity of hypothesis class and some confidence related constant. Using the inequality (13)  in [[3]](#references) and without being too strict, we can write

$$ \mathcal{L}_{p}~(h) \leq \mathcal{L}_{D}~(h) + \widehat{Rad_{n}}~(\mathcal{H}) + C(\delta,n) \tag{7} $$

which indicates that discrepancy between generalization loss and training loss is constrained by empirical rademacher complexity.

$$ \mathcal{L}_{p}~(h) - \mathcal{L}_{D}~(h) \leq \widehat{Rad_{n}}~(\mathcal{H}) + C(\delta,n) \tag{8} $$

That is to say if a hypothesis class with high complexity is used, $$ \widehat{Rad_{n}}~(\mathcal{H}) $$ approaches to 1 and difference between training and generalization loss diverge according to (8) so that overfitting occurs. For this reason it is expected that if a hypothesis class is too complex and able to memorize all the training data ($$ \mathcal{L}_{D}~(h) \to 0 $$), then it generalizes poorly in case the amount of training data is insufficient. However, it is known that deep neural networks generalize well despite the fact that they are shown to be able to even memorize random data in [[2]](#references). This indicates that our customary understanding of generalization doesn't exactly holds for deep neural networks and a new point of view is due.

## References

[[1]](#references) Peter L. Bartlett and Shahar Mendelson, "*Rademacher and Gaussian complexities: Risk bounds and structural results*", Journal of Machine Learning Research, 2002.

[[2]](#references) Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals, "*Understanding Deep Learning Requires Rethinking Generalization*", ICLR, [arXiv:1611.03530](https://arxiv.org/abs/1611.03530), 2017.

[[3]](#references) Pirmin Lemberger, "*On Generalization and Regularization in Deep Learning*", [arXiv:1704.01312](https://arxiv.org/abs/1704.01312), 2017.


