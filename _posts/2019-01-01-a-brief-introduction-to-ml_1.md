---
layout: post
title: A Brief Introduction to Machine Learning for Engineers 1
categories: ML
date:   2019-01-01 22:54:40 +0300
excerpt: This is a rather dense book by *Osvaldo Simeone* which favors readers with solid background in mathematics as the title suggests. In short, its coverage can be described as a subset of Pattern Recognition and Machine Learning by Christopher Bishop.

---


## All posts

- [Chapter 2: Part I]({{ site.url }}/ml/2019/01/08/a-brief-introduction-to-ml_2_i.html)
- [Chapter 2: Part II]({{ site.url }}/ml/2019/01/15/a-brief-introduction-to-ml_2_ii.html)
- [Chapter 2: Part III]({{ site.url }}/ml/2019/01/22/a-brief-introduction-to-ml_2_iii.html)


This is a rather dense book by *Osvaldo Simeone* which favors readers with solid background in mathematics as the title suggests. In short, its coverage can be described as a subset of *Pattern Recognition and Machine Learning* by Christopher Bishop.

However, it has its moments in terms of being terse, so be prepared to comb through [stack exchange](https://stats.stackexchange.com/) for certain derivations. In this series of posts, brief chapter notes and specific derivation of equations, formulas etc left out for the reader in the book will be explained. Main references are listed below:

- *A Brief Introduction to ML for Engineers* by Osvaldo Simeone
- *Pattern Recognition and Machine Learning* by Chrisopher Bisho
- *Mathematics for Machine Learning* by Marc Peter Deisenroth, A Aldo Faisal, and Cheng Soon Ong (This is a fantastic book for strengthening math background and freely available at [here](https://mml-book.github.io/)

**Note:** Numbering of chapter titles and subtitles is chosen according to the book for making referencing easy.

**Quick Notation:** The most common symbols are shown and the rest will be clarified as introduced.

- Random variables are denoted with roman typeface $${\rm x}$$ and their value regular font $$x$$
- Matrices are denoted with upper case $$X$$ and random matrices with roman typeface $${\rm X}$$
- Vectors are in column form
- $$\log$$ is base 2 and $$ \ln $$ is natural logarithm


# 1. Introduction

Definition of machine learning given by [mahtematicalmonk](http://jwmi.github.io/index.html) such that "algorithms for inferring unknowns from knowns" captures the essence of the field. As the saying goes "thinking is comparing", we can cement our understanding of machine learning by contrasting it with classical engineering:

Classical engineering approach: Domain knowledge and in-depth analysis $$\rightarrow$$ math model $$\rightarrow$$ hand crafted solutions.

ML approach: Train a generic model with substantial amount of data $$\rightarrow$$ predict result.

Machine learning as a field generally is divided under three topics:

**Supervised Learning:** We have labeled data and main tasks are:
- Classification
- Regression

**Unsupervised Learning:** We have unlabeled data and main tasks are:
- Clustering
- Dimensionality reduction

**Reinforcement Learning:** Predicting optimal decision based on positive or negative feedback.


In addition, learning from data is executed according to following settings:
- **Passive:** Data is given **Active:** Learner can choose the data
- **Offline:** Batch of training examples **Online:** Samples are provided in a stream.

Algorithms in this book are introduced based on the theoretical arguments based on information theoretic performance metrics.

**Note:** I strongly suggest the reader to open up a [kaggle](https://www.kaggle.com/) account, because all jupyter notebooks in this site are publicly published there and can be used easily without any hassle of system specific installments or requirements.



 |Next post: [Chapter 2: Part I]({{ site.url }}/ml/2019/01/08/a-brief-introduction-to-ml_2_i.html)|

