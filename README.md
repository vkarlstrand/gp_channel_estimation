
# Gaussian Process Methods for Estimating Channel Characteristics
For the course *Degree Project in Electrical Engineering, First Cycle* (course code EF112X) at KTH Royal Institute of Technology, Stockholm, Sweden.

Created by Viktor Karlstrand and Anton Ottosson, February 2020.


## Table of Contents
- [Repository content](#content)
- [References](#references)



## Abstract
Gaussian processes (GPs) as a Bayesian regression method have been around for some time. Since proven advantageous for sparse and noisy data, we explore the potential of Gaussian process regression (GPR) as a tool for estimating radio channel characteristics.

Specifically, we consider the estimation of a time-varying continuous transfer function from discrete samples. We introduce the basic theory of GPR, and employ both GPR and its deeplearning counterpart deep Gaussian process regression (DGPR) for estimation. We find that both perform well, even with few samples. Additionally, we relate the channel coherence bandwidth to a GPR hyperparameter called length-scale. The results show a tendency towards proportionality, suggesting that our approach offers an alternative way to approximate the coherence bandwidth.




 ## Repository content <a id="content"></a>
- `demos` - Different demonstrations of GP's and DGP's.
- `code` - Final implementation used in the paper.

To see table of contents for each folder, enter the folder and read the `README.md`.

**File naming convention:**
- `gpr` - Gaussian Process Regression
- `dgpr`- Deep Gaussian Process Regression
- `1d`, `2d1` - *input dimension*`d`*output dimension*



## References <a id="references"></a>
The project is heavily based on code from [GPyTorch](https://gpytorch.ai/) and [GPflow](https://github.com/GPflow/GPflow).

We recommend watching Kilian Q. Weinberger's lectures on [YouTube](https://www.youtube.com/watch?v=MrLPzBxG95I&list=PLl8OlHZGYOQ7bkVbuRthEsaLr7bONzbXS) for a general understandning of machine learning. Christopher M. Bishop's book *Pattern Recognition And Machine Learning (Springer 2006)* is also great.


**GPyTorch**
> J. R. Gardner, G. Pleiss, D. Bindel, K. Q. Weinberger and A. G. Wilson. GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration. Advances in Neural Information Processing Systems, 2018. (Link: https://arxiv.org/abs/1809.11165)

**GPflow**
> A. Matthews, M. van der Wilk, Z. Ghahramani and J. Hensman. GPflow: A Gaussian process library using TensorFlow. Journal of Machine Learning Research, vol. 18, no. 40, pp. 1–6, April 2017. (Link: http://www.jmlr.org/papers/volume18/16-537/16-537.pdf)

**Gaussian Processes**
> C. E. Rasmussen and C. K. I. Williams. Gaussian Processes for Machine Learning (Adaptive Computation and Machine Learning). The MIT Press. 2005. (Link: http://www.gaussianprocess.org/gpml/chapters/RW.pdf)

**Deep Gaussian Processes**
> A. C. Damianou and N. D. Lawrence. Deep Gaussian Processe. 2012. (Link: http://proceedings.mlr.press/v31/damianou13a.pdf)

 ## DISCLAIMER
 **Since we are new to machine learning, GP's and DGP's in general, there might be some errors and misunderstandings.
 Please have this in mind. Thank you!**
