---
layout: post
title:  "Notes on Linear Gaussian Models"
date:   2025-05-16 00:00:00 -0000
---

The goal of this note is to rehearse some classic results on linear Gaussian models. This mostly covers linear Gaussian state space models, a.k.a. Kalman filter. We will focus most on variational inference which in this setting yields an interesting connection with linear quadratic control. This massively simplifies the derivation.

To gain some motivation, the [Gaussian belief propagation](https://gaussianbp.github.io/) blog post provides some highly aesthetically pleasing visualizations.

## Brief intro to linear Gaussian models
Linear Gaussian models assume that the relationships between all modeled variables are linear with Gaussian noise. 

For example, consider two variables $$x \in \mathbb{R}^{m}, z \in \mathbb{R}^{n}$$. Let $$x$$ be dependent on $$z$$ according to:
<center>
$$
x = W^{\intercal}z + b + \epsilon, \epsilon \sim \mathcal{N}(\epsilon\vert  0, \Phi)
$$
</center>
where $$W \in \mathbb{R}^{n \times m}, b \in \mathbb{R}^{n}$$ are the weights and bias of the linear transformation, and $$\epsilon \in \mathbb{R}^{m}$$ is a noise sampled from a Gaussian distribution with covariance $$\Phi$$. 

We can express the same relationship using the following probability distributions:
<center>
$$
\begin{align}
P(x, z) &= P(x\vert z)P(z) \\
P(z) &= \mathcal{N}(z\vert  \mu, \Sigma) \\
P(x\vert z) &= \mathcal{N}(x\vert  W^{\intercal}z + b, \Phi)
\end{align}
$$
</center>
This is the model that underlies probabilistic PCA (pPCA), where $$x$$ represent observational data and $$z$$ represent latent factors, and $$W$$ is the loading matrix or principle components.

A linear Gaussian hidden Markov model (HMM) is the following:
<center>
$$
\begin{align}
P(x_{0:T}, z_{0:T}) &= \prod_{t=0}^{T}P(z_{t}\vert z_{t-1})P(x_{t}\vert z_{t}) \\
P(z_{t}\vert z_{t-1}) &= \mathcal{N}(z_{t}\vert  A^{\intercal}z_{t-1} + b, \Phi) \\
P(x_{t}\vert z_{t}) &= \mathcal{N}(x_{t}\vert  C^{\intercal}z_{t} + d, \Psi) \\
\end{align}
$$
</center>
where $$P(z_{0}\vert z_{-1}) = P(z_{0})$$ is the initial latent state distribution and $$A, b, C, d$$ are the weights and biases of shapes similar to pPCA. The transition distribution $$P(z_{t}\vert z_{t-1})$$ is usually referred to as a dynamical system.

In both pPCA and LG-HMM, an operation of major interest is inferring the latent factor or latent states from observations of $$x$$'s. This corresponds to computing the posterior distribution over the latents using the Bayes rule:
<center>
$$
P(Z\vert X) = \frac{P(X\vert Z)P(Z)}{\int_{Z}P(X\vert Z)P(Z)}
$$
</center>
Depending on the purpose, the $$X, Z$$ could be different things. For pPCA, $$Z = z, X = x$$. For HMM, usually $$Z = z_{t}$$ and either $$X = x_{0:t}$$ or $$X = x_{0:T}$$ depending on whether you are interested in filtering or smoothing. Doing these operations in linear Gaussian HMM is typically referred to Kalman filtering or smoothing.

## Predict and update in Bayesian inference
Two operations are particularly characteristics of Bayesian inference, especially sequential Bayesian inference of the kind in Kalman filtering. The predict step computes the distribution over future values after integrating out uncertainty over current values. The update step corrects the predictive distribution upon observing new signals.

### Predict step
Take Kalman filtering for example. Suppose your current belief about the latent state is $$P(z_{t}) = \mathcal{N}(z_{t}\vert  \mu, \Sigma)$$. The predictive distribution pushes the latent state through the linear transformation while marginalizing all uncertainty over $$z_{t}$$:
<center>
$$
P(z_{t+1}) = \int_{z_{t}}P(z_{t+1}\vert z_{t})P(z_{t})
$$
</center>
Computing this directly is no easy task. However, for linear Gaussian systems, we can arrive at a solution via the following steps.

First, suppose we only consider a know $$z_{t}$$ as opposed to a whole distribution over $$z_{t}$$. We can express the process generating $$z_{t+1}$$ as:
<center>
$$
z_{t+1} = A^{\intercal}z_{t} + b + \epsilon, \epsilon \sim \mathcal{N}(\epsilon\vert  0, \Phi)
$$
</center>
We know a linear transformation multiplies the variance by the weights (e.g., weights higher than 1 increases variance). Thus:
<center>
$$
A^{\intercal}z_{t} + b \sim \mathcal{N}(A^{\intercal}\mu + b, A^{\intercal}\Sigma A)
$$
</center>
Finally, we know the inherent noise of $$z_{t+1}$$ will corrupt its value regardless of the variance due to uncertainty about $$z_{t}$$. We thus arrive at the following:
<center>
$$
P(z_{t+1}) = \mathcal{N}(z_{t+1}\vert  A^{\intercal}\mu + b, A^{\intercal}\Sigma A + \Psi)
$$
</center>

### Update step
Let's now move one time step into the future. Suppose your predictive distribution of $$z_{t}$$ using the previous estimate is $$P(z_{t}) = \mathcal{N}(z_{t}\vert  \mu, \Sigma)$$. Upon observing $$x_{t}$$, you want to compute $$P(z_{t}\vert x_{t})$$. Iteratively doing so leads to Kalman filtering. 

One way to solve this problem is to leverage the Gaussian conditional density formula: Let a fully Gaussian distribution be $$\mathcal{N}(x; \mu, \Sigma)$$. Let us observe a subset of the variables $$x_{2}$$. We can partition the parameters as follows:
<center>
$$
\mu = \left[\begin{array}{c}
\mu_{1} \\
\mu_{2} \\
\end{array}\right], \Sigma = \left[\begin{array}{cc}
\Sigma_{11} & \Sigma_{12} \\
\Sigma_{21} & \Sigma_{22} \\
\end{array}\right]
$$
</center>
The conditional Gaussian $$P(x_{1}\vert x_{2}) = \mathcal{N}(x_{1}; \mu_{1\vert 2}, \Sigma_{1\vert 2})$$ has the following parameters:
<center>
$$
\begin{align}
\mu_{1\vert 2} &= \mu_{1} + \Sigma_{12}\Sigma_{22}^{-1}(x_{2} - \mu_{2}) \\
\Sigma_{1\vert 2} &= \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
\end{align}
$$
</center>
The full derivation can be found [here](https://statproofbook.github.io/P/mvn-cond.html).

We can get some intuition from a simpler bivariate case. Let $$\rho$$ denote the covariance between $$x_1$$ and $$x_2$$. The above formula reduce to:
<center>
$$
\begin{align}
\mu_{1\vert 2} &= \mu_{1} + \rho\frac{(x_{2} - \mu_{2})}{\sigma_{2}^{2}} \\
\sigma_{1\vert 2}^{2} &= \sigma_{11}^{2} - \frac{\rho^{2}}{\sigma_{2}^{2}}
\end{align}
$$
</center>
In other words, if $$x_1$$ and $$x_2$$ are correlated, then the new mean for $$x_1$$ gets adjusted by the precision weighted deviation in $$x_{2}$$ (or its prediction residual), and the new variance gets reduced by the precision weighted covariance.

To get the update step, let's first write down the joint distribution $$P(z, x)$$:
<center>
$$
P(z, x) = \mathcal{N}\left(
\left[\begin{array}{c}
z \\
x \\
\end{array}\right]; 
\left[\begin{array}{c}
\mu \\
C^{\intercal}\mu + d \\
\end{array}\right], 
\left[\begin{array}{cc}
\Sigma & \Sigma C\\
C^{\intercal}\Sigma & C^{\intercal}\Sigma C + \Psi \\
\end{array}\right], 
\right)
$$
</center>

Thus, upon observing $$x$$, we can obtain the posterior over $$z$$ using the conditional formula:
<center>
$$
\begin{align}
\mu_{z\vert x} &= \mu + \Sigma C(C^{\intercal}\Sigma C + \Psi)^{-1}(x - C^{\intercal}\mu - d)\\
\Sigma_{z\vert x} &= \Sigma - \Sigma C(C^{\intercal}\Sigma C + \Psi)^{-1}C^{\intercal}\Sigma\\
\end{align}
$$
</center>
The matrix $$\Sigma C(C^{\intercal}\Sigma C + \Psi)^{-1}$$ is called the Kalman gain. 

## Variational inference and linear quadratic control

Using the predict and update formulas is really inconvenient. For example, to derive Kalman smoothing, you will need some other formulas. A much more convenient method is to use variational inference where filtering and smoothing are treated in the almost the same way. 

### Variational inference
Considering the filtering problem as a first step. Variational filtering aims to find an approximate posterior $$Q(z_{t})$$ which minimizes the KL divergence from the true posterior. $$Q(z_{t})$$ is usually chosen to be Gaussians for continuous latents. From the literature, we know that this is equivalent to maximizing the evidence lower bound (ELBO):
<center>
$$
\mathcal{L}(Q) = \mathbb{E}_{Q(z_{t})}[\log P(x_{t}\vert z_{t}) + \log P(z_{t})] + \mathbb{H}[Q]
$$
</center>
where $$\mathbb{H}[Q]$$ is the entropy of $$Q$$. 

The nice thing about variational inference is that smoothing optimizes almost the same objective. The only difference is that instead of finding the posterior distribution over $$z_{t}$$, we want to find the posterior distribution over $$z_{0:T}$$. We will assume the mean-field factorization over the approximate posterior: $$Q(z_{0:T}) = \prod_{t=0}^{T}Q(z_{t})$$. The ELBO can be written as:
<center>
$$
\begin{align}
\mathcal{L}(Q) &= \mathbb{E}_{Q(z_{0:T})}[\log P(x_{0:T}\vert z_{0:T}) + \log P(z_{0:T})] + \mathbb{H}[Q] \\
&= \sum_{t=0}^{T}\left\{\mathbb{E}_{Q(z_{0:T})}[\log P(x_{t}\vert z_{t}) + \log P(z_{t}\vert z_{t-1})] + \mathbb{H}[Q_{t}]\right\}
\end{align}
$$
</center>

From the literature, we know the optimal posteriors have the following form:
<center>
$$
\begin{align}
Q^{filter}(z_{t}) &\propto \exp\left(\log P(x_{t}\vert z_{t}) + \log P(z_{t})\right) \\
Q^{smooth}(z_{t}) &\propto \exp\left(\log P(x_{t}\vert z_{t}) + \mathbb{E}_{Q^{smooth}(z_{t+1})}[\log P(z_{t+1}\vert z_{t})] + \mathbb{E}_{Q^{smooth}(z_{t-1})}[\log P(z_{t}\vert z_{t-1})]\right) \\
\end{align}
$$
</center>
The only difference in the smoothing posterior is that we need to average over the posterior of adjacent time steps.

The key insight is the following: **Given the model is linear Gaussian, the log likelihood and log prior in the ELBO are quadratic functions of the latent variables. Maximizing ELBO thus corresponds to solving a linear quadratic control problem, with an additional requirement of maximizing the entropy of the control policy.**

#### **Expectation of quadratic form**
Before we proceed, we will need to get good at working with expectation of quadratic forms. This will show up a lot in linear Gaussian systems because the log likelihood of Gaussian distributions is quadratic. The main property is the following (see proof [here](https://statproofbook.github.io/P/mean-qf.html)):
<center>
$$
\mathbb{E}_{X \sim \mathcal{N}(\mu, \Sigma)}[X^{\intercal}AX] = \mu^{\intercal}A\mu + \mathbf{Tr}(A\Sigma)
$$
</center>

Extending this to our setting with linear transformation:
<center>
$$
\begin{align}
\mathbb{E}_{X}[(B^{\intercal}X + b)^{\intercal}A(B^{\intercal}X + b)] = \mathbb{E}_{X}[(B^{\intercal}X)^{\intercal}A(B^{\intercal}X) + b^{\intercal}Ab + (B^{\intercal}X)^{\intercal}Ab + b^{\intercal}A(B^{\intercal}X)]
\end{align}
$$
</center>
Notice the second term is a constant and the last two terms are linear transformations of $$X$$. Thus, for those terms, we have:
<center>
$$
\mathbb{E}_{X}[b^{\intercal}Ab + (B^{\intercal}X)^{\intercal}Ab + b^{\intercal}A(B^{\intercal}X)] = b^{\intercal}Ab + (B^{\intercal}\mu)^{\intercal}Ab + b^{\intercal}A(B^{\intercal}\mu)
$$
</center>
The last two terms here can actually be combined because they are bilinear terms made up of the same quantities.

For the first term, let $$C = BAB^{\intercal}$$, we have:
<center>
$$
\begin{align}
\mathbb{E}_{X}[(B^{\intercal}X)^{\intercal}A(B^{\intercal}X)] &= \mu^{\intercal}C\mu + \mathbf{Tr}(C\Sigma) \\
&= (B^{\intercal}\mu)A(B^{\intercal}\mu) + \mathbf{Tr}(BAB^{\intercal}\Sigma)
\end{align}
$$
</center>
Putting together, we have:
<center>
$$
\mathbb{E}_{X}[(B^{\intercal}X + b)^{\intercal}A(B^{\intercal}X + b)] = (B^{\intercal}\mu + b)A(B^{\intercal}\mu + b) + \mathbf{Tr}(BAB^{\intercal}\Sigma)
$$
</center>
We will use this extensively below.

#### **Variational filtering**
Let's now derive the variational filtering solution in detail. Let us denote the variational posterior as $$Q(z) = \mathcal{N}(z\vert  \mu_{q}, \Sigma_{q})$$, dropping the time index. The ELBO objective is the following:
<center>
$$
\max_{\mu_{q}, \Sigma_{q}}\mathbb{E}_{Q(z)}[\log P(x\vert z) + \log P(z)] + \mathbb{H}[Q]
$$
</center>
The log likelihood and log prior can be written as:
<center>
$$
\begin{align}
\log P(x\vert z) &= -\frac{1}{2}\bigg[(x - C^{\intercal}z - d)^{\intercal}\Phi^{-1}(x - C^{\intercal}z - d) + \log\det(\Psi) + m\log(2\pi)\bigg] \\
\log P(z) &= -\frac{1}{2}\bigg[(z - \mu)^{\intercal}\Sigma^{-1}(z - \mu) + \log\det(\Sigma) + n\log(2\pi)\bigg]
\end{align}
$$
</center>
The entropy of $$Q(z)$$ is:
<center>
$$
-\mathbb{E}_{Q(z)}[\log Q(z)] = \frac{1}{2}\bigg[\log\det(\Sigma_{q}) + n\log(2\pi) + n\bigg]
$$
</center>

Applying the quadratic form identity to the expected log likelihood with constants dropped, we have:
<center>
$$
\begin{align}
&\mathbb{E}_{Q(z)}[(x - C^{\intercal}z - d)^{\intercal}\Phi^{-1}(x - C^{\intercal}z - d)] \\
&= (x - C^{\intercal}\mu_{q} - d)\Phi^{-1}(x - C^{\intercal}\mu_{q} - d) + \mathbf{Tr}(C\Psi^{-1}C^{\intercal}\Sigma_{q}) \\
\end{align}
$$
</center>
Similarly, for the expected log prior, we have:
<center>
$$
\begin{align}
\mathbb{E}_{Q(z)}[(z - \mu)^{\intercal}\Sigma^{-1}(z - \mu)] = (\mu_{q} - \mu)^{\intercal}\Sigma^{-1}(\mu_{q} - \mu) + \mathbf{Tr}(\Sigma^{-1}\Sigma_{q})
\end{align}
$$
</center>

Plugging back to the objective, dropping constants that don't depend on $$\mu_{q}, \Sigma_{q}$$, and multiplying all terms by 2, we need to solve the following optimization problem:
<center>
$$
\begin{align}
\min_{\mu_{q}, \Sigma_{q}} L(\mu_{q}, \Sigma_{q}) &= (x - C^{\intercal}\mu_{q} - d)\Phi^{-1}(x - C^{\intercal}\mu_{q} - d) + \mathbf{Tr}(C\Psi^{-1}C^{\intercal}\Sigma_{q}) \\
&\quad + (\mu_{q} - \mu)^{\intercal}\Sigma^{-1}(\mu_{q} - \mu) + \mathbf{Tr}(\Sigma^{-1}\Sigma_{q}) + \log\det(\Sigma_{q})
\end{align}
$$
</center>
Taking the gradient w.r.t. $$\mu_{q}$$ and set to zero, we get:
<center>
$$
\begin{align}
\nabla_{\mu_{q}}L &= \nabla_{\mu_{q}}\bigg\{(x - C^{\intercal}\mu_{q} - d)\Psi^{-1}(x - C^{\intercal}\mu_{q} - d) + (\mu_{q} - \mu)^{\intercal}\Sigma^{-1}(\mu_{q} - \mu)\bigg\}\\
&= -C\Psi^{-1}(x - C^{\intercal}\mu_{q} - d) + \Sigma^{-1}(\mu_{q} - \mu) \\
&= -C\Psi^{-1}(x - d) - \Sigma^{-1}\mu + (C\Psi^{-1}C^{\intercal} + \Sigma^{-1})\mu_{q} \\
&= -C\Psi^{-1}(x - d) + C\Psi^{-1}C^{\intercal}\mu - C\Psi^{-1}C^{\intercal}\mu - \Sigma^{-1}\mu + (C\Psi^{-1}C^{\intercal} + \Sigma^{-1})\mu_{q} \\
&= -C\Psi^{-1}(x - C^{\intercal}\mu - b) - (C\Psi^{-1}C^{\intercal} + \Sigma^{-1})\mu + (C\Psi^{-1}C^{\intercal} + \Sigma^{-1})\mu_{q} \\
&:= 0 \\
\mu_{q} &= \mu + (C\Psi^{-1}C^{\intercal} + \Sigma^{-1})^{-1}C\Psi^{-1}(x - C^{\intercal}\mu - d)
\end{align}
$$
</center>

Taking the gradient w.r.t $$\Sigma_{q}$$ and set to zero, we get:
<center>
$$
\begin{align}
\nabla_{\Sigma_{q}}L &= \nabla_{\Sigma_{q}}\bigg\{\mathbf{Tr}(C\Psi^{-1}C^{\intercal}\Sigma_{q}) + \mathbf{Tr}(\Sigma^{-1}\Sigma_{q}) + \log\det(\Sigma_{q})\bigg\} \\
&= C\Psi^{-1}C^{\intercal} + \Sigma^{-1} - \Sigma_{q}^{-1} \\
&:= 0 \\
\Sigma_{q} &= (C\Psi^{-1}C^{\intercal} + \Sigma^{-1})^{-1}
\end{align}
$$
</center>

We can show that $$\mu_{q} = \mu_{z\vert x}$$ by showing the coefficients in front of the residual $$x - C^{\intercal}\mu - d$$ is equivalent to that of the non-variational update:
<center>
$$
\begin{align}
(C\Psi^{-1}C^{\intercal} + \Sigma^{-1})^{-1}C\Psi^{-1} &= \Sigma C(C^{\intercal}\Sigma C + \Psi)^{-1} \\
C\Psi^{-1}(C^{\intercal}\Sigma C + \Psi) &= (C\Psi^{-1}C^{\intercal} + \Sigma^{-1})\Sigma C \\
C\Psi^{-1}C^{\intercal}\Sigma C + C &= C\Psi^{-1}C^{\intercal}\Sigma C + C
\end{align}
$$
</center>
We can also show $$\Sigma_{q} = \Sigma_{z\vert x}$$ in a similar way:
<center>
$$
\begin{align}
(C\Psi^{-1}C^{\intercal} + \Sigma^{-1})^{-1} &= \Sigma - \Sigma C(C^{\intercal}\Sigma C + \Psi)^{-1}C^{\intercal}\Sigma \\
I &= (C\Psi^{-1}C^{\intercal} + \Sigma^{-1})\Sigma - (C\Psi^{-1}C^{\intercal} + \Sigma^{-1})\Sigma C(C^{\intercal}\Sigma C + \Psi)^{-1}C^{\intercal}\Sigma \\
\cancel{I} &= C\Psi^{-1}C^{\intercal}\Sigma + \cancel{I} - (C\Psi^{-1}C^{\intercal} + \Sigma^{-1})\Sigma C(C^{\intercal}\Sigma C + \Psi)^{-1}C^{\intercal}\Sigma \\
C\Psi^{-1}C^{\intercal}\Sigma &= (C\Psi^{-1}C^{\intercal} + \Sigma^{-1})\Sigma C(C^{\intercal}\Sigma C + \Psi)^{-1}C^{\intercal}\Sigma \\
C\Psi^{-1}C^{\intercal}\Sigma(C^{\intercal}\Sigma C + \Psi) &= (C\Psi^{-1}C^{\intercal} + \Sigma^{-1})\Sigma CC^{\intercal}\Sigma \\
C\Psi^{-1}C^{\intercal}\Sigma C^{\intercal}\Sigma C + C\Psi^{-1}C^{\intercal}\Sigma\Psi &= C\Psi^{-1}C^{\intercal}\Sigma CC^{\intercal}\Sigma + \Sigma^{-1}\Sigma CC^{\intercal}\Sigma \\
C\Psi^{-1}C^{\intercal}\Sigma C^{\intercal}\Sigma C + CC^{\intercal}\Sigma &= C\Psi^{-1}C^{\intercal}\Sigma CC^{\intercal}\Sigma + CC^{\intercal}\Sigma \\
\end{align}
$$
</center>

#### **Variational smoothing**
Let's now witness how easy it is to do smoothing with the variational technique; no complicated backward pass! 

To start off, let's write down the objective function related to a single time step of $$Q(z_{t})$$:
<center>
$$
\max_{\mu_{t}, \Sigma_{t}}\mathbb{E}_{Q(z_{t})}[\log P(x_{t}\vert z_{t})] + \mathbb{E}_{Q(z_{t}, z_{t+1})}[\log P(z_{t+1}\vert z_{t})] + \mathbb{E}_{Q(z_{t-1}, z_{t})}[\log P(z_{t}\vert z_{t-1})] + \mathbb{H}[Q]
$$
</center>
We see that the difference from before is we need to take expectation over the latents from adjacent time steps. Furthermore, we need to make use of transition log likelihood of the form:
<center>
$$
\log P(z_{t}\vert z_{t-1}) = -\frac{1}{2}\bigg[(z_{t} - A^{\intercal}z_{t-1} - b)^{\intercal}\Psi^{-1}(z_{t} - A^{\intercal}z_{t-1} - b) + \log\det(\Phi) + n\log(2\pi)\bigg] \\
$$
</center>

Let us denote $$Q(z_{t-1}) = \mathcal{N}(z_{t-1}\vert  \mu_{t-1}, \Sigma_{t-1})$$, $$Q(z_{t+1}) = \mathcal{N}(z_{t+1}\vert  \mu_{t+1}, \Sigma_{t+1})$$ . We can write the quadratic terms in the expected transition log likelihoods as:
<center>
$$
\begin{align}
&\mathbb{E}_{Q(z_{t-1}, z_{t})}[(z_{t} - A^{\intercal}z_{t-1} - b)^{\intercal}\Phi^{-1}(z_{t} - A^{\intercal}z_{t-1} - b)] \\
&= \mathbb{E}_{Q(z_{t})}[(z_{t} - A^{\intercal}\mu_{t-1} - b)^{\intercal}\Phi^{-1}(z_{t} - A^{\intercal}\mu_{t-1} - b) + \mathbf{Tr}(A\Phi^{-1}A^{\intercal}\Sigma_{t-1})] \\
&= (\mu_{t} - A^{\intercal}\mu_{t-1} - b)^{\intercal}\Phi^{-1}(\mu_{t} - A^{\intercal}\mu_{t-1} - b) + \mathbf{Tr}(A\Phi^{-1}A^{\intercal}\Sigma_{t-1}) + \mathbf{Tr}(\Phi^{-1}\Sigma_{t})
\end{align}
$$
</center>

Similarly,
<center>
$$
\begin{align}
&\mathbb{E}_{Q(z_{t}, z_{t+1})}[(z_{t+1} - A^{\intercal}z_{t} - b)^{\intercal}\Phi^{-1}(z_{t+1} - A^{\intercal}z_{t} - b)] \\
&= \mathbb{E}_{Q(z_{t})}[(\mu_{t+1} - A^{\intercal}z_{t} - b)^{\intercal}\Phi^{-1}(\mu_{t+1} - A^{\intercal}z_{t} - b) + \mathbf{Tr}(\Phi^{-1}\Sigma_{t+1})] \\
&= (\mu_{t+1} - A^{\intercal}\mu_{t} - b)^{\intercal}\Phi^{-1}(\mu_{t+1} - A^{\intercal}\mu_{t} - b) + \mathbf{Tr}(A\Phi^{-1}A^{\intercal}\Sigma_{t}) + \mathbf{Tr}(\Phi^{-1}\Sigma_{t+1})
\end{align}
$$
</center>

Putting back to the objective function and dropping constants, we need to solve the following:
<center>
$$
\begin{align}
\min_{\mu_{t}, \Sigma_{t}} &L(\mu_{t}, \Sigma_{t}) = (x_{t} - C^{\intercal}\mu_{t} - d)\Psi^{-1}(x_{t} - C^{\intercal}\mu_{t} - d) + \mathbf{Tr}(C\Psi^{-1}C^{\intercal}\Sigma_{t}) \\
&\quad + (\mu_{t+1} - A^{\intercal}\mu_{t} - b)^{\intercal}\Phi^{-1}(\mu_{t+1} - A^{\intercal}\mu_{t} - b) + \mathbf{Tr}(A\Phi^{-1}A^{\intercal}\Sigma_{t}) \\
&\quad + (\mu_{t} - A^{\intercal}\mu_{t-1} - b)^{\intercal}\Phi^{-1}(\mu_{t} - A^{\intercal}\mu_{t-1} - b) + \mathbf{Tr}(\Phi^{-1}\Sigma_{t}) \\
&\quad + \log\det(\Sigma_{t})
\end{align}
$$
</center>

Taking the gradient w.r.t. $$\mu_{t}$$ and set to zero, we get:
<center>
$$
\begin{align}
\nabla_{\mu_{t}}L &= \nabla_{\mu_{t}}\bigg\{(x_{t} - C^{\intercal}\mu_{t} - d)\Psi^{-1}(x_{t} - C^{\intercal}\mu_{t} - d) \\
&\quad + (\mu_{t+1} - A^{\intercal}\mu_{t} - b)^{\intercal}\Phi^{-1}(\mu_{t+1} - A^{\intercal}\mu_{t} - b) \\
&\quad + (\mu_{t} - A^{\intercal}\mu_{t-1} - b)^{\intercal}\Phi^{-1}(\mu_{t} - A^{\intercal}\mu_{t-1} - b)\bigg\}\\
&= -C\Psi^{-1}(x_{t} - C^{\intercal}\mu_{t} - d) - A\Phi^{-1}(\mu_{t+1} - A^{\intercal}\mu_{t} - b) + \Phi^{-1}(\mu_{t} - A^{\intercal}\mu_{t-1} - b) \\
&= -C\Psi^{-1}(x - d) - A\Phi^{-1}(\mu_{t+1} - b) - \Phi^{-1}(A^{\intercal}\mu_{t-1} + b) \\
&\quad + (C\Psi^{-1}C^{\intercal} + A\Phi^{-1}A^{\intercal} + \Phi^{-1})\mu_{t} \\
&:= 0 \\
\mu_{t} &= (C\Psi^{-1}C^{\intercal} + A\Phi^{-1}A^{\intercal} + \Phi^{-1})^{-1}C\Psi^{-1}(x_{t} - d) \\
&\quad + (C\Psi^{-1}C^{\intercal} + A\Phi^{-1}A^{\intercal} + \Phi^{-1})^{-1}A\Phi^{-1}(\mu_{t+1} - b) \\
&\quad - (C\Psi^{-1}C^{\intercal} + A\Phi^{-1}A^{\intercal} + \Phi^{-1})^{-1}\Phi^{-1}(A^{\intercal}\mu_{t-1} + b)
\end{align}
$$
</center>

We can obtain a residual form for $$\mu_{t}$$ by adding and subtracting the prediction from each term using a previous estimate $$\mu'_{t}$$. Let $$\Lambda = (C\Psi^{-1}C^{\intercal} + A\Phi^{-1}A^{\intercal} + \Phi^{-1})^{-1}$$, we can rewrite the above as:
<center>
$$
\begin{align}
\mu_{t} &= \Lambda C\Psi^{-1}(x_{t} + C^{\intercal}\mu'_{t} - C^{\intercal}\mu'_{t} - d) \\
&\quad + \Lambda A\Phi^{-1}(\mu_{t+1} + A^{\intercal}\mu'_{t} - A^{\intercal}\mu'_{t} - b) \\
&\quad - \Lambda \Phi^{-1}(\mu'_{t} - \mu'_{t} + A^{\intercal}\mu_{t-1} + b) \\
&= \mu'_{t} + \Lambda C\Psi^{-1}(x_{t} - C^{\intercal}\mu'_{t} - d) \\
&\quad + \Lambda A\Phi^{-1}(\mu_{t+1} - A^{\intercal}\mu'_{t} - b) \\
&\quad + \Lambda \Phi^{-1}(\mu'_{t} - A^{\intercal}\mu_{t-1} - b) \\
\end{align}
$$
</center>
The result is a gradient descent like update rule where the estimate $$\mu_{t}$$ is updated in the direction of prediction error of both the current observation and latent states at adjacent time steps.

Taking the gradient w.r.t $$\Sigma_{t}$$ and set to zero, we get:
<center>
$$
\begin{align}
\nabla_{\Sigma_{t}}L &= \nabla_{\Sigma_{t}}\bigg\{\mathbf{Tr}(C\Psi^{-1}C^{\intercal}\Sigma_{t}) + \mathbf{Tr}(A\Phi^{-1}A^{\intercal}\Sigma_{t}) + \mathbf{Tr}(\Phi^{-1}\Sigma_{t})+ \log\det(\Sigma_{t})\bigg\} \\
&= C\Psi^{-1}C^{\intercal} + A\Phi^{-1}A^{\intercal} + \Phi^{-1} - \Sigma_{t}^{-1} \\
&:= 0 \\
\Sigma_{t} &= (C\Psi^{-1}C^{\intercal} + A\Phi^{-1}A^{\intercal} + \Phi^{-1})^{-1}
\end{align}
$$
</center>

One thing to notice is that the posterior covariance only depends on the transition and observation covariance and is independent of the observation or covariance at adjacent time steps. This is a correct but somewhat undesirable result due to the mean field factorization. One way to overcome this is to use the exact marginal prior $$\log \int_{z_{t}}P(z_{t+1}\vert z_{t})Q(z_{t})$$ rather than the lower bound $$\mathbb{E}_{Q(z_{t})}[\log P(z_{t+1}\vert z_{t})]$$. 