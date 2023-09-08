---
layout: post
title:  "Rationalizing Expected Free Energy"
date:   2023-09-07 00:00:00 -0000
---

# Another attempt to rationalize Expected Free Energy: Insights from Reinforcement Learning

Expected Free Energy (EFE) is at the core of active inference and the key differentiator between active inference and purely reward-driven agent modeling frameworks (e.g., reinforcement learning). For the optimization-savvy, EFE can be seen as an objective function optimized by active inference agents, similar to the expected utility objective which is optimized by RL agents. For Bayesians, EFE can be seen as a particular choice of prior on agent behavior. However, which view lies closer to the spirit of the active inference founders? 

Since its origin, the rationale for EFE has been at the center of debates by the curious. In [Active Inference And Epistemic Value](https://www.tandfonline.com/doi/full/10.1080/17588928.2015.1020053), one of the earliest papers to introduce this concept, EFE was described as the expectation of (variational) free energy in the future; thus any free energy-minimizing agent should believe a priori that they would take actions to minimize EFE. In [Whence the Expected Free Energy](https://direct.mit.edu/neco/article/33/2/447/95645/Whence-the-Expected-Free-Energy), the authors argued that EFE is not a straightforward extension of the Free Energy of the Future (FEF) and the difference between the two is an information gain term -- a celebrated symbol for epistemic behavior. In [The Two Kinds of Free Energy and the Bayesian Revolution](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008420), the authors highlighted the ambiguity in using the term "free energy" to refer to different quantities and that the variational free energy (VFE) in approximate Bayesian inference is not the same as the EFE used for action selection. However, recent efforts such as the [Geometric Methods](https://arxiv.org/abs/2203.10592), [The FEP Made Simpler](https://arxiv.org/abs/2201.06387), and [Strange Things](https://arxiv.org/abs/2210.12761) papers have presented arguments to justify EFE and its connection with VFE as dynamical systems in limiting cases (with special information theoretic conditions; remorsefully I have not fully wrapped my head around these ideas). These are just a few examples to show that **the origin of EFE is an unsolved question**.

In my opinion, a recent paper titled [Simplifying Model-Based RL:  Learning Representations, Latent-Space Models, And Policies With One Objective](https://arxiv.org/abs/2209.08466) represents by far the best attempt to rationalize EFE, including some of the cognitive and philosophical aspects of active inference and the free energy principle, even though this was most likely not their intention. Specifically, the authors proposed the Aligned Latent Models (ALM) for the purpose of solving the [objective mismatch](https://arxiv.org/abs/2002.04523) problem and improving agent capabilities on complex, high dimensional visual RL tasks. My goal here is to explain its connections with and implications for active inference.

(I will use the standard POMDP notations in terms of states $$s \in \mathcal{S}$$, actions $$a \in \mathcal{A}$$, and observations $$o \in \mathcal{O}$$.)

## The spirit of active inference

Underlying active inference, and more precisely the free energy principle (FEP), are two fundamental ideas. First, agents should minimize surprise by restricting themselves to a small region in the state space, such as fish remaining in water and humans maintaining a constant body temperature. This notion was formalized in one of the earliest works on active inference ([Action and Behavior: A Free Energy Formulation](https://link.springer.com/article/10.1007/s00422-010-0364-z)) as minimizing the entropy of observed sensory signals $$o \in O$$, defined as the long-term average of observation likelihood under the agent's generative model:
<center>
$$
\min H(o) = \lim_{T \rightarrow \infty} - \frac{1}{T}\int_{t=0}^{T}\log P(o_t)d_t
$$
</center>
However, from this objective alone, it is hard to tell what the generative model $$P(o)$$ should be (i.e., its structure and parameters) and what the agent is optimizing with respect to (i.e., the decision variables). The first ambiguity points to a missing piece in surprise minimization, namely, the agent should **not only restrict itself to a small region but also that this region should be desirable or characteristic for the agent**, such as water but not land for fish even though there is less land mass on earth. This notion of desirability should be a part of the generative model. 

The second fundamental idea responds to the second ambiguity with regard to decision variables. It is obvious that actions (i.e., what the agent can *do*) should be a part of the decision variables, so should be the agent's perceptual behavior, which takes sensory observations as an inputs and updates the agent's mental/internal states. In particular, active inference adopts an *enactive* approach to perception (see [A Tale of Two Densities](https://journals.sagepub.com/doi/full/10.1177/1059712319862774)), where perception is treated as an interface to adaptive behavior as opposed a mirror of the environment. In other words, **perception is not obligated to make Bayes-optimal prediction of future observations (although it can) as long as it enables optimal action selection for the purpose of surprise minimization**. This is the point of contact between active inference and objective mismatch in model-based RL, namely, that perception and action should optimize the same objective as opposed to two separate, and potentially misaligned, objectives.

## Expected Free Energy

Regardless of the spirit of the founders, active inference has developed into the formulation today where the agent builds an explicit generative model of environment observations and uses the generative model to find EFE-minimizing actions. 

A particular version of EFE is defined as (see my [previous post](https://rw422scarlet.github.io/2023/07/30/active-inference-introduction.html) on different versions of active inference and formulations of EFE):
<center>
$$
\underbrace{\mathbb{E}_{Q(o\vert a)}[-\log \tilde{P}(o)]}_{\text{Pragmatic value}} - \underbrace{\mathbb{E}_{Q(o\vert a)}D_{KL}[Q(s\vert o, a) \vert \vert  Q(s\vert a)]}_{\text{Epistemic value}}
$$
</center>
where the $$Q$$'s are the states and observations predicted by the generative model and $$\tilde{P}$$ is a distribution specifying preferred observations. The first term called "pragmatic value" represents a notion of preference-seeking. The second term called "epistemic value" represents a notion of information-seeking, and this term is the key feature of active inference. Removing epistemic value would make the objective indistinguishable from standard RL, where the log likelihood of observations serves as a reward function: $$R(o) = \log \tilde{P}(o)$$. So the question of interest here is why is this objective designed in this way and how is it related to the FEP?

## A RL perspective on Expected Free Energy

To motivate the exposition below, we start with a brief summary of the RL perspective on EFE as implied by [the ALM agent](https://arxiv.org/abs/2209.08466):
1. **EFE is an objective function (or the result thereof) which drives the agent's perceptual and control behavior simultaneously**
2. **Under the EFE objective, the agent tries to remain in a desired density over trajectories, in turn avoiding surprising trajectories**

To make these points concrete, let's formalize the problem with some mathematical jargons. We will consider an environment defined on a set of observation $$\mathcal{O}$$ and a set of actions $$\mathcal{A}$$. The agent interacts with the environment through the exchange of actions $$a_t$$ and observations $$o_t$$ at each time step. We denote the sequence of observations and actions until the current time step (also known as history) as $$h_t = (o_{0:t}, a_{0:t-1})$$. In the most general case, both the environment (denoted with $$P$$) and the agent (denoted with $$\pi$$) condition on the entire history, which results in the following interaction process:
<center>
$$
a_t \sim \pi(\cdot\vert h_t), \quad o_{t+1} \sim P(\cdot\vert h_t, a_t)
$$
</center>
To make the agent configuration precise, let's parameterize every module associated with the agent with parameters $$\theta$$. **Let's assume that the agent process the observations using internal latent states (or representations) $$z$$ such that conditioning on the entire history is mediated by $$z$$ through an encoder $$E_{\theta}(z_t\vert o_t, z_{t-1}, a_{t-1})$$ and a latent state policy $$\pi_{\theta}(a_t\vert z_t)$$**. In this way, action sampling upon a new observation is further broken down to the following process:
<center>
$$
z_t \sim E_{\theta}(\cdot\vert o_t, z_{t-1}, a_{t-1}), \quad a_t \sim \pi_{\theta}(\cdot\vert z_t)
$$
</center>
which over time generates infinite-length trajectories of the form $$\tau = (o_{0:\infty}, z_{0:\infty}, a_{0:\infty})$$. 

To represent the requirement that the agent should remain in desirable, characteristic states, we define a variable $$y \in \{0, 1\}$$. Using a reward function $$R(o)$$ to specify desirability, we defined the probability of a good trajectory as:
<center>
$$
P(y=1\vert \tau) = R(\tau) = \sum_{t=0}^{\infty}\gamma^{t}R(o_t)
$$
</center>
where $$\gamma$$ is a discount factor.

Given this setup, we can formulate the desire of following good trajectories as an optimization problem:
<center>
$$
\begin{align}
\max_{\theta} \log P_{\theta}(y=1) &= \log \int_{\tau}P_{\theta}(y=1, \tau) \\
&= \log\mathbb{E}_{P_{\theta}(\tau)}[P(y=1\vert \tau)] \\
&\geq \mathbb{E}_{Q_{\theta}(\tau)}[\log R(\tau) + \log P_{\theta}(\tau) - \log Q_{\theta}(\tau)] \\
&= \mathbb{E}_{Q_{\theta}(\tau)}\left[\log\sum_{t=0}^{\infty}\gamma^t R(o_t) + \log P_{\theta}(\tau) - \log Q_{\theta}(\tau)\right] \\
&\geq \mathbb{E}_{Q_{\theta}(\tau)}\left[\sum_{t=0}^{\infty}\gamma^t \log R(o_t) + \log P_{\theta}(\tau) - \log Q_{\theta}(\tau)\right]
\end{align}
$$
</center>
where the last line is a lower bound on the log marginal likelihood of good trajectories and $$Q_{\theta}(\tau)$$ is known as a variational distribution.

Familiar readers will recognize this as a slight twist of the [control-as-inference](https://arxiv.org/abs/1805.00909) framework, where the optimality variable is conditioned on the entire trajectory as opposed to a single time step. The key feature of the control-as-inference framework is that **if one can design good generative distribution $$P$$ and variational distribution $$Q$$, interesting properties can be derived for the agent**. 

To this end, the authors propose the following distributions:
<center>
$$
\begin{align}
P(\tau) &= P(o_{0})\prod_{t=0}^{\infty}P(o_{t+1}\vert h_t, a_t)\pi(a_t\vert z_t)E_{\theta}(z_t\vert o_t, z_{t-1}, a_{t-1}) \\
Q_{\theta}(\tau) &= P(o_0)\prod_{t=0}^{\infty}P(o_{t+1}\vert h_t, a_t)\pi_{\theta}(a_t\vert z_t)M_{\theta}(z_t\vert z_{t-1}, a_{t-1})
\end{align}
$$
</center>
where $$E_{\theta}(z_0\vert o_0, z_{-1}, a_{-1}) = E_{\theta}(z_0\vert o_0)$$ and $$M_{\theta}$$ is a latent dynamics model not conditioned on observations with $$M_{\theta}(z_0\vert z_{-1}, a_{-1}) = M_{\theta}(z_0)$$ (and we have ignored rigorous measure-theoretic treatments on infinite-length sequences).

This allows us to write down a **single** objective function (i.e., the log marginal likelihood lower bound) for both perception and action as:
<center>
$$
\mathcal{L}_{\text{ALM}}(\theta) = \mathbb{E}_{Q_{\theta}(\tau)}\left[\sum_{t=0}^{\infty}\gamma^{t}\left(\underbrace{\log R(o_t)}_{\text{Reward}} + \underbrace{\log E_{\theta}(z_t\vert o_t, z_{t-1}, a_{t-1}) - \log M_{\theta}(z_t\vert z_{t-1}, a_{t-1})}_{\text{Information gain}}\right)\right]
$$
</center>

The active inference-savvy should recognize its extreme resemblance to EFE, in particular the pragmatic-epistemic decomposition as defined above. However, there are some important differences, the most important of which is that the **agent does not build an explicit generative model of environment observations, but rather one of its own latent states**. For model-based planning, this introduces a slight inconvenience since the observations in the expectation are sampled from the environment, which is not available at planning time. But this can be easily overcome by training a log density ratio estimator $$\log C \approx \log \frac{E}{M}$$ through binary classification of the latents processed by the encoder on real observation samples and those generated by the latent dynamics model.

The more important difference is that rather than predicting environment observations, the latent representations $$z$$ will be developed in  a ["self-predictive"](https://arxiv.org/abs/2007.05929) manner (also see [Bootstrap Your Own Latent](https://arxiv.org/abs/2006.07733)), where the latent dynamics $$M$$ tries to predict the encoding of received observations and the encoder $$E$$ tries to match the predictions of the latent dynamics. Interestingly, [a recent paper](https://arxiv.org/abs/2212.03319) shows that this class of self-predictive representation learning implicitly performs eigen-decomposition of the environment dynamics (under assumptions of the initialization and relative updating speed of the encoder and latent dynamics), similar to Principal Component Analysis (a well-known linear-Gaussian latent variable model). This means that **by optimizing the ALM objective, the agent will behave as if it has a generative model of the environment and operates on the basis of beliefs**.

## Closing thoughts: The engineering and philosophical status of active inference

The practical implementations of active inference have always been intertwined with its philosophical status since its birth (which I have very little expertise but see a recent [MLST podcast](https://www.youtube.com/watch?v=bL00-jtRrMA) on this topic). At times, this can introduce ambiguities to engineers, for example on the notion of [self-evidencing](https://philarchive.org/rec/HOHTSB-2) vs [different kinds of free energies](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008420), which makes it difficult to navigate between the "high road" and "low road" of active inference (a reference from chapter 2 & 3 of [the active inference book](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind)). 

Of particular interest to this post is enactivism, which is closely related to the [objective mismatch](https://arxiv.org/abs/2002.04523) problem in RL. Given its relationship with predictive coding models of perception, computational models of active inference have often been presented as perception through explicit Bayesian inference. However, [some](https://link.springer.com/article/10.1007/s11229-016-1239-1) [authors](https://journals.sagepub.com/doi/full/10.1177/1059712319862774) have advocated for the interface perspective on perception under the FEP. The ALM objective seems to suggest that these different perspectives (e.g., vs constructivism) may end up with the same solution and thus indistinguishable. 

From an engineering perspective, building active inference agents with implicit generative models (i.e., without observation reconstruction) has been considered, and it is in fact the reason why ALM, a paper not related to active inference but bears extremely high resemblance, caught my attention. In [contrastive active inference](https://arxiv.org/abs/2110.10083), the authors used almost the same architecture as ALM:  an encoder, a latent dynamics model, and a classification method, for representation learning and EFE estimation. However, contrastive active inference takes the EFE objective as given and adapts it with novel contrastive estimation techniques (i.e., the "low road") rather than starting with the high-level premise for surprise minimization (i.e., the "high road"). (I'm sure the authors will forgive me for saying this.) 

Outside of enactivism, the ambiguation of belief and desire in active inference is also a subject of philosophical debates. The EFE has often been presented as a prior $$P(a) \propto \exp(-EFE(a))$$ such that all active inference agent update equations can be derived as variational Bayesian inference (see my [previous post](https://rw422scarlet.github.io/2023/07/30/active-inference-introduction.html) on the derivation). Sometimes, the preference distribution $$\tilde{P}(o)$$ is also referred to as a prior. Such framing has received various [supporters](https://www.tandfonline.com/doi/abs/10.1080/00048402.2019.1602661) and [doubters](https://link.springer.com/article/10.1007/s11229-016-1250-6). I think the ALM agent will be inclined to view EFE as an objective, but it will not be able to comment on whether any aspects of this objective can be considered as belief or desire. 

Lastly, it should be noted that taking the ALM perspective on EFE is limited in that the objective and agent properties discussed above ultimately came from the designer's specification of the generative and variational distributions, and changing them will most likely destroy the resemblance we observed. This highlights a large gap in the attempt to understand active inference and design adaptive RL agents which we should continue to fill. 