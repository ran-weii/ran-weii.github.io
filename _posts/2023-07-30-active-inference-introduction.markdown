---
layout: post
title:  "Making Sense of Active Inference: Optimal Control Without Cost Function"
date:   2023-07-30 00:00:00 -0000
---

Here is my attempt to introduce the mechanics of active inference, a framework for modeling and understanding sentient agents originated in neuroscience. My main interest will be centered on active inference's approach to unify perception and control under the same optimization objective. As we will explain below, a unified objective likely has many advantages, including simpler agent design and better performance. We will see that active inference approaches the unified objective from a very perception-centric perspective rather than a control-centric perspective, which is itself a recent movement in reinforcement learning and optimal control. 

## Why unify perception and control?

We will consider agents that build an explicit model of the environment and behave in a way such that a reward function is optimized (a.k.a., model-based reinforcement learning; MBRL). Perception can be understood as making predictions about the future states of the environment. 

The traditional recipe for building such agents is to let them interact with the environment and collect data, optimize the predictive accuracy of the model, and use the model to plan sequences of reward-maximizing actions. As the model becomes more accurate, the agent's ability to achieve high rewards should increase correspondingly. 

Unfortunately, a recent paper titled [Objective Mismatch in Model-based Reinforcement Learning](https://arxiv.org/abs/2002.04523) found that the correspondence between model accuracy and achieved rewards is often violated in practice, if not non-existent at all. There are numerous ways why this could be the case. For example, in model-base RL, there is a well-known phenomenon that inaccuracies in the model (due to overfitting on training data and lack of regularization on out-of-distribution data) tends to be exploited by the planner, which has been studied extensively (for example, see [When to Trust Your Model](https://arxiv.org/abs/1906.08253)). Model exploitation can lead to premature convergence, but more often catastrophic failure. The bottom line here is that the conflict arises because **the model is not trained for how it is supposed to be used: optimizing rewards**. 

To better understand and address the objective mismatch problem, a subfield has emerged within the MBRL community called **decision-aware MBRL** which aims to develop unified objective functions for model learning and action selection so that they are aware of each other's role in the optimization process. In a [recent survey](https://arxiv.org/abs/2310.06253), we studied all existing decision-aware MBRL approaches and found that these unified objective functions are developed based on the principle of either better estimating or directly optimizing rewards with respect to a pair of model and action selection policy. Notable approaches include *value-prediction* which trains the model to predict values (cumulative rewards) instead of environment observations (e.g., see [this paper](https://arxiv.org/abs/2011.03506)) and *distribution correction* which corrects for model error in reward estimation or optimization (e.g., see [this paper](https://arxiv.org/abs/2110.02758)). Experimental results have shown increased robustness of agents trained with these unified objective functions to model misspecification, distracting observations, and other perceptual challenges. But it's clear that these approaches are very control-centric, **models are developed solely for the purpose of control**. 

## A handwavy introduction of active inference

Active inference can be seen as a way to develop agent objectives using ideas from the [Free Energy Principle](https://www.nature.com/articles/nrn2787) (FEP). The FEP roughly states that the agent has a probabilistic model of the sensory observations it is supposed to receive; both perception and control should gather evidence for the probabilistic model (i.e., maximize its likelihood). 

The FEP is often brought up at the same time with [predictive processing](https://en.wikipedia.org/wiki/Predictive_coding#:~:text=In%20neuroscience%2C%20predictive%20coding%20). In the RL context, predictive processing can be understood as the idea that both perception and control should suppress prediction error, where perception achieves this objective by building a better model of the environment, control achieves this objective by changing the environment such that environment-generated signals are better predicted by the model. Under this view, **perception and control work towards the same goal: minimize prediction error**. In contrast to the control-centric approach to unified objectives in RL, models are developed for the purpose of prediction, and so is control. 

There are solid [cognitive](https://journals.sagepub.com/doi/full/10.1177/1059712319862774) and [neuroscience](https://www.sciencedirect.com/science/article/pii/S0896627311009305) motivations for this perception-centric view of active inference over utility-maximizing optimal control, mostly claiming advantages of active inference in agents with limited information storage and processing capacity. For example, in a paper titled [Nonmodular Architecture of Cognitive Systems Based On Active Inference](https://arxiv.org/abs/1903.09542), the authors found that an active inference controller is robust to model misspecification. 

Most relevant to the present discussion is a paper titled [Learning Action-oriented Models Through Active Inference](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007805), where the authors showed that an active inference agent with a misspecified model succeeds at a task while learning model parameters that deviate significantly from the actual environment statistics. And if the agent were to accurately capture the environment statistics in this setting, it would not solve the task. Furthermore, the parameters learned by the active inference agent are optimistic in the sense that the learned transition dynamics is biased towards solving the task. 

Despite these promising results, how to properly formulate perception and control as prediction error minimization has been an active research effort: active inference formulations have gone through drastic changes over the years and I think it is still not fully settled (see [this paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008420) on active inference and different kinds of free energy).  

I will unpack two versions of active inference in detail, an old and a new one, to illustrate the conceptual underpinning. 

## Optimal control without cost function

A paper published in 2012 titled [Active Inference And Agency: Optimal Control Without Cost Functions](https://link.springer.com/article/10.1007/s00422-012-0512-8) really embodies the idea that optimal control should be based on prediction rather than reward or cost functions. The authors proposed two agents: an agency-free agent and an agency-based agent. I will discuss the former agent and contrast it with the standard approach to (partially observable) Markov decision process. 

In partially observable Markov decision process (POMDP), we model the environment using a set of hidden states $$s \in \mathcal{S}$$, actions $$a \in \mathcal{A}$$, and observations $$o \in \mathcal{O}$$. Upon taking action $$a_t$$ at time step $$t$$, the environment transitions into a new state $$s_{t+1}$$ according to probability distribution $$P(s_{t+1}\vert s_t, a_t)$$. However, the agent cannot directly observe the environment states but only observations sampled from $$P(o_t\vert s_t)$$. 

The most established approach to planning in POMDP is based on performing backward dynamic programming in the belief space. Basically, the planner tries to think about **"given that I believe the environment is in a certain state, what will my next belief be under the POMDP model of the environment, and what value is associated with that belief"**. At every time step of interaction with the environment, the agent first updates its belief about the environment (e.g., using a Kalman filter), and then uses the updated belief to find the optimal plan. 

Active inference tries to formulate both the belief update and the planning process as inference or prediction (i.e., inference about the future). Similar to the optimal control formulation, the agent knows that observations are sampled from $$P(o_t\vert s_t)$$. However, the active inference agent models the environment transitions as $$P(s_{t+1}\vert s_t)$$, **eschewing the action variable in the optimal control formulation**. 

To be more precise, this paper considers solving episodic tasks with a maximum of $$T$$ time steps. The agent explicitly represents the hidden states at all time steps, and observations until the current time step $$\tau$$. The probabilistic model is defined as follows:
<center>
$$
P(o_{1:\tau}, s_{1:T}) = \prod_{t=1}^{\tau}P(o_t\vert s_t)\prod_{t=1}^{T}P(s_t\vert s_{t-1})
$$
</center>
where $$P(s_1\vert s_0) = P(s_1)$$ is the initial state distribution.

**Perception as hidden state inference:** As said before, the agent performs all mental activities for the purpose of maximizing the likelihood of the probabilistic model. For perception in the current context, this corresponds to updating beliefs about the hidden states by minimizing variational free energy, an upper bound on the (negative) model evidence. 

Specifically, the agent tries to form its belief about the hidden states by optimizing a parameterized distribution $$Q(s_{1:\tau}) = \prod_{t=1}^{\tau}Q(s_t)$$ towards minimal free energy (we assume beliefs about states at adjacent time steps are not correlated, a.k.a., the mean-field assumption).  The free energy function is defined as:
<center>
$$
\begin{align}
\mathcal{F}(o_{1:\tau}, Q) &= \mathbb{E}_{Q(s_{1:T})}[\log Q(s_{1:T}) - \log P(o_{1:\tau}, s_{1:T})]\\
&= \mathbb{E}_{Q(s_{1:T})}[\sum_{t=1}^{T}\log Q(s_{t}) - \sum_{t=1}^{\tau}\log P(o_t\vert s_t) - \sum_{t=1}^{T}\log P(s_t\vert s_{t-1})] \\
&= - \underbrace{\sum_{t=1}^{\tau}\mathbb{E}_{Q(s_t)}[\log P(o_t\vert s_t)]}_{\text{Past observations}} + \underbrace{\sum_{t=1}^{T}\mathbb{E}_{Q(s_{t-1:t})}[\log Q(s_{t}) - \log P(s_t\vert s_{t-1})]}_{\text{Past and future states}}
\end{align}
$$
</center>
Taking the derivative of $$\mathcal{F}$$ w.r.t. $$Q$$, we can show that the optimal $$Q$$ has the following form (see derivation at the end):
<center>
$$
Q^*(s_t) \propto \exp\left(\mathbb{I}[t \leq \tau]\log P(o_t\vert s_t) + \mathbb{E}_{Q^*(s_{t-1})}[\log P(s_t\vert s_{t-1})] + \mathbb{E}_{Q^*(s_{t+1})}[\log P(s_{t+1}\vert s_t)]\right)
$$
</center>
Arrange slightly differently, we have:
<center>
$$
Q^*(s_t) \propto \left\{\begin{array}{ll}\exp\left(\mathbb{E}_{Q^*(s_{t-1})}[\log P(o_t, s_t\vert s_{t-1})] + c_1\right) & t \leq \tau \\ \exp\left(\mathbb{E}_{Q^*(s_{t-1})}[\log P(s_t\vert s_{t-1})] + c_2\right) & t > \tau \end{array}\right.
$$
</center>
where $$c_1, c_2$$ correspond to the last term in the previous equation. **We see that these terms highly resemble Bayesian posterior and posterior predictive distributions.**

**Control as observation inference:** This part is where active inference deviates from optimal control. First, the authors introduce an additional mapping $$P(o_{t+1}\vert o_t, a_t)$$. I like to think about this as **"reflex"**, where actions are quickly mapped to observations (or the other way around assuming inverse dynamics uniqueness). They then require the agent to select actions such that the next observation under the reflex distribution has the least free energy under the current updated belief about hidden states:
<center>
$$
a_\tau = \arg\min_{a} \sum_{o_{\tau+1}}P(o_{\tau+1}\vert o_\tau, a)\mathcal{F}(o_{1:\tau+1}, Q^*)
$$
</center>
where $$\mathcal{F}(o_{1:t+1}, Q^*)$$ is the free energy after observing the hypothetical next observation, evaluated using the current belief.

### Connections to optimal control *with* cost function
Let's now try to understand what is meant by control as observation inference. The magic lies in not explicitly representing actions in the transition model.

Let us assume we have access to an (reward-maximizing) optimal policy $$\pi^*(a\vert s)$$ and a regular transition model $$P(s'\vert s, a)$$. Assuming we always choose actions from the optimal policy, the transition model eschewing the action variable can be constructed from:
<center>
$$
P^{\pi^*}(s'\vert s) = \sum_{a}P(s'\vert s, a)\pi^*(a\vert s)
$$
</center>
Let us also find the optimal control counterpart of the reflex model. Defining $$b(s_\tau\vert o_{\tau}) = P(s_t\vert o_{\tau}, o_{1:\tau-1})$$ as the Bayesian posterior, we can construct the reflex model as:
<center>
$$
\begin{align}
P(o_{\tau+1}\vert o_{\tau}, a) &= \sum_{s_{\tau+1}}P(o_{\tau+1}\vert s_{\tau+1})\sum_{s_{\tau}}P(s_{\tau+1}\vert s_{\tau}, a)b(s_{\tau}\vert o_{\tau}) \\
&\triangleq P(o_{\tau+1}\vert b, a)
\end{align}
$$
</center>

We are now able to simplify the hypothetical free energy $$\mathcal{F}(o_{1:\tau+1}, Q^*)$$. 

We are currently at time step $$t = \tau + 1$$. For future time steps $$t > \tau + 1$$, since we have minimized the corresponding terms in $$\mathcal{F}$$ (let's denote them with $$\mathcal{F}_{>}$$), we assume that it is approximately zero:
<center>
$$
\begin{align}
\mathcal{F}_{>}(Q^*) &= \sum_{t=\tau+1}^{T}\mathbb{E}_{Q^*(s_{t-1:t})}[\log Q^*(s_t) - \log P(s_t\vert s_{t-1})] \\
&= \sum_{t=\tau+1}^{T}\mathbb{E}_{Q^*(s_{t-1})}D_{KL}[Q^*(s_t) \vert \vert  P(s_t\vert s_{t-1})] \\
&\approx 0
\end{align}
$$
</center>
This is simply saying we can make good prediction of future states.

For past time steps $$t \leq \tau$$, we know that the corresponding terms in  $$\mathcal{F}$$ (let's denote them with $$\mathcal{F}_{<}$$) are approximately equal to the likelihood of past observations evaluated under the model:
<center>
$$
\mathcal{F}_{<}(Q^*) \approx \log P(o_{1:\tau})
$$
</center>
And this does not change no matter what the next observation $$o_{\tau+1}$$ will be. 

Thus, the only term that differentiates $$\mathcal{F}(o_{1:\tau+1}, Q^*)$$ for different $$o_{\tau+1}$$ is the current step, which equals:
<center>
$$
\begin{align}
\mathcal{F}_{\tau+1} &= -\mathbb{E}_{Q^*(s_{\tau+1})}[\log P(o_{\tau+1}\vert s_{\tau+1})] \\
&\approx -\mathbb{E}_{P(s_{\tau+1}\vert Q^*)}[\log P(o_{\tau+1}\vert s_{\tau+1})] \\
&\approx -\log \sum_{s_{\tau+1}}P(o_{\tau+1}\vert s_{\tau+1})\sum_{s_{\tau}}P^{\pi^*}(s_{\tau+1}\vert s_{\tau})Q^*(s_{\tau}) \\
&\triangleq -\log P(o_{t+1}\vert \pi^*)
\end{align}
$$
</center>
where $$P(s_{\tau+1}\vert Q^*) = \sum_{s_{\tau}}Q^*(s_{\tau})P(s_{\tau+1}\vert \tau)$$ is the posterior predictive distribution. 

Thus, $$\mathcal{F}_{\tau+1}$$ represents the (negative) predictive likelihood of the next observation under the optimal policy, and selecting actions comes down to finding:
<center>
$$
\begin{align}
a_{\tau+1} &= \arg\max_{a}\sum_{o_{\tau+1}}P(o_{\tau+1}\vert b, a)\log P(o_{t+1}\vert \pi^*)
\end{align}
$$
</center>
which is simply saying **let's find an action such that the next predicted observation coincide with the one that would be generated by the optimal policy.**

![](/assets/2023-07-30-active-inference-introduction/optimal_control_without_cost.png)

An illustration of optimal control as observation inference (2012) and optimal control as prior inference (2015), which we will unpack below.

## Modern active inference

In 2015, a paper titled [Active Inference And Epistemic Value](https://www.tandfonline.com/doi/full/10.1080/17588928.2015.1020053) marks the beginning of a new era for active inference, where the ability to handle epistemic uncertainty is claimed as a central property and a natural consequence of active inference. This new version was later refined in a paper titled [Active Inference: A Process Theory](https://direct.mit.edu/neco/article/29/1/1/8207/Active-Inference-A-Process-Theory) and the most updated version (the modern version) of active inference was comprehensively reviewed in [this paper](https://www.sciencedirect.com/science/article/pii/S0022249620300857) (highly recommended). 

The difference between the modern version and the previous version is that the agent now explicitly represents actions with no additional reflex mapping. We still consider an episodic setting with a maximum of $$T$$ time steps. We denote the action sequence to be modeled as $$\pi = a_{1:T-1}$$. The probabilistic model is defined as:
<center>
$$
P(o_{1:\tau}, s_{1:T}, \pi) = \prod_{t=1}^{\tau}P(o_t\vert s_t)\prod_{t=1}^{T}P(s_t\vert s_{t-1}, \pi)P(\pi)
$$
</center>
where $$P(s_1\vert s_0, \pi) = P(s_1)$$. 

The agent now represents beliefs about not only the hidden states but also the action sequence. This is captured in the distribution $$Q(s_{1:T}, \pi) = \prod_{t=1}^{T}Q(s_t\vert \pi)Q(\pi)$$. The free energy function is defined as:
<center>
$$
\begin{align}
\mathcal{F}(o_{1:\tau}, Q) &= \mathbb{E}_{Q}[\log Q(s_{1:T}, \pi) - \log P(o_{1:\tau}, s_{1:T}, \pi)] \\
&= D_{KL}[Q(\pi)\vert \vert P(\pi)] + \mathbb{E}_{Q}[\log Q(s_{1:T}\vert \pi) - \log P(o_{1:\tau}, s_{1:T}\vert \pi)] \\
&\triangleq D_{KL}[Q(\pi)\vert \vert P(\pi)] + \mathbb{E}_{Q(\pi)}[\mathcal{F}(o_{1:\tau}, Q\vert \pi)]
\end{align}
$$
</center>
where $$\mathcal{F}(o_{1:\tau}, Q\vert \pi)$$ denotes action-conditioned free energy. 

**Perception as hidden state inference:** Perception in the modern version is very similar to the previous version, except that we now have to find the optimal state estimates for each $$\pi$$. Borrowing the previous results, we have:
<center>
$$
Q^*(s_t\vert \pi) \propto \left\{\begin{array}{ll}\exp\left(\mathbb{E}_{Q^*(s_{t-1})}[\log P(o_t, s_t\vert s_{t-1}, \pi)] + c_1\right) & t \leq \tau \\ \exp\left(\mathbb{E}_{Q^*(s_{t-1})}[\log P(s_t\vert s_{t-1}, \pi)] + c_2\right) & t > \tau \end{array}\right.
$$
</center>
**Control as prior inference:** We saw in the previous version of active inference that *goal-directed actions are induced by generating optimistic predictions* - predictions about sensory consequences that would have been observed if the agent were to act optimally. This version of active inference induces such behavior using a *goal-directed prior over actions sequences*. 

Specifically, the prior over $$\pi$$ is defined as:
<center>
$$
P(\pi) \propto \exp\left(-\mathcal{G}(\pi\vert Q^*)\right)
$$
</center>
where $$\mathcal{G}(\pi\vert Q^*)$$ is known as the expected free energy (EFE) defined as:
<center>
$$
\mathcal{G}(\pi\vert Q^*) \triangleq \mathbb{E}_{Q^*(o_{\tau+1:T}, s_{\tau+1:T}\vert \pi)}[\log Q^*(s_{\tau+1:T}\vert \pi) - \log \tilde{P}(o_{\tau+1:T}, s_{\tau+1:T}\vert \pi)]
$$
</center>
$$Q^*(o_{\tau+1:T}, s_{\tau+1:T}\vert \pi) = \prod_{t=\tau+1}^{T}P(o_t\vert s_t)Q^*(s_t\vert \pi)$$ is the joint predictive distribution. 

The optimal posterior over action sequences under this prior is:
<center>
$$
Q^*(\pi) \propto \exp\left(-\mathcal{G}(\pi\vert Q^*) - \mathcal{F}(o_{1:\tau}, Q^*\vert \pi)\right)
$$
</center>
Thus, the posterior is simply a minor modification of the prior based on which action sequence likely generated the observed signals. If we assume the dynamics model fits the data well so that $$\mathcal{F} \approx 0$$, then we can see that the prior is doing the majority of the heavy-lifting.

### Control prior design choices

The choice of the EFE prior is usually justified as **"a free energy minimizing agent should a priori believe they will choose actions that minimize free energy"**. Note, however, the "probabilistic model" $$\tilde{P}(o_{\tau+1:T}, s_{\tau+1:T}\vert \pi)$$ in the EFE prior is not necessarily the same as the probabilistic model used to perform state estimation. There are thus many design decisions in defining $$\tilde{P}$$. 

We can definitely choose them to be the same: $$\tilde{P}(o_{\tau+1:T}, s_{\tau+1:T}\vert \pi) = \prod_{t=\tau+1}^{T}P(o_t\vert s_t)Q^*(s_t\vert \pi)$$. Then the EFE becomes:
<center>
$$
\begin{align}
\mathcal{G}(\pi\vert Q^*) &= \sum_{t=\tau+1}^{T}\mathbb{E}_{Q^*(o_t, s_t\vert \pi)}[ \log Q^*(s_t\vert \pi) - \log P(o_t\vert s_t) - \log Q^*(s_t\vert \pi)] \\
&= \sum_{t=\tau+1}^{T}\mathbb{E}_{Q^*(o_t, s_t\vert \pi)}[- \log P(o_t\vert s_t)] \\
&= \sum_{t=\tau+1}^{T}\mathbb{E}_{Q^*(s_t\vert \pi)}\mathcal{H}[P(o_t\vert s_t)]
\end{align}
$$
</center>
In other words, the EFE becomes the expected entropy $$\mathcal{H}$$ of future observations. 

We can alternatively choose $$\tilde{P}(o_{\tau+1:T}, s_{\tau+1:T}\vert \pi) = \prod_{t=\tau+1}^{T}P(o_t\vert s_t)\tilde{P}(s_t)$$, where
<center>
$$
\tilde{P}(s_t) \propto \exp(R(s_t))
$$
</center>
is a desired distribution over states, defined using a state-based reward function. Then, the EFE becomes:
<center>
$$
\begin{align}
\mathcal{G}(\pi\vert Q^*) &= \sum_{t=\tau+1}^{T}\mathbb{E}_{Q^*(o_t, s_t\vert \pi)}[ \log Q^*(s_t\vert \pi) - \log P(o_t\vert s_t) - \log \tilde{P}(s_t)] \\
&= \sum_{t=\tau+1}^{T} \underbrace{D_{KL}[Q^*(s_t\vert \pi)\vert \vert \tilde{P}(s_t)]}_{\text{Risk}} + \underbrace{\mathbb{E}_{Q^*(s_t\vert \pi)}\mathcal{H}[P(o_t\vert s_t)]}_{\text{Ambiguity}}
\end{align}
$$
</center>
This gives us the well-known risk-ambiguity decomposition. 

To get the final decompositions, we will choose $$\tilde{P}(o_{\tau+1:T}, s_{\tau+1:T}\vert \pi) = \prod_{t=\tau+1}^{T}\tilde{P}(o_t)\tilde{P}(s_t\vert o_t)$$, where 
<center>
$$
\tilde{P}(o_t) \propto \exp(R(o_t))
$$
</center>
We will then add and subtract the EFE with $$\sum_{t=\tau+1}^{T}\mathbb{E}_{Q^*(o_t, s_t\vert \pi)}[\log Q^*(s_t\vert o_t, \pi)]$$, where $$Q^*(s_t\vert o_t, \pi) \propto Q^*(s_t\vert \pi)P(o_t\vert s_t)$$. The EFE becomes the following:
<center>
$$
\begin{align}
\mathcal{G}(\pi\vert Q^*) &= \sum_{t=\tau+1}^{T}\mathbb{E}_{Q^*(o_t, s_t\vert \pi)}[ \log Q^*(s_t\vert \pi) - \log \tilde{P}(o_t) - \log \tilde{P}(s_t\vert o_t)] \\
&= \sum_{t=\tau+1}^{T}\mathbb{E}_{Q^*(o_t\vert \pi)}[- \log \tilde{P}(o_t)] + \mathbb{E}_{Q^*(o_t\vert \pi)}D_{KL}[Q^*(s_t\vert o_t, \pi) \vert \vert  \tilde{P}(s_t\vert o_t)] \\
&\quad - \mathbb{E}_{Q^*(o_t\vert \pi)}D_{KL}[Q^*(s_t\vert o_t, \pi) \vert \vert  Q^*(s_t\vert \pi)] \\
&\geq \sum_{t=\tau+1}^{T}\underbrace{\mathbb{E}_{Q^*(o_t\vert \pi)}[- \log \tilde{P}(o_t)]}_{\text{Pragmatic value}} - \underbrace{\mathbb{E}_{Q^*(o_t\vert \pi)}D_{KL}[Q^*(s_t\vert o_t, \pi) \vert \vert  Q^*(s_t\vert \pi)]}_{\text{Epistemic value}}
\end{align}
$$
</center>
where the last line is obtained by dropping the second term in the second line, because KL divergence is non-negative. This gives us the well-known pragmatic-epistemic value decomposition. But note that it is a **bound** on the EFE and not the EFE itself as defined in the prior. 

If we care doing a quick manipulation on the epistemic value as follows:
<center>
$$
\begin{align}
&\mathbb{E}_{Q^*(o_t\vert \pi)}D_{KL}[Q^*(s_t\vert o_t, \pi) \vert \vert  Q^*(s_t\vert \pi)] \\
&= \mathbb{E}_{Q^*(o_t, s_t\vert \pi)}[\log Q^*(s_t\vert \pi) + \log P(o_t\vert s_t) - \log Q^*(o_t\vert \pi) - Q^*(s_t\vert \pi)] \\
&= \mathbb{E}_{Q^*(s_t\vert \pi)P(o_t\vert s_t)}[\log P(o_t\vert s_t)] - \mathbb{E}_{Q^*(o_t, s_t\vert \pi)}[\log Q^*(o_t\vert \pi)] \\
&= -\mathbb{E}_{Q^*(s_t\vert \pi)}\mathcal{H}[P(o_t\vert s_t)] - \mathbb{E}_{Q^*(o_t\vert \pi)}[\log Q^*(o_t\vert \pi)]
\end{align}
$$
</center>
Plug this back to the previous decompositions, we have:
<center>
$$
\begin{align}
\mathcal{G}(\pi\vert Q^*) &\geq \sum_{t=\tau+1}^{T}\mathbb{E}_{Q^*(o_t\vert \pi)}[- \log \tilde{P}(o_t)] + \mathbb{E}_{Q^*(s_t\vert \pi)}\mathcal{H}[P(o_t\vert s_t)] + \mathbb{E}_{Q^*(o_t\vert \pi)}[\log Q^*(o_t\vert \pi)] \\
&= \sum_{t=\tau+1}^{T}\underbrace{D_{KL}[Q^*(o_t\vert \pi) \vert \vert  \tilde{P}(o_t)]}_{\text{Risk}} + \underbrace{\mathbb{E}_{Q^*(s_t\vert \pi)}\mathcal{H}[P(o_t\vert s_t)]}_{\text{Ambiguity}}
\end{align}
$$
</center>
This gives us a risk-ambiguity decomposition in the observation space.

Overall, this exercise suggests that whether an active inference will perform well on an actual (reward-seeking) task depends on whether the designer can specify a good prior for the task. 

### Connections to optimal control *without* cost function

A salient question at this point is **whether modern active inference has lost its root of "optimal control without cost function"**. 

To see the connection, let's modify the generative model slightly into the following factorized format:
<center>
$$
P(o_{1:\tau}, s_{1:T}, a_{1:T}) = \prod_{t=1}^{\tau}P(o_t|s_t)\prod_{t=1}^{T}P(s_t|s_{t-1}, a_{t-1})\pi(a_t|s_t)
$$
</center>
where $$P(s_1|s_0, a_0) = P(s_1)$$. We will correspondingly modify the belief distribution as $$Q(s_{1:T}, a_{1:T}) = \prod_{t=1}^{T}Q(s_t)Q(a_t|s_t)$$. The free energy function is defined as:
<center>
$$
\begin{align}
\mathcal{F}(o_{1:\tau}, Q) &= \mathbb{E}_{Q}[\log Q(s_{1:T}, a_{1:T}) - \log P(o_{1:\tau}, s_{1:T}, a_{1:T})] \\
&= \sum_{t=1}^{T}\mathbb{E}_{Q}[\log Q(s_t) + \log Q(a_t|s_t) - \log P(o_t, s_t|s_{t-1}, a_{t-1}) - \log \pi(a_t|s_t)] \\
&= \sum_{t=1}^{T}\mathbb{E}_{Q}[D_{KL}[Q(a_t|s_t)||\pi(a_t|s_t)]] + \mathbb{E}_{Q}[\log Q(s_t) - \log P(o_t, s_t|s_{t-1}, a_{t-1})]\\
&\triangleq \sum_{t=1}^{T} \mathbb{E}_{Q(s_t)}[D_{KL}[Q(a_t|s_t)||\pi(a_t|s_t)]] + \mathbb{E}_{Q(s_{t-1}, a_{t-1})}[\mathcal{F}(o_{t}, Q|a_{t-1})]
\end{align}
$$
</center>
where for $$t > \tau$$, $$o_t$$ does not exist.

**Perception as hidden state inference:**  It's easy to see that the optimal state estimates remain as approximate Bayesian posteriors:
<center>
$$
Q^*(s_t) \propto \left\{\begin{array}{ll}\exp\left(\mathbb{E}_{Q^*(s_{t-1}, a_{t-1})}[\log P(o_t, s_t|s_{t-1}, a_{t-1})] + c_1\right) & t \leq \tau \\ \exp\left(\mathbb{E}_{Q^*(s_{t-1}, a_{t-1})}[\log P(s_t|s_{t-1}, a_{t-1})] + c_2\right) & t > \tau \end{array}\right.
$$
</center>
**Control as prior inference:** Defining a similar notion of EFE for this factorized generative model:
<center>
$$
\pi(a_t|s_t) \propto \exp\left(-\mathcal{G}(a_t, s_t|Q^*)\right)
$$
</center>
where
<center>
$$
\mathcal{G}(a_t, s_t|Q^*) \triangleq \mathbb{E}_{Q^*(o_{\tau+1:T}, s_{\tau+1:T}|s_t, a_t)}[\log Q^*(s_{\tau+1:T}, a_{\tau+1:T}) - \log \tilde{P}(o_{\tau+1:T}, s_{\tau+1:T}, a_{\tau+1:T}|s_t, a_t)]
$$
</center>
we obtain a similar optimal policy where the posterior over actions roughly corresponds to the prior:
<center>
$$
Q^*(a_t|s_t) \propto \exp\left(-\mathcal{G}(a_t, s_t|Q^*) - \mathbb{E}_{Q^{*}(s_{t-1}, a_{t-1})}[\mathcal{F}(o_{t}, Q^*|a_{t-1})]\right)
$$
</center>
Familiar readers might recognize a lot of similarity between this factorization and standard notations in RL. This is indeed the factorization underlying most [deep RL-based implementations of active inference](https://arxiv.org/abs/2207.06415). 

To see the connection with "optimal control without cost function", let us define a new state variable $$\tilde{s} = [s, a]$$ which is the concatenation of the original state and action variables. We can then rewrite the factorized generative model as:
<center>
$$
P(o_{1:\tau}, \tilde{s}_{1:T}) = \prod_{t=1}^{\tau}P(o_t|\tilde{s}_t)\prod_{t=1}^{T}P(\tilde{s}_t|\tilde{s}_{t-1})
$$
</center>
where $$P(o_t|\tilde{s}_t) = P(o_t|s_t, a_t) = P(o_t|s_t)$$. We have visualized this and the 2012 model in the figure in the previous section.

Under this representation, **we dismantle the conventional treatment of states and actions as separate objects and instead understand actions as just a subset of states which we can directly affect or control if we put our mind to**. In this way, we realize active inference's motivation to unify both perception and control as hidden states inference, except that inference of active states is minimally affected by observations but rather largely modulated by the prior. But this is exactly the vision of active inference: actions are just **precise** predictions.

## Closing thoughts

We started with a motivation to understand how active inference formulates a **single** objective function for perception and control, which may in turn address the objective mismatch problem in RL. We have seen that active inference adopts a very perception-centric approach to the development of the unified objective as opposed to the control-centric approach in RL. 

On the surface, both versions of active inference appear to unify perception and control under a single objective of minimizing free energy, and yet the devil is in the detail of the generative model. Given the intricacies of the prior design choices, one may ask whether active inference still conforms to the mismatched perception-control paradigm, where the perception model is trained to predict environment observations and the action selection policy is optimized for some other criteria. I think the key here is that among all possible prior design choices, there is only a subset that is correct. As I discuss in [another post](https://rw422scarlet.github.io/2023/09/07/rationalize-efe.html), while the EFE objective may seem ad-hoc, it actually gives rise to a key property which makes agents robust to model error. In this sense, the EFE objective can be seen as a type of distribution correction objective (and can in fact be derived from this perspective). This further aligns the perception-centric approach with the control-centric approach in RL. 

## Other resources and code examples
Most of the contents in this post were taken from [my thesis](https://www.proquest.com/openview/34741b4f77c914241b634e5723891d29/1?pq-origsite=gscholar&cbl=18750&diss=y) and previous open sourced materials. In section 2.4.3 of the thesis, I gave an attempt to connect EFE and expected value and tried to understand the implicit assumptions made by EFE (although the connections were never validated). In section 2.4.5, I gave a brief overview of the history and neuroscience motivations behind active inference.

You can checkout my implementation of the old version (i.e., optimal control without cost) [here](https://www.kaggle.com/code/runway/active-inference-optimal-control-without-cost) and a slightly variation of the new version [here](https://www.kaggle.com/code/runway/active-inference-learning-action-oriented-models) (which implements the [action oriented model paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007805)). Both notebooks contain explanation of the formulations and experiments on concrete examples.

## Appendix

#### Derivation for optimal $$Q(s_{1:T})$$ in optimal control without cost
We start by finding the derivative of $$\mathcal{F}$$ w.r.t. to a specific element of vector $$Q(s_t)$$, for $$t \leq \tau$$:
<center>
$$
\begin{align}
\nabla_{Q(s_t)}\mathcal{F}(o_{1:\tau}, Q) &= \nabla_{Q(s_t)}\mathbb{E}_{Q(s_t)}[\log Q(s_t)] -\nabla_{Q(s_t)}\mathbb{E}_{Q(s_t)}[\log P(o_t\vert s_t)] \\
&\quad - \nabla_{Q(s_t)}\mathbb{E}_{Q(s_{t-1})Q(s_t)}[\log P(s_t\vert s_{t-1})] - \nabla_{Q(s_t)}\mathbb{E}_{Q(s_{t})Q(s_{t+1})}[\log P(s_{t+1}\vert s_t)] \\
&= \log Q(s_t) + 1 - \log P(o_t\vert s_t) - \mathbb{E}_{Q(s_{t-1})}[\log P(s_t\vert s_{t-1})] - \mathbb{E}_{Q(s_{t+1})}[\log P(s_{t+1}\vert s_t)]
\end{align}
$$
</center>
Setting the derivative to zero, we have:
<center>
$$
\log Q(s_t) \propto \log P(o_t\vert s_t) + \mathbb{E}_{Q(s_{t-1})}[\log P(s_t\vert s_{t-1})] + \mathbb{E}_{Q(s_{t+1})}[\log P(s_{t+1}\vert s_t)]
$$
</center>
We know that for $$t > \tau$$, we have not observed any $$o_t$$, thus the first term doesn't exist and we represent it using an indicator function.
