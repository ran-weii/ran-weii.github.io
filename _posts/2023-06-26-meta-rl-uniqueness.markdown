---
layout: post
title:  "The Uniqueness of Agent Beliefs in Meta and Bayesian Reinforcement Learning"
date:   2023-06-26 00:00:00 -0000
---

We previously studied [meta learning for sequential prediction](https://ran-weii.github.io/2023/06/23/meta-learning-uniqueness.html). Meta learning has an important advantage over the traditional Bayesian inference approach in that the **agent learns to perform implicit update of Bayesian statistics without require explicit generative modeling**. This is important because we can do away with the limitation of manual generative model specification and fully leverage the expressivity of function approximation. Afterwards, we can try to retrieve the implicitly learned Bayesian network for interpretability. We then showed that among all Bayesian agents parameterized with arbitrary generative models, **only a single agent with the correct generative model is optimal for the meta learning objective**. This is also important because it cuts down a significant amount of ambiguity if we were to interpret the agent's generative model. We recognize that both of these attractive properties may not be true if the agent does not fully optimize the meta learning objective. But let's set that aside for the moment. 

Beside meta-trained sequential prediction agents, there is also an important class of meta-trained decision making (RL) agents. Similar to the sequential prediction agents, these agents are parameterized with a memory module which takes in the entire observation history and output a policy (see [RL2](https://arxiv.org/abs/1611.02779) and [L2RL](https://arxiv.org/abs/1611.05763)). These agents are trained with simple RL algorithms but on a distribution of tasks which is not directly observable to the agent. The agent ends up discovering Bayes-optimal exploration strategies to uncover the underlying task variable. 

The [Meta-learning of Sequential Strategies](https://arxiv.org/abs/1905.03030) paper gave a relatively in-depth explanation of how Bayes-optimal behavior arises in sequential prediction agents, however it did not explain exactly how this behavior arise in meta RL agents. Furthermore, it begs the same question of **whether the implicitly learned Bayesian networks underlying meta RL agents are unique?** The decision making setting is more tricky, because the agent can exhibit the same behavior for different combinations of reward and world (dynamics) models. This is the reason why inverse reinforcement learning is in general unidentifiable even with infinite data. 

Nevertheless, implicit Bayesian inference in meta RL agents have received convincing empirical support in both standard meta RL with static task distribution (see [this paper](https://arxiv.org/abs/2010.11223)) and in general POMDP environments (see [this paper](https://arxiv.org/abs/2208.03520)). More subtly, [this paper](https://arxiv.org/abs/2010.04466) showed that whether the meta RL agent will learn an adaptive strategy to perform Bayesian inference or a (seemingly unintelligent) heuristic strategy depends on the environment uncertainty, task complexity, and training lifetime. 

We will try to unpack these insights in this post. The steps are surprisingly similar to the sequential prediction agent. 

## Meta Reinforcement Learning

In meta RL, we have a distribution of tasks index by $$\mu$$, where each task may correspond to a different transition dynamics $$P(s_{t+1} \vert s_t, a_t; \mu)$$ and reward $$R(s_t, a_t; \mu)$$. Similar to the sequential prediction setting, we assume the agent chooses actions using the entire observation history $$h_t := (s_{0:t}, a_{0:t-1}, r_{0:t-1})$$ from a policy $$\pi(a_t \vert h_t)$$. Unfortunately, the agent cannot directly observe the task. 

The agent's interaction with the environment induces a trajectory $$\tau = (s_{0:\infty}, a_{0:\infty}, r{0:\infty})$$ in the environment with distribution:
<center>
$$
P(\tau) = P(\mu)\prod_{t=0}^{\infty}P(s_t|s_{t-1}, a_{t-1}; \mu)\pi(a_t|h_t)R(s_{t}, a_{t}; \mu)
$$
</center>
The objective $$J$$ of the agent is find a policy which maximizes the expected cumulative reward (a.k.a return) of the trajectory:
<center>
$$
J(\pi) = \mathbb{E}_{P(\tau)}[\sum_{t=0}^{\infty}\gamma^{t}R(s_t, a_t; \mu)]
$$
</center>
where $$\gamma \in (0, 1)$$ is a discount factor. 

If the agent already knows the ground truth environment model, including the task distribution, transition distribution, and reward function, the standard approach is to plan in the belief space using dynamic programming:
<center>
$$
Q(b, a) = \sum_{\mu}b(\mu)R(s, a; \mu) + \mathbb{E}_{P(s'|s, a) \delta(b'|s, a, r, s', b)}[V(b')]
$$
</center>
where $$V$$ and $$Q$$ are the state and state-action value functions, $$b(\mu) := P(\mu|h)$$ is the Bayesian belief of the current task given history $$h$$, $$P(s'|s, a) := \sum_{\mu}P(s'|s, a; \mu)b(\mu)$$ is the posterior predictive of the next state, and $$\delta(b'|s, a, r, s', b)$$ is a deterministic transition of agent belief computed from Bayes' rule. Policy derived from this $$Q$$ function is known to handle exploration and exploitation in a Bayes optimal fashion, however, it requires the ground truth environment model to be known.

Instead, we consider the model-free RL setting, where the agent does not have an explicit model of the environment but only tries to find a history-conditioned policy to maximize the return $$J$$. What will this type of agent look like?

## Can Bayesian inference still emerge in meta RL agents?

The challenge of performing this analysis in the meta RL setting is that the agent has influence on the data distribution and the likelihood model we previously manipulated is now replaced with a fixed reward function. Interestingly, we can obtain a similar analysis as the sequential prediction setting if we assume the meta RL agent learns using the policy gradient method. 

In this setting, the agent updates its policy by taking the following gradient:
<center>
$$
\mathbb{E}_{P^{\pi}(h_t; \mu), a_t \sim \pi(\cdot|h_t)}[Q(h_t, a_t)\nabla\log\pi(a_t|h_t)]
$$
</center>
where $$P^{\pi}(h_t; \mu)$$ is a joint distribution of history and the task variable when deploying the current policy $$\pi$$ in the environment. Notice that this can be seen as a maximum likelihood estimation problem weighted by *on-policy* $$Q$$. We can assume that $$Q$$ can be estimated independently, for example using Monte Carlo rollouts. 

Let us denote the belief-based policy as:
<center>
$$
\pi(a|h) = \sum_{\mu}\pi(a|s; \mu)P(\mu|h)
$$
</center>
Note, however, that this policy (roughly the Thompson Sampling policy) may not be as optimal as belief-space planning if the optimal belief-based policy cannot be expressed as a linear combination over the task variable, but it makes the analysis tractable. 

Our goal is to show that the belief-based policy is better than any alternative policy $$p(a|h)$$ on the value-weighted likelihood. We can express this condition as:
<center>
$$
\mathbb{E}_{P^{\pi}(a_t, h_t; \mu)}\left[Q(h_t, a_t)\log\frac{\pi(a_t|h_t)}{p(a_t|h_t)}\right] \geq 0
$$
</center>
The proof proceeds in a way very similar to the sequential prediction setting from the [last post](https://ran-weii.github.io/2023/06/23/meta-learning-uniqueness.html):
<center>
$$
\begin{align}
&\mathbb{E}_{P^{\pi}(a_t, h_t; \mu)}\left[Q(h_t, a_t)\log\frac{\pi(a_t|h_t)}{r(a_t|h_t)}\right] \\
&= \sum_{\mu}\sum_{h_t}\sum_{a_t}Q(h_t, a_t)\log\frac{\pi(a_t|h_t)}{p(a_t|h_t)}\pi(a_t|s_t, \mu)P(h_t|\mu)P(\mu) \\
&= \sum_{\mu}\sum_{h_t}\sum_{a_t}Q(h_t, a_t)\log\frac{\pi(a_t|h_t)}{p(a_t|h_t)}\pi(a_t|s_t, \mu)P(\mu|h_t)P(h_t) \\
&= \sum_{h_t}\left[\sum_{a_t}Q(h_t, a_t)\log\frac{\pi(a_t|h_t)}{p(a_t|h_t)}\sum_{\mu}\pi(a_t|s_t, \mu)P(\mu|h_t)\right]P(h_t) \\
&= \sum_{h_t}\left[\sum_{a_t}Q(h_t, a_t)\log\frac{\pi(a_t|h_t)}{p(a_t|h_t)}\pi(a_t|h_t)\right]P(h_t) \\
&= \mathbb{E}_{P(h_t)}\mathbb{E}_{\pi(a_t|h_t)}\left[Q(h_t, a_t)\log\frac{\pi(a_t|h_t)}{p(a_t|h_t)}\right]\\
&\not\geq 0
\end{align}
$$
</center>
This derivation reveals two problems. First, the problem is not intrinsically meaningful in the sense that the policy $$\pi$$ is trying to reproduce itself. This is a similar problem to the sequential prediction setting. The second problem is that the last line is not necessarily greater than zero, which fails our proof. Even if we assume $$Q$$ is strictly positive, the composition of $$Q$$ and $$\log$$ is no longer concave because $$Q$$ can arbitrarily warp with $$\log$$ function. 

The [Meta-learning of Sequential Strategies](https://arxiv.org/abs/1905.03030) paper instead tries to show that the belief based policy is optimal for encoding an expert policy. This is undesirable since it excludes the more general RL setting. Furthermore, it brings in additional complexity due to causal intervention and distribution shift between the expert and learner policies. 

## Is the belief-based agent unique?

Given the empirical identification of implicit belief in meta RL agent and the failure of the previous attempt, it begs the question of how these beliefs emerge? Let's now try to answer this question from an optimization perspective. 

As before, let's assume our meta RL agent's policy parameterized with $$\theta$$ has the following form:
<center>
$$
\pi_{\theta}(a|h) = \sum_{\mu}\pi_{\theta}(a|s; \mu)b_{\theta}(\mu|h)
$$
</center>
For simplicity, we remove the reward from belief updating. But note that the general results do not change when it is incorporated. 

Similar to the sequential prediction setting, we will focus on analyzing one time slice of the Bayesian network describing the environment agent interaction. The corresponding value-weighted log likelihood is:
<center>
$$
\begin{align}
\mathbb{E}_{P(a_{t}, h_{t})}[\mathcal{L}(\theta)] &= \mathbb{E}_{P(a_{t}, h_{t})}\left[Q(h_t, a_t)\log\sum_{\mu}\pi_{\theta}(a_{t}|s_t; \mu)b_{\theta}(\mu|h_t)\right] \\
&= \mathbb{E}_{P(a_{t}, h_{t})}\left[Q(h_t, a_t)\log\sum_{\mu}\exp(\log \pi_{\theta}(a_{t}|s_t; \mu) + \log b_{\theta}(\mu|h_t))\right]
\end{align}
$$
</center>
The log likelihood gradient is:
<center>
$$
\begin{align}
\nabla\mathcal{L}(\theta) = \mathbb{E}_{\pi(\mu)}[Q(h_t, a_t)\left(\nabla\log \pi_{\theta}(a_{t}|s_t; \mu) + \nabla\log b_{\theta}(\mu|h_t)\right)]
\end{align}
$$
</center>
where $$\pi(\mu) \propto \exp(\log \pi_{\theta}(a_t|s_t; \mu) + \log b_{\theta}(\mu|h_t)) = b_{\theta}(\mu|a_{t}, x_t)$$ is a retrospective belief for what the task variable should have been if the agent had taken action $$a_t$$. 

For the belief term, we have:
<center>
$$
\begin{align}
\nabla\log b_{\theta}(\mu|h_t) &= \nabla\log\frac{P_{\theta}(s_t|s_{t-1}, a_{t-1}; \mu)b(\mu)}{\sum_{\mu'}P_{\theta}(s_t|s_{t-1}, a_{t-1}; \mu')b(\mu')} \\
&=\nabla\log P_{\theta}(s_t|s_{t-1}, a_{t-1}; \mu) + \log b(\mu) - \nabla\log \sum_{\mu'}P_{\theta}(s_t|s_{t-1}, a_{t-1}; \mu')b(\mu') \\
&=\nabla\log P_{\theta}(s_t|s_{t-1}, a_{t-1}; \mu) + \log b(\mu) - \nabla\log \sum_{\mu'}\exp(\log P_{\theta}(s_t|s_{t-1}, a_{t-1}; \mu') + \log b(\mu')) \\
&= \nabla\log P_{\theta}(s_t|s_{t-1}, a_{t-1}; \mu) + \log b(\mu) - \mathbb{E}_{\pi'(\mu)}[\nabla\log P_{\theta}(s_t|s_{t-1}, a_{t-1}; \mu) + \log b(\mu)]
\end{align}
$$
</center>
where $$\pi'(\mu) = b_{\theta}(\mu|h_t)$$. 

Putting all together, the log likelihood gradient is:
<center>
$$
\begin{align}
\nabla\mathcal{L}(\theta) &= \mathbb{E}_{\pi(\mu)}[Q(h_t, a_t)(\nabla\log \pi_{\theta}(a_t|s_t; \mu) + \nabla\log P_{\theta}(s_t|s_{t-1}, a_{t-1}; \mu) + \log b(\mu))] \\
&\quad - \mathbb{E}_{\pi'(\mu)}[Q(h_t, a_t)(\nabla\log P_{\theta}(s_t|s_{t-1}, a_{t-1}; \mu) + \log b(\mu))] \\
&= \mathbb{E}_{b_{\theta}(\mu|a_t, h_t)}[Q(h_t, a_t)\nabla\log \pi_{\theta}(a_t|s_t; \mu)] \\
&\quad + \sum_{\mu}\underbrace{\left(b_{\theta}(\mu|a_t, h_t) - b_{\theta}(\mu|h_t)\right)}_{\approx 0}Q(h_t, a_t)\nabla\log P_{\theta}(s_t|s_{t-1}, a_{t-1}; \mu) \\
&\quad + \sum_{\mu}\underbrace{\left(b_{\theta}(\mu|a_t, h_t) - b_{\theta}(\mu|h_t)\right)}_{\approx 0}Q(h_t, a_t)\log b(\mu)
\end{align}
$$
</center>
The only incentive for the agent to learn a model of the environment is in the second term, which only occurs in the beginning of a sequence if we assume adjacent beliefs to become increasingly similar as we did in the sequential prediction setting. We see that learning of transition probabilities depends on how different the two adjacent beliefs are and how large the expected value $$Q$$ is. If the retrospective beliefs are very different from the prior beliefs, it must be that the policy $$\pi(a|s; \mu)$$ is very decisive for a particular state $$s$$. In other words, **the most amount of dynamics learning will happen in crucial states with high value**, more likely in earlier part of the sequence. 

The analysis potentially explains some of the experimental findings. In [this paper](https://arxiv.org/abs/2208.03520), the authors found that recurrent neural network policies whose hidden states have higher mutual information with the ground truth state achieve higher returns. Furthermore, in environments with distracting states, where the state variables do not participate in the optimal policy, the RNN hidden states learn to ignore them. The first observation is most likely an optimization problem. We know that the estimated *on-policy* $$Q$$ function depends on the true belief. If the RNN hidden state, the $$b(\mu\vert a, h)$$ in our setting, cannot properly recognize the ground truth state, then the policy gradient will ambiguate between different beliefs and won't fully take advantage of the latent task variable. For the second observation, it is more likely that the neural network uses its available capacity towards modeling useful hidden states than an optimization issue. Because if it were an optimization issue, the beliefs over the distracting states should be very imprecise, causing difficulty of updating the transition probabilities over these variables. Then we should predict that the mutual information should be constant rather than decrease over training. 

In [this paper](https://arxiv.org/abs/2010.04466), the authors found that whether the meta RL agent learns a "heuristic" over an adaptive policy if the task is too complex, the environment is too uncertain, or the meta-training time is too short. First, when the environment is degenerate in the sense that all hidden states are distracting states so that they don't contribute to the optimal policy, one cannot distinguish heuristic and adaptive policies. When the environment or task is not degenerate, the number of exploration steps increases with the environment uncertainty and task complexity in order to disambiguate the task variable. Lastly, when the meta-training time is too short, it is as if the environment is too uncertain since the agent has not had enough time to disambiguate the environment and build an implicit model. Our analysis agrees with these observations. 