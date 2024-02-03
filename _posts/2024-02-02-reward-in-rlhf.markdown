---
layout: post
title:  "Do We Need Reward in RLHF? DPO and the Unlikelihood Family Curse"
date:   2024-02-02 00:00:00 -0000
---

The quick rise of [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) within the RLHF toolkit has spawned a sizable debate last year on whether RL is needed for RLHF. Luckily, the question is unresolved so that we can continue to think about it. After some thoughts, I realize the question might be slightly misleading and the true underlying question is **whether (explicit) reward modeling is needed?** 

To start appreciating this question, I did a deep dive (or detour?) in the [previous post](https://ran-weii.github.io/2024/01/15/why-rlhf.html) to understand what problem is RLHF even trying to solve and what role does reward modeling play in that process. The takeaway was that RLHF aims to at least 1) mitigate exposure bias due to training on data distribution but deploying on self-generated distribution (often resulting in producing degenerate outputs) and 2) outperform humans (suboptimal demonstrations) on some ground-truth reward. However, the former seems to be a bigger headache in practice and reward models can help by providing automated feedback in closed-loop training. 

With these goals and the role of reward in mind, we can finally examine whether DPO and the alike introduce any gaps in the process. While DPO is commonly understood as preference optimization without RL, the key to this post is that it is a representative of the **unlikelihood family algorithms** which maximizes the policy likelihood of some outputs and minimizes the policy likelihood of other outputs. This will allow us to link its optimization objective directly to the policy behavior, which is what eventually matters to the goals of RLHF. 

I will start by examining the relationship between DPO and RLHF, whether they learn the same reward and policy, and whether DPO addresses the problem RLHF aims to address. I will then examine the unlikelihood family and conjecture what's missing about RLHF. **In essence, current preference modeling approaches have a key property that prevents learning the right reward for out-of-distribution responses to correct exposure bias (i.e., it's cursed). However, the debate between DPO and RLHF, and more broadly on reward modeling and RL, seems misplaced as fails to emphasize an important point: learning from human feedback is fundamentally an active learning process, so what matters is acquiring relevant feedback and fast iterations.**

The math has been condensed in the main sections. Details can be found in the appendix.

## How close is DPO to RLHF?
### A brief recap of DPO
DPO aims to simplify the two-step reward learning + policy optimization process into a single step using the following insight: Given that the optimal KL-regularized policy has the following form:
<center>
$$
\begin{align}
&\arg\max_{\pi}\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)}[R(x, y) -\beta D_{KL}(\pi(y|x) || \pi_{\text{ref}}(y|x))] \\
&= \frac{\pi_{\text{ref}}(y|x)\exp(\frac{1}{\beta}R(x, y))}{\sum_{\tilde{y}}\pi_{\text{ref}}(\tilde{y}|x)\exp(\frac{1}{\beta}R(x, \tilde{y}))} \\
&= \frac{\exp(\frac{1}{\beta}R(x, y) + \log \pi_{\text{ref}}(y|x))}{\sum_{\tilde{y}}\exp(\frac{1}{\beta}R(x, \tilde{y}) + \log \pi_{\text{ref}}(\tilde{y}|x))}
\end{align}
$$
</center>
The reward function can be reparameterized using the respective optimal policy as:
<center>
$$
\begin{align}
&R(x, y) = \beta\log\pi(y|x) - \beta\log \pi_{\text{ref}}(y|x) + \beta\log Z(x) \\
&\text{ where } Z(x) = \sum_{\tilde{y}}\exp(\frac{1}{\beta}R(x, \tilde{y}) + \log \pi_{\text{ref}}(\tilde{y}|x))
\end{align}
$$
</center>
We can plug this reward parameterization into the preference model and as a result obtain the optimal policy without RL. The preference learning loss function operates directly in the space of policies:
<center>
$$
\begin{align}
&\max_{\pi} \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}[\log P(y_w \succ y_w|x)] \\ 
&= \max_{\pi}\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
\end{align}
$$
</center>
where $$\sigma(\cdot)$$ is the sigmoid function. 

### Does DPO learn the right reward?
Intuitively, DPO would learn the same reward function as RLHF if its (implicit) reward gradient coincides with the RLHF reward gradient, assuming the same initialization, same reward function class, and gradient-based optimization. 

The RLHF reward gradient has the following form (using softmax distribution identity):
<center>
$$
\mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}}\left[P(y_{l} \succ y_{w}|x)\left(\nabla_{R} R(y_{w}, x) - \nabla_{R} R(y_{l}, x)\right)\right]
$$
</center>
where $$P$$ is the current estimate of the preference model. 

To get the implicit DPO reward gradient, we simply take the DPO loss function but instead reparameterize the policy (i.e., the decision variables) with the reward function. Using the following identity of softmax policy likelihood gradient w.r.t. reward:
<center>
$$
\begin{align}
\nabla_{R}\log\pi(y|x) &= \frac{1}{\beta}\nabla_{R} R(x, y) - \frac{1}{\beta}\mathbb{E}_{\tilde{y} \sim \pi(\cdot|x)}[\nabla_{R} R(x, \tilde{y})]
\end{align}
$$
</center>
The DPO reward gradient can be written as:
<center>
$$
\begin{align}
&\mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}}\left[P(y_{l} \succ y_{w}|x)\left(\beta\nabla_{R} \log\pi(y_{w}|x) - \beta\nabla_{R} \log\pi(y_{l}|x)\right)\right] \\
&= \underbrace{\mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}}\left[P(y_{l} \succ y_{w}|x)\left(\nabla_{R} R(x, y_{w}) - \nabla_{R} R(x, y_{l})\right)\right]}_{\text{Same as RLHF reward gradient}}
\end{align}
$$
</center>
where the second line is due to canceling the second term in the policy likelihood gradient. The current preference estimate $$P$$, even though parameterized by the current policy and the reference policy, is the same as the reward parameterization in RLHF due to the reparameterization equivalence. 

This exercise seems almost redundant because we have assumed the reparameterization equivalence. But we did get exactly the same reward gradient. This means that if we are somehow able to retrieve the reward function from the DPO policy (e.g., via inverse RL), we should get the same reward function as the RLHF one. 

| ![](/assets/2024-02-02-reward-in-rlhf/likelihood_unlikelihood.png) | 
|:--:| 
| *An illustration of likelihood and unlikelihood. [HF stable diffusion 2.1](https://huggingface.co/spaces/stabilityai/stable-diffusion): "A greek god lifting/pushing the ocean wave upward/downward."* |

### What kind of policy does DPO learn and how?
In the above DPO objective gradient derivation, if we instead differentiate w.r.t. the policy instead of the reward, we get the following gradient:
<center>
$$
\mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}}\left[\beta P(y_{l} \succ y_{w}|x)\left(\nabla_{\pi} \log\pi(y_{w}|x) - \nabla_{\pi} \log\pi(y_{l}|x)\right)\right] \\
$$
</center>
In other words, the policy is trained to maximize the likelihood of response $$y_{w}$$ and minimize the likelihood of $$y_{l}$$, **i.e., likelihood and unlikelihood**, weighted by the preference estimate and $$\beta$$. 

This is almost the same equation as in the DPO paper. The important point here is that the preference estimate $$P$$ plays the role of the KL constraint. A non-negligible weight is applied only if the log ratio between the learner policy and the reference policy on the positive example is much less than the log ratio of the negative example:
<center>
$$
\log\pi(y_{w}|x) - \log \pi_{\text{ref}}(y_{w}|x) \ll \log\pi(y_{l}|x) - \log \pi_{\text{ref}}(y_{l}|x)
$$
</center>
which is when $$\pi(y_w|x)$$ is too small or $$\pi_{\text{ref}}(y_w|x)$$ is too large. 

**Does DPO learn the same policy as RLHF?** Empirically, it has been shown that DPO works well when initialized with the reference policy, but when initialized with a random policy, it significantly underperforms (see the [Zephyr paper](https://arxiv.org/abs/2310.16944)). So how can this happen if DPO incurs the same reward gradient as RLHF, and the DPO policy is by definition reward-optimal? **A sensible clue seems to be whether the DPO reparameterization puts it in the same reward function class as RLHF and whether the training data is sufficient to identify the reward function for the respective class.** 

Notice that the KL-regularized policy optimization objective in RLHF can be interpreted as an entropy-regularized policy optimization objective with a composite reward:
<center>
$$
\begin{align}
&\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)}[R(x, y) - \beta D_{KL}(\pi(y|x) || \pi_{\text{ref}}(y|x))] \\
&= \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)}[\underbrace{R(x, y) + \beta \log\pi_{\text{ref}}(y|x)}_{\text{Composite reward}} + \beta\mathcal{H}(\pi(y|x))]
\end{align}
$$
</center>
In other words, with the reference policy likelihood being a part of the composite reward function, the learned reward function only needs to model the residual. In this case, a much smaller and simpler reward function class can be used. [This paper](https://arxiv.org/abs/2312.11456) shows that the sample-complexity of RLHF depends on the reward function complexity. A less complex reward function class likely requires less data to identify, given that the set of reward function hypotheses inconsistent with the preference data gets eliminated exponentially fast as we sample data sufficiently diverse in the reward feature space (see [this paper](https://arxiv.org/abs/1907.03976)). Unfortunately, DPO does not directly control the implicit reward function complexity. 

**Does DPO achieve the goal of RLHF?** Ultimately, we do not care whether DPO gives us the same policy as RLHF. Rather, we care whether DPO can achieve the main goals of RLHF: 1) alleviate exposure bias and 2) outperform human demonstrations. Exposure bias still matters in the contextual bandit formulation of RLHF, because even though we denote a response using a single letter $$y$$, it is still a sequence of tokens sampled from an autoregressive model:
<center>
$$
\pi(y|x) = \prod_{i}\pi(y_i|y_{<i}, x)
$$
</center>
RLHF alleviates exposure bias by training on self-generated distribution, assuming the learned reward function is correct. However, DPO by definition is only ever trained on other-generated distribution (i.e., the dataset). So whether DPO can alleviate exposure bias likely depends on the preference data distribution. 

The likelihood and unlikelihood perspective of the DPO loss shows that, **to some extent,DPO is simply imitating the positive examples in the dataset**. [This paper](https://arxiv.org/abs/2312.11456) shows that the optimal preference data distribution is the expert response distribution (i.e., what we want to achieve with fine-tuning). This explains why DPO fine-tuning on preference data sampled from more powerful models (e.g., GPT4 in [UltraFeedback](https://arxiv.org/abs/2310.01377)) can significantly enhance performance. But the imitation perspective does suggest a ceiling on how much we should expect DPO to outperform humans. 

## A brief overview of the unlikelihood family
The term "unlikelihood" was perhaps first proposed by [Welleck et al, 2019](https://openreview.net/forum?id=SJeYe0NtvH). To address exposure bias (mostly repetitions) in maximum likelihood training of autoregressive language models, they proposed to additionally minimize the likelihood of previous (context) tokens:
<center>
$$
\max_{\pi} \mathbb{E}_{y_i \sim \mathcal{D}}\bigg[\log\pi(y_i|y_{<i}) - \underbrace{\alpha \sum_{k<i}\log\pi(y_k|y_{<i})}_{\text{unlikelihood}}\bigg]
$$
</center>
This method significantly reduced repetitions, but it was not clear how to easily find negative samples for the unlikelihood term for more general problems. 

Around the same time DPO was published, a method called sequence likelihood calibration was introduced to RLHF with the following ranking loss:
<center>
$$
\min_{\pi}\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\max(0, \beta - \log\pi(y_w|x) + \log\pi(y_l|x))\right]
$$
</center>

## Unlikelihood and generative modeling of human preferences
More recently, [this paper](https://arxiv.org/abs/2311.14115) proposed a density estimation perspective on preference modeling, where the preference model is defined by the log likelihood of the learner response policy:
<center>
$$
\begin{align}
P(y_w \succ y_l|x) &= \frac{\pi(y_w|x)}{\pi(y_w|x) + \pi(y_l|x)} \\
&= \frac{\exp(\log\pi(y_w|x))}{\exp(\log\pi(y_w|x)) + \exp(\log\pi(y_l|x))}
\end{align}
$$
</center>
This is the same preference model I studied in [this post](https://rw422scarlet.github.io/2024/01/15/why-rlhf.html) around the same time DPO was published, which has an appealing cognitive science interpretation as the human's **preferred response distribution** and generalizes the reward modeling approach.

The corresponding policy gradient for the preference learning objective (using softmax identity) is:
<center>
$$
\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[P(y_l \succ y_w|x)(\nabla_{\pi}\log\pi(y_w|x) - \nabla_{\pi}\log\pi(y_l|x))\right]
$$
</center>
At a first glance, this seems to be a more extreme version of DPO where even the reference policy constraint is no longer present since $$P$$ is just weighting how large the positive likelihood is compared to the negative likelihood. 

### Fixing DPO with regularized generative modeling?
The unlikelihood perspective puts DPO on a more similar ground with imitation learning, which makes us ask whether we could fix it (e.g., improve its robustness to exposure bias) using similar techniques from inverse RL. 

As I reviewed in the [previous post](https://ran-weii.github.io/2024/01/15/why-rlhf.html), inverse RL is simply a class of imitation learning approach with a constraint imposed on the policy class. The right constraint allows the policy to extrapolate or generalize to unseen states. In the case of inverse RL, constraining the policy to be reward-optimal allows the reward model to provide feedback to the learner policy in conjunction with a dynamics model (although I prefer the regularization perspective over the feedback perspective as it highlights that the regularization effect comes from the dynamics model; see [our paper](https://arxiv.org/abs/2309.08571) on this). 

To test this hypothesis, we impose the same constraint as DPO and RLHF that the policy is the optimal KL-regularized policy:
<center>
$$
\begin{align}
\max_{R} &\quad \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log\sigma(\log\pi(y_w|x) - \log\pi(y_l|x))\right] \\
\text{s.t.} &\quad \pi = \frac{\pi_{\text{ref}}(y|x)\exp(\frac{1}{\beta}R(x, y))}{\sum_{\tilde{y}}\pi_{\text{ref}}(\tilde{y}|x)\exp(\frac{1}{\beta}R(x, \tilde{y}))}
\end{align}
$$
</center>
This requires us to go back to the reward modeling and RL regime where in each optimization step, we must first update the reward along the following gradient:
<center>
$$
\mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}}\left[P(y_{l} \succ y_{w}|x)\left(\nabla R(x, y_{w}) - \nabla R(x, y_{l})\right)\right]
$$
</center>
and then find the optimal policy w.r.t. the updated reward using KL-regularized RL. Unfortunately, we encounter the same problem as DPO that the reward model is only ever trained on other-generated distribution. 


**What's wrong with RLHF?** In inverse RL, the key to mitigating exposure bias comes from the second term in the policy likelihood gradient:
<center>
$$
\begin{align}
\nabla_{R}\log\pi(y|x) &= \frac{1}{\beta}\nabla_{R} R(x, y) - \frac{1}{\beta}\underbrace{\mathbb{E}_{\tilde{y} \sim \pi(\cdot|x)}[\nabla_{R} R(x, \tilde{y})]}_{\text{Always canceled in RLHF}}
\end{align}
$$
</center>
which allows us to learn to provide feedback via reward on self-generated distribution. **However, in the RLHF preference model class, the second term is always canceled.** This is the key to the convenience of DPO (i.e., cancellation of log partition function $$Z$$), but it also means that we will never update the reward estimate on OOD samples. 

**Can this formulation still be more robust because of RL training on self-generated distribution?** The answer is likely no, because the benefit of training on self-generated distribution comes from having an accurate reward function. Since the reward is only ever trained on in-distribution samples, there is in general no assurance on the policy's behavior out-of-distribution. 

**Is there any benefit with this formulation?** It is possible that this formulation will provide more flexible in-distribution function approximation. Given the gradient weight is under-constrained as opposed to in DPO, it would more rarely encounter cases where the reward gradient is weighted by zero, e.g., when the reference policy likelihood is too high. In this case, the RL policy can increase the reward scale to fit certain data points, which means the $$\beta$$ parameter ([empirically shown to be sensitive](https://huggingface.co/blog/pref-tuning)) is not necessary. However, given the stochasticity of real datasets and that DPO often overfits, this advantage will most likely be inapplicable in practice. 

**What will safe RLHF?**  It seems like the only thing that will save degeneracy/reward hacking in the current RLHF pipeline is conservatism. This can be achieved through reward uncertainty estimation (e.g., via [ensembles](https://arxiv.org/abs/2310.02743)). In this way, theoretically the worst a RLHF policy will do is to imitate the positive examples and refrain from imitating the negative examples in the dataset. However, this would require the positive examples in the dataset to have high quality.

## Summary and outlook
The conclusion of the exercise is that DPO and the unlikelihood family seem absolutely fine, and yet fundamentally flawed. Reward may help in some places but is highly nuanced. However, the debate seems to miss the main point: **RLHF is fundamentally an active learning process from human feedback**. If we are in the active learning regime, then whether we use RL or unlikelihood doesn't really matter. People seem to have realized this with the use of iterative DPO. 

There are currently two notable research directions in RLHF. Iterative self-improvement (e.g., [self-instruct](https://arxiv.org/abs/2212.10560) and [self-reward](https://arxiv.org/abs/2401.10020)) aims to sift through noisy and sparse human data and generate artificial data, for which the reliability seems dubious but is the crux to learning from AI feedback and has been well performing in practice. The other direction is to get the most out of human data by carefully treating their heterogeneity and figuring out what to do subsequently. For example, [generative preference modeling](https://arxiv.org/abs/2311.14115) can be adapted to identify heterogeneous preferences via mixture models. The recently proposed [self-play preference optimization](https://arxiv.org/abs/2401.04056) deals with satisfying the majority's preference in the mixture. The game of more nuanced preference modeling and aggregation seem to be just getting started. But this is a regime that concerns more with stakeholder decisions than algorithmic development. 

### Appendix
#### Helpful identities
**Logsumexp gradient**:
<center>
$$
\begin{align}
\nabla_{f}\log\sum_{x}\exp(f(x)) &= \frac{1}{\sum_{\tilde{x}}\exp(f(\tilde{x}))}\nabla_{f}\sum_{x}\exp(f(x)) \\
&= \frac{1}{\sum_{\tilde{x}}\exp(f(\tilde{x}))}\sum_{x}\nabla_{f}\exp(f(x)) \\
&= \frac{1}{\sum_{\tilde{x}}\exp(f(\tilde{x}))}\sum_{x}\exp(f(x))\nabla f(x) \\
&= \sum_{x}\frac{\exp(f(x))}{Z}\nabla_{f} f(x) \\
&= \mathbb{E}_{x \sim \pi(\cdot)}[\nabla_{f} f(x)]
\end{align}
$$
</center>
where $$\pi(x) \propto \exp(f(x))$$.

**Log softmax gradient**: Let us denote the softmax function as:
<center>
$$
\sigma(f(x)) = \frac{\exp(f(x))}{\sum_{\tilde{x}}\exp(f(\tilde{x}))}
$$
</center>

Using the logsumexp gradient, the gradient of log softmax w.r.t. $$f(\cdot)$$ is:
<center>
$$
\begin{align}
\nabla\log \sigma(f(x)) &= \nabla f(x) - \nabla \log\sum_{\tilde{x}}\exp(f(x)) \\
&= \nabla f(x) - \mathbb{E}_{\tilde{x} \sim \pi(\cdot)}[\nabla f(\tilde{x})]
\end{align}
$$
</center>

#### RLHF reward gradient
Recall that the RLHF preference model is defined as:
<center>
$$
P(y_w \succ y_l|x) = \frac{\exp(R(x, y_w))}{\exp(R(x, y_w)) + \exp(R(x, y_l))}
$$
</center>

The maximum likelihood learning objective is:
<center>
$$
L(R) = \mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}}[\log P(y_{w} \succ y_{l}|x)]
$$
</center>

The gradient of the loss function w.r.t $$R$$ is:
<center>
$$
\begin{align}
\nabla_{R} L(R) &= \nabla_{R} \mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}}[\log P(y_{w} \succ y_{l}|x)] \\
&= \mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}}[\nabla_{R} \log P(y_{w} \succ y_{l}|x)] \\
&= \mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}}[\nabla_{R} R(y_{w}, x) - P(y_{w} \succ y_{l}|x)\nabla_{R} R(y_{w}, x) - P(y_{l} \succ y_{w}|x)\nabla_{R} R(y_{l}, x)] \\
&= \mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}}[(1 - P(y_{w} \succ y_{l}|x))\nabla_{R} R(y_{w}, x) - P(y_{l} \succ y_{w}|x)\nabla_{R} R(y_{l}, x)] \\
&= \mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}}[P(y_{l} \succ y_{w}|x)\left(\nabla_{R} R(y_{w}, x) - \nabla_{R} R(y_{l}, x)\right)] \\
\end{align}
$$
</center>
The loss function gradient has the same form for any choice of reward function. For example, we can set $$R(y, x) = \log\pi(y|x)$$ to get the last preference model based on preferred response distribution. 

It is also good to clarify where the training data are sampled from in RLHF. Typically, the prompts are sampled from a distribution of use cases (e.g., in the [OpenAI report](https://arxiv.org/abs/2203.02155)), and then a pair of responses are sampled from the reference model:
<center>
$$
x \sim P(x), y_{w} \sim \pi_{\text{ref}}(\cdot|x), y_{l} \sim \pi_{\text{ref}}(\cdot|x)
$$
</center>
This distribution will determine on what kind of inputs the learned reward function is accurate.

#### DPO policy log likelihood gradient
Recall that DPO reparameterize the reward function as:
<center>
$$
\begin{align}
R(x, y) &= \beta\log\pi(y|x) - \beta\log \pi_{\text{ref}}(y|x) + \beta\log Z(x) \\
&= \beta\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} + \log Z(x)
\end{align}
$$
</center>

Replacing the preference model with this reward function, we get:
<center>
$$
\begin{align}
P(y_w \succ y_l|x) &= \frac{\exp(\beta\log\frac{\pi(y_{w}|x)}{\pi_{\text{ref}}(y_{w}|x)} + \beta\log Z(x))}{\exp(\beta\log\frac{\pi(y_{w}|x)}{\pi_{\text{ref}}(y_{w}|x)} + \beta\log Z(x)) + \exp(\beta\log\frac{\pi(y_{l}|x)}{\pi_{\text{ref}}(y_{l}|x)} + \beta\log Z(x))} \\
&= \frac{\exp(\beta\log\frac{\pi(y_{w}|x)}{\pi_{\text{ref}}(y_{w}|x)})Z^{\beta}(x)}{\exp(\beta\log\frac{\pi(y_{w}|x)}{\pi_{\text{ref}}(y_{w}|x)})Z^{\beta}(x) + \exp(\beta\log\frac{\pi(y_{l}|x)}{\pi_{\text{ref}}(y_{l}|x)})Z^{\beta}(x)} \\
&= \frac{\exp(\beta\log\frac{\pi(y_{w}|x)}{\pi_{\text{ref}}(y_{w}|x)})}{\exp(\beta\log\frac{\pi(y_{w}|x)}{\pi_{\text{ref}}(y_{w}|x)}) + \exp(\beta\log\frac{\pi(y_{l}|x)}{\pi_{\text{ref}}(y_{l}|x)})}
\end{align}
$$
</center>

Plugging into the preference model objective to get the gradient w.r.t. $$\pi$$:
<center>
$$
\begin{align}
&\mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}}\left[P(y_{l} \succ y_{w}|x)\left(\nabla_{\pi} \beta\log\frac{\pi(y_{w}|x)}{\pi_{\text{ref}}(y_{w}|x)} - \nabla_{\pi} \beta\log\frac{\pi(y_{l}|x)}{\pi_{\text{ref}}(y_{l}|x)}\right)\right] \\
&=\mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}}\left[\beta P(y_{l} \succ y_{w}|x)\left(\nabla_{\pi} \log\pi(y_{w}|x) - \nabla_{\pi} \log\pi_{\text{ref}}(y_{w}|x) - \nabla_{\pi} \log\pi(y_{l}|x) + \nabla_{\pi} \log\pi_{\text{ref}}(y_{l}|x)\right)\right] \\
&= \mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}}\left[\beta P(y_{l} \succ y_{w}|x)\left(\nabla_{\pi} \log\pi(y_{w}|x) - \nabla_{\pi} \log\pi(y_{l}|x)\right)\right]
\end{align}
$$
</center>

#### DPO reward gradient
Recall that DPO assumes the policy is the optimal RLHF policy, which as the form:
<center>
$$
\begin{align}
\pi(y|x) &= \frac{\pi_{\text{ref}}(y|x)\exp(\frac{1}{\beta}R(x, y))}{\sum_{\tilde{y}}\pi_{\text{ref}}(\tilde{y}|x)\exp(\frac{1}{\beta}R(x, \tilde{y}))} \\
&= \frac{\exp(\frac{1}{\beta}R(x, y) + \log \pi_{\text{ref}}(y|x))}{\sum_{\tilde{y}}\exp(\frac{1}{\beta}R(x, \tilde{y}) + \log \pi_{\text{ref}}(\tilde{y}|x))}
\end{align}
$$
</center>

Using the softmax gradient identity, the gradient of the policy log likelihood w.r.t. the reward function is:
<center>
$$
\begin{align}
\nabla_{R}\log\pi(y|x) &= \nabla_{R}\left(\frac{1}{\beta}R(x, y) + \log\pi_{\text{ref}}(y|x)\right) - \mathbb{E}_{\tilde{y} \sim \pi(\cdot|x)}\left[\nabla_{R}\left(\frac{1}{\beta}R(x, \tilde{y}) + \log\pi_{\text{ref}}(\tilde{y}|x)\right)\right] \\
&= \frac{1}{\beta}\nabla_{R}R(x, y) - \frac{1}{\beta}\mathbb{E}_{\tilde{y} \sim \pi(\cdot|x)}\left[\nabla_{R}R(x, y)\right]
\end{align}
$$
</center>

Plugging into the DPO policy log likelihood gradient, we get:
<center>
$$
\begin{align}
&\mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}}\left[\beta P(y_{l} \succ y_{w}|x)\left(\nabla_{R} \log\pi(y_{w}|x) - \nabla_{R} \log\pi(y_{l}|x)\right)\right] \\
&= \mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}}\left[\beta P(y_{l} \succ y_{w}|x)\left(\frac{1}{\beta}\nabla_{R}R(x, y_{w}) - \frac{1}{\beta}\mathbb{E}_{\tilde{y} \sim \pi(\cdot|x)}\left[\nabla_{R}R(x, y)\right] - \frac{1}{\beta}\nabla_{R}R(x, y_{l}) + \frac{1}{\beta}\mathbb{E}_{\tilde{y} \sim \pi(\cdot|x)}\left[\nabla_{R}R(x, y)\right]\right)\right] \\
&= \mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}}\left[P(y_{l} \succ y_{w}|x)\left(\nabla_{R}R(x, y_{w}) - \nabla_{R}R(x, y_{l})\right)\right] \\
\end{align}
$$
</center>
We see that the second term in the policy log likelihood gradient canceled between the positive and the negative example. This term also cancels for the regularized generative model. This is in some sense the "curse" of RLHF.