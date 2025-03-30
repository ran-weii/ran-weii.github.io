---
layout: post
title:  "Introducing CleanIL for Imitation and Inverse Reinforcement Learning"
date:   2025-03-28 00:00:00 -0000
---

We are excited to introduce [CleanIL](https://github.com/ran-weii/cleanil), a repository of high quality, single-file implementations of state-of-the-art (SOTA) Imitation Learning (IL) and Inverse Reinforcement Learning (IRL) algorithms. 

![](/assets/2025-03-28-introduce-cleanil/cleanil.png)

By now, there already exist many high quality Reinforcement Learning (RL) libraries, such as [Spinning Up](https://github.com/openai/spinningup), [StableBaseline](https://github.com/DLR-RM/stable-baselines3), [Dopamine](https://github.com/google/dopamine),  [TianShou](https://github.com/thu-ml/tianshou). While the majority of these libraries focus on model-free online RL, dedicated repos for offline and model-based RL like [CORL](https://github.com/corl-team/CORL), [OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit), and [MBRL-Lib](https://github.com/facebookresearch/mbrl-lib) can still be found. There also exists libraries like [TorchRL](https://github.com/pytorch/rl) which implements an interesting mix of algorithm variety, on top of providing easy to use RL-specific helper functions. To allow users to quickly get a birds-eye view of the algorithms without being buried in deep code bases (like [RLLib](https://github.com/ray-project/ray/tree/master/rllib)), many repos have adopted the single-file implementation philosophy (perhaps most notably [CleanRL](https://github.com/vwxyzjn/cleanrl)). 

However, looking over the IL and IRL side, the picture is quite bleak. The only well-maintained library we could find is [Imitation](https://github.com/HumanCompatibleAI/imitation). Although the algorithms provided in Imitation are seminal and have stood the test of time, they are very limited in variety and missed important settings like offline and model-based algorithms. This is not because no one cares about or works on IL and IRL. Rather, it is because **SOTA algorithms are scattered all over the place on the internet - A centralized repo of baselines is missing!**

CleanIL aims to fill this gap! To start with, we have implemented the following algorithms (see table below), ranging from online to offline, model-free to model-based, and explicit to implicit reward parameterizations. This is not to say that algorithms not included at the moment are not important, but rather we thought that the listed algorithms provide a good variety to get started and obtain good results. Please visit the [CleanIL Github page](https://github.com/ran-weii/cleanil) for benchmark results, Wandb logs, and pre-trained models.

| Paper | Algorithm | On/offline | Model free/based | Explicit/implicit reward |
|-------|-----------|------------| ---------------- | ------------------------ |
| Behavior Cloning | bc | Off | MF | - |
| [Implicit Behavioral Cloning](https://arxiv.org/abs/2109.00137) | ibc | Off | MF | Implicit |
| [IQ-Learn: Inverse soft-Q Learning for Imitation](https://arxiv.org/abs/2106.12142) | iqlearn | Off | MF | Implicit |
| [Dual RL: Unification and New Methods for Reinforcement and Imitation Learning](https://arxiv.org/abs/2302.08560) | recoil | Off | MF | Implicit |
| [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476) | gail | On | MF | Explicit |
| [When Demonstrations Meet Generative World Models: A Maximum Likelihood Framework for Offline Inverse Reinforcement Learning](https://arxiv.org/abs/2302.07457) | omlirl | Off | MB | Explicit |
| [A Bayesian Approach to Robust Inverse Reinforcement Learning](https://arxiv.org/abs/2309.08571) | rmirl | Off | MB | Explicit |

Implementation-wise, we adopt a near single-file strategy and use TorchRL for basic RL helper functions (e.g., replay buffer). We have tried to optimize the implementations in terms of performance and speed and we will continue to do so. Some of the best practices we have adopted include gradient penalty, terminal state handling, and `torch.compile`. There are a number of other implementation tricks and observations that we noticed along the way, which we detail in this [blog post](https://ran-weii.github.io/2025/03/28/cleanil-implementation-tricks.html).

We welcome all users and readers to try out and star the repo, raise issues, and contribute your algorithms or use cases!

To close up, we discuss some practical use cases of IL and IRL and some future directions for CleanIL.

## IL and IRL in the wild
The most obvious and also most popular application for IL and IRL has been training robots and autonomous vehicles in physical and simulation environments. However, the applications of especially IRL are far beyond this. Below we list a number of interesting applications of IRL in a wide range of different application domains. This is by no means exhaustive, and please let us know about your innovative IL and IRL use cases.

| Paper | Domain | Use case |
| ----- | ------ | -------- |
| [Driving in Real Life with Inverse Reinforcement Learning](https://arxiv.org/abs/2206.03004) | Autonomous driving | A reward function was learned from expert human drivers to score trajectories generated by a planner in a context sensitive way. |
| [World scale inverse reinforcement learning in Google Maps](https://research.google/blog/world-scale-inverse-reinforcement-learning-in-google-maps/) | Maps, routing | IRL was used to tune route recommendation objectives from actual routes taken by the users. |
| [Inverse Reinforcement Learning for Team Sports: Valuing Actions and Players](https://www.ijcai.org/proceedings/2020/464) | Sports | IRL was used to score NHL hockey players such that non-scoring players' impact are better evaluated. |
| [Learning strategies in table tennis using inverse reinforcement learning](https://link.springer.com/article/10.1007/s00422-014-0599-1) | Sports | Reward functions were used to characterize player strategies of different expertise. |
| [Extracting Reward Functions from Diffusion Models](https://arxiv.org/abs/2306.01804) | Image generation | Reward functions were extracted from large diffusion models to fine tune small diffusion models to mitigate harmful image generation. |
| [Insights from the Inverse: Reconstructing LLM Training Goals Through Inverse RL](https://arxiv.org/abs/2410.12491) | LLM post-training, interpretability | Reward functions were extracted from post-trained LLM to understand the impact of toxicity reduction fine-tuning. |
| [How Private Is Your RL Policy? An Inverse RL Based Analysis Framework](https://arxiv.org/abs/2112.05495) | Privacy | Used IRL to assess privacy preserving RL agents' ability to protect private reward functions. |
| [M3Rec- An Offline Meta-level Model-based Reinforcement Learning Approach for Cold-Start Recommendation](https://arxiv.org/abs/2012.02476) | Recommendation system | A meta user model was trained using IRL for fast recommendation policy adaptation. |
| [Rationally Inattentive Inverse Reinforcement Learning Explains YouTube Commenting Behavior](https://arxiv.org/abs/1910.11703) | Recommendation system | IRL were used to estimate user commenting preference over video features and the cost of paying attention to video features. |

## The future of CleanIL
* **Support for more problem domains**: currently we have only implemented algorithms for state-based continuous state and action domains in MuJoCo as standard benchmarks. We plan to work on discrete and hybrid state and action domains and potentially other observation modalities depending on the problems and applications.
* **Implementation efficiency**: one way to make IL and IRL more practical is to make the algorithms easier and faster to run. We have implemented `torch.compile` to accelerate training. However, its optimal usage and other memory management techniques were not yet explored. Furthermore, highly parallelized RL (such as [JaxIRL](https://github.com/FLAIROx/jaxirl)) may further boost training speed. We plan to explore these options.
* **Partial observability and sequence models**: IL and IRL with fully observable MDP assumption cannot properly explain agent behavior in partially observable environments or with history dependent behavior. We plan to support sequence models and add partially observable domains as benchmarks in the future.
* **Heterogeneous behavior and rewards**: most IL and IRL works assume the dataset is collected from a single agent. This is often not true. There are also many applications where the goal is to extract rewards from different agents with both shared and different components. We aim to support joint reward/policy learning and behavior categorization in the future.
* **Reward learning from multiple feedback types**: demonstrations and expert trajectories are just one type of feedback through which humans convey latent preferences. There are many other potential feedback types, such as comparisons, corrections, language, etc. [Prior work](https://arxiv.org/abs/2002.04833) has found that different feedback types have different advantages, and combining multiple feedback types can be even more beneficial (see [[1]](https://arxiv.org/abs/2406.06874), [[2]](https://arxiv.org/abs/2006.14091)). We plan to implement examples of learning from multiple feedback types in the future.
* **Uncertainty quantification**: most IL and IRL algorithms extract a single reward function or policy from data, ignoring the fact that the data may not be sufficient to specify the reward function. This can be problematic when the reward function is transferred to a different environment. We plan to address this by supporting uncertainty quantification techniques.
* **Reward evaluations**: the quality of rewards should be evaluated not only by the performance of the jointly learned policy but also the ability to train policies from scratch both in the same and different domains. We plan to support reward transfer and policy recovery capabilities in the future as well as other reward evaluation techniques.
* **Multi-agent problems**: while multi-agent is an important problem domain. It is significantly more complex in both the algorithms and training environments. We will consider supporting multi-agent IL and IRL algorithms in the future.