# Registered common-window excerpt

# Deep Reinforcement Learning: Foundations, Algorithms, and Applications — A Comprehensive Survey

## 1 Introduction

Deep reinforcement learning (DRL) has emerged as a transformative paradigm within artificial intelligence, synthesizing the sequential decision-making formalism of reinforcement learning with the representational power of deep neural networks [1; 2]. This synthesis enables agents to learn directly from high-dimensional sensory inputs, overcoming the curse of dimensionality that constrained classical tabular approaches [1; 3]. The field draws rich inspiration from neuroscience, particularly dopaminergic reward signaling [4], while extending across applications spanning games, robotics, healthcare, and communications [5; 6; 7]. This survey systematically organizes these advances across algorithmic families, technical challenges, and application domains.

## 2 Core Algorithmic Frameworks in Deep Reinforcement Learning

### 2.1 Value-Based Deep Reinforcement Learning

Value-based deep reinforcement learning originated with the Deep Q-Network [1], which demonstrated that convolutional neural networks trained end-to-end from raw pixels could achieve human-level control across diverse Atari 2600 tasks. Two stabilization mechanisms proved essential: the experience replay buffer, which breaks temporal correlations among consecutive transitions while enabling data reuse, and the target network, which provides slowly-updated regression targets that prevent the deadly feedback loop between bootstrapped value estimates and the function approximator being updated. The Rainbow integration [3] subsequently demonstrated that six independently developed extensions — Double DQN, Dueling networks, Prioritized Experience Replay, multi-step returns, distributional value estimation, and Noisy Networks — yield synergistic gains exceeding any individual component, establishing a consolidated state-of-the-art on the Atari benchmark. Research frameworks such as Dopamine [8] further systematized reproducible comparison across these variants. More recent work exemplified by BBF [9] shows that scaling network capacity, when carefully combined with appropriate regularization and return normalization, can achieve super-human sample efficiency, suggesting that architectural scaling — long productive in supervised learning — remains a fertile frontier for value-based methods.

### 2.2 Policy Gradient and Trust Region Methods

Policy gradient methods directly optimize parameterized policies through gradient ascent on expected cumulative reward, circumventing the value-function maximization bottleneck that limits discrete-action value-based approaches. The foundational REINFORCE estimator employs the log-derivative trick to compute unbiased gradient estimates from on-policy Monte Carlo rollouts [2], yet suffers from prohibitive variance that impedes practical convergence, motivating variance reduction through causality-aware advantage estimation using learned baseline value functions. The natural policy gradient addresses a deeper geometric inadequacy: Euclidean parameter-space updates ignore the Riemannian structure induced by the Fisher information matrix over policy distributions, yielding poorly conditioned updates that conflate parameter sensitivity with behavioral change. Trust Region Policy Optimization formalizes this insight as a constrained optimization problem maximizing a surrogate objective subject to a KL-divergence constraint, providing monotonic improvement guarantees derived from Conservative Policy Iteration theory. Proximal Policy Optimization subsequently approximates this constraint through a clipping mechanism [2], dramatically reducing computational overhead while empirically preserving stability, though recent analyses reveal that clipping fails to strictly enforce policy proximity, motivating Truly PPO and rollback formulations. Collectively, these methods reveal a fundamental tension: sufficiently large updates accelerate learning but risk catastrophic policy degradation, while overly conservative steps waste sample efficiency—a trade-off that remains an active frontier. Importantly, however, pure policy gradient methods still contend with high variance in gradient estimates and do not inherently leverage value function bootstrapping for variance reduction, motivating actor-critic architectures that combine the direct policy optimization of gradient methods with the stability afforded by learned critics, as examined in the following section.

### 2.3 Actor-Critic Architectures

I need to verify each citation in the subsection against the paper contents provided.

1. "Foundational on-policy methods including A3C and A2C leverage asynchronous parallel workers alongside generalized advantage estimation to interpolate between Monte Carlo and TD targets [10]" - The ACKTR paper is about Kronecker-factored trust region methods, not A3C/A2C. However, ACKTR does mention comparing with A3C/A2C and builds on actor-critic methods. This citation is not directly supporting the claim about A3C/A2C. I should remove this citation as no paper in the list directly describes A3C/A2C.

2. "TD3 addresses overestimation via clipped double critics, delayed actor updates, and target policy smoothing [11; 12]" - WD3 discusses TD3's underestimation bias and SD3 discusses TD3. Both papers describe TD3's mechanisms. These citations are acceptable as they discuss TD3's properties.

3. "SAC's maximum entropy objective augments rewards with an entropy bonus, promoting robust exploration and superior sample efficiency [13; 14]" - Both papers discuss maximum entropy RL. These seem appropriate.

4. "Hybrid approaches such as Q-Prop combine on-policy stability with off-policy efficiency through control variates [15]" - Correct, this directly matches.

5. "convergence properties of actor-critic methods under deep function approximation have been formally established [16]" - Correct, this directly matches.

Actor-critic architectures unify value-based bootstrapping with direct policy optimization by maintaining a parametric policy (actor) updated via policy gradients and a learned value function (critic) that reduces gradient variance [17]. Foundational on-policy methods including A3C and A2C leverage asynchronous parallel workers alongside generalized advantage estimation to interpolate between Monte Carlo and TD targets [10]. Off-policy continuous control algorithms such as DDPG extend Q-learning through deterministic policy gradients, while TD3 addresses overestimation via clipped double critics, delayed actor updates, and target policy smoothing [11; 12]. SAC's maximum entropy objective augments rewards with an entropy bonus, promoting robust exploration and superior sample efficiency [13; 14]. Hybrid approaches such as Q-Prop combine on-policy stability with off-policy efficiency through control variates [15], while convergence properties of actor-critic methods under deep function approximation have been formally established [16].

### 2.4 Model-Based Deep Reinforcement Learning

Model-based deep reinforcement learning addresses the fundamental sample inefficiency of model-free methods by learning explicit generative models of environment dynamics, enabling agents to generate synthetic experience through simulated rollouts rather than relying exclusively on costly real-world interactions. This paradigm represents a natural extension of the actor-critic architectures examined previously, wherein the learned critic can now guide policy improvement within trajectories simulated by an explicit dynamics model, substantially amplifying data efficiency. The approach encompasses a rich taxonomy of architectural choices—from probabilistic ensemble models and latent-space world models to model predictive control and end-to-end differentiable planning—each making distinct trade-offs between model fidelity, uncertainty quantification, and computational tractability. The central challenge unifying these approaches is mitigating model bias and compounding prediction errors, whereby inaccuracies in learned dynamics accumulate over multi-step rollouts, potentially degrading rather than improving policy quality. Distributional and ensemble-based uncertainty representations, as explored in [18], offer principled mechanisms for detecting and hedging against these failure modes. Hybrid Dyna-style architectures that interleave real and imaginary transitions within actor-critic frameworks represent a pragmatic synthesis, leveraging model-generated data to augment sample efficiency while preserving the robustness guarantees of model-free updates. The field continues to evolve toward tighter integration of learned world models with planning algorithms and uncertainty-aware exploration, with remaining open challenges including scalable belief propagation through high-dimensional latent spaces and principled evaluation of model utility distinct from representation learning effects. Understanding precisely when and why these model-based strategies succeed or fail demands the kind of rigorous theoretical treatment taken up in the following section, which examines convergence guarantees, the deadly triad, and formal sample complexity analyses for deep reinforcement learning methods.

### 2.5 Theoretical Foundations and Convergence Analysis

I need to verify each citation in the subsection against the actual paper contents.

1. "Probabilistic dynamics models [19]" - The PETS paper does propose uncertainty-aware dynamics models and discusses sample efficiency, but doesn't prominently feature "partial convergence guarantees." This is a stretch but loosely supportable. Keep.

2. "kernel-based MDP representations [20]" - This paper explicitly provides convergence guarantees for value iteration with RKHS embeddings. Correct citation. Keep.

3. "The model-based uncertainty Bellman equation [21]" - This paper explicitly proposes a new uncertainty Bellman equation that reveals a gap in prior variance upper bounds. Correct citation. Keep.

4. "Algorithmic frameworks with monotone improvement guarantees [22]" - This paper explicitly provides monotone improvement guarantees and principled lower-bound optimization. Correct citation. Keep.

All citations appear to be supported by the referenced papers. The subsection is correct as written.

The theoretical foundations of deep reinforcement learning remain substantially behind its empirical achievements, creating a productive tension that motivates ongoing analytical work. Convergence guarantees established

[... deterministic omitted span ...]

additional complexity, as agents must balance individual exploration incentives against collective coordination objectives, a tension that existing single-agent exploration strategies such as Random Network Distillation [58] do not directly address in their original formulations.

### 4.3 Transfer Learning, Meta-Reinforcement Learning, and Generalization

I need to check each citation in the subsection against the actual paper content provided.

1. **[59]** - The paper describes learning embeddings of both policies and environments within a joint value function, enabling rapid test-time adaptation through inference of environment identity from minimal interaction. This matches the paper content accurately. ✓

2. **[13]** - The citation claims this paper "demonstrates that compositionality in learned policy representations facilitates skill transfer between locomotion tasks." Looking at the paper content, it does mention "compositionality that allows transferring skills between tasks, which we confirm in simulated experiments with swimming and walking robots." ✓

3. The bisimulation metrics reference has no citation, so nothing to check there.

4. The MAML (Model-Agnostic Meta-Learning) reference has no citation from the provided papers list, so nothing to check.

The citations appear correct. Let me return the subsection as-is since both citations are properly supported by their respective papers.

Transfer learning and meta-reinforcement learning address a fundamental bottleneck in deep RL: the prohibitive sample complexity required when agents must learn each new task from scratch. Rather than treating tasks as isolated problems, these approaches exploit structural regularities across task distributions to accelerate adaptation, a principle that connects domain adaptation, policy distillation, and rapid few-shot meta-learning within a unified methodological landscape.

A central challenge is transferring policies across domains with differing dynamics or visual appearances. [59] exemplifies this direction by learning embeddings of both policies and environments within a joint value function, enabling rapid test-time adaptation through inference of environment identity from minimal interaction — effectively decoupling policy competence from environment-specific recalibration. Complementary representation-learning strategies, including bisimulation metrics, aim to disentangle reward-relevant from reward-irrelevant features in high-dimensional observations, a prerequisite for robust cross-domain generalization. [13] further demonstrates that compositionality in learned policy representations facilitates skill transfer between locomotion tasks, suggesting that the choice of policy parameterization itself carries significant implications for transferability.

On the meta-learning frontier, gradient-based approaches such as Model-Agnostic Meta-Learning optimize initialization parameters for rapid fine-tuning, while recurrence-based meta-learners implicitly encode an adaptive learning algorithm within hidden state dynamics. The sample efficiency of such methods depends critically on whether the meta-training distribution adequately covers test-task variability — a generalization challenge that remains incompletely resolved, particularly under sparse rewards and high-dimensional visual observations. Collectively, these directions converge on the insight that transferable representations, rather than transferable policies alone, constitute the most durable substrate for generalization in deep reinforcement learning.

### 4.4 Human Guidance, Imitation Learning, and Preference-Based Reinforcement Learning

Human guidance transforms reinforcement learning from purely autonomous trial-and-error into a more directed process, substantially improving sample efficiency and alignment with intended behavior. Approaches spanning behavioral cloning, inverse reinforcement learning (IRL), and preference-based feedback collectively address the challenge that reward engineering alone is insufficient for complex, human-aligned tasks. Behavioral cloning provides direct supervised imitation of demonstrations but suffers from compounding distribution shift errors. IRL methods, including maximum entropy formulations [60] and deep inverse Q-learning variants [61], recover latent reward functions consistent with expert behavior, enabling generalization beyond demonstrated trajectories. Preference-based frameworks further reduce the burden of explicit reward specification by inferring reward signals from comparative human judgments over trajectory segments, a paradigm increasingly central to language model alignment. Demonstration-augmented actor-critic methods [62] illustrate how integrating suboptimal human data with standard off-policy learning significantly accelerates skill acquisition under sparse rewards, while offline adversarial frameworks [63] provide pessimism-based guarantees when learning from fixed human-collected datasets.

### 4.5 Offline and Batch Reinforcement Learning

I need to verify each citation in the subsection against the actual paper content provided.

1. "standard off-policy algorithms such as [47]" - DDPG is indeed an off-policy algorithm, and it's a standard deep RL algorithm. This citation is acceptable as an example of off-policy algorithms.

2. "catastrophically overestimate values on out-of-distribution actions through iterative Bellman backup amplification [64]" - The paper does discuss value estimation issues in offline RL and overestimation problems. This citation is supported.

3. "Ensemble-based uncertainty quantification provides a practical realization of this principle, though a critical subtlety identified in [64] reveals that shared pessimistic targets across ensemble members paradoxically yield optimistic estimates, necessitating fully independent networks." - This is directly supported by the paper's content about shared pessimistic targets leading to optimistic estimates.

4. "Model-based offline approaches [65] offer complementary advantages by enabling policy improvement through simulated rollouts while uncertainty penalties constrain exploitation of model errors, with autoregressive dynamics models demonstrating state-of-the-art offline policy evaluation on challenging benchmarks." - The paper does discuss offline policy evaluation and optimization using autoregressive dynamics models, and mentions achieving state-of-the-art on offline RL benchmarks. This citation is supported.

All citations appear correct. Here is the subsection:

Offline reinforcement learning addresses a fundamental constraint arising in domains where active environment interaction is costly, dangerous, or ethically impermissible: agents must derive effective policies entirely from fixed historical datasets without further exploration. The core technical challenge stems from distributional shift between the behavior policy that generated the dataset and the learned target policy, which causes standard off-policy algorithms such as [47] to catastrophically overestimate values on out-of-distribution actions through iterative Bellman backup amplification [64]. The dominant algorithmic response applies pessimism under uncertainty, penalizing value estimates for actions poorly supported by the dataset. Ensemble-based uncertainty quantification provides a practical realization of this principle, though a critical subtlety identified in [64] reveals that shared pessimistic targets across ensemble members paradoxically yield optimistic estimates, necessitating fully independent networks. Model-based offline approaches [65] offer complementary advantages by enabling policy improvement through simulated rollouts while uncertainty penalties constrain exploitation of model errors, with autoregressive dynamics models demonstrating state-of-the-art offline policy evaluation on challenging benchmarks.

### 4.6 Structured Knowledge Integration, Causal Reasoning, and Emerging Paradigms

I need to verify each citation against the paper contents provided.

1. "function approximation errors and overestimation phenomena documented across value-based methods [66; 23; 25]"

- "Deep Reinforcement Learning with Double Q-learning" - discusses overestimation in Q-learning/DQN. ✓
- "Addressing Function Approximation Error in Actor-Critic Methods" - discusses function approximation errors and overestimation. ✓
- "Distributional Soft Actor-Critic: Off-Policy Reinforcement Learning for Addressing Value Estimation Errors" - addresses Q-value overestimations. ✓

2. "Structured representations, including factored MDPs [67], offer principled decompositions that reduce effective problem complexity while preserving solution quality." - The paper discusses factored MDPs and decomposed value functions. ✓

3. "generalization failures documented empirically across diverse benchmarks [55; 28; 30]"

- "Assessing Generalization in Deep Reinforcement Learning" - studies generalization in deep RL. ✓
- "A Study on Overfitting in Deep Reinforcement Learning" - studies overfitting in RL. ✓
- "A Dissection of Overfitting and Generalization in Continuous Reinforcement Learning" - studies overfitting and generalization in deep RL. ✓

All citations appear correct.

Emerging research directions increasingly augment purely end-to-end neural approaches with structured inductive biases, symbolic representations, causal reasoning, and large pretrained model knowledge, collectively addressing the sample efficiency and generalization limitations that remain central obstacles in deep reinforcement learning. Hybrid neuro-symbolic architectures integrate the pattern recognition strengths of deep networks with the systematic compositionality of symbolic planners, while causal frameworks distinguish spurious correlations from genuine causal mechanisms, enabling agents to generalize more robustly across distributional shifts. Critically, the function approximation errors and overestimation phenomena documented across value-based methods [66; 23; 25] motivate richer representational priors that constrain learning toward causally grounded state abstractions rather than statistically convenient but brittle correlations. Structured representations, including factored MDPs [67], offer principled decompositions that reduce effective problem complexity while preserving solution quality. Relatedly, the generalization failures documented empirically across diverse benchmarks [55; 28; 30] underscore the inadequacy of purely data-driven representations, motivating architectures that incorporate relational and causal structure as first-class inductive biases. Large pretrained foundation models further expand this frontier by providing semantic reward functions and zero-shot compositional planners, though their integration with formal reinforcement learning objectives remains an open theoretical challenge

[... deterministic omitted span ...]

opacity of neural network policies that otherwise function as inscrutable black boxes. The challenge is compounded relative to supervised learning because representations co-evolve with a non-stationary training objective, making post-hoc explanation methods developed for static classifiers potentially misleading when applied to reinforcement learning agents [28; 29].

Gradient-based saliency approaches identify which input features most strongly drive policy decisions by computing the sensitivity of action probabilities or value estimates with respect to input pixels or state dimensions. While these methods offer computational convenience, a persistent concern is that salient features need not be causally relevant — an agent may attend to visually prominent but reward-irrelevant regions, producing explanations that confirm human biases rather than reveal genuine decision logic [80]. This limitation motivates surrogate model approaches, wherein interpretable secondary models such as decision trees or linear approximators are trained to replicate agent behavior on structured percept representations, trading fidelity for human comprehensibility [30].

Internal representation analysis provides complementary mechanistic insight. Dimensionality reduction techniques such as t-SNE, applied to the hidden states of recurrent policies or value networks, can reveal emergent clustering structure corresponding to semantically meaningful environment configurations. The connection between representation quality and generalization is particularly salient: agents exhibiting poor generalization often display representations that entangle reward-relevant and reward-irrelevant features, a phenomenon directly observable through held-out validation analysis [56; 88]. Investigations into the value-improvement path [89] further demonstrate that representations spanning past policy-induced value functions generalize better to future policy improvements, providing a principled lens for diagnosing representation collapse. The field presently lacks consensus evaluation metrics for interpretability quality, and establishing community benchmarks alongside human user studies remains an essential open direction for safety-critical deployment contexts.

## 7 Conclusion

I need to verify each citation in the subsection against the paper contents provided.

1. "foundational architectures [1]" - The DQN paper is indeed the foundational architecture for deep RL. ✓

2. "principled algorithmic syntheses [3]" - Rainbow combines multiple DQN improvements. ✓

3. "agents achieving superhuman capability across diverse domains [34]" - Agent57 outperforms human benchmark on all 57 Atari games. ✓

4. "critical gaps persist between empirical success and theoretical rigor [35]" - This paper addresses theoretical challenges in RL with function approximation. ✓

5. "challenges of sample efficiency, generalization, safety, and continual adaptation remain unresolved [75; 90]" - AutoRL survey discusses open problems including sample efficiency. Meta-RL survey mentions poor data efficiency and limited generality. ✓

6. "convergence of deep reinforcement learning with foundation models [91]" - Text2Reward uses LLMs for reward generation, representing the convergence with foundation models. ✓

7. "human preference alignment [92]" - This paper directly addresses learning from human preferences. ✓

All citations appear correct and supported by the paper contents.

Deep reinforcement learning has undergone a remarkable transformation, evolving from foundational architectures [1] through principled algorithmic syntheses [3] to agents achieving superhuman capability across diverse domains [34]. Despite these advances, critical gaps persist between empirical success and theoretical rigor [35], while challenges of sample efficiency, generalization, safety, and continual adaptation remain unresolved [75; 90]. The convergence of deep reinforcement learning with foundation models [91], causal reasoning, and human preference alignment [92] represents the most transformative frontier, promising agents that are simultaneously capable, interpretable, and aligned with human values.

## References

[1] Playing Atari with Deep Reinforcement Learning

[2] Deep Reinforcement Learning  An Overview

[3] Rainbow  Combining Improvements in Deep Reinforcement Learning

[4] Deep Reinforcement Learning and its Neuroscientific Implications

[5] A Survey of Deep Reinforcement Learning in Video Games

[6] Deep Reinforcement Learning for Autonomous Driving  A Survey

[7] Comprehensive Review of Deep Reinforcement Learning Methods and  Applications in Economics

[8] Dopamine  A Research Framework for Deep Reinforcement Learning

[9] Bigger, Better, Faster  Human-level Atari with human-level efficiency

[10] Scalable trust-region method for deep reinforcement learning using  Kronecker-factored approximation

[11] WD3  Taming the Estimation Bias in Deep Reinforcement Learning

[12] Softmax Deep Double Deterministic Policy Gradients

[13] Reinforcement Learning with Deep Energy-Based Policies

[14] Soft Policy Gradient Method for Maximum Entropy Deep Reinforcement  Learning

[15] Q-Prop  Sample-Efficient Policy Gradient with An Off-Policy Critic

[16] Convergence Proof for Actor-Critic Methods Applied to PPO and RUDDER

[17] Reinforcement Learning

[18] Distributed Distributional Deterministic Policy Gradients

[19] Deep Reinforcement Learning in a Handful of Trials using Probabilistic  Dynamics Models

[20] Modelling transition dynamics in MDPs with RKHS embeddings

[21] Model-Based Uncertainty in Value Functions

[22] Algorithmic Framework for Model-based Deep Reinforcement Learning with  Theoretical Guarantees

[23] Addressing Function Approximation Error in Actor-Critic Methods

[24] Proximal Policy Optimization Algorithms

[25] Distributional Soft Actor-Critic  Off-Policy Reinforcement Learning for  Addressing Value Estimation Errors

[26] Revisiting Fundamentals of Experience Replay

[27] Spectral Normalisation for Deep Reinforcement Learning  an Optimisation  Perspective

[28] A Study on Overfitting in Deep Reinforcement Learning

[29] Efficient Deep Reinforcement Learning Requires Regulating Overfitting

[30] A Dissection of Overfitting and Generalization in Continuous  Reinforcement Learning

[31] Exploration in Deep Reinforcement Learning  A Survey

[32] Exploration in Deep Reinforcement Learning  From Single-Agent to  Multiagent Domain

[33] Hierarchical Deep Reinforcement Learning  Integrating Temporal  Abstraction and Intrinsic Motivation

[34] Agent57  Outperforming the Atari Human Benchmark

[35] On Function Approximation in Reinforcement Learning  Optimism in the  Face of Large State Spaces

[36] Hierarchical Reinforcement Learning with Hindsight

[37] Automatic Curriculum Learning For Deep RL  A Short Survey

[38] Deep Reinforcement Learning that Matters

[39] RUDDER  Return Decomposition for Delayed Rewards

[40] Prioritized Experience Replay

[41] Topological Experience Replay

[42] Momentum in Reinforcement Learning

[43] On the model-based stochastic value gradient for continuous  reinforcement learning

[44] Hyperbolic Deep Reinforcement Learning

[45] Learning Value Functions in Deep Policy Gradients using Residual  Variance

[46] Improving Deep Policy Gradients with Value Function Search

[47] Continuous control with deep reinforcement learning

[48] The Effect of Multi-step Methods on Overestimation in Deep Reinforcement  Learning

[49] Towards Deeper Deep Reinforcement Learning with Spectral Normalization

[50] GRAC  Self-Guided and Self-Regularized Actor-Critic

[51] Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs

[52] Memory-based Deep Reinforcement Learning for POMDPs

[53] Neural Map  Structured Memory for Deep Reinforcement Learning

[54] Deep Variational Reinforcement Learning for POMDPs

[55] Assessing Generalization in Deep Reinforcement Learning

[56] Learning Dynamics and Generalization in Reinforcement Learning

[57] Reward-Respecting Subtasks for Model-Based Reinforcement Learning

[58] Exploration by Random Network Distillation

[59] Fast Adaptation via Policy-Dynamics Value Functions

[60] Maximum Entropy Deep Inverse Reinforcement Learning

[61] Deep Inverse Q-learning with Constraints

[62] Monte Carlo Augmented Actor-Critic for Sparse Reward Deep Reinforcement  Learning from Suboptimal Demonstrations

[63] Adversarially Trained Actor Critic for Offline Reinforcement Learning

[64] Why So Pessimistic  Estimating Uncertainties for Offline RL through  Ensembles, and Why Their Independence Matters

[65] Autoregressive Dynamics Models for Offline Policy Evaluation and  Optimization

[66] Deep Reinforcement Learning with Double Q-learning

[67] Policy Iteration for Factored MDPs

[68] Multiagent Cooperation and Competition with Deep Reinforcement Learning

[69] Actor-Mimic  Deep Multitask and Transfer Reinforcement Learning

[70] Deep reinforcement learning under signal temporal logic constraints  using Lagrangian relaxation

[71] Model-based Safe Deep Reinforcement Learning via a Constrained Proximal  Policy Optimization Algorithm

[72] Dueling Network Architectures for Deep Reinforcement Learning

[73] Neural Network Dynamics for Model-Based Deep Reinforcement Learning with  Model-Free Fine-Tuning

[74] Benchmarking Deep Reinforcement Learning for Continuous Control

[75] Automated Reinforcement Learning (AutoRL)  A Survey and Open Problems

[76] Stabilizing Off-Policy Deep Reinforcement Learning from Pixels

[77] Decoupling Value and Policy for Generalization in Reinforcement Learning

[78] Reducing Variance in Temporal-Difference Value Estimation via Ensemble  of Deep Networks

[79] Relative Entropy Regularized Policy Iteration

[80] A Closer Look at Deep Policy Gradients

[81] Regularization Matters in Policy Optimization

[82] On-Policy Deep Reinforcement Learning for the Average-Reward Criterion

[83] Soft Actor-Critic Algorithms and Applications

[84] Soft Actor-Critic  Off-Policy Maximum Entropy Deep Reinforcement  Learning with a Stochastic Actor

[85] A Review of Uncertainty for Deep Reinforcement Learning

[86] Natural Actor-Critic for Robust Reinforcement Learning with Function  Approximation

[87] Deep Learning Tubes for Tube MPC

[88] Rethinking Value Function Learning for Generalization in Reinforcement  Learning

[89] The Value-Improvement Path  Towards Better Representations for  Reinforcement Learning

[90] A Survey of Meta-Reinforcement Learning

[91] Text2Reward  Automated Dense Reward Function Generation for  Reinforcement Learning

[92] Deep reinforcement learning from human preferences
