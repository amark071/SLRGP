# Registered common-window excerpt

# Deep Reinforcement Learning: Foundations, Advances, and Applications — A Comprehensive Survey

## 1 Foundations of Deep Reinforcement Learning

### 1.1 Markov Decision Processes and Problem Formulation

## 1.1 Markov Decision Processes and Problem Formulation

The Markov Decision Process (MDP) provides the fundamental mathematical framework underlying virtually all reinforcement learning algorithms. Formally, an MDP is defined as a tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma)$, where $\mathcal{S}$ denotes the state space, $\mathcal{A}$ the action space, $\mathcal{T}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [1]$ the transition probability function, $\mathcal{R}: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ the reward function, and $\gamma \in [2]. At each timestep $t$, an agent observes state $s_t \in \mathcal{S}$, selects action $a_t \in \mathcal{A}$ according to its policy $\pi$, transitions to a new state $s_{t+1}$ sampled from $\mathcal{T}(\cdot | s_t, a_t)$, and receives scalar reward $r_t = \mathcal{R}(s_t, a_t)$. The agent's objective is to find a policy $\pi^*$ that maximizes the expected discounted cumulative return $\mathbb{E}_\pi[3]$. This objective, along with the recursive structure of the MDP, lays the groundwork for the Bellman equations discussed in the following section.

The **Markov property** is the cornerstone of this framework, stipulating that the future state distribution depends only on the current state and action, and not on the history of preceding states and actions: $\mathcal{T}(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = \mathcal{T}(s_{t+1} | s_t, a_t)$. This conditional independence assumption allows the sequential decision-making problem to be decomposed recursively via dynamic programming, enabling tractable solution methods [4]. As will be seen in Section 1.2, this recursive decomposition is precisely what the Bellman equations formalize. However, the Markov property also represents a significant modeling assumption, as many real-world environments exhibit non-Markovian characteristics where historical context meaningfully influences future dynamics [5].

### Reward Functions and Structures

The reward function encodes the task objective and is central to shaping agent behavior. Rewards may be dense, providing frequent feedback at every step, or sparse, delivering signal only upon rare task completion events [4]. The design of reward functions is a non-trivial challenge; poorly specified rewards can lead to unintended agent behaviors. Beyond scalar Markov rewards, research has examined the expressivity of multidimensional reward representations [6] and non-Markovian reward specifications where desired behaviors depend on the full trajectory of states and actions [7]. The structural properties of reward functions also carry important theoretical implications, informing concepts such as potential-based reward shaping and the sample complexity of policy evaluation [8]. These structural considerations directly influence the form and tractability of the Bellman equations that govern value function computation.

### State and Action Spaces

MDPs may be defined over discrete or continuous state and action spaces. In classical tabular settings, exact dynamic programming methods such as value iteration and policy iteration — both grounded in the Bellman optimality principle — are applicable when the state space is small and finite. However, as the state space grows — particularly in multi-component engineering systems where states scale exponentially with the number of components — exact methods become computationally intractable, motivating the use of function approximation [9]. In continuous state spaces, compact and efficient representations of the value function become essential for tractable solutions [10]. Deep neural networks address this scalability challenge by serving as powerful function approximators over high-dimensional inputs, forming the bridge between classical MDP theory and modern deep reinforcement learning.

### Partially Observable MDPs

A fundamental limitation of the standard MDP framework is the assumption of full state observability. In most real-world applications, an agent only receives partial, noisy, or incomplete observations of the true underlying state. The **Partially Observable Markov Decision Process (POMDP)** generalizes the MDP by introducing an observation space $\mathcal{O}$ and an observation emission function $\mathcal{Z}: \mathcal{S} \times \mathcal{A} \times \mathcal{O} \rightarrow [1]$, which specifies the probability of observing $o_t \in \mathcal{O}$ given the current state and action [11]. Because the true state is hidden, optimal decision-making in POMDPs requires maintaining a **belief state** — a probability distribution over possible underlying states given the history of observations and actions — which serves as a sufficient statistic for optimal planning [12].

POMDPs are notoriously difficult to solve due to the continuous and high-dimensional nature of the belief space, rendering exact computation intractable in general [13]. Planning algorithms for POMDPs typically rely on approximations, such as discretization of the belief space or point-based methods. Recent work has established quasipolynomial-time algorithms for planning in observable POMDPs under specific assumptions on filter stability [14], while spectral decomposition methods provide principled approaches for reinforcement learning in POMDPs with provable regret bounds [15]. The POMDP-lite subclass, in which hidden variables are constant or change only deterministically, admits more tractable solutions and is suitable for a broad range of robotic tasks [16].

In deep reinforcement learning, partial observability is often addressed by equipping agents with memory mechanisms such as recurrent neural networks or LSTM-based architectures that maintain a compressed history of past observations as a proxy for the belief state [17]. Representation learning approaches further aim to learn abstract state representations that approximately satisfy the Markov property from high-dimensional, partially observable inputs [18; 19].

### Semi-MDPs and Temporal Abstraction

The **Semi-Markov Decision Process (Semi-MDP)** extends the standard MDP by allowing actions to persist for variable durations rather than a fixed single timestep. In a Semi-MDP, an agent selects a temporally extended action (or **option**) that executes for a stochastic duration, after which the environment is observed and the next decision is made. This framework provides the theoretical basis for **hierarchical reinforcement learning** and temporal abstraction, enabling agents to reason and plan at multiple levels of temporal granularity [20]. The options framework, grounded in Semi-MDP theory, allows agents to compose primitive actions into macro-actions and reason over extended time horizons, significantly reducing the effective planning depth and improving computational efficiency [21; 22].

### Non-Markovian Extensions

Beyond POMDPs and Semi-MDPs, researchers have explored a variety of extensions that relax or modify the standard MDP assumptions to better capture real-world complexity. **Regular Decision Processes (RDPs)** and **omega-regular decision processes (ODPs)** extend the MDP framework to handle non-Markovian transition and reward functions that depend on regular or omega-regular properties of the past or future trajectory, respectively [23; 24]. **MDPs with unawareness (MDPUs)** address settings where the decision maker is unaware of certain available actions [25], while **parametric MDPs (pMDPs)** incorporate unknown transition parameters to account for stochastic environmental uncertainties [26]. **Dynamic Contextual MDPs (DCMDPs)** further generalize the contextual MDP framework to handle non-Markovian environments where context evolves over time [5].

Real-time extensions of the MDP formalism have also been proposed to address the mismatch between the classical assumption that the environment remains static during action selection and the reality of real-time systems where state and actions evolve simultaneously [27]. In multi-agent settings, the MDP framework is generalized to **Markov Games** and **Decentralized POMDPs (Dec-POMDPs)**, which model interactions among multiple agents with potentially conflicting or cooperative objectives, giving rise to additional challenges of non-stationarity and exponential scaling of the joint state-action space [22].

### Summary

The MDP and its extensions constitute the formal backbone of deep reinforcement learning. The richness of this framework — spanning fully observable and partially observable settings, finite and infinite horizons, discrete and continuous spaces, single-agent and multi-agent scenarios, and Markovian and non-Markovian dynamics — provides the conceptual language necessary to formalize an enormous diversity of sequential decision-making problems. Crucially, the core MDP formulation, with its Markov property and discounted return objective, furnishes the mathematical structure from which the Bellman equations emerge naturally. Understanding the assumptions, strengths, and limitations of each formulation is therefore essential not only for selecting appropriate algorithms, but also for appreciating how the theoretical guarantees and practical challenges of deep reinforcement learning are ultimately rooted in the recursive optimality principles explored in the following section.

### 1.2 Bellman Equations and Optimality Principles

## 1.2 Bellman Equations and Optimality Principles

Building directly upon the MDP formalism established in Section 1.1—where the agent's objective is to maximize the expected discounted cumulative return $\mathbb{E}_\pi[3]$—the Bellman equations provide the recursive decompositions of value functions that

[... deterministic omitted span ...]

exploration challenges introduced by each paradigm — including non-stationarity, sparse rewards, and partial observability — further motivate the need for paradigm-specific solutions.

### Cooperative Settings

In purely cooperative settings, all agents share a common objective and must coordinate their actions to maximize a joint reward signal. The central challenge in such environments is the **multi-agent credit assignment problem**: determining how each individual agent contributed to the collective outcome. When agents receive only a global team reward, it becomes difficult to evaluate individual contributions, which can slow down learning and lead to suboptimal coordination. Algorithms such as QMIX and VDN address this through value decomposition, factoring the joint action-value function into individual contributions while maintaining consistency with the global objective.

A prominent benchmark for cooperative multi-agent reinforcement learning is the StarCraft Multi-Agent Challenge (SMAC), which features heterogeneous agent types on complex maps and demands tight coordination. In this domain, agent modeling has been explored as an auxiliary learning task to improve coordination. For example, extending actor-critic methods with agent modeling — where agents learn to predict the policies of other agents alongside standard actor and critic objectives — has been shown to stabilize learning and improve expected rewards in cooperative object transportation tasks [209]. By leveraging representations of co-learners' behaviors, agents can better anticipate joint actions and coordinate more effectively.

Non-monotonic value function factorization presents another important avenue in cooperative settings. Standard QMIX enforces a monotonicity constraint on the mixing network, which limits the expressiveness of the joint value function and may fail to represent certain cooperative strategies. NQMIX addresses this limitation by introducing an off-policy policy gradient method that removes the monotonicity constraint, enabling more expressive non-monotonic joint value functions. This allows agents to represent richer cooperative strategies, particularly in environments with heterogeneous agent types [210]. NQMIX further mitigates overestimation issues inherent in QMIX by targeting state-values rather than overestimated Q-value targets.

Coordination in cooperative settings is also tightly linked to the communication protocols discussed in the preceding section. When agents can share information, coordination becomes substantially easier, but bandwidth and observability constraints frequently prevent unlimited communication. Communication-based cooperative methods learn to selectively share relevant information, improving joint decision-making under partial observability. These communication protocols must be co-optimized alongside cooperative policies to ensure that shared information is meaningful and actionable.

### Competitive Settings

In purely competitive settings, the goals of agents are directly opposed. The canonical example is a zero-sum two-player game, where one agent's gain is precisely another's loss. The theoretical foundation for competitive multi-agent learning is rooted in game theory, particularly in the concept of Nash equilibria, where no agent can unilaterally improve its outcome given the strategies of opponents.

A key challenge in competitive environments is **opponent modeling**: accurately representing and anticipating the strategies of adversaries to develop effective counter-strategies. Unlike cooperative settings where agents benefit from coordination, competitive agents must exploit weaknesses in opponent behavior while simultaneously protecting against exploitation themselves. Self-play is a prominent training paradigm in competitive settings, wherein agents improve by playing against copies or historical versions of themselves, gradually discovering stronger strategies through iterative refinement.

Emergent strategies in competitive settings can be highly sophisticated and counterintuitive, arising from the co-evolutionary dynamics of competing learners. In adversarial actor-critic formulations, the introduction of an adversarial agent that attempts to predict and counter the primary agent's actions has been shown to stimulate more diverse and exploratory behaviors. The Adversarially Guided Actor-Critic (AGAC) algorithm incorporates an adversary that mimics the actor by minimizing the KL-divergence between their action distributions, while the actor learns to differentiate itself from adversary predictions. This adversarial pressure incentivizes the actor to pursue strategies that are difficult to predict from historical trajectories, promoting innovation in hard-exploration and procedurally-generated competitive tasks [211]. Notably, such adversarial mechanisms also serve as an implicit form of coordinated exploration, complementing the exploration strategies discussed in the following section.

Competitive environments also introduce non-stationarity challenges from the perspective of each individual agent: as opponents improve their strategies, the effective environment dynamics change, invalidating value estimates and requiring continuous adaptation. This co-evolutionary instability can cause learning to cycle without converging to stable solutions, a problem that has motivated research into opponent-aware learning, best-response computation, and policy regularization methods that stabilize training against adaptive adversaries.

### Mixed Cooperative-Competitive Settings

Many real-world scenarios involve both cooperative and competitive elements simultaneously. Mixed settings arise when agents form teams that cooperate internally while competing against other teams, or when individual agents must balance shared group interests against personal incentives. Such environments are formalized as general-sum Markov games, where agent reward functions can exhibit partial alignment and partial conflict.

Mixed settings demand algorithms capable of managing the dual imperatives of coordination with allies and competition with opponents. This requires agents to simultaneously perform intra-team credit assignment, inter-team opponent modeling, and policy adaptation to a non-stationary multi-team environment. The MADDPG algorithm, a multi-agent extension of DDPG, addresses this by learning centralized critics conditioned on the observations and actions of all agents, allowing each agent to account for the behavior of both allies and opponents during training while executing decentralized policies at test time. This centralized training with decentralized execution (CTDE) paradigm is broadly applicable across cooperative, competitive, and mixed settings, and naturally integrates with the communication-based information sharing mechanisms described earlier.

Social dilemmas represent a particularly challenging class of mixed settings, where individually rational behavior leads to collectively suboptimal outcomes. Examples include the prisoner's dilemma, the tragedy of the commons, and public goods games. In these scenarios, agents face incentives to defect even though mutual cooperation would yield higher collective rewards. Designing reward structures and learning algorithms that promote cooperative behavior in the presence of defection incentives is an active area of research. Intrinsic motivation and social reward shaping have been proposed to encourage prosocial behavior, while mechanism design approaches modify the reward structure itself to align individual and collective incentives.

Agent modeling is particularly valuable in mixed settings, as understanding both allied and opponent intentions enables more nuanced strategy formation. Hierarchical approaches that decompose decision-making into high-level strategic reasoning and low-level execution have demonstrated promise in mixed competitive-cooperative tasks, allowing agents to reason about team strategy while adapting tactically to adversarial opponents [209].

### Reward Structure Considerations Across Paradigms

The reward structure is the primary determinant of the interaction paradigm and profoundly influences algorithmic design. Purely cooperative settings typically employ a shared team reward, which naturally aligns agent incentives but complicates credit assignment. Purely competitive settings employ zero-sum or adversarial rewards, demanding opponent awareness. Mixed settings employ individual or sub-group rewards that partially overlap, requiring careful reward decomposition.

In many practical mixed environments, reward shaping is used to augment sparse environmental rewards with dense auxiliary signals that guide agent behavior toward desired interaction patterns. The design of such auxiliary rewards must carefully account for the multi-agent structure to avoid misaligning cooperative incentives or inadvertently encouraging defection. Potential-field-based approaches have been explored to provide informative auxiliary critics that complement reward-based critics, offering structured guidance particularly in scenarios where direct reward signals are sparse or delayed [212]. This connection between reward shaping and exploration further bridges into the joint exploration strategies discussed in the following section, where sparse reward signals similarly motivate the use of intrinsic motivation and auxiliary objectives.

### Summary

Cooperative, competitive, and mixed multi-agent settings each pose distinct algorithmic challenges that require tailored solutions. Cooperative settings demand credit assignment mechanisms, value decomposition, and communication protocols. Competitive settings require opponent modeling, self-play, and strategies for managing co-evolutionary instability. Mixed settings demand the simultaneous management of intra-team coordination and inter-team competition, often requiring hierarchical reasoning and carefully designed reward structures. Together, these paradigms build directly upon the communication and information-sharing foundations discussed in the preceding section, and their associated challenges of sparse rewards, partial observability, and non-stationarity motivate the coordinated exploration strategies examined in the section that follows. The diversity of these paradigms underscores the richness of multi-agent deep reinforcement learning as a research field and highlights the importance of selecting algorithms whose inductive biases match the underlying

[... deterministic omitted span ...]

for Generalization in Reinforcement Learning

[106] Analysis of a Target-Based Actor-Critic Algorithm with Linear Function  Approximation

[107] Convergence Proof for Actor-Critic Methods Applied to PPO and RUDDER

[108] Decision-Aware Actor-Critic with Function Approximation and Theoretical  Guarantees

[109] Model-Augmented Actor-Critic  Backpropagating through Paths

[110] How to Learn a Useful Critic  Model-based Action-Gradient-Estimator  Policy Optimization

[111] TD-Regularized Actor-Critic Methods

[112] Actor critic learning algorithms for mean-field control with moment  neural networks

[113] Wasserstein Flow Meets Replicator Dynamics  A Mean-Field Analysis of  Representation Learning in Actor-Critic

[114] Learning Value Functions in Deep Policy Gradients using Residual  Variance

[115] DAC  The Double Actor-Critic Architecture for Learning Options

[116] Parameter-Based Value Functions

[117] Actor-Director-Critic  A Novel Deep Reinforcement Learning Framework

[118] The Actor-Advisor  Policy Gradient With Off-Policy Advice

[119] Playing Atari with Deep Reinforcement Learning

[120] Group Equivariant Deep Reinforcement Learning

[121] Analyzing the Hidden Activations of Deep Policy Networks  Why  Representation Matters

[122] On Improving Deep Reinforcement Learning for POMDPs

[123] Reinforcement Learning via Recurrent Convolutional Neural Networks

[124] Neural Map  Structured Memory for Deep Reinforcement Learning

[125] Representation Learning for Continuous Action Spaces is Beneficial for  Efficient Policy Learning

[126] Learning to Represent Action Values as a Hypergraph on the Action  Vertices

[127] Value-Consistent Representation Learning for Data-Efficient  Reinforcement Learning

[128] Frustratingly Easy Regularization on Representation Can Boost Deep  Reinforcement Learning

[129] Deep Reinforcement Learning meets Graph Neural Networks  exploring a  routing optimization use case

[130] Deep Reinforcement Learning in Parameterized Action Space

[131] Deep Multi-Agent Reinforcement Learning with Discrete-Continuous Hybrid  Action Spaces

[132] Deep Reinforcement Learning Using a Low-Dimensional Observation Filter  for Visual Complex Video Game Playing

[133] Learning to Factor Policies and Action-Value Functions  Factored Action  Space Representations for Deep Reinforcement learning

[134] Network Randomization  A Simple Technique for Generalization in Deep  Reinforcement Learning

[135] PAnDR  Fast Adaptation to New Environments from Offline Experiences via  Decoupling Policy and Environment Representations

[136] Reducing the Deployment-Time Inference Control Costs of Deep  Reinforcement Learning Agents via an Asymmetric Architecture

[137] VDSC  Enhancing Exploration Timing with Value Discrepancy and State  Counts

[138] Temporally-Extended ε-Greedy Exploration

[139] Dealing with uncertainty  balancing exploration and exploitation in deep  recurrent reinforcement learning

[140] ε-BMC  A Bayesian Ensemble Approach to Epsilon-Greedy  Exploration in Model-Free Reinforcement Learning

[141] Exploration Conscious Reinforcement Learning Revisited

[142] Unifying Count-Based Exploration and Intrinsic Motivation

[143] Exploration in Feature Space for Reinforcement Learning

[144] DORA The Explorer  Directed Outreaching Reinforcement Action-Selection

[145] Exploration Potential

[146] Intrinsically-Motivated Reinforcement Learning  A Brief Introduction

[147] Surprise-Based Intrinsic Motivation for Deep Reinforcement Learning

[148] Incentivizing Exploration In Reinforcement Learning With Deep Predictive  Models

[149] Information Content Exploration

[150] Information Maximizing Exploration with a Latent Dynamics Model

[151] Bayesian Curiosity for Efficient Exploration in Reinforcement Learning

[152] Implicit Generative Modeling for Efficient Exploration

[153] Exploration by Maximizing Rényi Entropy for Reward-Free RL Framework

[154] Successor-Predecessor Intrinsic Exploration

[155] Intrinsically motivated graph exploration using network theories of  human curiosity

[156] Reward Prediction Error as an Exploration Objective in Deep RL

[157] Temporal Difference Uncertainties as a Signal for Exploration

[158] Model-Based Bayesian Exploration

[159] Deep Bandits Show-Off  Simple and Efficient Exploration with Deep  Networks

[160] Sample Efficient Reinforcement Learning via Model-Ensemble Exploration  and Exploitation

[161] Intrinsic Exploration as Multi-Objective RL

[162] Redeeming Intrinsic Rewards via Constrained Optimization

[163] Deep Intrinsically Motivated Exploration in Continuous Control

[164] Goal-oriented Trajectories for Efficient Exploration

[165] ExTra  Transfer-guided Exploration

[166] Show me the Way  Intrinsic Motivation from Demonstrations

[167] The Challenges of Exploration for Offline Reinforcement Learning

[168] Influence-Based Multi-Agent Exploration

[169] Never Explore Repeatedly in Multi-Agent Reinforcement Learning

[170] MEET  A Monte Carlo Exploration-Exploitation Trade-off for Buffer  Sampling

[171] Self-supervised Sequential Information Bottleneck for Robust Exploration  in Deep Reinforcement Learning

[172] Modeling Human Exploration Through Resource-Rational Reinforcement  Learning

[173] Goal-conditioned Offline Planning from Curious Exploration

[174] Analysis and Optimisation of Bellman Residual Errors with Neural  Function Approximation

[175] Actor Prioritized Experience Replay

[176] DisCor  Corrective Feedback in Reinforcement Learning via Distribution  Correction

[177] GRAC  Self-Guided and Self-Regularized Actor-Critic

[178] Spectral Normalisation for Deep Reinforcement Learning  an Optimisation  Perspective

[179] Averaged-DQN  Variance Reduction and Stabilization for Deep  Reinforcement Learning

[180] Revisiting Plasticity in Visual Reinforcement Learning  Data, Modules  and Training Stages

[181] Loss of Plasticity in Continual Deep Reinforcement Learning

[182] Lenient Multi-Agent Deep Reinforcement Learning

[183] Stable deep reinforcement learning method by predicting uncertainty in  rewards as a subtask

[184] Direct and indirect reinforcement learning

[185] Single-Timescale Actor-Critic Provably Finds Globally Optimal Policy

[186] Policy Mirror Descent for Regularized Reinforcement Learning  A  Generalized Framework with Linear Convergence

[187] Bridging Physics-Informed Neural Networks with Reinforcement Learning   Hamilton-Jacobi-Bellman Proximal Policy Optimization (HJBPPO)

[188] A Nonparametric Off-Policy Policy Gradient

[189] Neural Lyapunov and Optimal Control

[190] Approximating two value functions instead of one  towards characterizing  a new family of Deep Reinforcement Learning algorithms

[191] Quinoa  a Q-function You Infer Normalized Over Actions

[192] Smoothed Action Value Functions for Learning Gaussian Policies

[193] ($S$,$N$,$T$)-Implications

[194] Policy Gradient Methods for Distortion Risk Measures

[195] Deep Value Model Predictive Control

[196] A Survey of Deep Reinforcement Learning in Video Games

[197] Structure in Deep Reinforcement Learning  A Survey and Open Problems

[198] Ensemble Reinforcement Learning in Continuous Spaces -- A Hierarchical  Multi-Step Approach for Policy Training

[199] Adaptive and Multiple Time-scale Eligibility Traces for Online Deep  Reinforcement Learning

[200] RLx2  Training a Sparse Deep Reinforcement Learning Model from Scratch

[201] Symmetric Replay Training  Enhancing Sample Efficiency in Deep  Reinforcement Learning for Combinatorial Optimization

[202] Never Worse, Mostly Better  Stable Policy Improvement in Deep  Reinforcement Learning

[203] A Unifying Framework for Reinforcement Learning and Planning

[204] Approximate dynamic programming with $(\min,+)$ linear function  approximation for Markov decision processes

[205] Orchestrated Value Mapping for Reinforcement Learning

[206] A Survey Analyzing Generalization in Deep Reinforcement Learning

[207] A Survey on Deep Reinforcement Learning-based Approaches for Adaptation  and Generalization

[208] Learning from Humans as an I-POMDP

[209] Agent Modeling as Auxiliary Task for Deep Reinforcement Learning

[210] NQMIX  Non-monotonic Value Function Factorization for Deep Multi-Agent  Reinforcement Learning

[211] Adversarially Guided Actor-Critic

[212] Potential Field Guided Actor-Critic Reinforcement Learning

[213] End-to-End Policy Gradient Method for POMDPs and Explainable Agents

[214] Distributional Advantage Actor-Critic

[215] Doubly Robust Off-Policy Actor-Critic Algorithms for Reinforcement  Learning

[216] Tactical Optimism and Pessimism for Deep Reinforcement Learning

[217] Physical Deep Reinforcement Learning Towards Safety Guarantee

[218] Robust Deep Reinforcement Learning Through Adversarial Attacks and  Training   A Survey

[219] Towards Optimal Adversarial Robust Q-learning with Bellman  Infinity-error

[220] Robust Adversarial Reinforcement Learning

[221] Reachability Verification Based Reliability Assessment for Deep  Reinforcement Learning Controlled Robotics and Autonomous Systems

[222] Towards Safe Reinforcement Learning via Constraining Conditional  Value-at-Risk

[223] Distributional Reinforcement Learning for Efficient Exploration

[224] Deep Reinforcement Learning with Weighted Q-Learning

[225] Deep Reinforcement Learning with Double Q-learning

[226] Rainbow  Combining Improvements in Deep Reinforcement Learning

[227] Reducing Variance in Temporal-Difference Value Estimation via Ensemble  of Deep Networks

[228] Statistical Inference of the Value Function for Reinforcement Learning  in Infinite Horizon Settings

[229] Reward Shaping with Dynamic Trajectory Aggregation

[230] Prediction and Control in Continual Reinforcement Learning

[231] Reinforcement Learning with Partially Known World Dynamics

[232] Bellman Gradient Iteration for Inverse Reinforcement Learning

[233] Reinforcement Learning in Economics and Finance

[234] A Study of Policy Gradient on a Class of Exactly Solvable Models

[235] Addressing Function Approximation Error in Actor-Critic Methods

[236] Are Gradient-based Saliency Maps Useful in Deep Reinforcement Learning 

[237] Towards Deep Symbolic Reinforcement Learning

[238] Open Problems and Modern Solutions for Deep Reinforcement Learning

[239] Soft Actor-Critic Algorithms and Applications

[240] Distributional Soft Actor-Critic  Off-Policy Reinforcement Learning for  Addressing Value Estimation Errors

[241] DSAC-T  Distributional Soft Actor-Critic with Three Refinements

[242] Better Exploration with Optimistic Actor-Critic

[243] Online Meta-Critic Learning for Off-Policy Actor-Critic Methods

[244] Revisiting Rainbow  Promoting more Insightful and Inclusive Deep  Reinforcement Learning Research

[245] GDI  Rethinking What Makes Reinforcement Learning Different From  Supervised Learning

[246] Diverse Projection Ensembles for Distributional Reinforcement Learning

[247] Normality-Guided Distributional Reinforcement Learning for Continuous  Control

[248] Exploration with Multi-Sample Target Values for Distributional  Reinforcement Learning

[249] A Comparative Analysis of Expected and Distributional Reinforcement  Learning

[250] The Benefits of Being Distributional  Small-Loss Bounds for  Reinforcement Learning
