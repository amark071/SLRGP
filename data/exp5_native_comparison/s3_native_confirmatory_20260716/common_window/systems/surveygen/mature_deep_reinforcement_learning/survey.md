# Registered common-window excerpt

# Deep Reinforcement Learning: A Comprehensive Survey of Foundations, Algorithms, and Applications

## Foundations of Reinforcement Learning and Deep Learning

### Markov Decision Processes and Problem Formulation

The Markov Decision Process (MDP) provides the formal mathematical framework underpinning reinforcement learning ref [1]. An MDP is defined by a tuple (S, A, P, R, γ), where S denotes the state space, A the action space, P: S × A × S → [0,1] the transition probability function, R: S × A → ℝ the reward function, and γ ∈ [0,1) the discount factor ref [26]. The Markov property stipulates that future states depend solely on the current state and action, rendering the process memoryless ref [33]. An agent interacts with the environment by selecting actions according to a policy π: S → A, receiving scalar rewards and transitioning between states. The objective is to find an optimal policy π* maximizing the expected cumulative discounted return ref [32]. Central to this formulation are the Bellman equations, which decompose value functions recursively, enabling iterative computation of optimal state-value V*(s) and action-value Q*(s,a) functions ref [1]. Extensions such as constrained MDPs incorporate safety requirements into this framework ref [37].

### Classical Reinforcement Learning Algorithms

Classical reinforcement learning algorithms provide the foundational computational machinery upon which modern deep RL systems are built. Dynamic programming methods, including policy iteration and value iteration, solve MDPs exactly when the environment model is known, leveraging the Bellman equations to compute optimal value functions ref [1]. When the model is unavailable, Monte Carlo methods estimate value functions through episodic sampling, averaging returns from complete trajectories without bootstrapping ref [1]. Temporal difference (TD) learning combines the sampling efficiency of Monte Carlo approaches with the bootstrapping of dynamic programming, enabling online, incremental updates ref [26]. Q-learning, a seminal off-policy TD algorithm, directly approximates the optimal action-value function by updating estimates using the maximum future reward, converging to optimal policies in tabular settings ref [1]. SARSA, an on-policy variant, updates based on the action actually taken. These tabular methods, while theoretically well-understood, become computationally intractable in high-dimensional spaces, motivating the integration of function approximation and deep neural networks ref [32].

### Deep Neural Network Architectures for RL

Deep neural network architectures constitute the representational backbone of modern DRL systems, enabling scalable function approximation across high-dimensional state and action spaces. Convolutional neural networks (CNNs) are foundational for processing visual inputs; the seminal DQN architecture employed stacked CNNs to learn control policies directly from raw pixel observations in Atari games ref [5]. For tasks involving partial observability or temporal dependencies, recurrent architectures—particularly Long Short-Term Memory (LSTM) networks—provide mechanisms for integrating information across time steps, mitigating the limitations of Markovian state representations ref [46]. The dueling network architecture introduced a structural decomposition of the Q-function into separate state-value and advantage streams, enabling more efficient policy evaluation across actions with similar values ref [13]. Actor-critic frameworks further leverage distinct network components for policy and value estimation ref [21]. End-to-end visuomotor policies, combining convolutional perception with motor control, have demonstrated effectiveness in robotic manipulation tasks ref [49].

### Function Approximation in Reinforcement Learning

Function approximation is indispensable for scaling reinforcement learning to high-dimensional state and action spaces where tabular representations are computationally infeasible ref [1]. Rather than maintaining explicit value estimates for every state-action pair, function approximators—parameterized by weights θ—generalize across states, enabling agents to operate in continuous or combinatorially large domains ref [26]. The policy gradient theorem formally establishes that policies can themselves be represented by differentiable function approximators and updated via gradient ascent on expected return, providing theoretical grounding for this paradigm ref [12]. Deep neural networks have emerged as the dominant approximator class, demonstrated compellingly by DQN, which approximates Q-values directly from raw pixel inputs ref [5]. However, function approximation introduces significant challenges, including overestimation bias in value estimates ref [14], instability arising from correlated updates, and convergence difficulties when combining bootstrapping with nonlinear approximators ref [32]. These practical challenges motivate stabilization mechanisms such as experience replay and target networks, which remain foundational to modern deep reinforcement learning systems ref [1].

### Exploration vs. Exploitation Trade-off

The exploration-exploitation trade-off represents one of the most fundamental dilemmas in reinforcement learning: an agent must balance gathering new information about the environment against leveraging existing knowledge to maximize cumulative reward ref [1]. In deep reinforcement learning, this tension is amplified by high-dimensional state spaces where exhaustive exploration is computationally infeasible. The simplest strategy, epsilon-greedy, selects random actions with probability epsilon and greedy actions otherwise, providing a straightforward mechanism for exploration ref [5]. However, such undirected exploration proves insufficient in sparse reward environments, where meaningful feedback is rarely encountered through random action selection ref [36]. More sophisticated approaches include intrinsic motivation, curiosity-driven exploration, and count-based methods that reward novelty ref [36]. Maximum entropy frameworks, as employed in Soft Actor-Critic, encourage stochastic policies that inherently promote exploration by maximizing both expected reward and policy entropy ref [6]. Deterministic policy gradient methods address exploration through off-policy behavior policies ref [7], while prioritized experience replay improves data utilization by emphasizing informative transitions ref [2].

## Core Deep Reinforcement Learning Algorithms

### Value-Based Methods: DQN and Its Extensions

The Deep Q-Network (DQN) represents a landmark advance in value-based deep reinforcement learning, demonstrating that convolutional neural networks trained with Q-learning can learn control policies directly from raw pixel inputs across diverse Atari 2600 games ref [5]. DQN introduced two critical stabilization mechanisms: experience replay, which breaks temporal correlations in training data, and target networks, which provide stable optimization targets ref [32]. Despite these innovations, DQN suffers from overestimation bias, wherein the max operator in Q-learning inflates value estimates. Double DQN addresses this by decoupling action selection from value estimation using separate networks ref [14]. The Dueling Network architecture further improves learning by decomposing Q-values into separate state-value and advantage streams, enabling more efficient policy evaluation ref [13]. Prioritized Experience Replay enhances sample efficiency by replaying transitions proportional to their temporal-difference error ref [2]. The Rainbow framework integrates these extensions alongside distributional RL and multi-step returns, achieving state-of-the-art performance on Atari benchmarks ref [11].

### Policy Gradient Methods

Policy gradient methods constitute a foundational paradigm in deep reinforcement learning, wherein the policy is explicitly parameterized and optimized by ascending the gradient of expected cumulative reward with respect to policy parameters. The seminal policy gradient theorem, formalized by Sutton et al. ref [12], established that the gradient of expected return can be expressed as an expectation over state-action visitation distributions, enabling gradient estimation from sampled trajectories. The REINFORCE algorithm exemplifies this approach, employing Monte Carlo rollouts to estimate policy gradients ref [1]. A central challenge in policy gradient methods is high variance in gradient estimates, which impedes stable learning. Variance reduction techniques, including the use of baseline functions and advantage estimation, have been widely adopted to address this issue ref [12]. These methods naturally accommodate stochastic policies and continuous action spaces, providing a flexible foundation upon which actor-critic architectures ref [21], trust region methods ref [3], and maximum entropy formulations ref [6] are subsequently built.

### Trust Region and Proximal Policy Optimization

Trust Region Policy Optimization (TRPO) addresses a fundamental instability in policy gradient methods: excessively large policy updates can catastrophically degrade performance. TRPO formulates policy optimization as a constrained problem, enforcing a Kullback-Leibler divergence constraint between successive policies to guarantee monotonic improvement ref [3]. By restricting updates to a trust region, TRPO provides theoretical guarantees of policy improvement while remaining effective for large nonlinear neural network policies across diverse tasks, including simulated locomotion and Atari game playing ref [3]. However, TRPO's computational overhead, arising from second-order optimization and constraint enforcement, limits its scalability. Proximal Policy Optimization (PPO) addresses this limitation by replacing the hard constraint with a clipped surrogate objective, penalizing updates that deviate excessively from the current policy without requiring expensive constraint computations ref [47]. PPO achieves comparable stability and performance to TRPO with significantly reduced complexity, making it widely adopted across robotics, continuous control, and game-playing applications ref [50].

### Actor-Critic Architectures

Actor-critic architectures represent a

[... deterministic omitted span ...]

Actor-Critic, encourage stochastic policies that inherently promote broader exploration ref [6]. Collectively, these complementary strategies substantially improve learning efficiency in environments where reward signals are exceptionally sparse.

### Stability and Convergence Issues

Training instability represents one of the most significant challenges in deep reinforcement learning, arising from the complex interplay between function approximation, bootstrapping, and non-stationary data distributions. A prominent manifestation is overestimation bias in value-based methods, where Q-learning systematically inflates action-value estimates, degrading policy quality ref [14]. Double DQN mitigates this by decoupling action selection from value estimation ref [15], while TD3 extends this principle to actor-critic settings by employing clipped double Q-learning ref [10]. The primacy bias describes agents' tendency to overfit early experiences, impairing subsequent learning; periodic network resets have been proposed as an effective remedy ref [48]. Complementarily, the dormant neuron phenomenon identifies progressive neuron inactivity that reduces network expressivity, addressable through neuron recycling strategies ref [57]. Soft Actor-Critic's maximum entropy framework provides inherent stability through stochastic policies and off-policy updates ref [6], while TRPO's trust region constraints ensure monotonic policy improvement ref [3].

### Loss of Plasticity and Continual Learning

Loss of plasticity refers to the progressive degradation of a deep RL agent's capacity to learn new behaviors over time, particularly in non-stationary environments. Ref [53] provides a systematic characterization of this phenomenon in value-based deep RL, demonstrating that agents cycling through sequences of Atari 2600 games progressively lose their ability to acquire good policies. This degradation manifests through several interrelated mechanisms, including implicit under-parameterization, capacity loss, and primacy bias ref [48]. Empirical analysis reveals that network activation footprints become increasingly sparse over training, contributing to diminishing gradients and reduced representational capacity ref [53]. The dormant neuron phenomenon further compounds this issue, as growing proportions of inactive neurons reduce network expressivity ref [57]. A notably effective mitigation strategy involves the Concatenated ReLUs (CReLU) activation function, which preserves representational diversity and facilitates continual learning across changing task distributions ref [53]. Periodic network resets represent a complementary remedy for counteracting primacy bias ref [48].

### Transfer Learning and Generalization

Transfer learning in deep reinforcement learning addresses the fundamental challenge of generalizing acquired knowledge across tasks and domains, thereby reducing sample complexity and improving deployment feasibility. A comprehensive taxonomy of transfer learning approaches in DRL has been systematically investigated, encompassing domain adaptation, policy transfer, and representation reuse strategies ref [23]. Sim-to-real transfer represents a particularly critical paradigm, wherein policies trained in simulation are adapted to physical environments; this approach has been demonstrated in mapless navigation ref [16] and target-driven visual navigation, where models trained in simulation generalize to real robotic scenarios with minimal fine-tuning ref [38]. Champion-level drone racing further exemplifies successful sim-to-real transfer by combining simulated deep RL training with real-world data collection ref [4]. Meta-learning frameworks extend these capabilities by enabling agents to rapidly adapt to novel tasks from limited experience, addressing the broader generalization challenge that remains central to deploying DRL in complex, non-stationary real-world environments ref [32].

### Multi-Agent Reinforcement Learning

Multi-agent reinforcement learning (MARL) extends the single-agent DRL framework to settings where multiple agents interact within a shared environment, introducing challenges of non-stationarity, partial observability, and coordination ref [45]. MARL encompasses cooperative, competitive, and mixed paradigms, each demanding distinct algorithmic treatments. A prominent approach is centralized training with decentralized execution (CTDE), wherein agents share information during training but act independently at deployment. The Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm exemplifies this paradigm by extending DDPG to multi-agent continuous control settings ref [34]. MARL has demonstrated significant utility in wireless communications, where multiple network entities must collaboratively optimize resource allocation and trajectory planning in UAV-assisted mobile edge computing ref [34]. Comprehensive tutorials on single and multi-agent DRL further highlight cooperative MARL's potential for building scalable, decentralized wireless networks ref [60]. Ongoing challenges include scalable credit assignment, emergent communication, and convergence guarantees in non-stationary multi-agent environments ref [45].

### Reproducibility and Experimental Rigor

Reproducibility constitutes a fundamental challenge in deep reinforcement learning, where non-determinism in benchmark environments and intrinsic algorithmic variance render result interpretation particularly difficult. Henderson et al. ref [20] systematically investigated these challenges, demonstrating that reported performance metrics vary substantially across implementations of identical algorithms, and that differences attributed to novel contributions may lack statistical significance. Their analysis revealed that hyperparameter choices, network architectures, and random seeds introduce considerable variance, making fair comparisons against baselines unreliable without rigorous controls. The authors advocate for standardized experimental reporting protocols, including confidence intervals, significance tests, and multiple evaluation seeds. This concern is amplified by the widespread adoption of benchmarks such as MuJoCo ref [25] and Atari environments ref [5], where subtle implementation differences yield divergent outcomes. Establishing community-wide norms for experimental rigor—encompassing transparent hyperparameter disclosure, code release, and statistically grounded evaluation—remains essential for sustaining meaningful scientific progress in the field.

## Applications of Deep Reinforcement Learning

### Game Playing and Simulated Environments

Game playing and simulated environments have served as foundational benchmarks for evaluating deep reinforcement learning algorithms. The seminal Deep Q-Network demonstrated that convolutional neural networks trained with Q-learning could achieve human-level performance on Atari 2600 games directly from raw pixel inputs ref [5]. Subsequent extensions, including Double DQN ref [14], Dueling Network Architectures ref [13], and Prioritized Experience Replay ref [2], progressively improved performance across the Atari benchmark, culminating in the Rainbow framework that integrates multiple complementary improvements ref [11]. Trust Region Policy Optimization further demonstrated robust performance on Atari alongside simulated locomotion tasks ref [3]. Physics-based simulation environments such as MuJoCo ref [25] became standard benchmarks for continuous control algorithms ref [8]. Beyond classical benchmarks, DRL achieved champion-level performance in drone racing through the Swift system ref [4] and outperformed world-class human competitors in Gran Turismo racing via GT Sophy ref [22], representing significant milestones in applying DRL to complex real-time competitive domains.

### Robotics and Autonomous Control

Deep reinforcement learning has demonstrated substantial promise in robotics, enabling autonomous acquisition of complex behaviors directly from sensory observations. Foundational work on continuous control via DDPG demonstrated that actor-critic methods could solve over twenty simulated manipulation and locomotion tasks, including dexterous manipulation and legged locomotion ref [8]. Extending this to real physical systems, asynchronous off-policy deep RL enabled robots to learn manipulation skills, including door opening, without prior demonstrations ref [17]. End-to-end visuomotor policy learning, mapping raw image observations directly to motor torques, further demonstrated the advantage of jointly training perception and control ref [49]. For high-dimensional dexterous manipulation, model-free DRL combined with human demonstrations successfully scaled to a 24-DoF robotic hand ref [54]. Mapless navigation using sparse range findings trained via asynchronous DRL transferred directly to real environments ref [16]. Comprehensive lessons from real-world robotic deployments highlight persistent challenges including sample inefficiency, safety, and the simulation-to-reality gap ref [39].

### Autonomous Driving and Navigation

Autonomous driving and navigation represent a prominent application domain for deep reinforcement learning, where agents must integrate perception and control in complex, dynamic environments. DRL methods have been applied to mapless navigation, wherein agents learn continuous steering commands directly from sparse range findings and target position information without requiring pre-built obstacle maps, demonstrating transferability to unseen real-world environments ref [16]. Target-driven visual navigation in indoor scenes has been addressed through actor-critic architectures conditioned on goal representations, enabling generalization across targets and scenes with efficient sim-to-real transfer ref [38]. Comprehensive surveys of DRL for autonomous driving provide taxonomies of tasks including lane keeping, intersection management, and multi-agent traffic scenarios, while addressing key computational challenges such as safety, partial observability, and the simulation-to-reality gap ref [56]. These works collectively highlight that integrating perception with decision-making end-to-end remains central to advancing autonomous navigation systems.

### Communications and Networking

Deep reinforcement learning has emerged as a transformative paradigm for addressing the complex, dynamic optimization challenges inherent in modern communications and networking systems. A comprehensive survey by ref [42] systematically documents DRL applications across dynamic spectrum access, data rate control, wireless caching, data offloading, network security, and connectivity preservation, demonstrating DRL's capacity to handle large state and action spaces intractable for classical RL. In resource allocation for

[... deterministic omitted span ...]

network policies raises fundamental questions regarding trust, reliability, auditability, and fairness ref [43]. Providing transparent explanations of agent behavior is essential for maintaining human oversight in high levels of automation, where both human operators and autonomous systems share decision-making responsibilities ref [43]. Formal specifications of the DRL explainability problem identify key components of a general explainable RL framework, categorizing methods according to the interpretable models employed and the surface representations of explanations provided ref [43]. Existing approaches span post-hoc explanation techniques, attention mechanisms, and inherently interpretable policy representations. Despite notable progress, significant open challenges remain, including scalability to complex environments, the fidelity of approximations, and the development of standardized evaluation metrics for explanation quality ref [43].

### Federated and Decentralized Deep Reinforcement Learning

Federated and decentralized deep reinforcement learning (DRL) addresses the challenge of training agents across distributed systems without centralizing sensitive data. In federated DRL frameworks, multiple agents—such as base stations or IoT devices—independently train local policies and periodically aggregate model parameters to construct a shared global model, thereby preserving data privacy while reducing communication overhead. A representative example is the FADE framework ref [29], which applies federated DRL to cooperative edge caching in IoT networks, demonstrating provable expectation convergence alongside improvements in cache hit rate and backhaul offloading. Decentralized multi-agent approaches further extend this paradigm by enabling autonomous decision-making at each node ref [60], which is particularly relevant for next-generation wireless networks requiring scalable, low-latency coordination. Applications span mobile edge computing, UAV-assisted networks ref [34], and resource allocation ref [42], where centralized training is infeasible. Open challenges include communication efficiency, heterogeneous data distributions, and formal convergence guarantees under non-stationary environments.

### Scalability and Real-World Deployment

Scaling deep reinforcement learning to real-world systems presents formidable challenges beyond those encountered in simulation. Hardware constraints, latency requirements, and safety guarantees impose strict operational boundaries that laboratory benchmarks rarely capture ref [39]. The simulation-to-reality gap remains a central obstacle: policies trained in environments such as MuJoCo ref [25] often fail to transfer directly due to unmodeled dynamics, sensor noise, and actuator limitations. Parallelizing learning across multiple physical robots has been proposed to mitigate sample inefficiency in real deployments ref [17]. Landmark demonstrations, including champion-level drone racing with Swift ref [4] and tokamak plasma control ref [30], illustrate that hybrid simulation-plus-real-world pipelines can bridge this gap. Nevertheless, ensuring reliable closed-loop performance under high-dimensional, high-frequency control demands remains non-trivial. Addressing these scalability barriers requires co-designing algorithms with deployment constraints, incorporating robust transfer mechanisms ref [23], and rigorously validating policies under realistic operating conditions ref [56].

### Integration with Large Pretrained Models

The integration of large pretrained models with deep reinforcement learning represents a transformative emerging direction, leveraging the rich representational knowledge encoded in large language models (LLMs) and vision-language models to address longstanding challenges in DRL. Pretrained models serve multiple complementary roles within DRL frameworks: as policy initializers that provide semantically meaningful starting points, as reward shapers that translate high-level task specifications into dense feedback signals, and as world models that encode prior knowledge about environment dynamics. A prominent exemplar is CodeRL ref [59], which treats a pretrained language model as an actor network augmented by a critic trained to predict functional correctness, demonstrating how pretraining substantially accelerates convergence on complex program synthesis tasks. Such hybrid architectures benefit from the generalization capacity of large pretrained models while retaining the adaptive optimization properties of RL. Future research must address challenges including computational overhead, alignment between pretraining objectives and RL reward structures, and maintaining plasticity during fine-tuning ref [53].

### Theoretical Foundations and Open Problems

Despite remarkable empirical achievements, deep reinforcement learning lacks comprehensive theoretical foundations, leaving fundamental questions unresolved. Classical RL theory, grounded in Markov decision processes and Bellman equations ref [1], provides convergence guarantees for tabular settings, but extending these guarantees to deep function approximators remains largely intractable ref [12]. The combination of bootstrapping, off-policy learning, and nonlinear function approximation—the so-called deadly triad—introduces instabilities whose theoretical characterization is incomplete ref [32]. Sample complexity bounds for deep RL algorithms remain poorly understood, with practical methods such as DQN ref [5], TRPO ref [3], and SAC ref [6] demonstrating strong empirical performance without tight theoretical justification. Furthermore, generalization in RL—understanding how policies trained in one environment transfer to another ref [23]—lacks formal treatment analogous to statistical learning theory. Open problems include convergence guarantees for actor-critic methods ref [21], formal understanding of overestimation bias ref [14], and theoretical characterization of plasticity loss ref [53], collectively representing critical gaps requiring rigorous mathematical investigation.

## References

[1] Reinforcement Learning: An Introduction. https://doi.org/https://doi.org/10.1109/tnn.2004.842673

[2] Prioritized Experience Replay. https://doi.org/https://doi.org/10.48550/arxiv.1511.05952

[3] Trust Region Policy Optimization. https://doi.org/https://doi.org/10.48550/arxiv.1502.05477

[4] Champion-level drone racing using deep reinforcement learning. https://doi.org/10.1038/s41586-023-06419-4

[5] Playing Atari with Deep Reinforcement Learning. https://doi.org/https://doi.org/10.48550/arxiv.1312.5602

[6] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. https://doi.org/https://doi.org/10.48550/arxiv.1801.01290

[7] Deterministic policy gradient algorithms.

[8] Continuous control with deep reinforcement learning.

[9] Continuous control with deep reinforcement learning. https://doi.org/https://doi.org/10.48550/arxiv.1509.02971

[10] Addressing Function Approximation Error in Actor-Critic Methods. https://doi.org/https://doi.org/10.48550/arxiv.1802.09477

[11] Rainbow: Combining Improvements in Deep Reinforcement Learning. https://doi.org/10.1609/aaai.v32i1.11796

[12] Policy Gradient Methods for Reinforcement Learning with Function Approximation.

[13] Dueling Network Architectures for Deep Reinforcement Learning. https://doi.org/https://doi.org/10.48550/arxiv.1511.06581

[14] Deep Reinforcement Learning with Double Q-Learning. https://doi.org/10.1609/aaai.v30i1.10295

[15] Deep Reinforcement Learning with Double Q-Learning. https://doi.org/https://doi.org/10.1609/aaai.v30i1.10295

[16] Virtual-to-real deep reinforcement learning: Continuous control of mobile robots for mapless navigation. https://doi.org/10.1109/IROS.2017.8202134

[17] Deep reinforcement learning for robotic manipulation with asynchronous off-policy updates. https://doi.org/https://doi.org/10.1109/icra.2017.7989385

[18] Deep reinforcement learning for robotic manipulation with asynchronous off-policy updates. https://doi.org/10.1109/ICRA.2017.7989385

[19] Deep Reinforcement Learning with Double Q-Learning. https://doi.org/https://doi.org/10.1609/aaai.v30i1.10295

[20] Deep Reinforcement Learning that Matters. https://doi.org/10.1609/aaai.v32i1.11694

[21] Actor-critic algorithms.

[22] Outracing champion Gran Turismo drivers with deep reinforcement learning. https://doi.org/10.1038/s41586-021-04357-7

[23] Transfer Learning in Deep Reinforcement Learning: A Survey. https://doi.org/10.1109/TPAMI.2023.3292075

[24] Optimization of Molecules via Deep Reinforcement Learning. https://doi.org/10.1038/s41598-019-47148-x

[25] MuJoCo: A physics engine for model-based control. https://doi.org/https://doi.org/10.1109/iros.2012.6386109

[26] Reinforcement learning. https://doi.org/10.4249/scholarpedia.1448

[27] Network planning with deep reinforcement learning. https://doi.org/10.1145/3452296.3472902

[28] Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward. https://doi.org/10.1609/aaai.v32i1.12255

[29] Federated Deep Reinforcement Learning for Internet of Things With Decentralized Cooperative Edge Caching. https://doi.org/10.1109/JIOT.2020.2986803

[30] Magnetic control of tokamak plasmas through deep reinforcement learning. https://doi.org/10.1038/s41586-021-04301-9

[31] Deep Reinforcement Learning-Based Intelligent Reflecting Surface for Secure Wireless Communications. https://doi.org/10.1109/TWC.2020.3024860

[32] An Introduction to Deep Reinforcement Learning. https://doi.org/10.1561/2200000071

[33] A Markovian Decision Process. https://doi.org/https://doi.org/10.1512/iumj.1957.6.56038

[34] Multi-Agent Deep Reinforcement Learning-Based Trajectory Planning for Multi-UAV Assisted Mobile Edge Computing. https://doi.org/10.1109/TCCN.2020.3027695

[35] Deep reinforcement learning for de novo drug design. https://doi.org/10.1126/sciadv.aap7885

[36] Exploration in Deep Reinforcement Learning: A Survey. https://doi.org/10.1016/j.inffus.2022.03.003

[37] Constrained EV Charging Scheduling Based on Safe Deep Reinforcement Learning. https://doi.org/10.1109/TSG.2019.2955437

[38] Target-driven visual navigation in indoor scenes using deep reinforcement learning. https://doi.org/10.1109/ICRA.2017.7989381

[39] How to train your robot with deep reinforcement learning: lessons we have learned. https://doi.org/10.1177/0278364920987859

[40] Deep Reinforcement Learning for Offloading and Resource Allocation in Vehicle Edge Computing and Networks. https://doi.org/10.1109/TVT.2019.2935450

[41] Deep Reinforcement Learning for Dialogue Generation. https://doi.org/10.18653/v1/D16-1127

[42] Applications of Deep Reinforcement Learning in Communications and Networking: A Survey. https://doi.org/10.1109/COMST.2019.2916583

[43] Explainable Deep Reinforcement Learning: State of the Art and Challenges. https://doi.org/10.1145/3527448

[44] Faster sorting algorithms discovered using deep reinforcement learning. https://doi.org/10.1038/s41586-023-06004-9

[45] A Comprehensive Survey of Multiagent Reinforcement Learning. https://doi.org/https://doi.org/10.1109/tsmcc.2007.913919

[46] Long Short-Term Memory. https://doi.org/https://doi.org/10.1162/neco.1997.9.8.1735

[47] Deep Reinforcement Learning: A Brief Survey. https://doi.org/https://doi.org/10.1109/msp.2017.2743240

[48] The Primacy Bias in Deep Reinforcement Learning. https://doi.org/10.48550/arXiv.2205.07802

[49] End-to-end training of deep visuomotor policies.

[50] A Brief Survey of Deep Reinforcement Learning. https://doi.org/10.1109/MSP.2017.2743240

[51] Deep reinforcement learning in medical imaging: A literature review. https://doi.org/10.1016/j.media.2021.102193

[52] DRN: A Deep Reinforcement Learning Framework for News Recommendation. https://doi.org/10.1145/3178876.3185994

[53] Loss of Plasticity in Continual Deep Reinforcement Learning. https://doi.org/10.48550/arXiv.2303.07507

[54] Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations. https://doi.org/10.15607/RSS.2018.XIV.049

[55] Deep Reinforcement Learning for Delay-Oriented IoT Task Scheduling in SAGIN. https://doi.org/10.1109/TWC.2020.3029143

[56] Deep Reinforcement Learning for Autonomous Driving: A Survey. https://doi.org/10.1109/TITS.2021.3054625

[57] The Dormant Neuron Phenomenon in Deep Reinforcement Learning. https://doi.org/10.48550/arXiv.2302.12902

[58] Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning. https://doi.org/10.1109/ICRA.2018.8463189

[59] CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning. https://doi.org/10.48550/arXiv.2207.01780

[60] Single and Multi-Agent Deep Reinforcement Learning for AI-Enabled Wireless Networks: A Tutorial. https://doi.org/10.1109/COMST.2021.3063822
