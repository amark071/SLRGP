# Registered common-window excerpt

# Continual Learning: A Comprehensive Survey of Foundations, Methods, Theory, Applications, and Future Directions

## 1 Introduction and Foundations of Continual Learning

### 1.1 What is Continual Learning? Definitions and Core Concepts

## 1.1 Defining Continual Learning

Continual learning (CL), also referred to as lifelong learning, incremental learning, or sequential learning, is a machine learning paradigm in which a model acquires knowledge from a continuous, non-stationary stream of data over time, building upon previously learned experiences without losing what it has already mastered [1; 2]. This paradigm stands in stark contrast to the dominant tradition of batch learning, where a model is trained once on a fixed dataset drawn from a stationary distribution assumed to be independent and identically distributed (i.i.d.), and remains static thereafter. In real-world deployments, however, data rarely conforms to such idealized assumptions; instead, it arrives sequentially, often reflecting a shifting and evolving environment [3; 4].

### Core Terminology

To reason precisely about continual learning, it is essential to establish a clear vocabulary of foundational concepts:

- **Task**: A task is a well-defined learning problem, typically characterized by a dataset, a loss function, and an associated objective. Tasks may correspond to distinct classification problems, domains, or environments. In many formulations, tasks arrive in a sequence, one after another, and the learner is expected to perform well on all tasks encountered so far [2; 1].

- **Task Sequence**: A task sequence $\mathcal{T} = \{T_1, T_2, \ldots, T_n\}$ is an ordered collection of tasks presented to the learner over time. The ordering may be fixed or vary, and tasks may differ substantially in their data distributions, label spaces, or underlying generative processes [4; 5].

- **Experience**: Drawing inspiration from neuroscience and cognitive science, the term *experience* refers to a learning episode comprising a subset of data associated with a particular task or time step. An experience encapsulates the data distribution, supervision signal, and context available to the learner at a given point in the task sequence [2; 6].

- **Non-Stationarity**: A defining property of continual learning settings is that the data distribution $p_t(x, y)$ changes across time steps or tasks. This non-stationarity can manifest as covariate shift, label shift, concept drift, or the emergence of entirely new classes and domains [7; 8].

- **Knowledge Retention and Transfer**: Beyond merely learning new tasks, a continual learner is expected to retain previously acquired knowledge (avoiding *catastrophic forgetting*) and ideally leverage past experience to accelerate or improve learning on future tasks (i.e., *forward transfer*), while new learning may also refine representations useful for old tasks (i.e., *backward transfer*) [1; 9].

### Biological Inspiration

The notion of continual learning is deeply inspired by biological systems. Humans and animals effortlessly acquire new skills and knowledge throughout their lifetimes without catastrophically erasing prior competencies [1]. This biological capacity for lifelong learning motivates the central aspiration of the field: to endow artificial neural networks with the same graceful ability to accumulate, organize, and generalize knowledge over extended periods of time [10; 11]. As will be discussed in the following section, however, artificial neural networks conspicuously fail to replicate this biological property — a failure that manifests as catastrophic forgetting, the central challenge of the field.

### Distinction from Traditional Batch Learning

In classical supervised learning, it is assumed that all training data is available simultaneously and drawn i.i.d. from a fixed underlying distribution. Models are optimized in full knowledge of this static dataset, and their performance is evaluated on a held-out test set drawn from the same distribution. Continual learning fundamentally violates these assumptions: data arrives sequentially, previous data may no longer be accessible due to memory constraints or privacy requirements, and the distribution of incoming data can shift arbitrarily between tasks [3; 12]. This shift in assumptions necessitates entirely new algorithmic strategies, as standard gradient-based optimization on new data tends to overwrite previously learned representations — a phenomenon known as catastrophic forgetting that is examined in depth in Section 1.2 [2].

Formally, let a continual learning system receive a sequence of experiences $\{(D_1, \mathcal{L}_1), (D_2, \mathcal{L}_2), \ldots, (D_n, \mathcal{L}_n)\}$, where $D_t$ is the dataset and $\mathcal{L}_t$ is the loss function associated with task $T_t$. At each time step $t$, the learner has access only to $D_t$ (and potentially a small memory of past samples), and must update its parameters $\theta$ so as to minimize $\mathcal{L}_t$ while preserving performance on all previous tasks $T_1, \ldots, T_{t-1}$. This formulation captures the core challenge of continual learning: learning without forgetting, under strict constraints on data access and computational resources [12; 4].

The goal of continual learning, broadly stated, is therefore to develop learning systems that are simultaneously *plastic* — capable of efficiently incorporating new information — and *stable* — capable of retaining previously acquired knowledge — across an open-ended sequence of learning experiences [1; 13]. This dual requirement, and the inherent tension between these two properties, constitutes the foundation upon which the entire field is built. Understanding precisely why this tension is so difficult to resolve in practice requires a closer examination of the mechanisms underlying catastrophic forgetting, which we turn to next.

### 1.2 Catastrophic Forgetting: The Central Challenge

## 1.2 Catastrophic Forgetting: The Central Challenge

Catastrophic forgetting (CF) — sometimes termed catastrophic interference — is arguably the most fundamental obstacle confronting continual learning systems. As introduced in the preceding section, the core aspiration of continual learning is to develop systems that are simultaneously plastic and stable; catastrophic forgetting is precisely what happens when plasticity overwhelms stability. The phenomenon describes the abrupt and dramatic degradation of a neural network's performance on previously learned tasks when the network is subsequently trained on new tasks or data distributions. Unlike the gradual, selective forgetting observed in biological memory systems, which tend to retain important information while gracefully discarding peripheral details, artificial neural networks trained via gradient-based optimization can lose previously acquired knowledge almost entirely and almost instantaneously upon exposure to new training signals. This phenomenon was first identified in the context of connectionist models and has since been recognized as a pervasive, deep-seated challenge across virtually all neural network architectures and learning settings [14].

### Root Causes: Gradient-Based Optimization and Shared Parameterization

The origins of catastrophic forgetting lie in two intertwined properties of modern neural networks. First, gradient-based learning algorithms such as stochastic gradient descent (SGD) and its variants optimize network parameters to minimize the loss on the current batch of training data, with no explicit mechanism to preserve performance on data from prior tasks. When the training distribution shifts — as it does when a new task is introduced in a sequential learning scenario — the gradient updates are shaped entirely by the new data, and adjustments to shared parameters that were previously tuned for old tasks can rapidly overwrite the representations and decision boundaries those tasks relied upon [15]. Second, standard neural networks employ a shared parameter space across all tasks: the same weights that encode knowledge about task A are also modified during learning of task B. Because no structural separation is enforced between task-specific representations, updates beneficial for the new task frequently conflict with — and thereby destructively overwrite — the parameter configurations critical for prior tasks [16].

From a geometric perspective, each task defines a region of parameter space where its loss is acceptably low. These regions may overlap substantially when tasks are similar, but for dissimilar tasks they can occupy largely disjoint subspaces of the high-dimensional parameter landscape. Gradient descent during task B drives the parameters toward the loss basin of task B without regard for the loss basin of task A, and if these basins are far apart, the model exits the region of good performance for task A, yielding catastrophic forgetting [17]. The degree of overlap between task-specific loss minima, which is closely related to the functional and distributional similarity between tasks, therefore plays a central role in determining how severely forgetting manifests [18]. This geometric intuition also helps clarify the formal continual learning objective introduced in Section 1.1: the requirement to minimize $\mathcal{L}_t$ while

[... deterministic omitted span ...]

a principled approach to address this tension: by identifying parameters critical to general knowledge and applying differential learning rates or gradient constraints based on parameter importance, these methods enable selective plasticity that updates domain-relevant parameters while preserving those encoding foundational language understanding. These selective update strategies share conceptual grounding with the parameter-efficient tuning methods reviewed in Section 5.1, where similar principles of restricting which parameters are modified underpin adapter-based and prompt-based continual fine-tuning approaches.

Knowledge integration techniques offer complementary strategies by explicitly combining representations or knowledge structures from the pre-trained model and the domain-adapted version. These approaches draw conceptual parallels to regularization-based continual learning methods such as Elastic Weight Consolidation and Synaptic Intelligence, adapting the core principle of penalizing deviation from important prior parameters to the pre-training context. The Fisher information matrix and related second-order approximations of parameter importance, well-established in the broader continual learning literature, have been explored in the language model pre-training context to identify which parameters encode the most critical general knowledge and should therefore be most strongly protected during domain adaptation. As noted in Section 5.1, computing exact Fisher information matrices remains computationally prohibitive at billion-parameter scale, and this challenge is equally acute—if not more so—in the continual pre-training setting.

An important consideration in continual domain-adaptive pre-training is the sequential nature of domain exposure: models may be required to adapt to a series of new domains over time, each introducing its own distribution shift, without access to data from previously seen domains. This setting closely mirrors the task-incremental continual learning scenario and introduces the risk of cumulative forgetting across multiple domain shifts. Methods that maintain episodic memory buffers storing representative samples from prior domain corpora, or that employ generative mechanisms to synthesize pseudo-data approximating past distributions, provide practical solutions but introduce their own challenges related to memory capacity and sample fidelity. These concerns echo those encountered with replay-based strategies in supervised NLP settings, where memory constraints and sample representativeness similarly limit practical applicability. The difficulty of scaling these approaches to the massive parameter counts and data volumes characteristic of modern LLMs remains an active research concern.

Cross-domain knowledge transfer is another central benefit enabled by continual pre-training strategies. When a model has been pre-trained on a broad general corpus and subsequently adapted to specialized domains, forward transfer allows domain-specific knowledge to benefit downstream tasks in related areas even when those tasks have not been explicitly encountered. This positive transfer effect is facilitated by the compositional and hierarchical nature of linguistic representations learned during general pre-training, which provide reusable feature abstractions applicable across diverse specialized contexts. Understanding the conditions under which continual domain-adaptive pre-training yields positive versus negative transfer across domain sequences constitutes an important open question, with task similarity, domain overlap, and corpus size all playing significant roles.

The interplay between continual pre-training and instruction tuning introduces additional complexity that serves as a natural bridge to the instruction tuning challenges examined in Section 5.3: models that undergo domain-adaptive pre-training must subsequently be instruction-tuned for task-following behavior, and the ordering and interaction of these training phases can substantially affect both domain knowledge retention and instruction-following capability [26]. Empirical investigations have shown that catastrophic forgetting during continual fine-tuning of LLMs is widespread across model scales ranging from one billion to seven billion parameters, with forgetting severity tending to increase with model size, underscoring the importance of carefully managing the pre-training and adaptation pipeline to preserve the broad capabilities developed during initial pre-training [26]. This finding directly motivates the dedicated treatment of continual instruction tuning that follows.

Overall, continual pre-training and domain-adaptive pre-training represent a practically significant frontier in continual learning for NLP, offering pathways to maintain and expand the capabilities of large language models as new domain knowledge becomes available over time. The methodological principles developed here—selective parameter updating, importance-weighted regularization, and memory-augmented replay—form a coherent thread running from the supervised continual learning methods of Section 5.1 through to the instruction tuning paradigm of Section 5.3. The field continues to develop more principled methods for balancing general and domain-specific knowledge, reducing computational costs associated with large-scale continual pre-training, and ensuring that adapted models retain the broad linguistic competencies that make them valuable across diverse downstream applications.

### 5.3 Continual Instruction Tuning for Large Language Models

## 5.3 Continual Instruction Tuning for Large Language Models

Building upon the foundations established by continual pre-training and domain-adaptive pre-training, the emergence of instruction tuning as a dominant paradigm for aligning large language models (LLMs) with human intent has introduced a new and particularly challenging dimension to continual learning research. While continual pre-training focuses on preserving general linguistic representations across evolving domain corpora, continual instruction tuning must additionally safeguard the model's acquired capacity to follow natural language instructions—a capability that sits at a higher level of abstraction and is uniquely vulnerable to sequential fine-tuning. When LLMs are sequentially fine-tuned on diverse instruction-following datasets spanning different NLP tasks and domains, they exhibit pronounced catastrophic forgetting, whereby performance on previously mastered instruction templates degrades sharply as new task-specific instructions are absorbed. This phenomenon is distinct from conventional continual learning in that it involves not merely forgetting discriminative feature representations, but rather degrading the model's generalized capacity to follow natural language instructions—a capability painstakingly acquired during large-scale instruction pre-training.

The interplay between catastrophic forgetting and instruction alignment degradation is a central concern in this setting. Unlike standard task-incremental classification, instruction-tuned LLMs must maintain broad generalization across heterogeneous task formats, including summarization, commonsense reasoning, question answering, and dialogue, while continuously acquiring new instruction-following competencies. Sequential fine-tuning without appropriate safeguards tends to erode both task-specific knowledge and the meta-level instruction comprehension that constitutes the core value of instruction-tuned models. The stability-plasticity dilemma [29; 1] manifests here with particular severity because the distributional diversity of instruction datasets is substantially greater than that encountered in typical image classification benchmarks. This challenge is compounded by the empirical finding noted in the preceding section that catastrophic forgetting during continual fine-tuning is widespread across model scales ranging from one billion to seven billion parameters, with forgetting severity tending to increase with model size [26], making the management of instruction alignment a concern that grows more pressing as models scale.

To systematically evaluate and compare methods in this emerging paradigm, the Continual Instruction Tuning Benchmark (CITB) and related evaluation frameworks have been proposed. These benchmarks sequence multiple NLP datasets as distinct instruction-following tasks and measure both forward and backward transfer alongside forgetting metrics, providing a principled basis for comparing diverse algorithmic approaches. Evaluation dimensions typically encompass average accuracy across all seen tasks, backward transfer capturing performance recovery or degradation on past tasks, and forward transfer measuring how instruction learning on one task aids performance on future tasks.

A range of methodological families have been explored for continual instruction tuning. **Data replay** approaches maintain a buffer of exemplars from prior instruction datasets and interleave them with current task data during fine-tuning, a strategy directly analogous to experience replay in standard continual learning [29]. While conceptually straightforward, replay in the instruction tuning context demands careful curation to ensure that replayed examples faithfully represent the diversity of prior instruction formats rather than merely their surface statistics.

**Model expansion** strategies address forgetting by augmenting the LLM architecture with task-specific modules or adapter layers for each new instruction domain, thereby avoiding direct overwriting of shared parameters. Such approaches draw inspiration from parameter isolation principles [1] and have been adapted to the LLM setting through lightweight adapter insertion, mixture-of-experts layers, and low-rank decomposition modules that can be selectively activated based on task identity or inferred from instruction context.

**Regularization-based methods** penalize updates to parameters deemed important for prior instruction tasks, building on the foundational intuition of weight importance estimation [1]. In the LLM context, computing exact Fisher information matrices is computationally prohibitive given the scale of modern transformer architectures, motivating approximate importance estimation schemes and structured sparsity constraints that operate efficiently at billion-parameter scale. This challenge mirrors the difficulties encountered in continual domain-adaptive pre-training, where second-order importance measures must similarly be approximated to

[... deterministic omitted span ...]

Backward Knowledge  Transfer

[62] Schematic Memory Persistence and Transience for Efficient and Robust  Continual Learning

[63] Representation Ensembling for Synergistic Lifelong Learning with  Quasilinear Complexity

[64] Continual Learning of a Mixed Sequence of Similar and Dissimilar Tasks

[65] AFEC  Active Forgetting of Negative Transfer in Continual Learning

[66] Continual Learning in the Teacher-Student Setup  Impact of Task  Similarity

[67] A Conceptual Framework for Lifelong Learning

[68] Computationally Budgeted Continual Learning  What Does Matter 

[69] Real-Time Evaluation in Online Continual Learning  A New Hope

[70] Class-Incremental Continual Learning for General Purpose Healthcare  Models

[71] Cost-effective On-device Continual Learning over Memory Hierarchy with  Miro

[72] LifeLearner  Hardware-Aware Meta Continual Learning System for Embedded  Computing Platforms

[73] Continual Learning for Real-World Autonomous Systems  Algorithms,  Challenges and Frameworks

[74] Rehearsal-Free Continual Learning over Small Non-I.I.D. Batches

[75] PIVOT  Prompting for Video Continual Learning

[76] Preventing Zero-Shot Transfer Degradation in Continual Learning of  Vision-Language Models

[77] CLIP model is an Efficient Continual Learner

[78] Continual Learning for Autonomous Robots  A Prototype-based Approach

[79] Continual Learning through Human-Robot Interaction -- Human Perceptions  of a Continual Learning Robot in Repeated Interactions

[80] Exploring System Performance of Continual Learning for Mobile and  Embedded Sensing Applications

[81] Lifelong Adaptive Machine Learning for Sensor-based Human Activity  Recognition Using Prototypical Networks

[82] Learning Fast, Learning Slow  A General Continual Learning Method based  on Complementary Learning System

[83] DualNet  Continual Learning, Fast and Slow

[84] Deep Generative Dual Memory Network for Continual Learning

[85] Sparse Coding in a Dual Memory System for Lifelong Learning

[86] SYNERgy between SYNaptic consolidation and Experience Replay for general  continual learning

[87] On-device Synaptic Memory Consolidation using Fowler-Nordheim  Quantum-tunneling

[88] Continual learning benefits from multiple sleep mechanisms  NREM, REM,  and Synaptic Downscaling

[89] Wake-Sleep Consolidated Learning

[90] A Study on Efficiency in Continual Learning Inspired by Human Learning

[91] Learning to Modulate Random Weights  Neuromodulation-inspired Neural  Networks For Efficient Continual Learning

[92] Learning to Continually Learn

[93] Uncertainty-based Modulation for Lifelong Learning

[94] SHARP  Sparsity and Hidden Activation RePlay for Neuro-Inspired  Continual Learning

[95] Saliency-Guided Hidden Associative Replay for Continual Learning

[96] Task-Aware Information Routing from Common Representation Space in  Lifelong Learning

[97] Is Multi-Task Learning an Upper Bound for Continual Learning 

[98] Learning to Learn without Forgetting by Maximizing Transfer and  Minimizing Interference

[99] Self-Attention Meta-Learner for Continual Learning

[100] Meta-Consolidation for Continual Learning

[101] Recasting Continual Learning as Sequence Modeling

[102] Advances and Challenges in Meta-Learning  A Technical Review

[103] Online Continual Learning via the Knowledge Invariant and Spread-out  Properties

[104] Federated Continual Learning with Weighted Inter-client Transfer

[105] Masked Autoencoders are Efficient Continual Federated Learners

[106] A distillation-based approach integrating continual learning and  federated learning for pervasive services

[107] A Unified and General Framework for Continual Learning

[108] Continual Learning with Neuron Activation Importance

[109] Learning Bayesian Sparse Networks with Full Experience Replay for  Continual Learning

[110] Re-evaluating Continual Learning Scenarios  A Categorization and Case  for Strong Baselines

[111] Continual Learning in Human Activity Recognition  an Empirical Analysis  of Regularization

[112] GCR  Gradient Coreset Based Replay Buffer Selection For Continual  Learning

[113] Online Continual Learning with Maximally Interfered Retrieval

[114] The Effectiveness of Memory Replay in Large Scale Continual Learning

[115] Optimal Continual Learning has Perfect Memory and is NP-hard

[116] Practical Recommendations for Replay-based Continual Learning Methods

[117] Generative Negative Replay for Continual Learning

[118] Class-Incremental Learning Using Generative Experience Replay Based on  Time-aware Regularization

[119] Bayesian Structure Adaptation for Continual Learning

[120] Ada-QPacknet -- adaptive pruning with bit width reduction as an  efficient continual learning method without forgetting

[121] Continual Learning with Dynamic Sparse Training  Exploring Algorithms  for Effective Model Updates

[122] An Introduction to Lifelong Supervised Learning

[123] Continual Learning in Low-rank Orthogonal Subspaces

[124] The Ideal Continual Learner  An Agent That Never Forgets

[125] Hierarchical Decomposition of Prompt-Based Continual Learning   Rethinking Obscured Sub-optimality

[126] Read Between the Layers  Leveraging Intra-Layer Representations for  Rehearsal-Free Continual Learning with Pre-Trained Models

[127] Adversarial Continual Learning

[128] Facilitating Bayesian Continual Learning by Natural Gradients and Stein  Gradients

[129] Experience Replay for Continual Learning

[130] AdaER  An Adaptive Experience Replay Approach for Continual Lifelong  Learning

[131] Prototype-Guided Memory Replay for Continual Learning

[132] Continual Classification Learning Using Generative Models

[133] Variational Prototype Replays for Continual Learning

[134] Continual Learning via Online Leverage Score Sampling

[135] Learning to Prompt for Continual Learning

[136] Convolutional Prompting meets Language Models for Continual Learning

[137] Towards guarantees for parameter isolation in continual learning

[138] Continual Reinforcement Learning with Complex Synapses

[139] Overcoming Catastrophic Forgetting by Incremental Moment Matching

[140] Task Agnostic Continual Learning Using Online Variational Bayes

[141] An Empirical Investigation of the Role of Pre-training in Lifelong  Learning

[142] Create and Find Flatness  Building Flat Training Spaces in Advance for  Continual Learning

[143] Explain to Not Forget  Defending Against Catastrophic Forgetting with  XAI

[144] Premonition  Using Generative Models to Preempt Future Data Changes in  Continual Learning

[145] Sub-network Discovery and Soft-masking for Continual Learning of Mixed  Tasks

[146] Investigating the Impact of Weight Sharing Decisions on Knowledge  Transfer in Continual Learning

[147] Preserving Earlier Knowledge in Continual Learning with the Help of All  Previous Feature Extractors

[148] Efficient Rehearsal Free Zero Forgetting Continual Learning using  Adaptive Weight Modulation

[149] Continual Learning with Adaptive Weights (CLAW)

[150] Mixture-of-Variational-Experts for Continual Learning

[151] Hierarchically Structured Task-Agnostic Continual Learning

[152] Contrastive Continual Learning with Feature Propagation

[153] Prior-Free Continual Learning with Unlabeled Data in the Wild

[154] Online Continual Learning under Extreme Memory Constraints

[155] Distal Interference  Exploring the Limits of Model-Based Continual  Learning

[156] Probing Representation Forgetting in Supervised and Unsupervised  Continual Learning

[157] CoSCL  Cooperation of Small Continual Learners is Stronger than a Big  One

[158] Balancing Stability and Plasticity through Advanced Null Space in  Continual Learning

[159] Learning a Low-Rank Feature Representation  Achieving Better Trade-Off  between Stability and Plasticity in Continual Learning

[160] Entropy-based Stability-Plasticity for Lifelong Learning

[161] Learn to Bind and Grow Neural Structures

[162] SCL(FOL) Revisited

[163] Quantum continual learning of quantum data realizing knowledge backward  transfer

[164] Batch Model Consolidation  A Multi-Task Model Consolidation Framework

[165] Continual Learning on a Diet  Learning from Sparsely Labeled Streams  Under Constrained Computation

[166] Continual Learning for Recurrent Neural Networks  an Empirical  Evaluation

[167] DualPrompt  Complementary Prompting for Rehearsal-free Continual  Learning

[168] Continuum  Simple Management of Complex Continual Learning Scenarios

[169] Online Continual Learning on Sequences

[170] SimCS  Simulation for Domain Incremental Online Continual Segmentation

[171] Essentials for Class Incremental Learning

[172] Gradient Episodic Memory with a Soft Constraint for Continual Learning

[173] Reducing catastrophic forgetting with learning on synthetic data

[174] Challenging Common Assumptions about Catastrophic Forgetting

[175] Centroids Matching  an efficient Continual Learning approach operating  in the embedding space

[176] Deep SimNets

[177] Representational Continuity for Unsupervised Continual Learning

[178] Regularization Shortcomings for Continual Learning

[179] Scalable Language Model with Generalized Continual Learning

[180] Introducing Language Guidance in Prompt-based Continual Learning

[181] Evaluating and Improving Continual Learning in Spoken Language  Understanding

[182] Diffusion-based neuromodulation can eliminate catastrophic forgetting in  simple neural networks

[183] Progress & Compress  A scalable framework for continual learning

[184] Unicorn  Continual Learning with a Universal, Off-policy Agent

[185] On robustness of generative representations against catastrophic  forgetting

[186] New Insights on Relieving Task-Recency Bias for Online Class Incremental  Learning

[187] Asynchronous Federated Continual Learning

[188] Neuro-mimetic Task-free Unsupervised Online Learning with Continual  Self-Organizing Maps

[189] Learning to Continually Learn Rapidly from Few and Noisy Data

[190] Continual learning  a feature extraction formalization, an efficient  algorithm, and fundamental obstructions

[191] Online Continual Learning on Hierarchical Label Expansion

[192] AdaptCL  Adaptive Continual Learning for Tackling Heterogeneity in  Sequential Datasets

[193] Continual Prompt Tuning for Dialog State Tracking

[194] Achieving Forgetting Prevention and Knowledge Transfer in Continual  Learning

[195] Generative Continual Concept Learning

[196] Hebbian Continual Representation Learning

[197] Cognitively Inspired Learning of Incremental Drifting Concepts

[198] Self-paced Weight Consolidation for Continual Learning

[199] Sequoia  A Software Framework to Unify Continual Learning Research

[200] SequeL  A Continual Learning Library in PyTorch and JAX

[201] Continual Learning in Medical Image Analysis  A Comprehensive Review of  Recent Advancements and Future Prospects
