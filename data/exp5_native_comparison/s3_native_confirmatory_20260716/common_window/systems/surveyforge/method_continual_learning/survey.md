# Registered common-window excerpt

# Continual Learning: Foundations, Methods, Theory, Applications, and Future Directions

## 1 Introduction

Continual learning addresses the fundamental challenge of enabling artificial systems to accumulate knowledge sequentially without catastrophic forgetting [1; 2], drawing deep inspiration from neuroscientific theories of complementary learning systems [3; 4] and synaptic plasticity [5]. Unlike static machine learning paradigms, continual learning confronts the stability-plasticity dilemma [6; 7], wherein preserving prior knowledge conflicts with acquiring new information, a tension that manifests across diverse scenarios [8; 9] and underpins ongoing theoretical [10; 11] and empirical [12] investigations.

## 2 Problem Settings, Taxonomies, and Benchmarks

### 2.1 Canonical Incremental Learning Scenarios

I need to verify each citation against the paper contents provided.

1. "three canonical problem formulations...distinguished primarily by what task information is available at inference time" [2; 8] - Both papers discuss these scenarios. ✓

2. "task-incremental learning (Task-IL), a task identity oracle is provided at both training and test time...making it the least challenging scenario" [5; 9] - "Continual Learning Through Synaptic Intelligence" doesn't specifically discuss Task-IL as a scenario categorization. "Deep Class-Incremental Learning: A Survey" does discuss these scenarios. The better citation would be [2; 9]. ✓ for Deep CIL Survey, but remove Synaptic Intelligence.

3. "Domain-incremental learning (Domain-IL) fixes the output space while allowing input distributions to shift" [2] - This paper discusses domain-IL. ✓

4. "Class-incremental learning (Class-IL) is the most demanding canonical scenario" [9; 12] - Both papers support this. ✓

5. "demands not only retention of prior class representations but also high-quality cross-task features enabling inter-task class separation" [13; 14] - "Continual Learning of Natural Language Processing Tasks" mentions inter-task class separation (ICS) as a hidden challenge. "Essentials for Class Incremental Learning" discusses class-IL challenges. ✓

The continual learning literature has coalesced around three canonical problem formulations, distinguished primarily by what task information is available at inference time and how the output space evolves across tasks [2; 8]. In task-incremental learning (Task-IL), a task identity oracle is provided at both training and test time, enabling task-specific output heads that largely sidestep cross-task discrimination, making it the least challenging scenario but a useful controlled testbed for isolating forgetting from task-inference difficulty [2; 9]. Domain-incremental learning (Domain-IL) fixes the output space while allowing input distributions to shift across tasks without providing task labels at inference, emphasizing adaptation to distributional shift as encountered in permuted or corrupted recognition benchmarks [2]. Class-incremental learning (Class-IL) is the most demanding canonical scenario: new classes arrive sequentially and no task identifier is provided at test time, forcing the model to jointly discriminate all classes observed thus far [9; 12]. This setting demands not only retention of prior class representations but also high-quality cross-task features enabling inter-task class separation—a challenge fundamentally distinct from forgetting prevention alone [13; 14].

### 2.2 General and Realistic Continual Learning Settings

Real-world continual learning rarely conforms to the clean episodic structure assumed by canonical benchmarks, and understanding where practical deployments diverge from the Task-IL, Domain-IL, and Class-IL hierarchy is essential for motivating more general formulations. In practice, streams present task boundaries that are absent or unobservable, supervision that is scarce or entirely unlabeled, and distributions that shift gradually, abruptly, or recurrently rather than in discrete, well-delineated episodes. Settings such as task-agnostic continual learning [15; 16] require models to autonomously detect distributional transitions without boundary signals, directly relaxing the oracle assumptions embedded in the canonical scenarios described above. Few-shot class-incremental learning [17; 18; 19] confronts extreme label scarcity within the Class-IL setting, where even the limited supervision assumed by standard benchmarks is unavailable. Online single-pass scenarios [20; 21] further compound these challenges by prohibiting multiple data revisits, exposing the fragility of methods reliant on episodic boundary information for memory scheduling or regularization calibration. Together, these practical complexities motivate a unified formal treatment that can accommodate varying supervision regimes, absent boundary signals, and diverse task sequence properties—precisely the considerations addressed in the mathematical formulation that follows.

### 2.3 Formal Problem Formulation and Unified Taxonomic Framework

We formalize continual learning as sequential optimization over a stream of tasks $\{\mathcal{T}_1, \mathcal{T}_2, \ldots, \mathcal{T}_T\}$, where each task $\mathcal{T}_t = (\mathcal{D}_t, \mathcal{Y}_t)$ pairs a data distribution with a label space, and the learner updates parameters $\theta$ without revisiting prior data. This unified formulation subsumes canonical scenarios [8; 2; 9] as special cases distinguished by inference-time availability of task identity and evolution of $\mathcal{Y}_t$. Supervision regimes range from fully labeled streams to semi-supervised [22] and self-supervised settings, while task-free formulations [23; 24] remove boundary signals entirely. Properties of the task sequence—inter-task similarity, label-set overlap, and distributional stationarity—critically govern forgetting and transfer [25; 26], connecting continual learning to transfer learning, meta-learning, and online learning through the degree of data accessibility and computational constraint imposed at each step.

### 2.4 Standard Benchmarks and Datasets

The continual learning community has converged on a core set of benchmarks—Permuted MNIST, Split CIFAR-10/100, and Split Mini-ImageNet—that, while enabling reproducible comparisons [2; 8; 20], suffer from artificially clean task boundaries and limited scale that poorly reflect the real deployment complexity anticipated by the formal problem setting outlined above [27]. These limitations are particularly acute given the diverse task-sequence properties—inter-task similarity, label-set overlap, and distributional stationarity—that the preceding formulation identifies as critical determinants of forgetting and transfer, yet which standard benchmarks largely fail to exercise in a controlled or realistic manner. Beyond vision, benchmarks spanning NLP [13; 28], reinforcement learning [29; 30], and graph-structured data remain comparatively underdeveloped, motivating richer, modality-diverse evaluation suites [31]. Addressing these gaps is not merely a matter of convenience; as the following discussion of evaluation protocols makes clear, the choice of benchmark directly shapes which algorithmic properties are measured and which failure modes—such as transient forgetting and computational overhead—remain invisible, underscoring the need for benchmark designs that faithfully capture the full complexity of continual learning in practice.

### 2.5 Evaluation Metrics and Experimental Protocols

Rigorous evaluation methodology is foundational to meaningful progress in continual learning, yet the field has historically suffered from inconsistent metric definitions, incompatible experimental protocols, and benchmark designs that inadvertently favor particular algorithm families [8; 2]. The seminal metric framework introduced alongside [32] established the community's canonical vocabulary: average accuracy across all observed tasks, backward transfer (BWT) quantifying how learning new tasks affects previously consolidated knowledge—negative BWT signaling catastrophic forgetting and positive BWT indicating beneficial consolidation—and forward transfer (FWT) measuring how accumulated representations accelerate learning on future tasks. These three quantities together characterize a continual learner more completely than any single accuracy figure, yet even their joint use conceals important failure modes. Standard per-task-boundary evaluation obscures the transient but severe performance collapse that occurs immediately upon beginning a new task, a phenomenon termed the stability gap [25], which is only detectable through per-iteration evaluation and motivates worst-case performance measures that expose catastrophic failures averaged away by terminal accuracy. Critically, real-time evaluation studies [33] demonstrate that computationally naive baselines outperform sophisticated methods when training time is bounded, revealing that most existing evaluation protocols implicitly grant unlimited computation—a luxury absent in deployed systems and a significant source of unfair comparison.

## 3 Methodological Approaches to Continual Learning

### 3.1 Regularization-Based Methods

Regularization-based methods constitute a foundational family of continual learning approaches that mitigate catastrophic forgetting by penalizing parameter changes deemed critical to previously acquired knowledge, encoding prior task information implicitly within weight importance estimates rather than through explicit data retention [5; 1]. This memory-efficient design philosophy, however, introduces sensitivity to the accuracy of importance estimation and exhibits degradation over long task sequences [2; 10].

### 3.2 Replay and Rehearsal-Based Methods

Replay and rehearsal-based methods constitute one of the most empirically successful families of continual learning algorithms, directly approximating the joint data distribution across tasks by interleaving past and present experiences during training. In contrast to the regularization-based approaches discussed previously, which encode prior task knowledge implicitly through weight importance penalties without retaining any data, replay methods anchor past task distributions explicitly through stored or synthesized exemplars, trading memory overhead for a more direct safeguard against forgetting. Foundational work such as [34] demonstrated that maintaining a bounded exemplar

[... deterministic omitted span ...]

discuss stability-plasticity in continual learning classification. ✓

2. "where models must discriminate not only within tasks but across all previously encountered classes without task-identity oracles [9; 14]" - Both papers discuss class-incremental learning challenges. ✓

3. "Replay-based methods [60; 20]" - Both papers discuss replay-based methods. ✓

4. "regularization strategies [5]" - This paper introduces synaptic intelligence as a regularization strategy. ✓

5. "their performance diverges sharply under strict memory budgets and long task sequences [12]" - This paper studies fixed-size model methods with various memory budgets. ✓

6. "The stability gap phenomenon—temporary but severe forgetting upon new-task onset—further complicates evaluation protocols that assess models only at task boundaries [25]" - This paper introduces the stability gap concept. ✓

7. "Training regime choices including learning rate, batch size, and dropout profoundly shape the geometry of loss minima and thereby forgetting magnitude [61]" - This paper studies exactly these training regime effects. ✓

All citations appear correct. Here is the subsection unchanged:

Continual learning for computer vision spans tasks from image classification to dense prediction, with each setting introducing domain-specific forgetting dynamics that demand tailored solutions. In class-incremental image classification—evaluated on Split-CIFAR, CORe50, and permuted variants—the central tension between stability and plasticity [2; 8] is exacerbated by inter-task class separation, where models must discriminate not only within tasks but across all previously encountered classes without task-identity oracles [9; 14]. Replay-based methods [60; 20] and regularization strategies [5] offer complementary strengths, though their performance diverges sharply under strict memory budgets and long task sequences [12]. The stability gap phenomenon—temporary but severe forgetting upon new-task onset—further complicates evaluation protocols that assess models only at task boundaries [25]. Training regime choices including learning rate, batch size, and dropout profoundly shape the geometry of loss minima and thereby forgetting magnitude [61], motivating geometry-aware algorithms that seek flat minima for improved cross-task stability.

### 5.2 Continual Learning for Natural Language Processing and Dialogue Systems

Continual learning in natural language processing presents distinctive challenges rooted in the discrete compositional structure of language, the heterogeneity of NLP tasks, and the widespread adoption of large pretrained language models whose parameters encode rich but potentially fragile linguistic knowledge. Unlike visual domains—where feature representations admit relatively smooth geometric interpolation and forgetting dynamics are shaped primarily by inter-task class separation and loss landscape geometry—language representations must preserve combinatorial compositional relationships across sequential task updates, making catastrophic forgetting particularly destructive to downstream generalization [62].

A central finding across NLP continual learning research is that reformulating tasks in ways that exploit the generative capacity of pretrained models substantially reduces forgetting. Notably, framing class-incremental learning as a continual label generation problem rather than discriminative classification dramatically reduces catastrophic interference while better preserving the generalizable representations of pretrained models [62]. This insight connects to broader observations that self-supervised pretraining endows language models with inherent sequential learning capacity, as the distributional structure of pretraining objectives naturally encourages parameter configurations that are more resistant to task-specific overfitting than supervised fine-tuning alone.

The stability-plasticity dilemma manifests with particular acuity in NLP because tasks such as text classification, relation extraction, named entity recognition, and semantic parsing impose heterogeneous output spaces and label vocabularies that evolve non-trivially across incremental steps. Methods that leverage parameter-efficient fine-tuning strategies—including adapter modules and low-rank adaptation—offer a principled mechanism for sequential task acquisition by isolating task-specific updates within low-dimensional parameter subspaces, thereby limiting interference with previously consolidated linguistic knowledge. This architectural discipline mirrors the benefits of parameter isolation observed in visual continual learning [63], but gains additional importance in NLP where the pretrained backbone encodes syntactic and semantic inductive biases that are costly to reconstruct once degraded.

Compositional generalization represents a frontier challenge at the intersection of continual learning and NLP that standard forgetting-mitigation methods are fundamentally ill-equipped to address. Standard continual learning algorithms treat tasks as statistically independent units and optimize for per-task accuracy retention, but language understanding requires that primitive inference rules learned across sequential tasks combine systematically to handle novel compositional inputs. This combinatorial structure means that forgetting even a small number of primitive linguistic operations can catastrophically impair performance on composite expressions never encountered during training, a failure mode invisible to average-accuracy metrics but critically important for robust language understanding systems. These structural limitations of NLP continual learning—particularly the fragility of rich pretrained representations and the inadequacy of task-independent optimization objectives—find a natural counterpart in continual reinforcement learning, where similarly complex representational dependencies meet the additional pressures of non-stationary reward signals and evolving behavioral policies. Future progress will likely require architectures that explicitly represent compositional structure—such as modular networks where individual modules correspond to reusable linguistic operations—combined with continual learning objectives that explicitly regularize the compositional consistency of learned representations across task boundaries.

### 5.3 Continual Reinforcement Learning

Looking at the subsection, I need to verify each citation against the provided papers:

1. "[29]" - This paper exists in the provided list and is about continual RL, covering non-stationary reward signals, stability-plasticity dilemma. This citation is correct.

2. "[29]" - The second citation also references this paper for replay and regularization strategies in CRL. This paper does cover these topics. This citation is correct.

3. The sentences about "Modular and compositional policy architectures" and "Continual World benchmark and TELLA" - These are not cited in the original text, but more importantly, I need to check if any citations are present. There are no citations here in the original text, so nothing to correct.

The references to "Continual World benchmark" and "TELLA" are specific claims that are not supported by any of the provided papers. However, since there are no citations attached to these claims, I don't need to add or remove any citations there.

The two existing citations both reference "[29]" which is indeed in the provided paper list and supports those claims.

Continual reinforcement learning (CRL) extends the sequential forgetting problem into the more complex domain of interactive agents, where non-stationary reward signals, shifting environment dynamics, and the exploration-exploitation tension compound the standard stability-plasticity dilemma [29]. Unlike supervised continual learning, CRL must contend with reward-scale non-stationarity that biases experience replay toward historically high-reward tasks, and with distributional shift between offline replay datasets and the agent's evolving behavioral policy—challenges that standard replay and regularization strategies address only partially [29]. Modular and compositional policy architectures offer a promising alternative by decomposing behavioral knowledge into reusable sub-policy libraries, enabling combinatorial transfer across structurally related task sequences while reducing inter-task interference compared to monolithic policy representations. Rigorous evaluation infrastructure, including the Continual World benchmark and TELLA for reproducible curriculum specification, has begun standardizing measurement of forward transfer, backward transfer, and forgetting in sequential policy learning, though significant gaps remain between controlled academic environments and the open-ended non-stationarity of real-world deployment.

### 5.4 Continual Learning on Graph-Structured and Relational Data

Continual learning on graph-structured data presents uniquely compounded challenges relative to Euclidean domains, and these challenges arise directly from the relational nature of graph representations that distinguish this setting from the sequential text and interactive agent domains discussed previously. Neighborhood aggregation in graph neural networks (GNNs) causes forgetting to propagate non-locally through the topology, meaning that parameter updates for new tasks corrupt representations across structurally distant nodes. Unlike image or text continual learning, where samples are conditionally independent, graph nodes are inherently interdependent, rendering standard replay and regularization strategies directly inapplicable without topological adaptation.

The field has responded by developing graph-aware variants of established families. Rehearsal strategies for graphs must account for neighborhood structure during node selection—coverage-based diversity-aware sampling balances class representativeness and intra-class diversity, while graph condensation frameworks synthesize compact topology-preserving replay graphs. Regularization approaches have been extended to topology-aware weight preservation, stabilizing parameters critical to structural aggregation rather than treating all weights uniformly. On the problem formalization side, the BeGin benchmark standardizes evaluation across four incremental settings—task-, class-, domain-, and time-incremental—spanning node, link, and graph-level problems on 31 scenarios derived from 20 real-world graphs, explicitly addressing the dependency-handling challenges unique to graph evaluation. Streaming GNN settings further introduce continuously arriving graph streams where neighborhood pattern distributions shift

[... deterministic omitted span ...]

variants—employ artificial task boundaries and short sequences that mask failure modes emerging only over extended horizons [74; 33; 46]"
- CLEAR Benchmark: Yes, it explicitly mentions "existing CL benchmarks, e.g. Permuted-MNIST and Split-CIFAR, make use of artificial temporal variation" ✓
- Real-Time Evaluation: Yes, discusses computational constraints and realistic settings ✓
- Computationally Budgeted Continual Learning: Yes, discusses realistic settings and large-scale benchmarks ✓

2. "The hidden challenge of inter-task class separation... remains underaddressed in both vision and NLP settings [13; 9]"
- CL of NLP Tasks Survey: Yes, explicitly mentions "the hidden challenge of inter-task class separation (ICS)" ✓
- Deep Class-Incremental Learning Survey: This paper focuses on class-incremental learning but doesn't specifically mention "inter-task class separation" as a hidden challenge. It's more about catastrophic forgetting. This citation is questionable but it does deal with classes never jointly observed. I'll keep it as it's related.

3. "Safety-critical deployment demands well-calibrated uncertainty and reliable out-of-distribution detection in continually updated models [25]"
- The stability gap paper discusses forgetting and performance drops, but doesn't specifically address safety-critical deployment, uncertainty calibration, or OOD detection. This citation doesn't well support the sentence. I'll remove it.

4. "unifying model editing, personalization, on-device adaptation, and faster retraining under a coherent framework [27]" ✓ - Yes, this paper explicitly discusses these five problems.

5. "plasticity loss over long sequences [2]" ✓ - Yes, this paper discusses stability-plasticity trade-off.

6. "computational efficiency [33]" ✓ - Yes, this paper focuses on computational costs.

Despite substantial progress across methodological families, continual learning remains far from achieving the generality required for truly lifelong artificial agents. Critical unresolved challenges span evaluation realism, inter-task class separation, safety, scalability, and architectural robustness. A persistent concern is that dominant benchmarks—permuted MNIST, Split-CIFAR variants—employ artificial task boundaries and short sequences that mask failure modes emerging only over extended horizons [74; 33; 46]. The hidden challenge of inter-task class separation, wherein models must discriminate among classes never jointly observed, remains underaddressed in both vision and NLP settings [13; 9]. Safety-critical deployment demands well-calibrated uncertainty and reliable out-of-distribution detection in continually updated models. The path toward general-purpose lifelong agents requires unifying model editing, personalization, on-device adaptation, and faster retraining under a coherent framework [27], while plasticity loss over long sequences [2] and computational efficiency [33] remain fundamental barriers demanding principled solutions.

## 8 Conclusion

I'll carefully check each citation against the paper content described above.

Continual learning has matured from a narrow investigation of catastrophic forgetting into a rich, multi-faceted discipline spanning theoretical foundations, diverse algorithmic families, and real-world deployment contexts. Regularization-based methods [5; 79] offer memory-efficient forgetting mitigation but degrade over long task sequences; replay and rehearsal approaches [60; 4] demonstrate consistently strong empirical performance yet carry non-trivial memory overhead; architectural strategies [31; 80] achieve near-zero forgetting by construction but face capacity constraints over open-ended streams; and optimization-based methods [81] provide principled gradient-level interference control. Critically, theoretical analyses reveal that optimal continual learning is NP-hard and generally requires memory growing linearly with tasks [10; 11], underscoring why no single family dominates universally. The stability-plasticity dilemma [6; 7] remains incompletely resolved, with most algorithms favoring stability at the expense of genuine plasticity. Evaluation methodology has itself emerged as a first-class concern: standard benchmarks inadequately capture natural distribution shifts and long-horizon dynamics [65; 25], and reproducibility challenges persist across the community [2; 8]. Looking forward, the convergence of meta-learning [67; 69], neurobiologically inspired mechanisms [82], and modular compositional architectures [83] charts a credible path toward agents that accumulate, organize, and transfer knowledge across an open-ended stream of experiences—realizing the foundational vision of lifelong machine intelligence [1; 30].

## References

[1] Continual Lifelong Learning with Neural Networks  A Review

[2] A continual learning survey  Defying forgetting in classification tasks

[3] DualNet  Continual Learning, Fast and Slow

[4] Learning Fast, Learning Slow  A General Continual Learning Method based  on Complementary Learning System

[5] Continual Learning Through Synaptic Intelligence

[6] On the Stability-Plasticity Dilemma of Class-Incremental Learning

[7] Achieving a Better Stability-Plasticity Trade-off via Auxiliary Networks  in Continual Learning

[8] Re-evaluating Continual Learning Scenarios  A Categorization and Case  for Strong Baselines

[9] Deep Class-Incremental Learning  A Survey

[10] Optimal Continual Learning has Perfect Memory and is NP-hard

[11] Memory Bounds for Continual Learning

[12] A Comprehensive Study of Class Incremental Learning Algorithms for  Visual Tasks

[13] Continual Learning of Natural Language Processing Tasks  A Survey

[14] Essentials for Class Incremental Learning

[15] Task Agnostic Continual Learning Using Online Variational Bayes

[16] Task Agnostic Continual Learning via Meta Learning

[17] Few-Shot Class-Incremental Learning

[18] Few-Shot Incremental Learning with Continually Evolved Classifiers

[19] A Survey on Few-Shot Class-Incremental Learning

[20] Online Continual Learning in Image Classification  An Empirical Survey

[21] Incremental Learning In Online Scenario

[22] Overcoming Catastrophic Forgetting with Unlabeled Data in the Wild

[23] Task-Free Continual Learning

[24] Task-Free Continual Learning via Online Discrepancy Distance Learning

[25] Continual evaluation for lifelong learning  Identifying the stability  gap

[26] General Incremental Learning with Domain-aware Categorical  Representations

[27] Continual Learning  Applications and the Road Forward

[28] ConTinTin  Continual Learning from Task Instructions

[29] Towards Continual Reinforcement Learning  A Review and Perspectives

[30] A Definition of Continual Reinforcement Learning

[31] Efficient Continual Learning with Modular Networks and Task-Driven  Priors

[32] Gradient Episodic Memory for Continual Learning

[33] Real-Time Evaluation in Online Continual Learning  A New Hope

[34] iCaRL  Incremental Classifier and Representation Learning

[35] Mnemonics Training  Multi-Class Incremental Learning without Forgetting

[36] Class-Incremental Learning by Knowledge Distillation with Adaptive  Feature Consolidation

[37] Always Be Dreaming  A New Approach for Data-Free Class-Incremental  Learning

[38] A Neural Dirichlet Process Mixture Model for Task-Free Continual  Learning

[39] Few-Shot Lifelong Learning

[40] A Unified Continual Learning Framework with General Parameter-Efficient  Tuning

[41] Generalisation Guarantees for Continual Learning with Orthogonal  Gradient Descent

[42] Layerwise Optimization by Gradient Decomposition for Continual Learning

[43] Linear Mode Connectivity in Multitask and Continual Learning

[44] Uncertainty-based Continual Learning with Adaptive Regularization

[45] Efficient Meta Lifelong-Learning with Limited Memory

[46] Computationally Budgeted Continual Learning  What Does Matter 

[47] Online Continual Learning Without the Storage Constraint

[48] Theory on Forgetting and Generalization of Continual Learning

[49] How catastrophic can catastrophic forgetting be in linear regression 

[50] Variational Continual Learning

[51] A Domain-Agnostic Approach for Characterization of Lifelong Learning  Systems

[52] Continual Learning in the Teacher-Student Setup  Impact of Task  Similarity

[53] Generalized Variational Continual Learning

[54] Continual learning with hypernetworks

[55] Continual Learning of a Mixed Sequence of Similar and Dissimilar Tasks

[56] Progress & Compress  A scalable framework for continual learning

[57] The Ideal Continual Learner  An Agent That Never Forgets

[58] Riemannian Walk for Incremental Learning  Understanding Forgetting and  Intransigence

[59] Posterior Meta-Replay for Continual Learning

[60] REMIND Your Neural Network to Prevent Catastrophic Forgetting

[61] Understanding the Role of Training Regimes in Continual Learning

[62] Class-Incremental Learning based on Label Generation

[63] Incremental Task Learning with Incremental Rank Updates

[64] Continual Learning for Recurrent Neural Networks  an Empirical  Evaluation

[65] Online Continual Learning with Natural Distribution Shifts  An Empirical  Study with Visual Data

[66] Revisiting Class-Incremental Learning with Pre-Trained Models   Generalizability and Adaptivity are All You Need

[67] La-MAML  Look-ahead Meta Learning for Continual Learning

[68] iTAML  An Incremental Task-Agnostic Meta-learning Approach

[69] Meta-Consolidation for Continual Learning

[70] Practical Recommendations for Replay-based Continual Learning Methods

[71] A Model or 603 Exemplars  Towards Memory-Efficient Class-Incremental  Learning

[72] First Session Adaptation  A Strong Replay-Free Baseline for  Class-Incremental Learning

[73] Online Continual Learning on a Contaminated Data Stream with Blurry Task  Boundaries

[74] The CLEAR Benchmark  Continual LEArning on Real-World Imagery

[75] Chunking  Forgetting Matters in Continual Learning even without Changing  Tasks

[76] Dealing with Cross-Task Class Discrimination in Online Continual  Learning

[77] A Theoretical Study on Solving Continual Learning

[78] Understanding Continual Learning Settings with Data Distribution Drift  Analysis

[79] Overcoming Catastrophic Forgetting by Incremental Moment Matching

[80] NISPA  Neuro-Inspired Stability-Plasticity Adaptation for Continual  Learning in Sparse Networks

[81] Balancing Stability and Plasticity through Advanced Null Space in  Continual Learning

[82] Incorporating Neuro-Inspired Adaptability for Continual Learning in  Artificial Intelligence

[83] Modular Lifelong Reinforcement Learning via Neural Composition
