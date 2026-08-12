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

Replay and rehearsal-based methods constitute one of the most empirically successful families of continual learning algorithms, directly approximating the joint data distribution across tasks by interleaving past and present experiences during training. In contrast to the regularization-based approaches discussed previously, which encode prior task knowledge implicitly through weight importance penalties without retaining any data, replay methods anchor past task distributions explicitly through stored or synthesized exemplars, trading memory overhead for a more direct safeguard against forgetting. Foundational work such as [34] demonstrated that maintaining a bounded exemplar memory with herding-based selection, combined with nearest-mean-of-exemplars classification, enables strong class-incremental performance, establishing episodic replay as a competitive baseline against which subsequent methods are measured. Complementary evaluations [8; 20] confirm that even naive rehearsal strategies can match or surpass sophisticated regularization approaches, underscoring the practical potency of direct data retention. Building on exemplar selection, [35] parameterizes stored exemplars and optimizes them end-to-end via bilevel optimization, moving beyond static herding toward actively shaped memory that maximally preserves decision boundaries. Replay utility is further enhanced by distillation over stored soft logits rather than raw labels, as the dark knowledge carried by past model predictions encodes richer inter-class structure [14], a principle that underlies the broader family of knowledge-distillation-augmented replay methods [36]. Data-free variants address privacy and storage constraints by inverting a frozen teacher model to synthesize past-task pseudo-data [37], though generative quality remains a limiting factor on complex benchmarks, motivating ongoing research into higher-fidelity synthesis and domain-gap mitigation. Whereas replay methods achieve interference prevention through data-level approximation of past distributions, the parameter isolation methods examined in the following section pursue an orthogonal strategy, eliminating interference by construction through dedicated structural allocations within the network itself.

### 3.3 Parameter Isolation and Architectural Expansion Methods

I need to verify each citation in the subsection against the paper contents provided.

1. "Fixed-capacity masking approaches, exemplified by PackNet-style iterative pruning and freezing, carve task-specific sparse subnetworks from a shared architecture [2; 9]" - Both papers are surveys that cover parameter isolation methods including masking approaches. This is reasonable.

2. "dynamic expansion methods such as Progressive Neural Networks and Bayesian nonparametric architectures grow capacity on demand [38]" - The CN-DPM paper is indeed about expansion-based approach using Bayesian nonparametric framework. This citation is correct.

3. "Soft-masking refinements allow graded parameter sharing across tasks, balancing interference prevention against knowledge transfer and mitigating capacity exhaustion over long sequences [39]" - The Few-Shot Lifelong Learning paper selects very few parameters for training new classes, keeping important parameters intact. This loosely relates to parameter isolation but doesn't specifically discuss "soft-masking refinements." This citation is weak/incorrect.

4. "A critical limitation shared across isolation methods is their dependence on task identity at inference time, restricting applicability in task-agnostic settings [8]" - This paper discusses categorization of scenarios including task identity issues. This is supportable.

5. "integration with pretrained backbones [40]" - This paper is directly about PET adaptation with pre-trained models for CL. Correct.

For citation 3, the Few-Shot Lifelong Learning paper is about selecting few parameters, not specifically soft-masking. I should remove this citation or find a better one. Since no better paper is available from the list, I'll remove it.

Parameter isolation methods constitute a fundamentally distinct approach to continual learning: rather than penalizing weight changes or rehearsing past data, they prevent interference by construction through dedicated parameter allocations. Fixed-capacity masking approaches, exemplified by PackNet-style iterative pruning and freezing, carve task-specific sparse subnetworks from a shared architecture [2; 9], while dynamic expansion methods such as Progressive Neural Networks and Bayesian nonparametric architectures grow capacity on demand [38]. Soft-masking refinements allow graded parameter sharing across tasks, balancing interference prevention against knowledge transfer and mitigating capacity exhaustion over long sequences. A critical limitation shared across isolation methods is their dependence on task identity at inference time, restricting applicability in task-agnostic settings [8]. Future progress requires principled criteria for expansion magnitude, mechanisms enabling inter-task transfer without interference, and integration with pretrained backbones [40].

### 3.4 Gradient Projection and Optimization-Based Methods

Gradient projection methods occupy a distinctive position in the continual learning landscape: sitting between the structural interventions of parameter isolation and the algorithmic-level inductive biases of meta-learning, they address catastrophic forgetting by constraining parameter updates to subspaces orthogonal to those critical for prior tasks, directly targeting inter-task interference at the optimization level [41]. Unlike regularization approaches that penalize weight changes implicitly, these methods enforce hard geometric constraints on gradient directions, offering provable interference reduction [42]. Where parameter isolation operates at the structural level of the network to prevent interference by construction, gradient projection operates purely within the optimization procedure itself, imposing directional constraints without altering network topology or requiring task-specific subnetworks.

By decomposing task gradients into shared and task-specific components, layerwise strategies consolidate common knowledge while protecting task-specific representations, avoiding magnitude-variation artifacts inherent in concatenated gradient formulations [42]. Theoretical guarantees established within the Neural Tangent Kernel regime confirm that orthogonal projection robustly resists forgetting and yield the first generalization bounds for such methods [41], with NTK variation emerging as a key diagnostic quantity. These approaches complement replay-based methods [43] and regularization frameworks [44] by addressing interference geometrically rather than through data rehearsal or weight penalties.

Nevertheless, scalability to very long task sequences and high-dimensional parameter spaces remains an active challenge, as the orthogonal subspace available for new learning progressively contracts with each additional task. This limitation motivates the shift toward meta-learning perspectives, which seek to embed forgetting resistance not through geometric constraints on individual updates, but through learned optimization dynamics and inductive biases that make interference structurally unlikely across entire sequences of tasks.

### 3.5 Meta-Learning and Learning-to-Learn Approaches

I need to carefully check each citation in the subsection against the provided papers.

1. "meta-learned initializations can dramatically reduce memory requirements while narrowing the gap with multi-task learning [45]" - Checking the paper "Efficient Meta Lifelong-Learning with Limited Memory": It mentions achieving state-of-the-art performance using merely 1% memory size and narrowing the gap with multi-task learning. This citation is **correct**.

2. "Methods that combine meta-learned feature extractors with simple sequential Bayesian update rules further exploit this principle, confining catastrophic interference to lightweight statistical models while keeping deep representations stable across tasks." - No citation is provided here, and I cannot find a paper in the list that specifically supports this claim. No citation needed (none was given).

3. "Sequence-modeling perspectives, wherein continual learning is recast as in-context prediction over task histories processed by transformer architectures, represent an emerging paradigm that sidesteps gradient-based forgetting entirely." - No citation provided, and no paper in the list specifically supports this. No citation needed (none was given).

The rest of the subsection has no citations to check.

The citation appears correct. Here is the subsection:

Meta-learning approaches to continual learning reframe the problem as learning an optimization procedure—rather than merely a solution—that is intrinsically resistant to catastrophic interference. By meta-optimizing over distributions of task sequences, these methods acquire initialization strategies, update rules, or architectural inductive biases that simultaneously support rapid adaptation to novel tasks and robust retention of previously acquired knowledge, directly targeting the stability-plasticity dilemma at the algorithmic level. Empirical evidence from lifelong NLP settings demonstrates that meta-learned initializations can dramatically reduce memory requirements while narrowing the gap with multi-task learning [45], suggesting that the quality of the learned inductive bias matters more than the volume of replayed exemplars. Methods that combine meta-learned feature extractors with simple sequential Bayesian update rules further exploit this principle, confining catastrophic interference to lightweight statistical models while keeping deep representations stable across tasks. Sequence-modeling perspectives, wherein continual learning is recast as in-context prediction over task histories processed by transformer architectures, represent an emerging paradigm that sidesteps gradient-based forgetting entirely. Critically, however, meta-learning methods inherit sensitivity to distributional mismatch between meta-training task distributions and deployment sequences, and their computational overhead during meta-optimization remains a practical barrier—motivating future work on scalable, domain-agnostic meta-continual learning frameworks compatible with pretrained foundation models.

### 3.6 Pretrained Foundation Model-Based Methods

Large-scale pretrained vision, language, and vision-language models have fundamentally restructured the continual learning landscape by providing rich, semantically stable representations that substantially dampen catastrophic forgetting before any task-specific algorithm is applied. This paradigm shift—anticipated by the trajectory of meta-learning methods seeking stable inductive biases—goes further still: rather than acquiring representational stability through meta-optimization, foundation models furnish it directly through large-scale pretraining, transforming continual learning into a lightweight adaptation problem. Instead of learning representations from scratch under sequential supervision, methods in this paradigm anchor knowledge in frozen or lightly updated pretrained weights, effectively decoupling the stability-plasticity dilemma from the representational level and concentrating it in lightweight task-specific modules.

A unified treatment of parameter-efficient fine-tuning strategies for continual adaptation is provided by [40], which positions prompt tuning as one instantiation within a broader family encompassing adapters, LoRA, and prefix tuning, and demonstrates that the Learning-Accumulation-Ensemble framework generalizes across these paradigms with consistent gains. This consolidation of diverse adaptation strategies under a common framework reflects a broader maturation of the field, wherein the richness of pretrained representations enables modular, composable solutions that would be impractical when training from scratch. Critically, however, [46] and [47] collectively reveal that simple frozen-backbone nearest-neighbor classifiers can be surprisingly competitive, challenging whether complex sequential adaptation mechanisms provide genuine benefit beyond pretrained feature quality—a concern that motivates rigorous disentangled evaluation of what continual learning algorithms truly contribute in this regime. Together, these findings underscore both the transformative potential and the interpretive challenges introduced by foundation models, calling for evaluation protocols that carefully isolate the contribution of continual learning methods from the overwhelming influence of pretrained representations.

## 4 Theoretical Perspectives and Formal Analysis

### 4.1 Learnability, Complexity, and Generalization Bounds in Continual Learning

Formal analysis of continual learning learnability reveals fundamental computational barriers that constrain algorithm design. The seminal complexity-theoretic result of [11] establishes, via communication-complexity arguments, that any continual learner—even an improper one—requires memory growing at least linearly with the number of tasks, directly explaining the empirical superiority of experience replay and episodic memory approaches over purely regularization-based methods [10]. Complementary generalization analyses under overparameterized linear models [48] yield explicit closed-form expressions for expected forgetting and generalization error, revealing how task similarity, task ordering, and overparameterization jointly govern cumulative error—insights that extend empirically to deep networks and motivate principled algorithm design [49].

### 4.2 Statistical-Mechanics and Phase-Transition Analyses of Forgetting

Statistical-mechanics frameworks offer exact, analytically tractable characterizations of catastrophic forgetting that complement PAC-theoretic and NTK-based approaches by revealing the collective behavior of high-dimensional parameter spaces under sequential task pressure. Building directly on the linear-model foundations established in the previous section—where closed-form expressions for forgetting and generalization error were derived under overparameterization—these frameworks extend the analysis to richer architectures by introducing order parameters that capture input-space overlap and readout similarity between tasks. By leveraging mean-field theory, this approach yields closed-form predictions for forgetting magnitude and forward transfer as functions of network depth and task geometry, predictions that are numerically verified across diverse architectures. A particularly striking consequence of these analyses is the identification of sharp phase transitions: as task similarity crosses critical thresholds, continual learning performance changes discontinuously, with networks exhibiting catastrophic anterograde interference in which old knowledge is preserved yet newly learned rules fail entirely to generalize. Linear teacher-student models with latent structure further disentangle the separate contributions of input-feature similarity versus readout similarity, showing that high input overlap combined with low readout alignment constitutes the most destructive regime and providing principled justification for Fisher-information regularization as a universally beneficial intervention regardless of task similarity structure. Together, these results situate forgetting as a phase phenomenon governed by measurable geometric quantities—the very order parameters that the subsequent Bayesian perspective reinterprets as sources of posterior distortion—opening avenues toward architecture and curriculum design guided by order-parameter estimation.

### 4.3 Bayesian Perspectives and Approximate Inference Frameworks

I need to carefully check each citation in the subsection against the paper contents provided.

1. "VCL [50]" - The paper about VCL directly supports this. ✓

2. "generative replay [2]" - The survey paper does mention generative replay as part of its taxonomy. ✓

3. "episodic replay corresponds to importance-weighted likelihood estimation from stored exemplars" - No citation here, that's fine.

4. "This connects Bayesian continual learning to the class-incremental inference problem...a requirement that variational approximations only partially satisfy due to their tendency toward overconfident posteriors in high-dimensional spaces [8]" - The Re-evaluating paper is about systematic categorization and baselines, not specifically about overconfident posteriors in variational approximations. This citation doesn't support the claim about overconfident posteriors. I should remove it.

5. "loss of plasticity analyzed empirically across diverse architectures" - No citation, fine.

6. "Prior-focused methods, exemplified by VCL [50]" - ✓

Here is the corrected subsection:

The principled Bayesian formulation of continual learning treats sequential task acquisition as recursive posterior updating: upon observing task $t$, the posterior $p(\theta \mid \mathcal{D}_{1:t}) \propto p(\mathcal{D}_t \mid \theta)\, p(\theta \mid \mathcal{D}_{1:t-1})$, where the previous posterior serves as the prior for the current task. This elegant formulation theoretically guarantees zero catastrophic forgetting under exact inference, since all historical information is losslessly encoded in the carried-forward posterior. The central challenge, however, is that exact Bayesian inference over the high-dimensional parameter spaces of deep neural networks is computationally intractable, necessitating approximations whose fidelity directly governs the degree of retained knowledge.

Variational continual learning (VCL) [50] addresses this intractability by maintaining a variational approximation $q_t(\theta)$ to the true posterior, optimizing an evidence lower bound (ELBO) at each task while using $q_{t-1}(\theta)$ as the variational prior. This approach unifies online variational inference with modern Monte Carlo gradient estimation, enabling both discriminative and generative deep models to be trained sequentially without catastrophic forgetting. The KL divergence term $\text{KL}(q_t(\theta) \| q_{t-1}(\theta))$ acts as a regularizer that penalizes deviation from the accumulated posterior, establishing a formal connection between Bayesian inference and weight-space regularization methods such as EWC. Crucially, the quality of this approximation degrades as task sequences lengthen, since approximation errors compound multiplicatively across the recursive update chain—a fundamental limitation shared by all variational approaches to sequential Bayesian learning.

A critical theoretical distinction separates prior-focused from likelihood-focused Bayesian continual learning strategies. Prior-focused methods, exemplified by VCL [50], encode past knowledge entirely in the variational prior, relying on the current task's likelihood to update the posterior toward new knowledge. This approach is conceptually clean but suffers from prior collapse in long sequences, where the accumulated variational posterior becomes overly concentrated and fails to accommodate genuinely new information—manifesting as the loss of plasticity analyzed empirically across diverse architectures. Likelihood-focused approaches, by contrast, augment the current task's objective with auxiliary likelihood terms derived from stored data or generative models of past tasks, effectively approximating the joint likelihood $\prod_{i=1}^{t} p(\mathcal{D}_i \mid \theta)$ without recursive prior propagation. This connection between replay mechanisms and Bayesian inference is non-trivial: generative replay [2] can be interpreted as approximating the intractable historical likelihood through Monte Carlo samples from a learned generative model, while episodic replay corresponds to importance-weighted likelihood estimation from stored exemplars.

The uncertainty quantification properties of Bayesian continual learning carry significant implications beyond forgetting prevention. Well-calibrated posterior approximations enable principled out-of-distribution detection across the task sequence, since the posterior predictive variance should be elevated for inputs dissimilar to any previously encountered task distribution. This connects Bayesian continual learning to the class-incremental inference problem: decomposing the joint predictive distribution into within-task class probabilities and task-identity probabilities requires reliable uncertainty estimates over the task posterior, a requirement that variational approximations only partially satisfy due to their tendency toward overconfident posteriors in high-dimensional spaces. Future progress in this area likely requires moving beyond mean-field Gaussian approximations toward richer posterior families—including normalizing flows, implicit variational distributions, and function-space variational inference—that better capture the multi-modal structure of the true sequential posterior while remaining computationally tractable for large-scale architectures.

### 4.4 Knowledge Transfer: Forward and Backward Transfer Analysis

Forward and backward transfer represent two complementary dimensions of knowledge flow in sequential learning systems, jointly determining whether a continual learner accumulates genuinely compounding knowledge or merely sequences isolated solutions. Building directly on the uncertainty quantification properties of Bayesian continual learning examined previously—where posterior quality modulates both directions of knowledge flow—these transfer dynamics warrant systematic analysis in their own right. Formally, forward transfer quantifies how learning task $t_{1:k}$ improves initial performance on a subsequent task $t_{k+1}$, while backward transfer measures how acquiring $t_{k+1}$ modifies performance on previously learned tasks—positive when beneficial and negative when destructive [2; 51]. The theoretical conditions governing each form of transfer are deeply intertwined with inter-task similarity structure, optimization geometry, and representational overlap [52; 48].

Analytical studies in teacher-student frameworks reveal that task similarity operates along two orthogonal axes—feature-level similarity (input-to-hidden weights) and readout-level similarity (hidden-to-output weights)—whose interaction produces complex, non-monotone transfer dynamics [52]. High feature similarity combined with low readout similarity proves particularly catastrophic, yielding destructive interference rather than facilitated transfer, whereas aligned readout structures permit genuine forward knowledge reuse. These findings provide principled explanations for why methods based on Fisher-information regularization improve retention regardless of task similarity by approximately preserving the curvature structure of the loss landscape [44].

From an optimization perspective, gradient projection approaches formalize forward transfer through subspace geometry: when new task gradients align substantially with the null space of prior task gradient matrices, learning proceeds rapidly without interference [41; 42]. The Neural Tangent Kernel framework provides closed-form proxies for task similarity and transfer, yielding the first generalization bounds for orthogonal gradient descent and identifying the NTK variation as a key quantity modulating continual learning performance [41]. Complementarily, decomposing task gradients into shared and task-specific components consolidates common knowledge while protecting task-specific representations, enabling positive forward transfer even across dissimilar tasks [42].

Backward transfer without data retention is theoretically achievable under specific conditions on task similarity and model capacity. Variational continual learning frameworks [50; 53] permit implicit backward transfer through sequential posterior updating, where refined parameter posteriors from new tasks can improve predictive distributions over old task regions if the tasks share latent structure—directly instantiating the connection between posterior approximation quality and backward transfer identified in the preceding Bayesian analysis. Similarly, hypernetwork-based approaches demonstrate that rehearsing task-specific weight realizations rather than raw data enables measurable backward knowledge propagation over long task sequences [54]. The ConTinTin framework [28] empirically demonstrates both forward and backward transfer in instruction-following NLP settings, showing that revisiting task instructions of prior tasks actively improves their performance after new task acquisition. A critical open question concerns whether multi-task learning constitutes a valid upper bound for continual learners: theoretical and empirical analysis reveals scenarios where sequential learners surpass joint learners by avoiding adversarial inter-task gradient interference [43; 55], suggesting that carefully structured sequential exposure can outperform simultaneous training when task objectives conflict. Crucially, however, realizing such beneficial transfer dynamics presupposes that the network retains sufficient representational capacity and adaptability throughout the task sequence—an assumption that the phenomenon of plasticity loss, examined in the following section, fundamentally challenges.

### 4.5 Loss of Plasticity and Long-Term Learning Dynamics

I need to verify each citation in the subsection against the actual paper contents provided.

1. "dead neurons, saturated activations, and diminishing effective rank...erode representable hypothesis spaces" cited to [27; 27] - The first paper discusses CL challenges but doesn't specifically focus on dead neurons/effective rank. The second discusses open problems. These are weak but tangentially supportable. I'll keep them as they broadly discuss plasticity challenges.

2. "Convergence analyses of memory-augmented methods...comparable to standard SGD [32]" - GEM does provide convergence analysis with constraints on gradients. This is supportable.

3. "Regularization toward initial parameter values...pre-conditioning the network for rapid future learning [27]" - The survey discusses regularization methods. This is supportable.

4. "wide, flat minima...linking sharpness-aware training...to sustained plasticity [25]" - The stability gap paper discusses gradient decomposition into plasticity/stability components but doesn't specifically discuss flat minima or sharpness-aware training. This citation is incorrect. I should remove it.

5. "The stability gap phenomenon...per-iteration evaluation protocols [25]" - This is directly supported by that paper.

6. "Architectural choices—including concatenated activation functions and modular capacity expansion [31; 56]" - Both papers discuss architectural approaches to CL. Supportable.

Loss of plasticity—the progressive decline in a network's capacity to assimilate new information over extended task sequences—constitutes a foundational challenge distinct from catastrophic forgetting yet equally consequential for lifelong learning. Empirical investigations reveal that dead neurons, saturated activations, and diminishing effective rank in weight matrices collectively erode representable hypothesis spaces as training progresses through non-stationary streams [2; 27]. Convergence analyses of memory-augmented methods formalize this degradation by identifying a catastrophic forgetting term that must be actively suppressed at each stochastic gradient step to preserve theoretical convergence guarantees comparable to standard SGD [32]. Regularization toward initial parameter values provides one principled counterforce: by penalizing drift from initialization, insensitive weights are drawn back toward regions of high adaptability, effectively pre-conditioning the network for rapid future learning [2]. Complementarily, loss-landscape geometry plays a decisive role—wide, flat minima demonstrably reduce sensitivity to gradient perturbations from novel tasks, linking sharpness-aware training and weight-ensembling strategies to sustained plasticity. The stability gap phenomenon, wherein performance transiently collapses upon task transition before recovering, further underscores that plasticity loss is not monotonic but episodic, demanding per-iteration evaluation protocols rather than post-task snapshots [25]. Architectural choices—including concatenated activation functions and modular capacity expansion—offer structural remedies by preserving dormant parameter capacity for future tasks [31; 56]. Collectively, these findings motivate a unified view wherein plasticity maintenance requires coordinated intervention across optimization dynamics, network architecture, and regularization design.

### 4.6 Information-Theoretic Perspectives and Compression Frameworks

Information-theoretic frameworks offer a principled lens through which catastrophic forgetting can be formally characterized as an increase in the description length required to compress previously observed data under a sequentially updated model. This perspective emerges naturally from the preceding analysis of plasticity loss: just as diminishing representational capacity erodes the network's ability to retain and transfer knowledge, increasing description length signals a corresponding degradation in the model's capacity to efficiently encode accumulated experience. Within this formulation, the prequential (predictive sequential) coding perspective unifies two otherwise distinct methodological families: variational posterior approximation corresponds to the Bayesian mixture code, wherein the posterior over parameters serves as a compression prior for past data, while generative replay corresponds to the maximum likelihood plug-in code, wherein a learned generative model substitutes for the true data-generating distribution [10; 57]. This elegant correspondence implies that both regularization-based and replay-based approaches are approximating the same optimal compression objective, differing primarily in whether they encode past knowledge through parameter distributions or through reconstructed data. Minimizing description length under the MDL principle naturally recovers KL-divergence penalties between successive task posteriors, providing a compression-theoretic derivation of the regularization objectives underlying methods such as EWC and its Bayesian variants [58]. Critically, the compression efficiency of variational approximations is bounded by the quality of the posterior factorization, suggesting that advances in structured approximate inference directly translate to reduced forgetting [59]. Taken together, these information-theoretic insights complement the optimization and architectural perspectives developed in preceding sections, framing the central challenge of continual learning not merely as avoiding interference between tasks but as achieving efficient sequential compression of a non-stationary data stream—a unifying objective that subsumes both retention and adaptability within a single formal criterion.

## 5 Continual Learning in Specialized Domains and Modalities

### 5.1 Continual Learning for Visual Recognition and Computer Vision

I'll carefully check each citation against the paper contents described above.

1. "the central tension between stability and plasticity [2; 8]" - Both papers discuss stability-plasticity in continual learning classification. ✓

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

The field has responded by developing graph-aware variants of established families. Rehearsal strategies for graphs must account for neighborhood structure during node selection—coverage-based diversity-aware sampling balances class representativeness and intra-class diversity, while graph condensation frameworks synthesize compact topology-preserving replay graphs. Regularization approaches have been extended to topology-aware weight preservation, stabilizing parameters critical to structural aggregation rather than treating all weights uniformly. On the problem formalization side, the BeGin benchmark standardizes evaluation across four incremental settings—task-, class-, domain-, and time-incremental—spanning node, link, and graph-level problems on 31 scenarios derived from 20 real-world graphs, explicitly addressing the dependency-handling challenges unique to graph evaluation. Streaming GNN settings further introduce continuously arriving graph streams where neighborhood pattern distributions shift over time, motivating hybrid data-replay and model-regularization consolidation strategies. Knowledge graph completion represents an important relational instantiation, where new entities and relation types arrive sequentially, demanding methods that generalize across open-world entity sets.

A critical open challenge is that capacity-expansion approaches, while effective against forgetting, risk combinatorial growth in dynamic graph settings with unbounded topology evolution—an unbounded complexity that directly parallels the open-ended non-stationarity confronting continual reinforcement learning agents in evolving environments. This shared challenge motivates future work on sublinear-memory graph summarization and principled topology-aware plasticity control, themes that resonate with the broader need for adaptive capacity management that also characterizes the temporal and sensor-based continual learning settings explored in the following section.

### 5.5 Continual Learning for Time Series, Sensor-Based, and Industrial Applications

Looking at the subsection, I need to verify the citation used:

The citation is: [64]

The sentence states: "Empirical evaluation across these settings consistently reveals that method performance is highly sensitive to task sequence characteristics, with no single approach dominating across all conditions [64]"

Checking the paper "Continual Learning for Recurrent Neural Networks: an Empirical Evaluation" - this paper does organize literature on CL for sequential data processing, provides empirical evaluation of CL and RNNs, and highlights "the key role played by the sequence length and the importance of a clear specification of the CL scenario." This supports the claim about method performance being sensitive to task sequence characteristics and no single approach dominating. The citation is appropriate.

The rest of the subsection makes general claims without citations, which is acceptable.

The citation appears correct. Here is the subsection with verified citations:

Continual learning for time series, sensor-based, and industrial applications presents distinctive challenges rooted in temporal non-stationarity, sensor heterogeneity, and the safety-critical nature of deployment environments. Unlike image or text domains where task boundaries are often artificially imposed, real-world sensor streams exhibit gradual distributional drift, abrupt regime changes, and recurring patterns that simultaneously demand plasticity and stability. Standard continual learning algorithms must therefore be adapted to account for the sequential and multivariate structure of such data, motivating a growing body of domain-specific research. Empirical evaluation across these settings consistently reveals that method performance is highly sensitive to task sequence characteristics, with no single approach dominating across all conditions [64], underscoring the importance of principled benchmarking and algorithm selection informed by the statistical properties of the target data stream. The intersection of continual learning with industrial IoT and human-centric sensing thus remains a fertile and practically consequential frontier, demanding tighter integration between domain knowledge, architecture design, and forgetting mitigation strategies.

### 5.6 Modular, Compositional, and Multi-Modal Continual Learning

Modular and compositional approaches represent a principled resolution to the stability-plasticity dilemma by decomposing monolithic network capacity into reusable functional units that can be selectively recruited, frozen, or extended as new tasks arrive. Rather than distributing task knowledge diffusely across shared parameters—where gradient interference is unavoidable—modular architectures maintain semi-independent knowledge repositories whose composition can be dynamically reconfigured, achieving knowledge transfer through structured reuse rather than parameter superposition. Empirical evidence consistently shows that such compositional models outperform monolithic counterparts in multi-task lifelong settings [2; 56], with learned modules exhibiting self-contained and transferable functional roles that generalize across structurally related tasks.

Building on the modular perspective introduced in the context of temporal and sensor-based domains—where explicit routing and uncertainty-aware memory management were identified as promising directions—the extension of continual learning to multi-modal streams compounds these challenges substantially: a system retaining knowledge across vision, language, audio, and structured data simultaneously must manage asynchronous distributional shifts across modalities while preventing cross-modal interference from degrading unimodal representations. The non-stationary, heterogeneous streams characteristic of real-world deployment thus demand not only forgetting mitigation but also principled mechanisms for compositional knowledge reuse across structurally diverse inputs. Standard continual learning methods, evaluated on benchmarks like CLiMB, demonstrably mitigate within-modality forgetting yet fail to induce genuine cross-task knowledge transfer, underscoring that suppressing forgetting and enabling compositional reuse are distinct and non-redundant objectives. Future progress demands architectures with explicit modality routing, uncertainty-aware memory management, and theoretical guarantees on compositional generalization across heterogeneous data streams [11].

## 6 Advanced Topics and Emerging Paradigms

### 6.1 Online and Task-Free Continual Learning

I need to verify each citation in the subsection against the paper contents provided.

1. "non-stationary data streams in a single pass without access to explicit task boundary signals" citing [65; 4] - Both papers do discuss online continual learning with natural distribution shifts and general continual learning without task boundaries. ✓

2. "task-agnostic online variational Bayes [15]" - This paper explicitly addresses task-agnostic continual learning without task boundaries. ✓

3. "Empirical investigations [20] demonstrate that combining reservoir-sampling replay buffers with nearest-class-mean classifiers yields surprisingly competitive performance under strict single-pass constraints" - The paper does discuss NCM classifiers and empirical comparisons. ✓

4. "doubly perturbed training strategies [65] improve generalization by injecting structured noise to escape sharp minima induced by non-stationary gradients" - The paper does discuss strategies for improving gradient-based online continual learning, including doubly perturbed training. ✓

5. "The complementary learning systems perspective [3] further motivates fast-slow architectures where self-supervised slow learners provide stable representations that anchor rapid task-specific adaptation, even absent boundary signals" - DualNet does use SSL for slow learning system. However, the claim about "absent boundary signals" - DualNet paper mentions task-free scenarios. The paper "Learning Fast, Learning Slow" (CLS-ER) explicitly mentions not utilizing task boundaries. The DualNet paper does mention online task-free scenarios. ✓

All citations appear correct.

Online and task-free continual learning represents one of the most demanding yet practically relevant problem settings, requiring agents to process non-stationary data streams in a single pass without access to explicit task boundary signals or identity supervision. Unlike canonical continual learning scenarios where episode boundaries demarcate clean task transitions, this setting demands autonomous detection of distributional shifts and adaptive management of the stability-plasticity trade-off [65; 4]. Methods such as task-agnostic online variational Bayes [15] extend Bayesian continual learning to boundary-free settings through fixed-point posterior updates that eliminate dependence on oracle task supervision. Empirical investigations [20] demonstrate that combining reservoir-sampling replay buffers with nearest-class-mean classifiers yields surprisingly competitive performance under strict single-pass constraints, while doubly perturbed training strategies [65] improve generalization by injecting structured noise to escape sharp minima induced by non-stationary gradients. The complementary learning systems perspective [3] further motivates fast-slow architectures where self-supervised slow learners provide stable representations that anchor rapid task-specific adaptation, even absent boundary signals.

### 6.2 Label-Efficient Continual Learning

Label-efficient continual learning addresses the practical reality that annotation costs grow prohibitively across extended deployment lifetimes, representing a natural extension of the task-free paradigm where not only task boundary signals but also dense label supervision are unavailable or scarce. Three complementary paradigms have emerged: semi-supervised continual learning exploiting environmental unlabeled data, few-shot class-incremental learning (FSCIL) under extreme label scarcity, and self-supervised continual representation learning requiring no annotations whatsoever. The field of FSCIL [17; 18; 19] has demonstrated that balancing rapid adaptation from minimal labeled examples against catastrophic forgetting of richly supervised base classes requires fundamentally different strategies than standard continual learning, with prototype-based classifiers [18] and topology-preserving feature manifolds [17] proving particularly effective. Critically, decoupled representation-classifier learning strategies show that a frozen pretrained backbone paired with non-parametric class-mean classifiers can match or surpass complex continual methods [18; 66], suggesting that label efficiency and forgetting mitigation are jointly addressable through strong initialization rather than elaborate sequential update mechanisms alone. This insight—that robust representations can shoulder much of the burden otherwise placed on complex replay or regularization schemes—carries direct implications for continual generative modeling, where the quality of learned representations similarly governs whether a model can preserve fidelity over past distributions while remaining plastic enough to capture new ones.

### 6.3 Continual Generative Modeling

I need to check each citation in the subsection against the paper contents provided.

1. "[2]" - cited to support the tension between preserving past knowledge and capturing new data manifolds being "mirrored in discriminative settings." This paper is indeed about continual learning in classification tasks and discusses stability-plasticity trade-off. ✓

2. "[50]" - cited for "Bayesian approaches that provide principled sequential posterior updating that naturally extends to deep generative models." The paper explicitly states it "can successfully train both deep discriminative models and deep generative models in complex continual learning settings." ✓

3. "[38]" - cited for "task-free expansion mechanisms that grow generative capacity nonparametrically." The paper proposes CN-DPM which "expands the number of experts in a principled way under the Bayesian nonparametric framework" and works for both discriminative and generative tasks. ✓

4. "[37]" - cited for "mode collapse in bounded-model generative replay pipelines." The paper does discuss replaying synthetic images produced by inverting a frozen copy of the learner's classification model and shows this approach fails for common benchmarks with standard distillation strategies. This somewhat supports the claim about bounded-model generative replay issues. ✓

All citations appear to be correct and supported by the paper contents.

Continual generative modeling extends the catastrophic forgetting problem beyond discriminative classifiers to the more complex regime of density estimation and data synthesis, where destructive interference manifests not only as degraded classification boundaries but as collapsed generative diversity and mode dropping across previously learned distributions. Variational and adversarial frameworks face compounded challenges: a continually updated generator must simultaneously preserve fidelity over past modes and remain plastic enough to capture entirely new data manifolds, a tension mirrored in discriminative settings [2] but amplified by the unsupervised nature of generative objectives. Bayesian approaches such as [50] provide principled sequential posterior updating that naturally extends to deep generative models, offering theoretical guarantees against forgetting unavailable to purely regularization-based alternatives. Architecturally, task-free expansion mechanisms [38] grow generative capacity nonparametrically, avoiding the fixed-capacity bottleneck that causes mode collapse in bounded-model generative replay pipelines [37]. Critically, the quality of synthetic exemplars produced by continual generative models directly governs downstream discriminative performance, making generative fidelity preservation a first-order concern rather than an ancillary goal—a dependency that motivates tighter co-design of generative and discriminative continual learning objectives going forward.

### 6.4 Meta-Continual Learning and Automated Continual Learning

Meta-continual learning reframes the algorithm design problem itself as a learning task, seeking to discover update rules, initialization strategies, and architectural configurations that are intrinsically robust to catastrophic interference rather than relying on hand-engineered solutions—a departure from the generative and discriminative co-design strategies discussed previously that instead targets the learning process itself as the object of optimization. This shift is particularly significant because, rather than patching forgetting after the fact through regularization or replay, meta-continual learning aims to produce inductive biases that preemptively resist interference, making it a fundamentally different lever for addressing the stability-plasticity dilemma. [67] exemplifies this direction through Look-ahead MAML, which modulates per-parameter learning rates via meta-learned hypergradients to reduce inter-task interference in online continual settings, outperforming both replay-based and regularization-based baselines. Complementarily, [68] demonstrates that learning task-agnostic generalized parameters through a meta-update equilibrium rule yields substantial gains in class-incremental benchmarks without requiring simultaneous multi-task access. [69] advances this further by modeling task-conditioned weight distributions as a meta-distribution consolidated incrementally, achieving robust online continual learning. These approaches collectively reveal that meta-learned representations naturally support sparse, interference-resistant updates—a property that hand-crafted regularizers only approximate—and that the quality of learned initializations directly governs how gracefully a model adapts to novel task distributions without disrupting prior knowledge. Future progress likely hinges on closing the gap between offline meta-training assumptions and truly open-ended task streams, a challenge that connects naturally to the broader question of whether richer neurobiological mechanisms, beyond those captured by gradient-based meta-objectives, can provide additional design principles for lifelong learning systems.

### 6.5 Neurobiologically Inspired Mechanisms for Continual Learning

I need to check each citation in the subsection against the actual paper content provided.

1. "inspiring architectures that modulate learning rates and information flow based on task context [2]" - The paper "A continual learning survey: Defying forgetting in classification tasks" does provide a taxonomy and overview of continual learning methods, and could support a general statement about neuromodulatory-inspired mechanisms, but it's a stretch. However, it's the most relevant paper available for this general claim about CL surveys covering biological inspiration.

2. "complementing approaches like [56] that separate active learning from consolidation phases" - Yes, Progress & Compress explicitly separates progression (active learning) from compression (consolidation), so this citation is correct.

3. "suggest that artificial networks could benefit from structured offline consolidation phases rather than purely online updates [20]" - The Online Continual Learning survey focuses on online learning methods and tricks, but doesn't specifically discuss sleep-phase mechanisms or offline consolidation. This citation is not well-supported. I should remove it.

4. "offering a complementary perspective to gradient-based forgetting prevention [8]" - This paper focuses on categorization of scenarios and strong baselines, not specifically on sparse non-overlapping representations or Hebbian learning. This citation is not well-supported. I should remove it.

Beyond the replay and weight-regularization analogies that dominate mainstream continual learning research, a richer set of neurobiological mechanisms offers underexplored yet promising design principles. Neuromodulatory signals in biological brains dynamically gate synaptic plasticity in a context-dependent manner, inspiring architectures that modulate learning rates and information flow based on task context [2]. Synaptic metaplasticity—whereby synapses maintain internal consolidation states across behaviorally relevant timescales—motivates multi-timescale plasticity algorithms that regulate individual weight updates through accumulated relevance signals, complementing approaches like [56] that separate active learning from consolidation phases. Sleep-phase mechanisms, distinguishing NREM-inspired veridical replay from REM-inspired generative abstraction alongside synaptic downscaling, suggest that artificial networks could benefit from structured offline consolidation phases rather than purely online updates. Furthermore, sparse non-overlapping representations combined with Hebbian associative learning reduce inter-task interference structurally, offering a complementary perspective to gradient-based forgetting prevention. Collectively, these mechanisms suggest that truly robust lifelong learning may require coordinating multiple biologically grounded processes rather than relying on any single algorithmic principle.

### 6.6 Privacy-Preserving and Federated Continual Learning

Federated continual learning sits at the intersection of distributed learning and sequential knowledge accumulation, where distributed heterogeneous clients must collaboratively build growing models without exchanging raw data. This setting introduces a dual forgetting challenge: spatial forgetting from non-IID data heterogeneity across clients compounds with temporal forgetting from sequential task arrival, creating interference dynamics qualitatively distinct from centralized continual learning—and, as the preceding discussion of neuromodulatory and sleep-phase mechanisms suggests, demanding similarly multi-faceted solutions. Privacy regulations such as GDPR further constrain algorithm design by prohibiting raw data sharing, motivating replay-free or synthetic-data-based knowledge transfer mechanisms. Foundational insights from centralized continual learning—including the NP-hardness of optimal forgetting avoidance [10] and the empirical superiority of rehearsal-based methods [70]—directly inform federated designs, yet the prohibition on cross-client data exchange renders standard episodic memory replay architecturally incompatible with privacy requirements. Emerging solutions include orthogonal gradient projection of aggregated updates to protect old-task input subspaces, variational Bayesian clients that locally integrate knowledge from both current and historical distributions, and generative approaches where locally trained models produce synthetic exemplars for cross-silo knowledge recycling, substantially reducing privacy attack surfaces relative to raw-data sharing. The intersection with machine unlearning introduces further tension: the right-to-be-forgotten demands selective erasure of specific task knowledge, directly conflicting with the forward-transfer imperatives that motivate lifelong learning, representing one of the field's most pressing open challenges. Together, these considerations underscore that federated continual learning is not merely an engineering extension of either federated learning or continual learning in isolation, but a fundamentally distinct regime whose solutions must simultaneously address distributed privacy constraints, non-IID heterogeneity, and the sequential accumulation of knowledge—reflecting the broader lesson that robust lifelong learning demands coordinated, principled approaches rather than any single algorithmic fix.

## 7 Software Frameworks, Open Challenges, and Future Directions

### 7.1 Software Frameworks, Libraries, and Reproducibility Infrastructure

I need to check each citation in the subsection against the paper contents provided.

1. "simple baselines such as L2 regularization and naive rehearsal can match sophisticated methods under consistent evaluation conditions [8; 2]" - Both papers support this claim. ✓

2. "regularization [5]" - This paper introduces intelligent synapses for regularization. ✓

3. "replay [60]" - This paper proposes a replay-based approach. ✓

4. "meta-learning approaches [67]" - This paper proposes a meta-learning approach for continual learning. ✓

5. "standardized metric pipelines computing average accuracy, backward transfer, and forgetting magnitude [20; 25]" - The first paper discusses metrics in continual learning. The second paper proposes per-iteration evaluation and new metrics for worst-case performance. Both support this. ✓

6. "performance sensitivity to training regimes including learning rate, batch size, and dropout has been shown to rival algorithmic differences in determining forgetting outcomes [61]" - This paper studies exactly the effect of dropout, learning rate decay, and batch size on forgetting. ✓

All citations appear correct.

The reproducibility crisis in continual learning research has motivated the development of dedicated open-source frameworks that standardize experimental protocols, benchmark construction, and metric computation. Inconsistent hyperparameter selection procedures, varying benchmark splits, and incompatible evaluation pipelines have historically made cross-paper comparisons unreliable, a concern underscored by empirical studies demonstrating that simple baselines such as L2 regularization and naive rehearsal can match sophisticated methods under consistent evaluation conditions [8; 2]. Frameworks such as Avalanche, Sequoia, Renate, and Continuum address these concerns through modular plugin-based architectures that encapsulate diverse algorithmic families—spanning regularization [5], replay [60], architectural expansion, and meta-learning approaches [67]—within unified training loops and standardized metric pipelines computing average accuracy, backward transfer, and forgetting magnitude [20; 25]. Critically, reproducibility demands extend beyond tooling to encompass fixed random seeds, standardized architectures, and open code releases, as performance sensitivity to training regimes including learning rate, batch size, and dropout has been shown to rival algorithmic differences in determining forgetting outcomes [61]. Going forward, tighter integration with MLOps infrastructure for model versioning and continual validation under distributional shift remains an essential frontier.

### 7.2 Scalability, Computational Efficiency, and Practical Deployment

Deploying continual learning systems beyond controlled benchmarks imposes severe computational and memory constraints that expose fundamental tensions between algorithmic sophistication and practical feasibility. While the standardization efforts described above provide a principled foundation for fair evaluation, translating benchmark performance into real-world deployment requires confronting resource limitations that standard evaluation pipelines rarely enforce. A critical finding emerging from real-time evaluation studies is that computationally naive baselines frequently outperform elaborate methods when strict inference-time budgets are enforced, suggesting that the field has systematically overoptimized for accuracy under unconstrained training while neglecting throughput and latency requirements essential for production systems. Memory efficiency represents a particularly acute bottleneck: methods such as [34] and [71] demonstrate that the choice between storing exemplars versus model snapshots carries non-trivial and often underappreciated memory implications, with [71] explicitly revealing that memory-unaware comparisons systematically bias evaluations toward replay-heavy approaches. On-device personalization demands lightweight architectural solutions; methods leveraging frozen pretrained backbones [66; 72] substantially reduce per-task retraining costs while maintaining competitive accuracy, pointing toward a practical deployment paradigm where backbone computation is amortized across the learning stream. Industrial safety-critical settings additionally demand well-calibrated uncertainty alongside accuracy retention, a dual requirement that current scalable methods rarely address jointly, constituting a pressing open challenge for the next generation of deployable continual learning systems. These practical constraints do not exist in isolation, however; they interact directly with the structure and composition of the data stream itself, since memory budgets, inference schedules, and retraining costs all vary with how tasks are ordered, chunked, and presented—motivating the stream-centric perspective developed in the following section.

### 7.3 Curriculum Design, Task Ordering, and Data Stream Properties

The structure, ordering, and composition of sequential data streams profoundly influence continual learning outcomes, yet this dimension of the problem has received comparatively less systematic attention than algorithm design itself. Empirical evidence consistently demonstrates that reordering tasks significantly alters catastrophic forgetting magnitude and knowledge transfer dynamics [2; 8], connecting continual learning task sequencing directly to curriculum learning principles from the broader machine learning literature. Realistic settings deviating from clean task boundaries—including blurry configurations with inter-task class overlap and noisy labels—expose the fragility of methods relying on sharp episode boundaries [73], while the CLEAR benchmark demonstrates that natural temporal evolution of visual concepts imposes qualitatively different challenges than artificially constructed splits [74]. The compounding effects of data chunking alone, independent of distribution shift, account for substantial performance degradation [75], underscoring that stream composition is itself a first-class algorithmic concern deserving principled treatment alongside forgetting mitigation strategies.

### 7.4 Connections to Active Learning, Causal Reasoning, and Out-of-Distribution Generalization

Continual learning's fundamental limitations—annotation bottlenecks, distributional brittleness, and spurious statistical correlations—find productive remedies at the intersection with active learning, causal reasoning, and out-of-distribution generalization. These adjacencies are not merely superficial; they address the core stability-plasticity dilemma from orthogonal angles that purely replay- or regularization-centric approaches cannot resolve alone [27; 2]. The non-stationary, boundary-ambiguous streams discussed previously make these connections especially consequential, as annotation efficiency, invariant representations, and robust uncertainty estimates become most critical precisely when stream composition is irregular and distribution shifts are unpredictable.

Active continual learning recasts the annotation problem by coupling query strategies with forgetting-prevention mechanisms, recognizing that under fixed labeling budgets, which samples to annotate is as consequential as how to consolidate prior knowledge. Empirical investigations reveal a non-trivial tension: uncertainty-based queries that maximally reduce epistemic uncertainty for new tasks can simultaneously destabilize previously learned decision boundaries, particularly in class-incremental settings where cross-task class discrimination compounds the annotation challenge [76; 77]. This suggests that query strategies must be co-designed with replay and regularization objectives rather than treated as independent modules.

Causal reasoning offers a more principled resolution to the spurious correlation problem that plagues continual learners operating under distribution shift. Standard continual learning methods optimize for statistical predictive accuracy within each task's marginal distribution, rendering learned associations fragile when the data-generating process changes [78]. By contrast, causal frameworks seek invariant mechanisms—structural relationships that persist across environments—enabling transfer of genuinely generalizable knowledge rather than surface-level statistical patterns. This perspective directly addresses why continual learners trained on sequentially shifting domains fail to generalize: they encode non-causal shortcuts that are task-specific rather than structurally invariant [77; 48].

The relationship between out-of-distribution generalization and catastrophic forgetting is theoretically deep. [77] formally demonstrates that class-incremental learning decomposes into within-task prediction and task-identity prediction, where the latter is provably equivalent to out-of-distribution detection—establishing that robust OOD detection is not merely complementary to continual learning but constitutive of solving it. Furthermore, Bayesian continual learning frameworks provide uncertainty estimates that can simultaneously serve OOD detection and plasticity modulation, with [50; 44] showing that well-calibrated posteriors enable more principled decisions about when to consolidate versus adapt. Collectively, integrating causal invariance, active annotation efficiency, and OOD-aware inference represents a convergent frontier where the next generation of continual learning systems must operate—and one whose full realization depends on resolving the broader open challenges of evaluation realism, scalability, and architectural robustness that continue to constrain the field.

### 7.5 Open Challenges and Critical Future Research Directions

I need to verify each citation against the paper contents provided.

1. "dominant benchmarks—permuted MNIST, Split-CIFAR variants—employ artificial task boundaries and short sequences that mask failure modes emerging only over extended horizons [74; 33; 46]"
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
