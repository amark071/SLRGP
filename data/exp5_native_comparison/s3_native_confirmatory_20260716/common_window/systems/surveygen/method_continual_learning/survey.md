# Registered common-window excerpt

# Continual Learning: Foundations, Methods, and Applications — A Comprehensive Survey

## Foundations and Problem Formulation of Continual Learning

### Definition and Motivation

Continual learning (CL) is the paradigm whereby machine learning models incrementally acquire knowledge from non-stationary data streams without catastrophically forgetting previously learned information ref [40]. This capability mirrors biological intelligence, where organisms naturally adapt to evolving environments while retaining prior experience ref [24]. In contrast, standard deep neural networks trained sequentially suffer from catastrophic forgetting, wherein new task learning overwrites previously acquired parameters ref [42]. This limitation severely restricts real-world AI deployment across domains including healthcare ref [1], cybersecurity ref [25], and predictive maintenance ref [22]. CL addresses fundamental challenges in model editing, personalization, and on-device learning ref [37], making it indispensable for building lifelong intelligent systems capable of operating under continuously shifting data distributions ref [5].

### Learning Scenarios and Taxonomies

Continual learning encompasses several formally distinct scenarios that differ in what information is available at inference time. Task-incremental learning (TIL) provides explicit task identifiers during testing, reducing the problem to within-task prediction ref [49]. Class-incremental learning (CIL) requires the model to discriminate among all classes encountered without task identity, demanding both within-task prediction and task-identity inference ref [49]. Domain-incremental learning (DIL) maintains a fixed output space while input distributions shift across tasks ref [47]. Beyond these canonical settings, online versus offline variants distinguish whether data may be revisited ref [57], and blurry task boundary scenarios relax the assumption of disjoint class sets per task, reflecting more realistic deployment conditions ref [56]. Multimodal and NLP settings further extend this taxonomy ref [17].

### Catastrophic Forgetting and the Stability-Plasticity Dilemma

Catastrophic forgetting denotes the abrupt degradation of previously acquired knowledge when neural networks are trained sequentially on new tasks, arising from gradient interference and parameter overwriting that disrupt earlier learned representations ref [40]. This phenomenon is empirically confirmed across diverse domains, including medical imaging ref [1] and cybersecurity ref [25], where naive fine-tuning causes severe performance deterioration. Underlying this challenge is the stability-plasticity dilemma: stability preserves prior knowledge while plasticity enables adaptation to new information, and optimizing one typically compromises the other ref [44]. Gradient projection methods explicitly address this tension by constraining updates to subspaces orthogonal to previous task features ref [12], ref [36]. Theoretical analysis further demonstrates that catastrophic forgetting intensifies with task dissimilarity, and that regularization can partially mitigate but not fully resolve this fundamental trade-off ref [51].

### Theoretical Analysis of Continual Learning

Theoretical analysis of continual learning has yielded foundational insights into the conditions necessary for successful incremental learning. A pivotal contribution decomposes class-incremental learning (CIL) into two sub-problems: within-task prediction (WP) and task-identity prediction (TP), proving that good WP and good TP—equivalent to closed-world out-of-distribution (OOD) detection—are necessary and sufficient conditions for strong CIL performance ref [49]. This unification extends to open-world settings, connecting CIL and novelty detection under a common theoretical framework ref [26]. For regularization-based methods, formal risk bounds in the fixed-design linear regression setting reveal a provable forgetting-intransigence trade-off governed by the regularization parameter and inter-task dissimilarity ref [51]. Additionally, hierarchical decomposition of the continual learning objective identifies within-task prediction, task-identity inference, and task-adaptive prediction as distinct optimizable components ref [58].

### Evaluation Protocols and Metrics

Rigorous evaluation of continual learning methods requires standardized benchmarks and well-defined metrics. Commonly adopted benchmarks include Split CIFAR-100, Split ImageNet-R, Split CUB-200, and Split Cars-196 ref [2], with large-scale evaluations on ImageNet-1K ref [6] and domain-incremental variants ref [27]. Key metrics encompass average accuracy across all tasks, the forgetting measure quantifying performance degradation on previous tasks, forward transfer reflecting facilitation of new task learning, and backward transfer capturing improvements on old tasks ref [46]. Emerging benchmarks address multimodal settings such as CLiMB ref [17], NLP-oriented evaluation via TRACE ref [31], and blurry task boundary scenarios through Rainbow Memory ref [56]. Frameworks like Avalanche ref [29] and PILOT ref [45] provide standardized, reproducible evaluation pipelines supporting diverse continual learning scenarios.

### Relationship to Adjacent Learning Paradigms

Continual learning intersects with several adjacent paradigms, each sharing conceptual overlap yet maintaining distinct objectives. Transfer learning focuses on adapting pre-trained representations to new domains ref [2], whereas continual learning additionally requires preserving prior task performance. Multi-task learning optimizes jointly over all tasks simultaneously, an assumption continual learning explicitly relaxes ref [40]. Meta-learning seeks rapid adaptation to new tasks, and its principles inform continual learning through improved plasticity ref [37]. Online learning addresses sequential data streams and concept drift, forming a natural intersection termed Online Streaming Continual Learning ref [4]. Reinforcement learning shares the sequential decision-making structure, with continual learning formalized as computationally constrained reinforcement learning ref [5]. Federated learning introduces privacy and non-IID constraints that compound forgetting challenges ref [3].

## Core Methodological Approaches to Continual Learning

### Regularization-Based Methods

Regularization-based methods mitigate catastrophic forgetting by imposing constraints on parameter updates to preserve knowledge acquired from previous tasks. Elastic Weight Consolidation (EWC) and Synaptic Intelligence penalize modifications to parameters deemed important for prior tasks, while Learning without Forgetting (LwF) employs knowledge distillation on current data to retain old task outputs ref [40]. Theoretical analysis of ℓ₂-regularized continual learning reveals a provable stability-plasticity trade-off: stronger regularization reduces forgetting but induces intransigence toward new tasks ref [51]. Gradient projection methods extend this paradigm by constraining updates to the null space of previous task feature covariances, as in Adam-NSCL ref [36]. The Space Decoupling (SD) algorithm further refines this approach by partitioning the feature space into complementary stability and plasticity subspaces, enabling more flexible gradient directions while preserving task-shared representations ref [12].

### Replay and Memory-Based Methods

Replay-based methods mitigate catastrophic forgetting by maintaining a memory buffer of past data for rehearsal. Sample selection criteria critically determine buffer utility: gradient-based strategies select samples whose gradients are most interfered by new data ref [59], while diversity-based approaches such as Rainbow Memory leverage per-sample uncertainty and augmentation to maximize representational diversity ref [56]. Proxy-based contrastive replay (PCR) addresses class imbalance in online class-incremental settings by replacing contrastive samples with class proxies, improving convergence stability ref [21]. Second-order influence regularization identifies and corrects bias amplification across successive selection rounds ref [18]. Dual-view consistency exploration further enriches semantic information from single-pass streams ref [59]. DualHSIC augments rehearsal by explicitly modeling inter-task relationships through HSIC-based bottleneck and alignment components, reducing interference while promoting task-invariant knowledge sharing ref [14].

### Generative Replay and Brain-Inspired Replay

Generative replay addresses catastrophic forgetting by training generative models to synthesize pseudo-rehearsal data from previous tasks, interleaving synthetic samples with current task data during training. Brain-Inspired Replay (BIR) advances this paradigm by replaying internal hidden representations generated through context-modulated feedback connections, achieving state-of-the-art performance on challenging benchmarks such as class-incremental learning on CIFAR-100 without storing any raw data ref [42]. BIR has demonstrated particular promise in privacy-sensitive domains, including medical image classification for retinal disease detection ref [1]. For diffusion-based generative models, standard generative replay causes catastrophic degradation in denoising capabilities; generative distillation—which distils the entire reverse diffusion process—substantially mitigates this limitation with modest computational overhead ref [13]. Federated settings further exploit generative replay through synthetic data generation to simulate global distributions without sharing private client data ref [3].

### Parameter Isolation and Dynamic Architecture Methods

Parameter isolation and dynamic architecture methods address catastrophic forgetting by allocating dedicated parameters or sub-networks per task, thereby preventing inter-task interference. DyTox ref [16] employs a transformer-based encoder-decoder framework with dynamic token expansion, specializing each forward pass on a task distribution while maintaining strict parameter growth control, achieving state-of-the-art scalability on ImageNet100. PI-GNN ref [8] extends parameter isolation to dynamic graph learning, expanding model parameters to capture emerging graph patterns while freezing parameters corresponding to previously learned patterns via optimization-based identification. Complementarily, soft-masking approaches such as SPG ref [60] partially block parameter updates based on task-specific importance scores, enabling knowledge transfer across tasks while mitigating forgetting without monopolizing sub-networks, thus addressing capacity limitations inherent in hard isolation strategies.

### Optimization and Gradient-Based Methods

Optimization and gradient-based methods address catastrophic forgetting by operating directly in the parameter update

[... deterministic omitted span ...]

pre-training dataset is typically private and inaccessible, conventional replay methods cannot directly mitigate this degradation. To address this, ref [27] proposes Zero-Shot Continual Learning (ZSCL), which introduces a semantically diverse reference dataset for feature-space distillation between the current and initial model, alongside weight averaging in parameter space to prevent large parameter shifts. ZSCL demonstrates substantial improvements on the proposed Multi-domain Task Incremental Learning (MTIL) benchmark, outperforming competing methods by 9.7% average score, establishing that preserving generalization requires explicit regularization anchored to the original pre-trained representations.

### Language Guidance and Multimodal Continual Learning

Language semantics offer a powerful inductive bias for continual learning by grounding visual representations in a shared embedding space. LGCL (Language Guidance for Continual Learning) introduces language guidance as a plug-in for prompt-based methods, operating at both the task level within the prompt pool and at the class level on vision encoder outputs, achieving consistent performance improvements without additional learnable parameters ref [48]. Complementarily, CLiMB provides a systematic multimodal continual learning benchmark built upon a Vision-Language Transformer, evaluating how upstream continual learning generalizes to new multimodal and unimodal tasks ref [17]. Empirical findings from CLiMB reveal that while standard continual learning methods mitigate forgetting, they fail to enable meaningful cross-task knowledge transfer in multimodal settings, motivating the development of dedicated multimodal continual learning algorithms ref [17].

## Continual Learning in Specialized Domains and Emerging Settings

### Continual Learning for Natural Language Processing

Continual learning for natural language processing presents distinct challenges compared to vision-based settings, as language models must preserve linguistic knowledge and generalization capabilities while adapting to sequential tasks. A comprehensive survey of CL in NLP covers catastrophic forgetting prevention, knowledge transfer, and inter-task class separation ref [35]. For large language models (LLMs), the TRACE benchmark evaluates continual learning across domain-specific tasks, multilingual capabilities, code generation, and mathematical reasoning, revealing significant capability degradation after sequential training ref [31]. To address this, Reasoning-augmented Continual Learning (RCL) integrates task-specific cues with meta-rationales to mitigate forgetting ref [31]. O-LoRA further proposes orthogonal low-rank adaptation, learning tasks in mutually orthogonal subspaces to minimize interference while preserving generalization on unseen tasks ref [32]. Progressive Prompts extends prompt-based CL to language models by sequentially concatenating task-specific soft prompts, enabling forward transfer without data replay ref [28].

### Continual Learning for Computer Vision Beyond Classification

Beyond image classification, continual learning has been extended to dense prediction tasks including semantic and instance segmentation, where catastrophic forgetting is compounded by the background shift problem—previously annotated foreground classes become background in subsequent tasks. CISDQ addresses this by introducing dynamic queries with adaptive background modeling and a class/instance-aware query-guided knowledge distillation strategy, achieving state-of-the-art performance on both continual semantic and instance segmentation benchmarks ref [38]. Class imbalance and incremental query-based architectures further complicate forgetting mitigation in object detection settings. These challenges necessitate specialized architectural designs that decouple old and new knowledge representations, highlighting that solutions developed for classification transfer imperfectly to structured, spatially-sensitive vision tasks requiring pixel-level or region-level predictions.

### Continual Learning on Graphs

Continual learning on graphs presents unique challenges beyond standard image classification, as real-world graph-structured data is inherently dynamic, with new nodes, edges, and topology patterns emerging over time. Dynamic graph learning methods commonly suffer from catastrophic forgetting, where knowledge acquired for previous graph states is overwritten during updates for new graph configurations ref [8]. Existing continual graph learning approaches attempt to learn new patterns while preserving old ones within a fixed parameter set, creating a fundamental stability-plasticity tradeoff. To circumvent this limitation, Parameter Isolation GNN (PI-GNN) proposes parameter isolation and expansion, identifying parameters corresponding to unaffected graph patterns via optimization and freezing them, while expanding model capacity to accommodate emerging patterns ref [8]. Experiments across eight real-world datasets corroborate PI-GNN's effectiveness over state-of-the-art baselines.

### Continual Learning for Generative Models

Continual learning for generative models presents distinct challenges beyond those encountered in discriminative settings. Incrementally training generative adversarial networks risks mode collapse, wherein the model forgets previously learned data distributions upon exposure to new tasks. Diffusion models, despite achieving state-of-the-art image synthesis quality, are particularly susceptible to catastrophic degradation of denoising capabilities under standard generative replay ref [13]. To address this, generative distillation has been proposed, which distils the entire reverse diffusion process from a frozen copy of the prior model, substantially improving continual learning performance with modest computational overhead ref [13]. More broadly, forgetting in generative models due to generator shifts represents a recognized challenge across deep learning paradigms ref [54], motivating principled continual learning solutions tailored to generative architectures.

### Continual Learning in Healthcare and Medical Imaging

Healthcare and medical imaging present compelling use cases for continual learning, where models must adapt to emerging disease categories without retaining sensitive patient data. ref [1] provides a systematic comparative evaluation of twelve privacy-preserving, non-storage continual learning algorithms for retinal disease classification using optical coherence tomography (OCT) images across a class-incremental scenario. Their findings demonstrate that Brain-Inspired Replay (BIR) ref [42] achieves the highest accuracy of 62.00% on OCT retinal disease classification, while Efficient Feature Transformations (EFT) attains 66.82% on colon cancer histology, both substantially below joint retraining baselines. These results underscore the persistent accuracy gap between continual and joint training, while confirming that non-storage algorithms can meaningfully mitigate catastrophic forgetting, offering a viable pathway for privacy-compliant, long-term clinical deployment of deep learning diagnostic systems.

### Continual Learning for Cybersecurity and Predictive Maintenance

Cybersecurity and predictive maintenance represent two critical real-world domains where continual learning must contend with non-stationary, evolving data distributions. In phishing detection, feature distributions shift substantially due to emerging attack techniques and technological evolution; ref [25] demonstrates that continual learning algorithms maintain accuracy with only 2.45% deterioration over three years, compared to over 20.65% degradation in conventional models. For predictive maintenance, ref [22] provides a comprehensive overview of challenges arising from dynamic operational environments, proposing benchmark construction methodologies that reflect realistic non-stationary conditions and discussing how continual learning enables adaptive model updates post-deployment. Both domains share fundamental challenges including concept drift, domain shift, and the impracticality of full retraining, underscoring continual learning as an essential paradigm for sustainable deployment in safety-critical industrial and security applications.

### Open-World and Novelty-Aware Continual Learning

Open-world continual learning (OwCL) extends conventional class-incremental learning (CIL) by requiring models to simultaneously retain knowledge of known classes and detect novel, unseen categories. Ref [26] provides a theoretical unification of out-of-distribution (OOD) detection and CIL, proving that successful CIL necessitates effective closed-world OOD detection by decomposing CIL into within-task prediction and task-identity prediction sub-problems. This theoretical framework demonstrates that good OOD detection is both necessary and sufficient for robust CIL performance. Building upon these foundations, ref [20] proposes Pro-KT, a prompt-enhanced knowledge transfer model for OwCL that incorporates a prompt bank encoding task-generic and task-specific knowledge alongside a task-aware open-set boundary mechanism, achieving superior performance in both known-class classification and unknown-class detection on real-world benchmarks.

## Federated, Distributed, and Privacy-Preserving Continual Learning

### Federated Class-Continual Learning

Federated Class-Continual Learning (FCCL) addresses the compound challenge of dynamically emerging classes within federated learning environments, where data is distributed across clients and cannot be centrally aggregated. A critical finding is that non-IID data distributions across clients substantially exacerbate catastrophic forgetting, as local model updates reflect heterogeneous class distributions that conflict during global aggregation ref [3]. Existing FCCL approaches often require auxiliary datasets or retention of private task data, limiting their applicability in privacy-sensitive deployments. TARGET addresses these limitations by employing a previously trained global model for knowledge distillation at the model level, while training a generator to synthesize data approximating the global distribution at the data level, thereby eliminating reliance on stored real data ref [3]. This exemplar-free distillation paradigm renders FCCL viable for data-sensitive scenarios including healthcare and legal applications.

### Privacy Constraints in Continual Learning

Privacy constraints impose fundamental tensions on continual learning systems, particularly for rehearsal-based methods that require retaining raw data from previous tasks. Regulations such as GDPR prohibit the storage and reuse of sensitive personal data, rendering standard experience replay infeasible in healthcare, legal, and financial domains ref [1]. This

[... deterministic omitted span ...]

on a single GPU while matching offline learner performance. SparCL ref [11] addresses edge deployment constraints by synergizing weight sparsity, dynamic data removal, and gradient sparsity, achieving up to 23× reduction in training FLOPs while simultaneously improving accuracy. CRNet ref [19] proposes closed-form random theory solutions enabling single-pass learning with fast convergence, demonstrating superior training speed on CIFAR-100 and ImageNet benchmarks. Collectively, these approaches establish complementary strategies—architectural efficiency, sparsity-driven acceleration, and analytical solutions—for practical continual learning deployment.

### Memory-Efficient Strategies

Memory efficiency is a critical concern in continual learning, where strict resource budgets necessitate compact and principled approaches to knowledge retention. Rainbow Memory (RM) addresses this by managing episodic buffers through per-sample classification uncertainty and data augmentation to maximize sample diversity under fixed memory constraints ref [56]. SIESTA employs latent rehearsal with memory indexing, storing compressed representations rather than raw inputs to substantially reduce storage overhead ref [6]. SparCL further reduces memory demands via dynamic data removal, eliminating less informative training samples during continual training ref [11]. Exemplar-free methods such as DualPrompt and RanPAC circumvent rehearsal entirely, avoiding memory buffers through prompt-based tuning and frozen pre-trained representations respectively ref [15], ref [30], thereby offering strong performance in privacy-sensitive and resource-constrained deployment scenarios.

### On-Device and Edge Continual Learning

Deploying continual learning on resource-constrained edge devices introduces unique challenges beyond algorithmic forgetting prevention, encompassing strict limitations on computational budget, memory, and energy consumption. SparCL ref [11] addresses these constraints through the synergy of weight sparsity, dynamic data removal, and gradient sparsity, achieving up to 23× reduction in training FLOPs while maintaining competitive accuracy, with practical validation on mobile hardware. SIESTA ref [6] further advances edge-oriented continual learning via a wake/sleep framework employing backpropagation-free online updates during wake phases and compute-restricted rehearsal during sleep, enabling ImageNet-1K continual learning in under two hours on a single GPU. CRNet ref [19] complements these approaches through closed-form random-theory solutions enabling single-pass training with fast convergence. Collectively, these methods demonstrate that efficiency-aware design is essential for practical continual learning deployment ref [37].

### Continual Learning Software Frameworks and Toolboxes

Open-source software frameworks have become essential infrastructure for reproducible continual learning research. Avalanche, maintained by the ContinualAI non-profit organization, extends PyTorch with first-class support for dynamic architectures, streaming datasets, and incremental training and evaluation, providing a comprehensive suite of predefined benchmarks and modular algorithm implementations ref [7]. Its end-to-end design facilitates fast prototyping and standardized evaluation across diverse continual learning scenarios ref [29]. Complementing Avalanche, PILOT is a toolbox specifically designed for pre-trained model-based continual learning, implementing state-of-the-art class-incremental algorithms such as L2P, DualPrompt, and CODA-Prompt, while also benchmarking classical methods within pre-trained model contexts ref [45]. Together, these frameworks lower barriers to entry, promote reproducibility, and enable systematic comparison across algorithmic families.

### Scalability to Long Task Sequences and Large-Scale Datasets

Scalability to long task sequences and large-scale datasets represents a critical challenge for continual learning systems. Dynamic architecture methods such as DyTox address this through controlled parameter expansion via dynamic token expansion in transformer architectures, achieving state-of-the-art performance on ImageNet100 while maintaining negligible memory and time overheads ref [16]. Prompt-based approaches like Progressive Prompts demonstrate robustness under longer task sequences without proportional parameter growth ref [28]. SLCA reveals progressive overfitting as a fundamental bottleneck in continual learning on pre-trained models, achieving substantial improvements on Split ImageNet-R ref [2]. Gradient projection methods such as SD decouple feature spaces to prevent unbounded expansion across tasks ref [12], while CRNet provides closed-form solutions enabling efficient scaling on ImageNet ref [19].

### Future Directions and Open Challenges

Continual learning faces several open challenges that define its future trajectory. Model editing and personalization remain underexplored, requiring systems to selectively update knowledge without global retraining ref [37]. Scaling to long task sequences demands efficient architectures ref [16] and compute-aware strategies ref [6], ref [11]. Multimodal and vision-language continual learning introduces new forgetting dimensions, including zero-shot degradation ref [27] and cross-modal transfer ref [17]. Integration with large language models necessitates robust benchmarks ref [31] and subspace-based adaptation ref [32]. Open-world settings demand unified frameworks combining OOD detection with incremental learning ref [26], ref [20]. Privacy-preserving deployment in sensitive domains such as healthcare ref [1] and federated environments ref [3] remains critical. Finally, reinforcement learning integration and biologically inspired adaptability ref [24] represent promising directions toward truly lifelong intelligent agents.

## References

[1] Privacy-preserving continual learning methods for medical image classification: a comparative analysis. https://doi.org/10.3389/fmed.2023.1227515

[2] SLCA: Slow Learner with Classifier Alignment for Continual Learning on a Pre-trained Model. https://doi.org/10.1109/ICCV51070.2023.01754

[3] TARGET: Federated Class-Continual Learning via Exemplar-Free Distillation. https://doi.org/10.1109/ICCV51070.2023.00441

[4] Survey on Online Streaming Continual Learning. https://doi.org/10.24963/ijcai.2023/743

[5] Continual Learning as Computationally Constrained Reinforcement Learning. https://doi.org/10.48550/arXiv.2307.04345

[6] SIESTA: Efficient Online Continual Learning with Sleep. https://doi.org/10.48550/arXiv.2303.10725

[7] Avalanche: A PyTorch Library for Deep Continual Learning. https://doi.org/10.48550/arXiv.2302.01766

[8] Continual Learning on Dynamic Graphs via Parameter Isolation. https://doi.org/10.1145/3539618.3591652

[9] CODA-Prompt: COntinual Decomposed Attention-Based Prompting for Rehearsal-Free Continual Learning. https://doi.org/10.1109/CVPR52729.2023.01146

[10] Rethinking Momentum Knowledge Distillation in Online Continual Learning. https://doi.org/10.48550/arXiv.2309.02870

[11] SparCL: Sparse Continual Learning on the Edge. https://doi.org/10.48550/arXiv.2209.09476

[12] Rethinking Gradient Projection Continual Learning: Stability/Plasticity Feature Space Decoupling. https://doi.org/10.1109/CVPR52729.2023.00362

[13] Continual Learning of Diffusion Models with Generative Distillation. https://doi.org/10.48550/arXiv.2311.14028

[14] DualHSIC: HSIC-Bottleneck and Alignment for Continual Learning. https://doi.org/10.48550/arXiv.2305.00380

[15] DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning. https://doi.org/10.48550/arXiv.2204.04799

[16] DyTox: Transformers for Continual Learning with DYnamic TOken eXpansion. https://doi.org/10.1109/CVPR52688.2022.00907

[17] CLiMB: A Continual Learning Benchmark for Vision-and-Language Tasks. https://doi.org/10.48550/arXiv.2206.09059

[18] Regularizing Second-Order Influences for Continual Learning. https://doi.org/10.1109/CVPR52729.2023.01931

[19] CRNet: A Fast Continual Learning Framework With Random Theory. https://doi.org/10.1109/TPAMI.2023.3262853

[20] Learning to Prompt Knowledge Transfer for Open-World Continual Learning. https://doi.org/10.48550/arXiv.2312.14990

[21] PCR: Proxy-Based Contrastive Replay for Online Class-Incremental Continual Learning. https://doi.org/10.1109/CVPR52729.2023.02322

[22] Continual Learning for Predictive Maintenance: Overview and Challenges. https://doi.org/10.1016/j.iswa.2023.200251

[23] Continual Learning via Sequential Function-Space Variational Inference. https://doi.org/10.48550/arXiv.2312.17210

[24] Incorporating neuro-inspired adaptability for continual learning in artificial intelligence. https://doi.org/10.1038/s42256-023-00747-w

[25] Life-long phishing attack detection using continual learning. https://doi.org/10.1038/s41598-023-37552-9

[26] Open-World Continual Learning: Unifying Novelty Detection and Continual Learning. https://doi.org/10.48550/arXiv.2304.10038

[27] Preventing Zero-Shot Transfer Degradation in Continual Learning of Vision-Language Models. https://doi.org/10.1109/ICCV51070.2023.01752

[28] Progressive Prompts: Continual Learning for Language Models. https://doi.org/10.48550/arXiv.2301.12314

[29] Avalanche: an End-to-End Library for Continual Learning. https://doi.org/10.1109/CVPRW53098.2021.00399

[30] RanPAC: Random Projections and Pre-trained Models for Continual Learning. https://doi.org/10.48550/arXiv.2307.02251

[31] TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models. https://doi.org/10.48550/arXiv.2310.06762

[32] Orthogonal Subspace Learning for Language Model Continual Learning. https://doi.org/10.48550/arXiv.2310.14152

[33] A Unified Continual Learning Framework with General Parameter-Efficient Tuning. https://doi.org/10.1109/ICCV51070.2023.01055

[34] Continual Learning with Lifelong Vision Transformer. https://doi.org/10.1109/CVPR52688.2022.00027

[35] Continual Learning of Natural Language Processing Tasks: A Survey. https://doi.org/10.48550/arXiv.2211.12701

[36] Training Networks in Null Space of Feature Covariance for Continual Learning. https://doi.org/10.1109/CVPR46437.2021.00025

[37] Continual Learning: Applications and the Road Forward. https://doi.org/10.48550/arXiv.2311.11908

[38] Continual Learning for Image Segmentation With Dynamic Query. https://doi.org/10.1109/TCSVT.2023.3337884

[39] Variational Continual Learning. https://doi.org/10.17863/CAM.35471

[40] Embracing Change: Continual Learning in Deep Neural Networks.. https://doi.org/10.1016/j.tics.2020.09.004

[41] Adaptive Plasticity Improvement for Continual Learning. https://doi.org/10.1109/CVPR52729.2023.00755

[42] Brain-inspired replay for continual learning with artificial neural networks. https://doi.org/10.1038/s41467-020-17866-2

[43] ICICLE: Interpretable Class Incremental Continual Learning. https://doi.org/10.1109/ICCV51070.2023.00181

[44] Achieving a Better Stability-Plasticity Trade-off via Auxiliary Networks in Continual Learning. https://doi.org/10.1109/CVPR52729.2023.01148

[45] PILOT: a pre-trained model-based continual learning toolbox. https://doi.org/10.1007/s11432-024-4276-4

[46] Beyond Not-Forgetting: Continual Learning with Backward Knowledge Transfer. https://doi.org/10.48550/arXiv.2211.00789

[47] FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning. https://doi.org/10.48550/arXiv.2309.14062

[48] Introducing Language Guidance in Prompt-based Continual Learning. https://doi.org/10.1109/ICCV51070.2023.01053

[49] A Theoretical Study on Solving Continual Learning. https://doi.org/10.48550/arXiv.2211.02633

[50] Improving Plasticity in Online Continual Learning via Collaborative Learning. https://doi.org/10.1109/CVPR52733.2024.02214

[51] Fixed Design Analysis of Regularization-Based Continual Learning. https://doi.org/10.48550/arXiv.2303.10263

[52] Continual Learning for Generative Retrieval over Dynamic Corpora. https://doi.org/10.1145/3583780.3614821

[53] Generating Instance-level Prompts for Rehearsal-free Continual Learning. https://doi.org/10.1109/ICCV51070.2023.01088

[54] A Comprehensive Survey of Forgetting in Deep Learning Beyond Continual Learning. https://doi.org/10.1109/TPAMI.2024.3498346

[55] BiRT: Bio-inspired Replay in Vision Transformers for Continual Learning. https://doi.org/10.48550/arXiv.2305.04769

[56] Rainbow Memory: Continual Learning with a Memory of Diverse Samples. https://doi.org/10.1109/CVPR46437.2021.00812

[57] Online Continual Learning in Image Classification: An Empirical Survey. https://doi.org/10.1016/j.neucom.2021.10.021

[58] Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality. https://doi.org/10.48550/arXiv.2310.07234

[59] Not Just Selection, but Exploration: Online Class-Incremental Continual Learning via Dual View Consistency. https://doi.org/10.1109/CVPR52688.2022.00729

[60] Parameter-Level Soft-Masking for Continual Learning. https://doi.org/10.48550/arXiv.2306.14775
