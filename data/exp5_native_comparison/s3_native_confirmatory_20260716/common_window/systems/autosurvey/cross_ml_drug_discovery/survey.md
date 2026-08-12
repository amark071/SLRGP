# Registered common-window excerpt

# Machine Learning for Drug Discovery: Methods, Applications, and Future Directions

## 1 Introduction and Background

### 1.1 The Traditional Drug Discovery Pipeline

The traditional drug discovery and development pipeline represents one of the most complex, resource-intensive, and sequential scientific endeavors undertaken by modern medicine. Bringing a new therapeutic agent from initial concept to patient bedside typically spans more than a decade and demands billions of dollars in investment, with no guarantee of ultimate success [1; 2]. Understanding the individual stages of this pipeline is essential for appreciating both the magnitude of the challenge and the transformative potential that machine learning approaches offer.

**Target Identification and Validation**

The pipeline begins with target identification, where researchers seek to identify biological molecules—most commonly proteins such as enzymes, receptors, ion channels, or transporters—whose modulation is hypothesized to produce a therapeutically beneficial effect in a disease of interest. This stage draws heavily on advances in genomics, proteomics, transcriptomics, and systems biology, employing techniques such as genome-wide association studies (GWAS), gene expression profiling, and network-based analyses to nominate candidate targets [3]. Target validation then follows, requiring experimental and computational evidence that the proposed target is causally linked to the disease and that its modulation is both safe and efficacious. This phase is inherently uncertain; a large proportion of drug failures in later clinical stages can be traced back to insufficient target validation, underscoring the foundational importance of this step.

**Hit Discovery**

Once a validated target has been established, the next stage involves hit discovery—the identification of small molecules or biological entities that demonstrate measurable binding or modulatory activity against the target. Traditionally, this is accomplished through high-throughput screening (HTS) of large chemical libraries, which may contain millions of compounds, or through fragment-based drug discovery and virtual screening approaches [4]. The output of this stage is a collection of "hit" compounds that show promising initial activity but require extensive further optimization before they can be considered viable drug candidates.

**Lead Optimization**

Hit compounds are then subjected to lead optimization, a process in which medicinal chemists systematically modify molecular structures to improve potency, selectivity, pharmacokinetic properties, and safety profiles while reducing liabilities such as toxicity or poor metabolic stability [5; 6]. This stage is particularly resource-intensive, often consuming the majority of the preclinical research budget, and involves iterative cycles of synthesis, biological testing, and structural refinement guided by structure-activity relationship (SAR) analyses. The optimization of multiple molecular properties simultaneously—including absorption, distribution, metabolism, excretion, and toxicity (ADMET) profiles—poses a formidable multi-objective challenge that requires sophisticated experimental and computational tools.

**Preclinical Testing**

Candidate compounds that survive lead optimization are advanced to preclinical testing, which encompasses a comprehensive series of in vitro and in vivo studies designed to characterize the safety, pharmacokinetics, pharmacodynamics, and mechanism of action of the candidate. These studies are conducted in cell-based assays and animal models to generate the data package required to support an Investigational New Drug (IND) application to regulatory authorities. Preclinical testing is not only costly and time-consuming but also raises important questions about the translational relevance of animal models to human disease, contributing to subsequent failures in clinical development.

**Clinical Trials**

Clinical development is organized into three successive phases, each with distinct objectives and increasing scale. Phase I trials primarily assess safety and tolerability in a small number of healthy volunteers or patients, often including dose-escalation studies to determine the maximum tolerated dose [7]. Phase II trials evaluate preliminary efficacy and further characterize safety in a larger patient population, while Phase III trials involve large-scale, randomized, controlled studies designed to confirm efficacy and monitor adverse events across diverse patient populations. The sequential nature of these phases means that failures at any stage result in the loss of all preceding investment, contributing to the extraordinarily high overall cost of drug development [8]. Notably, clinical trial design itself is a complex undertaking requiring careful attention to patient enrollment, endpoint selection, and protocol optimization [9; 10].

**Regulatory Approval and Post-Market Surveillance**

Drugs that successfully complete Phase III trials must then navigate a rigorous regulatory review process conducted by agencies such as the U.S. Food and Drug Administration (FDA) or the European Medicines Agency (EMA). Regulatory submissions require comprehensive documentation of all preclinical and clinical data, chemistry, manufacturing, and controls (CMC) information, and risk management plans [11]. Following approval, post-market surveillance (Phase IV) continues to monitor long-term safety and effectiveness in the broader patient population.

**The Sequential and Resource-Intensive Nature of the Pipeline**

Collectively, these stages form a highly sequential pipeline in which each phase builds upon the outputs of its predecessor, creating critical dependencies and bottleneck points that amplify the consequences of failure at any individual stage [12]. The attrition of candidate compounds across these stages is dramatic; it has been estimated that only a tiny fraction of initially identified hits ultimately achieve regulatory approval. This sequential, resource-intensive architecture, combined with the biological complexity of human disease and the challenge of translating preclinical findings into clinical outcomes, defines the fundamental challenge that modern computational and machine learning approaches seek to address [1; 13]. As the following section elaborates, these compounding inefficiencies—spanning enormous financial costs, protracted timelines, high failure rates, and limited translational success—provide the central motivation for integrating machine learning and artificial intelligence throughout the drug discovery process.

### 1.2 Challenges in Traditional Drug Discovery: Cost, Time, and Attrition

The traditional drug discovery pipeline is widely recognized as one of the most resource-intensive and high-risk endeavors in modern science and industry. As described in the preceding section, the pipeline encompasses a series of highly sequential and interdependent stages—from target identification and hit discovery through lead optimization, preclinical testing, and multi-phase clinical trials—each of which introduces its own distinct challenges and failure modes. Bringing a single new drug to market typically requires an investment exceeding \$2–3 billion and spans a timeline of more than a decade [12; 14]. These staggering costs arise from the cumulative expenses of each pipeline stage, all of which demand substantial human, financial, and infrastructural resources [13].

A particularly acute challenge in this pipeline is the extraordinarily high attrition rate. The vast majority of candidate compounds that enter preclinical development never successfully complete clinical trials, and only a small fraction of those that do enter clinical testing ultimately receive regulatory approval [1]. Clinical failure rates have been attributed to a complex interplay of scientific and translational obstacles, including inadequate efficacy in human populations, unforeseen toxicity, poor pharmacokinetic and pharmacodynamic profiles, and the fundamental difficulty of translating findings from animal models to human disease [15]. These translational gaps are compounded by the limited predictive power of early-stage in vitro and in vivo assays, which often fail to capture the full complexity of human biology.

The attrition problem is further exacerbated by the inherent complexity of the chemical space that must be explored during drug design. It has been estimated that the space of drug-like molecules spans between 10²³ and 10⁶⁰ candidates [16], making exhaustive experimental screening practically infeasible. Conventional high-throughput screening campaigns, while valuable, are costly and time-consuming, and typically yield only a small number of viable lead compounds from vast compound libraries.

From a systemic perspective, the pharmaceutical industry has also faced a productivity crisis characterized by stagnant numbers of new drug approvals despite continuously escalating R&D expenditures [17]. Investigations into clinical trial meta-data have further revealed that innovation in drug discovery is constrained by network effects such as preferential attachment and local network dynamics, which bias exploration toward previously validated drug targets and limit the search for genuinely novel therapeutics [8]. Indeed, analyses suggest that only a small fraction of discovered compounds or drug candidates successfully transition from basic science into clinical research, and the average translational lag from initial discovery to first clinical application has been estimated at over 14 years [18].

These converging challenges—enormous financial costs, protracted timelines, high failure rates, and limited translational success—define the central motivation for integrating computational and artificial intelligence-driven strategies throughout the drug discovery process. As the following section traces, the field has a rich history of

[... deterministic omitted span ...]

and synthesizability assessment.

**Active Learning Integration**

Active learning provides a powerful mechanism for efficiently navigating the multi-objective optimization landscape by iteratively selecting the most informative molecules for experimental or computational evaluation and using the resulting feedback to update the generative model. In the context of multi-objective drug design, active learning frameworks prioritize candidates that are predicted to improve simultaneously across all objectives, effectively guiding the generative model toward regions of chemical space that are both promising and underexplored. Bayesian optimization, which naturally models uncertainty over objective functions, is particularly well-suited to active learning in multi-objective settings, as it enables principled exploration-exploitation trade-offs when evaluating expensive oracle functions such as molecular docking or ADMET prediction.

Deep generative models integrated with active learning loops—sometimes referred to as closed-loop or model-in-the-loop design cycles—have shown considerable promise in early drug discovery campaigns. In these frameworks, a generative model proposes candidate molecules, a property prediction oracle evaluates their multi-objective scores, and the resulting labeled data is used to retrain or fine-tune both the generative model and the property predictors. This iterative cycle progressively improves the quality and diversity of generated candidates while reducing the number of expensive evaluations required.

**Balancing Diversity and Constraint Satisfaction**

An inherent tension in multi-objective generative modeling is the trade-off between chemical diversity and constraint satisfaction. Highly constrained generation—particularly when enforcing synthesizability via reaction-based frameworks—may limit the diversity of produced molecules, concentrating the search around familiar chemical scaffolds. Strategies to mitigate this include diversity-promoting regularization terms in the generative objective, curiosity-driven exploration bonuses in RL frameworks, and GFlowNet-based methods that sample from distributions proportional to multi-objective rewards, enabling proportional exploration of the entire space of high-reward molecules rather than converging to a single mode. As will be discussed in Section 4.11, this tension between diversity and constraint satisfaction also poses significant challenges for benchmarking, since standard evaluation metrics must capture both the quality and the breadth of the Pareto-optimal solutions found.

**Evaluation and Benchmarking**

Evaluating multi-objective generative models requires metrics that capture both the quality and diversity of the Pareto-optimal solutions found. Standard benchmarks such as GuacaMol and MOSES provide multi-objective optimization tasks with predefined target profiles, allowing systematic comparison of generative approaches. However, as elaborated in the following section, these benchmarks often inadequately capture synthesizability constraints, underscoring the need for richer evaluation frameworks that incorporate retrosynthetic feasibility scores alongside traditional property-based metrics such as QED, SA score, and docking affinity.

In summary, multi-objective optimization and synthesizability represent two of the most practically important and technically challenging frontiers in molecular generative modeling. Progress in this area requires tight integration of reaction-aware generation, multi-objective optimization algorithms, active learning feedback loops, and robust evaluation frameworks that reflect the real-world constraints of pharmaceutical synthesis and development—challenges that motivate the comprehensive benchmarking discussion in the section that follows.

### 4.11 Benchmarking, Evaluation, and Challenges in Molecular Generation

### 4.11 Benchmarking, Evaluation, and Challenges in Molecular Generation

Building on the multi-objective optimization and synthesizability frameworks discussed in the preceding section, rigorous benchmarking and standardized evaluation are essential for measuring progress in molecular generative modeling and enabling fair comparisons across diverse architectures and training strategies. The field has converged on a core set of quantitative metrics that collectively assess the quality, diversity, and drug-likeness of generated molecular libraries. These include **validity** (the fraction of generated strings or graphs that correspond to chemically valid molecules), **uniqueness** (the fraction of valid molecules that are non-duplicate), and **novelty** (the fraction of valid, unique molecules not present in the training set). Beyond these basic measures, the **Fréchet ChemNet Distance (FCD)** has been widely adopted as a distribution-level metric that compares the statistical properties of generated molecules against reference datasets in the latent space of a pretrained neural network, providing a more holistic measure of chemical realism. Additional property-based metrics such as the **Quantitative Estimate of Drug-likeness (QED)**, synthetic accessibility scores (SAS), and **molecular docking scores** are used to assess the pharmaceutical relevance and binding potential of generated candidates, particularly in structure-based generation tasks. Critically, these metrics also serve as the very objectives that multi-objective optimization frameworks—discussed in Section 4.10—seek to jointly maximize, underscoring the deep connection between evaluation methodology and generative model design.

Several benchmarking platforms and datasets have been established to facilitate systematic comparisons. The **MOSES** (Molecular Sets) benchmark provides standardized train and test splits derived from ZINC and ChEMBL, alongside a comprehensive suite of metrics covering distribution learning, property optimization, and conditional generation. **GuacaMol** similarly offers a suite of goal-directed and distribution-learning benchmarks, testing models on their ability to optimize multiple drug-like objectives simultaneously. Notably, both MOSES and GuacaMol provide the multi-objective optimization tasks referenced in Section 4.10, serving as practical testbeds for evaluating the Pareto-optimal generation strategies and reward-shaping approaches described there. For structure-based drug design, the **CrossDocked2020** dataset is widely used to evaluate pocket-conditioned generative models, providing protein-ligand complexes with associated docking scores. The **QM9** dataset serves as a reference for evaluating 3D molecular generative models on quantum chemical properties, supporting comparisons of generated geometry quality [135; 136].

Despite these efforts, significant open challenges remain. **Chemical diversity** is a persistent concern: many generative models tend to produce molecules that cluster around the training distribution or seed compounds, limiting their ability to explore underrepresented regions of chemical space. This is compounded by the challenge of **out-of-distribution (OOD) generalization**, where models trained on known bioactive compounds may fail to generate structurally novel scaffolds with desired biological activity. The problem is particularly acute for structure-based generative models conditioned on protein pockets, where the diversity of generated ligands is inherently constrained by the pocket geometry. These diversity limitations also directly affect the practical utility of multi-objective optimization pipelines, as a lack of scaffold diversity can cause even sophisticated Pareto-based methods to converge prematurely.

**Bias in training and seed molecules** presents another fundamental challenge. Training datasets such as ZINC and ChEMBL reflect historical biases in medicinal chemistry, overrepresenting certain chemical scaffolds, heteroatom distributions, and molecular weight ranges. Generative models trained on such data may inadvertently reproduce these biases, limiting their utility for genuinely novel drug discovery campaigns. Similarly, reinforcement learning-based optimization approaches are sensitive to the choice of seed or starting molecules, and may converge to narrow regions of chemical space through reward hacking—a failure mode that is especially problematic in the multi-objective RL frameworks discussed in Section 4.10.

**Reproducibility** is a significant concern in the field, as many reported results are sensitive to implementation details, random seeds, dataset preprocessing choices, and hyperparameter settings. The lack of standardized open-source implementations and consistent evaluation protocols makes direct comparison across publications difficult. Efforts such as MOSES and GuacaMol have partially addressed this by providing reference implementations and fixed evaluation pipelines, but variability in how models are trained and evaluated persists.

Perhaps the most critical challenge is the **gap between computational generation and experimental validation**. Metrics such as QED, SAS, and even docking scores are imperfect proxies for actual biological activity, and molecules that score well in silico frequently fail in experimental assays due to issues of synthesis feasibility, off-target effects, metabolic instability, and cellular permeability. This disconnect directly motivates the synthesizability-constrained generation and active learning integration strategies outlined in Section 4.10, but closing this validation gap remains an open and active research frontier. Addressing these benchmarking and evaluation challenges—from standardizing diversity metrics to better capturing synthesizability in benchmark tasks—will be essential for translating the impressive in silico capabilities of molecular generative models into tangible advances in drug discovery.

## 5 Drug-Target Interaction, Drug-Drug Interaction, and Knowledge Graph Reasoning

### 5.1 Drug-Target Interaction Prediction: Foundations and Graph-Based Methods

## 5.1 Drug-Target Interaction Prediction: Foundations and Graph-Based Methods

Drug-target interaction (DTI) prediction is a cornerstone problem in computational drug discovery, aiming to determine whether a given small molecule drug binds to a specific biological target protein. Accurately identifying such interactions is critical for hit identification, lead optimization, and drug repurposing, as the interaction between a drug and its target largely determines therapeutic efficacy and potential off-target effects. Given the vast combinatorial space of possible drug-target pairs and the high

[... deterministic omitted span ...]

chemistry  drug discovery with deep learning generative  models

[41] Structure-based drug discovery with deep learning

[42] Knowledge-augmented Graph Machine Learning for Drug Discovery  A Survey  from Precision to Interpretability

[43] Large AI Models in Health Informatics  Applications, Challenges, and the  Future

[44] Towards Unified AI Drug Discovery with Multiple Knowledge Modalities

[45] Physical formula enhanced multi-task learning for pharmacokinetics  prediction

[46] TorchDrug  A Powerful and Flexible Machine Learning Platform for Drug  Discovery

[47] New Horizons  Pioneering Pharmaceutical R&D with Generative AI from lab  to the clinic -- an industry perspective

[48] Bioinformatics and Medicine in the Era of Deep Learning

[49] Large-scale ligand-based virtual screening for SARS-CoV-2 inhibitors  using deep neural networks

[50] Deep Learning for Virtual Screening  Five Reasons to Use ROC Cost  Functions

[51] A Novel Framework Integrating AI Model and Enzymological Experiments  Promotes Identification of SARS-CoV-2 3CL Protease Inhibitors and  Activity-based Probe

[52] Repurpose Open Data to Discover Therapeutics for COVID-19 using Deep  Learning

[53] Drug repurposing for COVID-19 using graph neural network and harmonizing  multiple evidence

[54] Scalable HPC and AI Infrastructure for COVID-19 Therapeutics

[55] Deep learning for drug repurposing  methods, databases, and applications

[56] Machine-Learning Driven Drug Repurposing for COVID-19

[57] Evening the Score  Targeting SARS-CoV-2 Protease Inhibition in Graph  Generative Models for Therapeutic Candidates

[58] Bridging the gap between target-based and cell-based drug discovery with  a graph generative multi-task model

[59] Optimizing Drug Design by Merging Generative AI With Active Learning  Frameworks

[60] On How AI Needs to Change to Advance the Science of Drug Discovery

[61] Analyzing Learned Molecular Representations for Property Prediction

[62] Step Change Improvement in ADMET Prediction with PotentialNet Deep  Featurization

[63] Benchmarking Deep Graph Generative Models for Optimizing New Drug  Molecules for COVID-19

[64] Molecule Generation for Drug Design  a Graph Learning Perspective

[65] HiGraphDTI  Hierarchical Graph Representation Learning for Drug-Target  Interaction Prediction

[66] Comprehensive evaluation of deep and graph learning on drug-drug  interactions prediction

[67] Multi-View Self-Attention for Interpretable Drug-Target Interaction  Prediction

[68] Improved Drug-target Interaction Prediction with Intermolecular Graph  Transformer

[69] A Systematic Survey in Geometric Deep Learning for Structure-based Drug  Design

[70] Structure-based drug design with geometric deep learning

[71] Few-shot link prediction via graph neural networks for Covid-19  drug-repurposing

[72] Utilising Graph Machine Learning within Drug Discovery and Development

[73] Learn molecular representations from large-scale unlabeled molecules for  drug discovery

[74] KPGT  Knowledge-Guided Pre-training of Graph Transformer for Molecular  Property Prediction

[75] MultiModal-Learning for Predicting Molecular Properties  A Framework  Based on Image and Graph Structures

[76] Multi-Modal Representation Learning for Molecular Property Prediction   Sequence, Graph, Geometry

[77] Validating the Validation  Reanalyzing a large-scale comparison of Deep  Learning and Machine Learning models for bioactivity prediction

[78] Drug discovery with explainable artificial intelligence

[79] Attention-based Multi-Input Deep Learning Architecture for Biological  Activity Prediction  An Application in EGFR Inhibitors

[80] Multi-Objective Molecule Generation using Interpretable Substructures

[81] Improving detection of protein-ligand binding sites with 3D segmentation

[82] Meaningful machine learning models and machine-learned pharmacophores  from fragment screening campaigns

[83] Deep Denerative Models for Drug Design and Response

[84] The Synthesizability of Molecules Proposed by Generative Models

[85] An open unified deep graph learning framework for discovering drug leads

[86] Drug-Target Indication Prediction by Integrating End-to-End Learning and  Fingerprints

[87] A hybrid quantum-classical fusion neural network to improve  protein-ligand binding affinity predictions for drug discovery

[88] Stepping Back to SMILES Transformers for Fast Molecular Representation  Inference

[89] SMILES Transformer  Pre-trained Molecular Fingerprint for Low Data Drug  Discovery

[90] SELFormer  Molecular Representation Learning via SELFIES Language Models

[91] Large-Scale Chemical Language Representations Capture Molecular  Structure and Properties

[92] Integrating Chemical Language and Molecular Graph in Multimodal Fused  Deep Learning for Drug Property Prediction

[93] Molecular Joint Representation Learning via Multi-modal Information

[94] Graph Neural Networks for Molecules

[95] Molecular Property Prediction  A Multilevel Quantum Interactions  Modeling Perspective

[96] Motif-based Graph Self-Supervised Learning for Molecular Property  Prediction

[97] Heterogeneous Molecular Graph Neural Networks for Predicting Molecule  Properties

[98] Neural Message Passing with Edge Updates for Predicting Properties of  Molecules and Materials

[99] Flexible dual-branched message passing neural network for quantum  mechanical property prediction with molecular conformation

[100] Simplicial Message Passing for Chemical Property Prediction

[101] Heterogeneous relational message passing networks for molecular dynamics  simulations

[102] HiGNN  Hierarchical Informative Graph Neural Networks for Molecular  Property Prediction Equipped with Feature-Wise Attention

[103] 3D Infomax improves GNNs for Molecular Property Prediction

[104] GeomGCL  Geometric Graph Contrastive Learning for Molecular Property  Prediction

[105] ChemRL-GEM  Geometry Enhanced Molecular Representation Learning for  Property Prediction

[106] Molecular Hypergraph Neural Networks

[107] Comparison of Atom Representations in Graph Neural Networks for  Molecular Property Prediction

[108] Motif-aware Attribute Masking for Molecular Graph Pre-training

[109] Unified 2D and 3D Pre-Training of Molecular Representations

[110] Automated 3D Pre-Training for Molecular Property Prediction

[111] Bi-level Contrastive Learning for Knowledge-Enhanced Molecule  Representations

[112] Molecular Representation Learning via Heterogeneous Motif Graph Neural  Networks

[113] Uni-QSAR  an Auto-ML Tool for Molecular Property Prediction

[114] AI-Bind  Improving Binding Predictions for Novel Protein Targets and  Ligands

[115] Expanding Chemical Representation with k-mers and Fragment-based  Fingerprints for Molecular Fingerprinting

[116] CheMixNet  Mixed DNN Architectures for Predicting Chemical Properties  using Multiple Molecular Representations

[117] Predicting Chemical Properties using Self-Attention Multi-task Learning  based on SMILES Representation

[118] ToDD  Topological Compound Fingerprinting in Computer-Aided Drug  Discovery

[119] Multiparameter Persistent Homology for Molecular Property Prediction

[120] SE(3)-Invariant Multiparameter Persistent Homology for Chiral-Sensitive  Molecular Property Prediction

[121] Pharmacoprint -- a combination of pharmacophore fingerprint and  artificial intelligence as a tool for computer-aided drug design

[122] Sort & Slice  A Simple and Superior Alternative to Hash-Based Folding  for Extended-Connectivity Fingerprints

[123] Molecular Graph Convolutions  Moving Beyond Fingerprints

[124] SPECTRe  Substructure Processing, Enumeration, and Comparison Tool  Resource  An efficient tool to encode all substructures of molecules  represented in SMILES

[125] Augmenting Molecular Images with Vector Representations as a  Featurization Technique for Drug Classification

[126] Geometry-aware Line Graph Transformer Pre-training for Molecular  Property Prediction

[127] Multi-View Graph Neural Networks for Molecular Property Prediction

[128] Directed Message Passing Based on Attention for Prediction of Molecular  Properties

[129] Zero-Shot 3D Drug Design by Sketching and Generating

[130] Energy-based Generative Models for Target-specific Drug Discovery

[131] NovoMol  Recurrent Neural Network for Orally Bioavailable Drug Design  and Validation on PDGFRα Receptor

[132] Deep Learning Methods for Small Molecule Drug Discovery  A Survey

[133] Advanced Graph and Sequence Neural Networks for Molecular Property  Prediction and Drug Discovery

[134] Grammars and reinforcement learning for molecule optimization

[135] Symmetry-adapted generation of 3d point sets for the targeted discovery  of molecules

[136] Augmenting Molecular Deep Generative Models with Topological Data  Analysis Representations

[137] Drug Synergistic Combinations Predictions via Large-Scale Pre-Training  and Graph Structure Learning

[138] Explainable Artificial Intelligence for Drug Discovery and Development  -- A Comprehensive Survey

[139] Otter-Knowledge  benchmarks of multimodal knowledge graph representation  learning from different sources for drug discovery

[140] Leveraging binding-site structure for drug discovery with point-cloud  methods

[141] PIGNet2  A Versatile Deep Learning-based Protein-Ligand Interaction  Prediction Model for Binding Affinity Scoring and Virtual Screening

[142] PotentialNet for Molecular Property Prediction

[143] CT-ADE  An Evaluation Benchmark for Adverse Drug Event Prediction from  Clinical Trial Results

[144] Zero-shot Learning of Drug Response Prediction for Preclinical Drug  Screening

[145] We Should at Least Be Able to Design Molecules That Dock Well

[146] RECOVER  sequential model optimization platform for combination drug  repurposing identifies novel synergistic compounds in vitro

[147] Visualizing Deep Graph Generative Models for Drug Discovery

[148] Knowledge-informed Molecular Learning  A Survey on Paradigm Transfer

[149] Can Large Language Models Empower Molecular Property Prediction 

[150] Comparative Analysis of LLaMA and ChatGPT Embeddings for Molecule  Embedding

[151] GPT-MolBERTa  GPT Molecular Features Language Model for molecular  property prediction

[152] MolXPT  Wrapping Molecules with Text for Generative Pre-training

[153] Multilingual Molecular Representation Learning via Contrastive  Pre-training

[154] IMG2SMI  Translating Molecular Structure Images to Simplified  Molecular-input Line-entry System

[155] Domain-Agnostic Molecular Generation with Chemical Feedback

[156] Text-Guided Molecule Generation with Diffusion Language Model

[157] Chemical-Reaction-Aware Molecule Representation Learning

[158] Recent advances in the Self-Referencing Embedding Strings (SELFIES)  library

[159] SMILES2Vec  An Interpretable General-Purpose Deep Neural Network for  Predicting Chemical Properties

[160] Molecular Contrastive Learning with Chemical Element Knowledge Graph
