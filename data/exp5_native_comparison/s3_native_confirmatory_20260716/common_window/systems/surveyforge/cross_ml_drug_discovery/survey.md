# Registered common-window excerpt

# Machine Learning for Drug Discovery: Methods, Applications, and Future Directions

## 1 Introduction

Drug discovery represents one of the most complex and resource-intensive endeavors in modern science, with traditional pipelines requiring over a decade of development and exceeding \$2.5 billion per approved therapeutic, yet suffering from attrition rates exceeding 90% in clinical stages. These systemic inefficiencies have catalyzed the adoption of computational approaches, evolving from early quantitative structure-activity relationship models and rule-based cheminformatics toward sophisticated data-driven machine learning frameworks [1; 2]. The emergence of deep learning, graph neural networks, and large-scale pretrained architectures has fundamentally transformed this landscape, enabling unprecedented predictive accuracy across molecular property prediction, virtual screening, and generative molecular design [3; 4; 5]. This survey comprehensively examines machine learning methods spanning molecular representations, property prediction, generative design, drug-target interaction modeling, and systems-level discovery, supported by standardized benchmarking platforms and open-source infrastructure [6; 7; 8] that collectively enable reproducible and translatable computational drug discovery research.

## 2 Molecular Representations and Data Infrastructure

### 2.1 Classical Molecular Descriptors and Fingerprint-Based Encodings

I need to verify each citation against the paper contents provided.

1. "Hand-crafted molecular representations...remain highly competitive, particularly in low-data pharmaceutical settings [3; 8]" - MoleculeNet does benchmark molecular representations. AMPL discusses benchmarking. Both support this. ✓

2. "Physicochemical descriptors...serve as interpretable baselines in quantitative structure-activity relationship modeling [1; 2]" - Both papers discuss physicochemical descriptors and QSAR. ✓

3. "Structural fingerprints such as Morgan/extended-connectivity fingerprints...enabling Tanimoto-based similarity computations central to ligand-based virtual screening [9]" - This paper discusses NLP for chemicals and mentions text-based representations, but doesn't specifically focus on Morgan fingerprints and Tanimoto similarity for virtual screening. This citation is weak. The paper about virtual screening or MoleculeNet would be more appropriate. However, from available papers, none specifically focus on Morgan fingerprints and Tanimoto similarity. I'll remove this citation or check if MoleculeNet supports it better - MoleculeNet does discuss molecular featurization including fingerprints. I'll change to [3].

4. "benchmarking studies confirm that physicochemical descriptors and graph representations collectively outperform traditional fingerprints in large-data regimes [8]" - AMPL explicitly states this finding. ✓

5. "Topological extensions leveraging multiparameter persistence homology...demonstrating statistically significant gains [10]" - ToDD is exactly about this. ✓

Hand-crafted molecular representations encode chemical structure into fixed-length numerical vectors without requiring neural network training, offering computational efficiency and interpretability that remain highly competitive, particularly in low-data pharmaceutical settings [3; 7]. Physicochemical descriptors—including molecular weight, topological polar surface area, hydrogen bond donor and acceptor counts, logP, and rotatable bond counts—directly encode Lipinski-style drug-likeness criteria and serve as interpretable baselines in quantitative structure-activity relationship modeling [1; 2]. Structural fingerprints such as Morgan/extended-connectivity fingerprints, MACCS keys, and Daylight-like hashed fingerprints map circular atomic neighborhoods to fixed-length bit vectors through hashing procedures, enabling Tanimoto-based similarity computations central to ligand-based virtual screening [3]. Notably, benchmarking studies confirm that physicochemical descriptors and graph representations collectively outperform traditional fingerprints in large-data regimes, yet fingerprint-based methods remain competitive when labeled data is scarce [7]. Topological extensions leveraging multiparameter persistence homology further enrich classical fingerprints by capturing multiscale structural information, demonstrating statistically significant gains over canonical representations on screening benchmarks [10]. Despite their limitations in encoding conformation-dependent and context-sensitive properties, classical descriptors retain indispensable value as efficient, interpretable baselines against which all learned representations must be rigorously benchmarked.

### 2.2 String-Based Molecular Representations and Chemical Language Models

SMILES strings and related linear notations treat molecules as sequences amenable to natural language processing architectures, establishing a productive analogy between chemical grammar and linguistic syntax that complements the physicochemical and fingerprint-based representations discussed above [11]. Early recurrent approaches demonstrated that LSTM networks trained on large SMILES corpora could learn valid chemical syntax and generate novel drug-like structures, with pretraining on unlabeled molecular databases proving especially beneficial in low-data regimes [12]. The SMILES Transformer extended this paradigm through unsupervised sequence-to-sequence pretraining, yielding transferable fingerprints that outperformed rule-based encodings on small datasets [12]. Critically, non-canonical SMILES augmentation—randomizing the traversal order of molecular graphs to generate multiple valid string representations of identical structures—substantially improves model generalization by exposing the encoder to diverse syntactic realizations of the same chemical entity [13]. Transformer-based chemical language models further advanced representation quality through masked pretraining objectives analogous to BERT, with systematic surveys confirming that complementary featurizations combining 1D strings with 2D and 3D information outperform any single modality [14; 15]. Nevertheless, comparative benchmarking reveals that pretrained molecular embeddings do not universally surpass classical fingerprints, underscoring that data scale, pretraining curriculum, and downstream task characteristics collectively govern transfer effectiveness [16; 17]. While sequence-based encodings offer powerful expressiveness through language model pretraining, they inherently linearize molecular topology—a limitation that graph-based representations, discussed in the following section, directly address by preserving the full two-dimensional structure of chemical compounds.

### 2.3 Graph-Based Molecular Representations and Message-Passing Architectures

I need to verify each citation against the paper contents provided.

1. "making graph neural networks (GNNs) an architecturally well-matched framework for molecular machine learning [18; 19]" - Both papers support this. ✓

2. "Node features typically encode atomic number, hybridization state, formal charge, chirality, and aromaticity, while edge features capture bond type, conjugation, and ring membership—choices that substantially influence downstream predictive performance [19; 20]" - Both papers discuss molecular graph features and their influence on performance. ✓

3. "The message-passing neural network (MPNN) framework iteratively updates node representations by aggregating neighborhood information through learnable functions, enabling end-to-end learning of chemically meaningful local and global structural features [21]" - ChemRL-GEM discusses GNN architecture for molecular representation but the MPNN framework specifically is not the primary focus. However, it does describe GNN architecture with atom-bond relations. The citation is reasonable but let me check if Graph-based Molecular Representation Learning would be more appropriate. The Graph-based survey would better support MPNN description. I'll add it.

4. "well-designed GNNs match or outperform fixed fingerprint baselines [19; 7]" - Both papers support this claim. ✓

5. "geometry-enhanced variants incorporating bond angles and three-dimensional information providing further improvements [21]" - ChemRL-GEM explicitly discusses bond angles and 3D information. ✓

Molecules are naturally representable as attributed graphs in which atoms constitute nodes and chemical bonds constitute edges, making graph neural networks (GNNs) an architecturally well-matched framework for molecular machine learning [18; 19]. Node features typically encode atomic number, hybridization state, formal charge, chirality, and aromaticity, while edge features capture bond type, conjugation, and ring membership—choices that substantially influence downstream predictive performance [19; 20]. The message-passing neural network (MPNN) framework iteratively updates node representations by aggregating neighborhood information through learnable functions, enabling end-to-end learning of chemically meaningful local and global structural features [18; 21]. Empirical benchmarking across diverse ADMET and bioactivity datasets consistently demonstrates that well-designed GNNs match or outperform fixed fingerprint baselines [19; 7], with geometry-enhanced variants incorporating bond angles and three-dimensional information providing further improvements [21].

### 2.4 Three-Dimensional Geometric Representations and Equivariant Neural Architectures

Three-dimensional geometric representations encode atomic coordinates explicitly, capturing the conformational and spatial information that governs binding selectivity and pharmacological activity far beyond what topological graphs convey. Building naturally upon the graph-based molecular representations established in the preceding discussion, where geometry-enhanced variants already demonstrated improvements through bond angle encoding, fully three-dimensional approaches make atomic coordinates a first-class component of the representation rather than an auxiliary feature. Approaches range from voxelized density grids and point clouds to distance-angle matrices, each offering distinct trade-offs between computational cost and geometric fidelity [22; 23]. A central architectural imperative is respecting the physical symmetry group SE(3): predictions of scalar properties must be invariant to rotation and translation, while intermediate vector-valued features must transform equivariantly, a requirement that has driven the development of specialized message-passing frameworks [21; 24]. Geometry-enhanced representations that simultaneously encode bond lengths, angles, and torsions demonstrably improve quantum property prediction and binding affinity estimation over purely topological counterparts [21; 23; 25]. Joint 2D–3D generative and predictive frameworks further illustrate that topological and geometric modalities contain complementary information whose fusion yields superior molecular design outcomes [26; 24]. Together, these advances point toward unified architectures that co-learn structure and property simultaneously—a trajectory that directly motivates the need for

[... deterministic omitted span ...]

with the citations retained:

Computational drug repurposing leverages existing pharmacological knowledge to identify novel therapeutic indications, substantially reducing development timelines and costs relative to de novo discovery. Similarity-based approaches exploit chemical, target, and side-effect profile overlap to infer drug-disease relationships, while network propagation algorithms diffuse known associations through protein interaction graphs to surface mechanistically plausible candidates. Transcriptomics-based methods, inspired by connectivity map frameworks, match drug-induced gene expression signatures against disease profiles, with deep learning architectures trained on approved drug-disease pairs demonstrating marked improvements over classical baselines. Multi-modal evidence integration—combining structural molecular features, knowledge graph embeddings, and real-world pharmacovigilance records—further enriches repurposing hypotheses, as exemplified by multi-agent systems that coordinate drug-target interaction modeling with biomedical literature search. The COVID-19 pandemic accelerated adoption of these pipelines, highlighting both their speed advantages and the critical need for causal inference frameworks to distinguish genuine therapeutic signals from confounding associations in observational data. Future progress hinges on tighter coupling between computational predictions and prospective clinical validation, alongside richer multi-modal representations [38; 73; 9] that capture the full complexity of drug-disease biology.

### 5.4 Drug Combination Synergy Prediction and Multi-Target Therapy

The combinatorial explosion of possible drug pairs and triplets renders exhaustive experimental synergy profiling computationally and economically intractable, motivating machine learning models that integrate multi-omics cell line profiles with molecular drug representations to predict synergistic interactions at scale. Building upon the drug-cell interaction frameworks established in single-agent response modeling, deep learning architectures encoding drug structural features alongside genomic, transcriptomic, and copy number variation profiles of cancer cell lines have demonstrated strong predictive performance, with graph neural networks and attention mechanisms capturing complex drug-cell interactions that simpler concatenation-based baselines cannot resolve. Biological network-informed approaches further enrich synergy prediction by embedding gene regulatory and signaling pathway graphs, enabling identification of mechanistically meaningful subnetworks rather than purely statistical correlations, thereby supporting interpretable oncology decision-making. Despite progress, a critical generalization gap persists between interpolation over known drug-cell combinations and extrapolation to entirely novel compounds or unseen cell lines, underscoring the need for few-shot and large language model-based strategies that leverage pre-trained biological priors to predict synergy with minimal labeled examples [14; 74]. Future advances will require standardized evaluation protocols distinguishing these distinct prediction scenarios alongside tighter integration of patient omics profiling to translate computational synergy predictions into personalized combination treatment selection, a challenge that directly connects to the broader imperative of reconciling heterogeneous molecular data layers addressed in multi-omics integration frameworks for cellular drug response modeling.

### 5.5 Multi-Omics Integration and Cellular Drug Response Modeling

Looking at the subsection, I need to verify the two citations:

1. `[3]` - This citation is used to support the claim about "Pharmacogenomic screens such as GDSC and CCLE have enabled deep learning models to jointly encode drug molecular structure alongside cell line genomic, transcriptomic, and copy number profiles." Looking at the MoleculeNet paper, it is about benchmarking molecular machine learning with curated datasets, metrics, and implementations. It does not specifically discuss GDSC, CCLE, or integrating cell line genomic/transcriptomic profiles for drug response prediction. This citation is incorrect and should be removed.

2. `[2]` - This citation is used to support "Integrating proteomic layers alongside transcriptomic signals further enriches causal resolution, moving beyond correlative genotype-phenotype mappings toward mechanistic models of how perturbations propagate through biological networks to produce observable phenotypes." The Deep Learning for Computational Chemistry paper is a review of deep learning in computational chemistry covering QSAR, virtual screening, protein structure prediction, etc. It does not specifically address multi-omics integration or proteomic/transcriptomic integration for drug response. This citation is also incorrect and should be removed.

Here is the corrected subsection:

Multi-omics integration for cellular drug response modeling represents a critical frontier where heterogeneous molecular data layers must be reconciled to yield mechanistically grounded, clinically translatable predictions. Machine learning frameworks addressing this challenge span concatenation-based fusion, autoencoder-driven dimensionality reduction, and attention-based multi-modal architectures, each offering distinct trade-offs between expressivity and interpretability. Pharmacogenomic screens such as GDSC and CCLE have enabled deep learning models to jointly encode drug molecular structure alongside cell line genomic, transcriptomic, and copy number profiles, establishing the empirical foundation for data-driven response prediction. Transfer learning strategies bridging bulk RNA-seq screens to single-cell resolution—exemplified by encoder-decoder architectures that leverage architecture surgery to incorporate high-throughput perturbational knowledge—substantially expand the predictable perturbation space beyond experimentally covered compounds, enabling in silico extrapolation critical for resource-constrained experimental designs. Integrating proteomic layers alongside transcriptomic signals further enriches causal resolution, moving beyond correlative genotype-phenotype mappings toward mechanistic models of how perturbations propagate through biological networks to produce observable phenotypes.

### 5.6 Systems Pharmacology and Network-Based Target Identification

Systems pharmacology extends beyond isolated drug-target pairs to model drugs, targets, pathways, and disease biology as interconnected networks, enabling identification of novel therapeutic targets and rational multi-target intervention strategies. Building upon the multi-omics cellular response modeling described above, these network-centric approaches provide a complementary systems-level perspective that contextualizes molecular perturbations within broader biological architectures. Network-based propagation methods applied to protein-protein interaction and gene regulatory networks have demonstrated strong capacity for disease module identification and target prioritization, with graph neural network-based approaches increasingly replacing classical network proximity measures by learning expressive representations over multi-relational biological graphs [30; 35]. Knowledge graph embedding frameworks that encode drug-target-disease-pathway quadruples further support polypharmacology inference and repurposing signal detection at population scale [75; 76]. Multi-omics integration, including genomic, transcriptomic, and proteomic layers, strengthens target nomination for multigenic diseases where single-target strategies have historically failed [77; 78]. Together, these network and knowledge graph frameworks naturally converge with the cellular drug response models discussed previously, as understanding how drugs propagate signals through biological networks ultimately informs both target selection and the mechanistic interpretation of phenotypic outcomes. Future advances lie in causally grounded network models that distinguish driver perturbations from passenger correlations, enabling more reliable prioritization of therapeutically actionable targets within complex disease networks.

## 6 Advanced Learning Paradigms and Specialized Methodologies

### 6.1 Transfer Learning, Pre-Training, and Self-Supervised Learning for Molecular Applications

I need to check each citation against the paper contents provided.

1. "...pre-trained on large unlabeled or broadly labeled corpora encode transferable chemical and biological knowledge that substantially reduces the labeled data requirements for downstream fine-tuning [4; 3]" - Massively Multitask Networks is about multitask learning (relevant). MoleculeNet is about benchmarking. Both are relevant but not directly about transfer learning reducing labeled data requirements. Keep as reasonable support.

2. "...models such as MolE adapting the DeBERTa architecture with a two-stage strategy combining structural self-supervision with large-scale multi-task biological property learning to achieve state-of-the-art performance across diverse ADMET benchmarks [38]" - Directly supported. Correct.

3. "...HelixADMET system incorporates self-supervised pre-training followed by multi-task and multi-stage fine-tuning... [79]" - Directly supported. Correct.

4. "...pre-trained graph representations broadly outperform models trained from scratch in low-data regimes [3; 7]" - MoleculeNet supports learnable representations being powerful. AMPL supports this. Correct.

5. "...dataset size directly correlates with predictive performance, and that single-task deep learning only surpasses shallow learners when sufficient labeled data are available [7]" - Directly supported by AMPL paper content. Correct.

6. "...one-shot learning architectures combining residual LSTM embeddings with graph convolutional networks... [72]" - Directly supported. Correct.

The last sentence about "physics-informed pre-training...joint 2D-3D self-supervised objectives" has no citation, which is fine.

All citations appear correct. Here is the subsection:

Transfer learning and self-supervised pre-training have emerged as transformative paradigms for addressing the pervasive data scarcity challenge in pharmaceutical machine learning, where labeled bioactivity measurements are expensive and sparse relative to the vastness of chemical space. The central insight is that expressive molecular representations pre-trained on large unlabeled or broadly labeled corpora encode transferable chemical and biological knowledge that substantially reduces the labeled data requirements for downstream fine-tuning [4; 3]. Transformer-based chemical language models pre-trained on massive SMILES corpora through masked token prediction and related self-supervised objectives have demonstrated that sequence-level pre-training captures meaningful substructural patterns, with models such as MolE adapting the DeBERTa architecture with a two-stage strategy combining structural self-supervision with large-scale multi-task biological property learning to achieve state-of-the-art performance across diverse ADMET benchmarks [38]. Complementarily, the HelixADMET system

[... deterministic omitted span ...]

Hybrid human-in-the-loop workflows, where medicinal chemistry expertise intervenes at critical decision boundaries, remain pragmatically essential for ensuring scientific rigor and synthesizability [27]. Establishing community benchmarks that fairly evaluate autonomous platforms against traditional workflows constitutes an urgent open challenge for the field.

## 8 Conclusion

I need to check each citation in the subsection to verify it supports the corresponding claim.

1. "from classical fingerprint-based QSAR models to equivariant geometric deep learning, generative architectures, and autonomous closed-loop platforms [2; 3]" - Both papers cover deep learning for computational chemistry and molecular machine learning benchmarks, supporting this historical overview. ✓

2. "data efficiency [72]" - This paper is specifically about one-shot learning to address low data problems in drug discovery. ✓

3. "generalization under distribution shift [71]" - This paper is specifically about OOD generalization. ✓

4. "interpretability [96]" - This paper is specifically about interpretability in drug discovery deep learning. ✓

5. "foundation models, federated learning [4]" - The Massively Multitask Networks paper is about multitask learning, not federated learning specifically. The citation for "federated learning" is incorrect. I should remove this citation or keep it only for the foundation models/multitask aspect. Since the paper doesn't cover federated learning, I'll remove the citation from "federated learning."

6. "synthesis-aware optimization [46]" - This paper is specifically about synthesis-aware molecular design. ✓

7. "autonomous discovery systems [97]" - This paper covers autonomous discovery. ✓

This survey has traced the remarkable transformation of drug discovery through machine learning, from classical fingerprint-based QSAR models to equivariant geometric deep learning, generative architectures, and autonomous closed-loop platforms [2; 3]. Cross-cutting themes of data efficiency [72], generalization under distribution shift [71], and interpretability [96] persistently challenge the field, while foundation models, federated learning, and synthesis-aware optimization [46] chart compelling forward trajectories toward genuinely predictive, clinically translatable, and autonomous discovery systems [97].

## References

[1] Artificial Intelligence in Drug Discovery  Applications and Techniques

[2] Deep Learning for Computational Chemistry

[3] MoleculeNet  A Benchmark for Molecular Machine Learning

[4] Massively Multitask Networks for Drug Discovery

[5] Structure-based drug discovery with deep learning

[6] Therapeutics Data Commons  Machine Learning Datasets and Tasks for Drug  Discovery and Development

[7] AMPL  A Data-Driven Modeling Pipeline for Drug Discovery

[8] TorchDrug  A Powerful and Flexible Machine Learning Platform for Drug  Discovery

[9] Exploring Chemical Space using Natural Language Processing Methodologies  for Drug Discovery

[10] ToDD  Topological Compound Fingerprinting in Computer-Aided Drug  Discovery

[11] Automatic chemical design using a data-driven continuous representation  of molecules

[12] SMILES Transformer  Pre-trained Molecular Fingerprint for Low Data Drug  Discovery

[13] Improving Chemical Autoencoder Latent Space and Molecular De novo  Generation Diversity with Heteroencoders

[14] A Systematic Survey of Chemical Pre-trained Models

[15] Improving Molecular Pretraining with Complementary Featurizations

[16] Using Molecular Embeddings in QSAR Modeling  Does it Make a Difference 

[17] Can Pre-trained Models Really Learn Better Molecular Representations for  AI-aided Drug Discovery 

[18] Graph-based Molecular Representation Learning

[19] Analyzing Learned Molecular Representations for Property Prediction

[20] KPGT  Knowledge-Guided Pre-training of Graph Transformer for Molecular  Property Prediction

[21] ChemRL-GEM  Geometry Enhanced Molecular Representation Learning for  Property Prediction

[22] Generating 3D Molecules for Target Protein Binding

[23] Structure-aware Interactive Graph Neural Networks for the Prediction of  Protein-Ligand Binding Affinity

[24] An Equivariant Generative Framework for Molecular Graph-Structure  Co-Design

[25] GFlowNets for AI-Driven Scientific Discovery

[26] Learning Joint 2D & 3D Diffusion Models for Complete Molecule Generation

[27] A Systematic Survey in Geometric Deep Learning for Structure-based Drug  Design

[28] GEFA  Early Fusion Approach in Drug-Target Affinity Prediction

[29] DeepAffinity  Interpretable Deep Learning of Compound-Protein Affinity  through Unified Recurrent and Convolutional Neural Networks

[30] A Review of Biomedical Datasets Relating to Drug Discovery  A Knowledge  Graph Perspective

[31] DeepDTA  Deep Drug-Target Binding Affinity Prediction

[32] Machine learning prediction of cancer cell sensitivity to drugs based on  genomic and chemical properties

[33] Predicting drug response of tumors from integrated genomic profiles by  deep neural networks

[34] GEOM  Energy-annotated molecular conformations for property prediction  and molecular generation

[35] Understanding the Performance of Knowledge Graph Embeddings in Drug  Discovery

[36] PotentialNet for Molecular Property Prediction

[37] Advanced Graph and Sequence Neural Networks for Molecular Property  Prediction and Drug Discovery

[38] MolE  a molecular foundation model for drug discovery

[39] Machine Learning Small Molecule Properties in Drug Discovery

[40] Benchmarking Accuracy and Generalizability of Four Graph Neural Networks  Using Large In Vitro ADME Datasets from Different Chemical Spaces

[41] FP-GNN  a versatile deep learning architecture for enhanced molecular  property prediction

[42] Predicting Aqueous Solubility of Organic Molecules Using Deep Learning  Models with Varied Molecular Representations

[43] Deep Learning Based Regression and Multi-class Models for Acute Oral  Toxicity Prediction with Automatic Chemical Feature Extraction

[44] Improving VAE based molecular representations for compound property  prediction

[45] Molecule-Morphology Contrastive Pretraining for Transferable Molecular  Representation

[46] Molecular Design in Synthetically Accessible Chemical Space via Deep  Reinforcement Learning

[47] GAUCHE  A Library for Gaussian Processes in Chemistry

[48] Learning Invariant Molecular Representation in Latent Discrete Space

[49] Generative chemistry  drug discovery with deep learning generative  models

[50] Deep Reinforcement Learning for De-Novo Drug Design

[51] Utilizing Reinforcement Learning for de novo Drug Design

[52] LIMO  Latent Inceptionism for Targeted Molecule Generation

[53] Latent Molecular Optimization for Targeted Therapeutic Design

[54] Artificial Intelligence for Drug Discovery  Are We There Yet 

[55] Molecule Generation for Drug Design  a Graph Learning Perspective

[56] Molecular Graph Convolutions  Moving Beyond Fingerprints

[57] Deep Learning Methods for Small Molecule Drug Discovery  A Survey

[58] An efficient graph generative model for navigating ultra-large  combinatorial synthesis libraries

[59] MAGNet  Motif-Agnostic Generation of Molecules from Shapes

[60] Generation of 3D Molecules in Pockets via Language Model

[61] Scalable Fragment-Based 3D Molecular Design with Reinforcement Learning

[62] Graph Convolutional Policy Network for Goal-Directed Molecular Graph  Generation

[63] GraphAF  a Flow-based Autoregressive Model for Molecular Graph  Generation

[64] MolGAN  An implicit generative model for small molecular graphs

[65] Differentiable Scaffolding Tree for Molecular Optimization

[66] Learning to Extend Molecular Scaffolds with Structural Motifs

[67] A 3D Generative Model for Structure-Based Drug Design

[68] Navigating the Design Space of Equivariant Diffusion-Based Generative  Models for De Novo 3D Molecule Generation

[69] Geometry-Complete Diffusion for 3D Molecule Generation and Optimization

[70] Structure-based drug design with geometric deep learning

[71] DrugOOD  Out-of-Distribution (OOD) Dataset Curator and Benchmark for  AI-aided Drug Discovery -- A Focus on Affinity Prediction Problems with Noise  Annotations

[72] Low Data Drug Discovery with One-shot Learning

[73] MolTrans  Molecular Interaction Transformer for Drug Target Interaction  Prediction

[74] DrugChat  Towards Enabling ChatGPT-Like Capabilities on Drug Molecule  Graphs

[75] SumGNN  Multi-typed Drug Interaction Prediction via Efficient Knowledge  Graph Summarization

[76] Mining On Alzheimer's Diseases Related Knowledge Graph to Identity  Potential AD-related Semantic Triples for Drug Repurposing

[77] Deep Learning in Pharmacogenomics  From Gene Regulation to Patient  Stratification

[78] Large-scale benchmark study of survival prediction methods using  multi-omics data

[79] HelixADMET  a robust and endpoint extensible ADMET system incorporating  self-supervised knowledge transfer

[80] Quantitative Evaluation of Explainable Graph Neural Networks for  Molecular Property Prediction

[81] Learning Graph-Level Representation for Drug Discovery

[82] CheMixNet  Mixed DNN Architectures for Predicting Chemical Properties  using Multiple Molecular Representations

[83] Pre-training of Equivariant Graph Matching Networks with Conformation  Flexibility for Drug Binding

[84] PIGNet2  A Versatile Deep Learning-based Protein-Ligand Interaction  Prediction Model for Binding Affinity Scoring and Virtual Screening

[85] EquiBind  Geometric Deep Learning for Drug Binding Structure Prediction

[86] Property-Aware Relation Networks for Few-Shot Molecular Property  Prediction

[87] Molecular representation learning with language models and  domain-relevant auxiliary tasks

[88] Validating the Validation  Reanalyzing a large-scale comparison of Deep  Learning and Machine Learning models for bioactivity prediction

[89] Learning Topology-Specific Experts for Molecular Property Prediction

[90] Calibration and generalizability of probabilistic models on low-data  chemical datasets with DIONYSUS

[91] Persistent spectral based machine learning (PerSpect ML) for drug design

[92] Learning Subpocket Prototypes for Generalizable Structure-based Drug  Design

[93] On the String Kernel Pre-Image Problem with Applications in Drug  Discovery

[94] Geometric Deep Learning on Molecular Representations

[95] Molformer  Motif-based Transformer on 3D Heterogeneous Molecular Graphs

[96] Interpretable Deep Learning in Drug Discovery

[97] Autonomous discovery in the chemical sciences part II  Outlook
