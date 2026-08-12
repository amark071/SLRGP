# Registered common-window excerpt

# Machine Learning for Drug Discovery: A Comprehensive Survey

## Introduction and Background

### The Traditional Drug Discovery Pipeline

The traditional drug discovery pipeline is a multi-stage, resource-intensive process encompassing target identification, hit discovery, lead optimization, preclinical evaluation, and clinical trials ref [28]. Bringing a novel therapeutic to market typically requires 10–15 years and investments exceeding one billion USD, with high attrition rates persisting across all development phases ref [33]. Scientists conventionally identify disease-relevant biological targets and systematically screen large chemical libraries to identify active compounds, followed by iterative optimization of pharmacokinetic and safety profiles ref [2]. Preclinical studies assess efficacy and toxicity before candidates advance through Phase I–III clinical trials ref [21]. Despite substantial investment, the overall success rate remains critically low, underscoring the urgent need for more efficient, data-driven methodologies to transform conventional drug development ref [49].

### Challenges and Limitations of Conventional Approaches

Conventional drug discovery faces profound bottlenecks that severely constrain its productivity and sustainability. The process is characterized by prohibitively high costs, with the R&D phase alone requiring hundreds of millions to over a billion dollars, and timelines spanning a decade or more ref [55]. Attrition rates remain alarmingly high, as the vast majority of candidate compounds fail during preclinical or clinical stages ref [33]. Traditional high-throughput screening yields hit rates of merely 1–2%, reflecting the immense difficulty of navigating expansive chemical and biological spaces ref [20]. Furthermore, experimental approaches for identifying drug-target interactions are labor-intensive, costly, and temporally inefficient ref [34]. These compounding limitations—low throughput, high failure rates, and resource intensity—underscore the urgent need for transformative computational strategies ref [21].

### The Role of Machine Learning and Artificial Intelligence

Machine learning (ML) and artificial intelligence (AI) represent transformative paradigms that fundamentally reorient drug discovery from hypothesis-driven experimentation toward data-driven decision-making ref [28]. By leveraging vast repositories of biological, chemical, and clinical data, ML algorithms can identify complex patterns imperceptible to conventional analysis ref [15]. Deep learning architectures have demonstrated superior performance in bioactivity prediction, molecular design, and synthesis planning ref [9]. ML methods accelerate every pipeline stage, from target identification and virtual screening to ADMET prediction and drug repurposing ref [16]. Furthermore, AI enables cost-effective screening on limited datasets ref [5], addresses low-data challenges through transfer and few-shot learning ref [3], and facilitates explainable, knowledge-augmented frameworks that enhance clinical translatability ref [6].

### Scope and Organization of the Survey

This survey provides a comprehensive examination of machine learning applications across the drug discovery pipeline, synthesizing advances spanning classical algorithms ref [31], deep learning architectures ref [9], graph-based methods ref [45], generative models ref [44], and emerging paradigms such as quantum machine learning ref [60] and federated learning ref [47]. The survey is organized into six principal sections: foundational ML methodologies ref [1], pipeline-stage applications from target identification ref [36] through ADMET prediction ref [16] and synthesis planning, data resources and benchmarking ref [22], domain-specific case studies including oncology ref [56], infectious diseases ref [7], and neglected tropical diseases ref [27], and finally critical challenges encompassing interpretability ref [6], reproducibility ref [4], and future directions ref [47].

## Machine Learning Foundations and Methodologies

### Classical Machine Learning Algorithms

Classical machine learning algorithms constitute the foundational toolkit for drug discovery applications, encompassing supervised and unsupervised methods that have demonstrated sustained utility across diverse tasks. Naïve Bayes, k-Nearest Neighbors (kNN), Support Vector Machines (SVM), Random Forests (RF), Decision Trees, and ensemble methods such as AdaBoost, gradient boosting, and XGBoost have been extensively evaluated for virtual screening, QSAR modeling, and bioactivity prediction ref [8]. Comparative studies across thousands of ChEMBL datasets demonstrate that SVM and RF consistently achieve competitive performance ref [53]. SVM is particularly valued for its ability to operate in high-dimensional feature spaces ref [41], while RF and gradient boosting methods offer robustness and interpretability ref [31]. These algorithms remain competitive benchmarks even as deep learning advances ref [54].

### Deep Learning Architectures

Deep learning architectures have emerged as transformative tools in drug discovery, offering superior capacity to extract hierarchical representations from complex molecular and biological data ref [9]. Convolutional Neural Networks (CNNs) excel at capturing local structural patterns in molecular graphs and biological images, demonstrating utility in phenotypic drug discovery ref [24]. Recurrent Neural Networks (RNNs) and Long Short-Term Memory networks (LSTMs) process sequential molecular representations such as SMILES strings, enabling generative molecular design ref [32]. Autoencoders facilitate unsupervised feature learning and latent space exploration for molecular optimization ref [44]. Generative Adversarial Networks (GANs) enable de novo molecule generation ref [50]. Transformer-based architectures leverage attention mechanisms for drug-protein interaction modeling, providing interpretable binding site predictions ref [23], while large language model-based approaches further advance structure-based drug discovery ref [12].

### Graph Machine Learning and Graph Neural Networks

Graph machine learning (GML) has emerged as a powerful paradigm for modeling the inherently graph-structured nature of molecular and biological data in drug discovery ref [45]. Molecules are naturally represented as graphs, with atoms as nodes and bonds as edges, enabling Graph Neural Networks (GNNs) such as Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs) to capture rich structural and relational information ref [30]. GML has demonstrated utility across target identification, small molecule design, drug-protein interaction prediction, and drug repurposing ref [45]. Structure-aware GNN approaches have been applied to protein contact maps for improved drug-protein interaction prediction ref [23]. Furthermore, integrating external biomedical knowledge graphs into GML pipelines—termed Knowledge-augmented Graph Machine Learning—enhances interpretability and predictive precision under data-scarce conditions ref [37].

### Transfer Learning and Few-Shot Learning

Transfer learning addresses the critical challenge of sparse labeled data in drug discovery by leveraging generalizable knowledge from related tasks to enable learning on small datasets ref [3]. Deep transfer learning represents the most prevalent instantiation of this paradigm, enabling pre-trained representations to be fine-tuned for specific molecular property prediction tasks ref [3]. Complementarily, few-shot learning approaches, including meta-learning frameworks such as Prototypical Networks and Relation Networks, have been introduced to ligand-based virtual screening and lead optimization contexts ref [51]. Studies demonstrate that graph convolutional network-derived embeddings consistently outperform traditional fingerprints in few-shot settings, particularly for toxicity prediction ref [51]. These techniques are especially valuable in low-data drug discovery scenarios where conventional supervised learning methods are inapplicable ref [47].

### Generative Models for Molecular Design

Generative machine learning frameworks have emerged as transformative tools for de novo molecular design, enabling the automated generation of novel drug-like molecules with desired properties ref [44]. Prominent architectures include Variational Autoencoders (VAEs), which encode molecular representations into continuous latent spaces to facilitate interpolation and optimization ref [32], and Generative Adversarial Networks (GANs), which employ adversarial training to produce chemically valid structures ref [9]. Reinforcement learning-based methods further enable goal-directed molecular generation by optimizing reward functions encoding multi-property objectives ref [44]. Recurrent neural networks, particularly LSTM-RNNs operating on SMILES strings, have also demonstrated strong performance in generating syntactically valid and diverse molecular libraries ref [44]. Collectively, these approaches address key challenges including synthesizability, multi-property optimization, and chemical space exploration ref [35].

### Advanced and Emerging Techniques

Beyond classical and deep learning paradigms, several advanced and emerging techniques are reshaping drug discovery. Multi-task learning enables simultaneous optimization across related biological endpoints, improving generalization under data scarcity ref [47]. Active learning iteratively selects the most informative compounds for experimental validation, substantially reducing screening costs ref [5]. Federated learning facilitates collaborative model training across institutions without sharing proprietary data, addressing privacy constraints ref [4]. Quantum machine learning (QML) represents a particularly transformative frontier, leveraging quantum computing to handle exponentially complex molecular calculations, with proposed frameworks suggesting dramatic reductions in R&D timelines and costs ref [55]. Despite its nascent stage, QML demonstrates considerable promise for accelerating hit generation and lead optimization ref [60]. Collectively, these paradigms expand the applicability of ML across diverse drug discovery challenges.

### Molecular Representations and Featurization

Molecular representations and featurization are foundational to ML-based drug discovery, as the choice of encoding directly determines model performance. Classical approaches include molecular fingerprints such as Extended Connectivity Fingerprints (ECFP) and MACCS keys, which encode structural features as binary vectors ref

[... deterministic omitted span ...]

strategies have emerged as powerful frameworks for systematically identifying repurposing candidates by integrating drug structure, target profiles, and multi-omics data ref [4]. KGML-xDTD exemplifies this paradigm, combining knowledge graph-based machine learning with graph reinforcement learning to predict drug-disease treatment probabilities while providing interpretable, path-based mechanistic explanations ref [6]. Graph machine learning approaches further enable repurposing by modeling complex biomolecular interaction networks ref [45]. Applications to COVID-19 and oncology demonstrate both the promise and translational challenges of ML-driven repurposing, underscoring the need for rigorous validation and accessible web-based platforms ref [4].

### Synthesis Planning and Reaction Prediction

Synthesis planning and reaction prediction represent critical yet challenging stages in translating computationally designed drug candidates into experimentally realizable compounds. Machine learning has demonstrated considerable promise in automating retrosynthetic analysis, predicting reaction outcomes, and optimizing synthetic routes ref [9]. Deep learning architectures, particularly transformer-based and graph neural network models, have been applied to predict reaction yields, regioselectivity, and feasible synthetic pathways ref [32]. These approaches leverage large reaction databases to learn chemical transformation rules, enabling automated route planning ref [54]. Integration of ML-based synthesis prediction with cheminformatics toolkits, such as the BCL, further supports reaction-based library design ref [13], facilitating end-to-end computational pipelines that connect molecular design with practical synthetic accessibility.

## Data Resources, Benchmarks, and Cheminformatics Infrastructure

### Bioactivity and Chemical Databases

Public bioactivity and chemical databases constitute the foundational infrastructure for ML-driven drug discovery, providing the large-scale, curated datasets essential for model training and validation. ChEMBL, PubChem, and BindingDB aggregate millions of experimentally determined bioactivity measurements across diverse compound-target pairs, enabling robust supervised learning ref [15]. PDBbind supplies high-quality protein-ligand complex structures with binding affinity annotations, supporting structure-based approaches ref [12]. DrugBank and UniProt provide complementary pharmacological and proteomic annotations, facilitating drug-target interaction modeling ref [11]. These repositories collectively enable the integration of chemical structure, biological activity, and clinical data ref [34]. However, data heterogeneity, inconsistent assay conditions, and variable annotation quality necessitate rigorous curation and standardization prior to ML model development ref [4].

### Benchmark Datasets and Evaluation Standards

Benchmark datasets and standardized evaluation protocols are critical for assessing and comparing ML models in drug discovery. Widely used benchmarks include DUD-E and LIT-PCBA for virtual screening, where DUD-E, despite known biases, remains effective for training when bias is carefully managed ref [10]. MUV and Tox21 datasets are frequently employed for few-shot learning and toxicity prediction tasks ref [51]. The MF-PCBA collection introduces multifidelity benchmarking across 60 datasets encompassing over 16.6 million molecule–protein interactions, capturing both primary and confirmatory HTS modalities ref [22]. MISATO provides quantum mechanics-refined protein–ligand complexes with molecular dynamics trajectories for structure-based discovery ref [12]. Rigorous external validation and unbiased splitting strategies are essential to avoid overly optimistic performance estimates ref [25].

### High-Throughput Screening Data and Multifidelity Datasets

High-throughput screening (HTS) generates hundreds of thousands of activity measurements per project, providing substantial data for ML model training ref [22]. However, conventional ML-ready datasets largely ignore primary screening data—the most voluminous yet noisiest modality—focusing exclusively on confirmatory results ref [22]. To address this, the Multifidelity PubChem BioAssay (MF-PCBA) benchmark was introduced, comprising 60 curated datasets integrating both primary and confirmatory screening modalities, totaling over 16.6 million unique molecule–protein interactions ref [22]. This multifidelity framework challenges ML models to integrate low- and high-fidelity measurements through molecular representation learning ref [22]. Such large-scale HTS data, combined with deep learning, demonstrate improved drug activity predictions and more effective experimental design ref [15], underscoring the critical importance of leveraging all available HTS modalities.

### Cheminformatics Toolkits and Software Platforms

Cheminformatics toolkits and software platforms constitute essential infrastructure for ML-driven drug discovery, enabling seamless integration of molecular processing with predictive modeling. The BioChemical Library (BCL) represents a prominent open-source toolkit that combines traditional cheminformatics functionality—such as computing chemical properties and estimating druglikeness—with ML-based QSAR/QSPR modeling, supporting reaction-based library design and modular pipeline construction ref [13]. RDKit remains widely adopted for molecular featurization, descriptor computation, and preprocessing tasks ref [27]. Beyond standalone toolkits, web-based platforms are increasingly critical for democratizing access to complex ML models, particularly for researchers lacking programming expertise ref [4]. Integrated platforms that consolidate multi-omics, structural, and pharmacological data into accessible interfaces are urgently needed to bridge the gap between computational predictions and practical drug discovery workflows ref [34].

### Data Curation, Preprocessing, and Quality Control

Data curation, preprocessing, and quality control are foundational prerequisites for robust ML model development in drug discovery. Raw bioactivity datasets frequently contain inconsistencies, duplicate entries, and measurement errors that can severely compromise model reliability ref [4]. Standardized preprocessing pipelines encompassing compound structure normalization, salt stripping, and activity threshold harmonization are essential for ensuring data integrity ref [22]. Handling class imbalance—prevalent in high-throughput screening datasets where actives constitute a small minority—requires strategies such as oversampling, undersampling, and cost-sensitive learning ref [27]. Furthermore, rigorous train-test splitting protocols that account for structural similarity bias are critical to prevent data leakage and overly optimistic performance estimates ref [25]. Community-wide adoption of standardized reporting and curation guidelines remains imperative for reproducibility ref [4].

### Knowledge Graphs and Biomedical Ontologies

Knowledge graphs (KGs) and biomedical ontologies constitute a critical infrastructure for knowledge-augmented machine learning in drug discovery. By integrating heterogeneous biomedical entities—including genes, proteins, diseases, drugs, and pathways—into structured relational frameworks, KGs enable sophisticated reasoning over complex biological networks ref [30]. Frameworks such as KGML-xDTD leverage KG-based machine learning to predict drug-disease treatment relationships while providing interpretable, path-based mechanistic explanations, addressing the 'black-box' limitations of conventional models ref [6]. Knowledge-augmented graph machine learning (KaGML) further combines structural biomedical data with external domain knowledge to improve prediction precision under data-scarce conditions ref [37]. Such approaches are increasingly integral to drug repurposing, target identification, and multi-omics data integration ref [45].

## Domain-Specific Applications and Case Studies

### Oncology and Cancer Drug Discovery

Oncology represents one of the most active domains for ML-driven drug discovery, given the complexity of cancer biology and the urgent need for effective therapeutics. ML approaches have demonstrated substantial utility in cancer target identification, anticancer compound screening, drug repurposing, and treatment outcome prediction ref [4]. Precision oncology has particularly benefited from supervised learning and feature selection for predictive biomarker identification, enabling marker-based clinical trial design and improved patient stratification ref [4]. Drug Discovery Maps illustrate how ML models can predict kinase inhibitor activity landscapes, yielding experimentally validated FLT3 inhibitors relevant to acute myeloid leukemia ref [18]. Furthermore, ML algorithms have been applied to predict drug efficacy and toxicity in oncology, integrating disease-state representations with therapeutic agent profiles to derive clinically relevant insights ref [56].

### Infectious Disease and Antimicrobial Drug Discovery

Machine learning has demonstrated significant utility in infectious disease and antimicrobial drug discovery, addressing urgent challenges including antibiotic resistance and emerging pathogens. ML-based virtual screening has substantially improved hit rates for antibacterial compounds; for instance, a directed-message passing neural network trained on growth inhibitory data against antibiotic-resistant Burkholderia cenocepacia achieved hit rates of 26% for FDA-approved compounds, representing a 14-fold improvement over traditional screening ref [40]. For neglected tropical diseases, ML models trained on high-throughput screening datasets have successfully predicted biological activities against Leishmania, Trypanosoma cruzi, and Trypanosoma brucei ref [27]. The COVID-19 pandemic further accelerated ML applications in antiviral discovery, with AI and ML approaches expediting identification and optimization of novel antivirals ref [7], though rigorous validation remains essential to ensure clinical translational impact ref [4].

### Central Nervous System and Neurological Disorders

Central nervous system (CNS) disorders represent one of the most challenging therapeutic areas, characterized by exceptionally high attrition rates and prolonged development timelines ref [43]. AI and ML have emerged as indispensable tools for accelerating CNS drug discovery, encompassing target identification, compound screening, hit/lead optimization, de novo drug design, and drug repurposing for neurological diseases including Alzheimer's and Parkinson's disease ref [43]. A particularly critical application is blood-brain barrier (BBB) permeability prediction, which is essential for CNS drug candidates ref [43]. ML-based virtual screening has been systematically applied to Alzheimer's disease drug discovery, leveraging algorithms such as Naïve Bayes, SVM, Random Forests, and neural networks to identify novel therapeutic

[... deterministic omitted span ...]

standards ref [22]. Rigorous external validation against well-curated datasets and community-wide benchmarking standards remain essential prerequisites for ensuring that ML models achieve genuine translational value ref [4].

### Reproducibility, Standardization, and Best Practices

Reproducibility and standardization represent critical challenges in ML-based drug discovery, where methodological inconsistencies and inadequate reporting undermine the reliability of published findings. A significant concern is that many ML models developed for biomedical applications, including COVID-19 prognostication, demonstrated limited clinical utility due to methodological flaws and dataset biases ref [4]. Community-wide efforts have established guidelines for improving model quality, including benchmarking against well-curated external validation datasets, standardized reporting metrics, and best practices for code publication and workflow automation ref [4]. Adoption of open-source implementations, such as those exemplified by accessible toolkits ref [13], alongside transparent data curation protocols ref [28], is essential. Rigorous cross-validation strategies, unbiased dataset splitting, and avoidance of data leakage are fundamental requirements for ensuring that ML models achieve genuine translational value ref [25].

### Integration of Multi-Modal and Multi-Scale Data

Integrating multi-modal and multi-scale data represents a critical yet underexplored challenge in ML-driven drug discovery. Effective drug discovery requires synthesizing heterogeneous data types—including genomics, proteomics, chemical structures, imaging, and clinical outcomes—across multiple biological scales ref [4]. While learning approaches have been developed for systematic identification of drug repurposing leads combining drug structure, target profiles, and multi-omics data from cell-based models, accessible web-based platforms integrating all these modalities remain lacking ref [4]. Graph machine learning frameworks offer promising capabilities for modeling biomolecular structures and integrating multi-omic datasets ref [45]. Multimodal deep learning models, such as STAMP-DPI, demonstrate the benefit of combining structural and sequence-based representations for improved drug-protein interaction prediction ref [23]. Multifidelity datasets like MF-PCBA further illustrate how integrating low- and high-fidelity experimental measurements enhances model performance ref [22].

### Translational Challenges and Clinical Impact

Despite remarkable computational advances, a substantial gap persists between ML-based predictions and clinically actionable outcomes. Models trained on preclinical data frequently fail to generalize to in vivo and clinical settings due to distributional shifts, incomplete biological context, and dataset biases ref [4]. As demonstrated during COVID-19, even rapidly developed ML models showed limited clinical utility owing to methodological flaws and biased training data ref [4]. Regulatory frameworks have not yet fully accommodated ML-derived evidence, creating additional barriers to clinical adoption ref [28]. Furthermore, the 'black-box' nature of complex models undermines clinician trust and interpretability ref [6]. Bridging this translational gap requires rigorous external validation, standardized reporting, and integration of real-world clinical data to ensure ML models achieve meaningful therapeutic impact ref [38].

### Emerging Paradigms and Future Opportunities

Several transformative paradigms are poised to redefine ML-driven drug discovery. Quantum machine learning (QML) offers the potential to dramatically reduce computational costs and timelines in preclinical development by leveraging quantum supremacy for complex molecular simulations ref [55], though significant practical challenges remain ref [60]. Large language models and foundation models are increasingly enabling richer molecular and biological representations, as evidenced by their integration into structure-based discovery datasets ref [12]. Knowledge-augmented graph machine learning combines structural biomedical data with external knowledge graphs for more interpretable and data-efficient predictions ref [30]. Furthermore, federated and open-science approaches facilitate collaborative model training across institutions ref [5], while physics-informed ML scoring functions continue bridging data-driven and mechanistic paradigms ref [17], collectively charting a promising trajectory for next-generation drug discovery.

## References

[1] A review on machine learning approaches and trends in drug discovery. https://doi.org/10.1016/j.csbj.2021.08.011

[2] Machine Learning in Drug Discovery. https://doi.org/10.55529/jaimlnn.32.53.60

[3] Transfer Learning for Drug Discovery.. https://doi.org/10.1021/acs.jmedchem.9b02147

[4] What are the current challenges for machine learning in drug discovery and repurposing?. https://doi.org/10.1080/17460441.2022.2050694

[5] Discovery of senolytics using machine learning. https://doi.org/10.1038/s41467-023-39120-1

[6] KGML-xDTD: a knowledge graph–based machine learning framework for drug treatment prediction and mechanism description. https://doi.org/10.1093/gigascience/giad057

[7] Application of artificial intelligence and machine learning for COVID-19 drug discovery and vaccine design. https://doi.org/10.1093/bib/bbab320

[8] Machine Learning-based Virtual Screening and Its Applications to Alzheimer’s Drug Discovery: A Review. https://doi.org/10.2174/1381612824666180607124038

[9] The rise of deep learning in drug discovery.. https://doi.org/10.1016/j.drudis.2018.01.039

[10] MILCDock: Machine Learning Enhanced Consensus Docking for Virtual Screening in Drug Discovery. https://doi.org/10.1021/acs.jcim.2c00705

[11] Machine Learning for Drug-Target Interaction Prediction. https://doi.org/10.3390/molecules23092208

[12] MISATO - Machine learning dataset of protein-ligand complexes for structure-based drug discovery. https://doi.org/10.1101/2023.05.24.542082

[13] Introduction to the BioChemical Library (BCL): An Application-Based Open-Source Toolkit for Integrated Cheminformatics and Machine Learning in Computer-Aided Drug Discovery. https://doi.org/10.3389/fphar.2022.833099

[14] Virtual Screening Algorithms in Drug Discovery: A Review Focused on Machine and Deep Learning Methods. https://doi.org/10.3390/ddc2020017

[15] Advancing computer-aided drug discovery (CADD) by big data and data-driven machine learning modeling. https://doi.org/10.1016/j.drudis.2020.07.005

[16] Artificial Intelligence in Drug Discovery: A Comprehensive Review of Data-driven and Machine Learning Approaches. https://doi.org/10.1007/s12257-020-0049-y

[17] New machine learning and physics-based scoring functions for drug discovery. https://doi.org/10.1038/s41598-021-82410-1

[18] Drug Discovery Maps, a Machine Learning Model That Visualizes and Predicts Kinome–Inhibitor Interaction Landscapes. https://doi.org/10.1021/acs.jcim.8b00640

[19] Applications of Machine Learning in Drug Discovery. https://doi.org/10.26717/bjstr.2019.23.003831

[20] A Machine Learning Strategy for Drug Discovery Identifies Anti-Schistosomal Small Molecules. https://doi.org/10.1021/acsinfecdis.0c00754

[21] Survey of Machine Learning Techniques in Drug Discovery.. https://doi.org/10.2174/1389200219666180820112457

[22] MF-PCBA: Multifidelity High-Throughput Screening Benchmarks for Drug Discovery and Machine Learning. https://doi.org/10.1021/acs.jcim.2c01569

[23] Structure-Aware Multimodal Deep Learning for Drug-Protein Interaction Prediction. https://doi.org/10.1021/acs.jcim.2c00060

[24] Deep learning in image-based phenotypic drug discovery.. https://doi.org/10.1016/j.tcb.2022.11.011

[25] How to approach machine learning-based prediction of drug/compound–target interactions. https://doi.org/10.1186/s13321-023-00689-w

[26] Insights into Machine Learning-based approaches for Virtual Screening in Drug Discovery: Existing strategies and streamlining through FP-CADD.. https://doi.org/10.2174/1570163817666200806165934

[27] Machine learning and drug discovery for neglected tropical diseases. https://doi.org/10.1186/s12859-022-05076-0

[28] Applications of machine learning in drug discovery and development. https://doi.org/10.1038/s41573-019-0024-5

[29] Machine Learning Assisted Design of Highly Active Peptides for Drug Discovery. https://doi.org/10.1371/journal.pcbi.1004074

[30] Knowledge-augmented Graph Machine Learning for Drug Discovery: A Survey. https://doi.org/10.1145/3744237

[31] A combined drug discovery strategy based on machine learning and molecular docking. https://doi.org/10.1111/cbdd.13494

[32] Deep Learning for Drug Design: an Artificial Intelligence Paradigm for Drug Discovery in the Big Data Era. https://doi.org/10.1208/s12248-018-0210-0

[33] Automating Drug Discovery using Machine Learning.. https://doi.org/10.2174/1570163820666230607163313

[34] Recent applications of deep learning and machine intelligence on in silico drug discovery: methods, tools and databases. https://doi.org/10.1093/bib/bby061

[35] Applications of machine learning in computer-aided drug discovery. https://doi.org/10.1017/qrd.2022.12

[36] Application of Machine Learning on Drug Target Discovery.. https://doi.org/10.2174/1567201817999200728142023

[37] Knowledge-augmented Graph Machine Learning for Drug Discovery: From Precision to Interpretability. https://doi.org/10.1145/3580305.3599563

[38] Machine Learning in Drug Discovery: A Review. https://doi.org/10.1007/s10462-021-10058-4

[39] Artificial Intelligence, Big data and Machine Learning approaches in Preci-sion Medicine & Drug Discovery.. https://doi.org/10.2174/1389450122999210104205732

[40] A machine learning model trained on a high-throughput antibacterial screen increases the hit rate of drug discovery. https://doi.org/10.1371/journal.pcbi.1010613

[41] Evolution of Support Vector Machine and Regression Modeling in Chemoinformatics and Drug Discovery. https://doi.org/10.1007/s10822-022-00442-9

[42] Application of Artificial Intelligence and Machine Learning in Drug Discovery and Development. https://doi.org/10.22270/jddt.v13i1.5867

[43] Artificial intelligence and machine learning‐aided drug discovery in central nervous system diseases: State‐of‐the‐arts and future directions. https://doi.org/10.1002/med.21764

[44] Generative machine learning for de novo drug discovery: A systematic review. https://doi.org/10.1016/j.compbiomed.2022.105403

[45] Utilizing graph machine learning within drug discovery and development. https://doi.org/10.1093/bib/bbab159

[46] Machine learning for target discovery in drug development.. https://doi.org/10.1016/j.cbpa.2019.10.003

[47] Advanced machine-learning techniques in drug discovery.. https://doi.org/10.1016/j.drudis.2020.12.003

[48] Exploring the Artificial Intelligence and Machine Learning Models in the Context of Drug Design Difficulties and Future Potential for the Pharmaceutical Sectors.. https://doi.org/10.1016/j.ymeth.2023.09.010

[49] Machine Learning and Artificial Intelligence: A paradigm shift in Big Data-Driven Drug Design and Discovery.. https://doi.org/10.2174/1568026622666220701091339

[50] Artificial intelligence to deep learning: machine intelligence approach for drug discovery. https://doi.org/10.1007/s11030-021-10217-3

[51] Few-Shot Learning for Low-Data Drug Discovery. https://doi.org/10.1021/acs.jcim.2c00779

[52] Machine Learning Methods in Drug Discovery. https://doi.org/10.3390/molecules25225277

[53] Bioactivity Comparison Across Multiple Machine Learning Algorithms Using Over 5000 Datasets for Drug Discovery. https://doi.org/10.1021/acs.molpharmaceut.0c01013

[54] Exploiting Machine Learning for End-To-End Drug Discovery and Development. https://doi.org/10.1038/s41563-019-0338-z

[55] The New Answer to Drug Discovery: Quantum Machine Learning in Preclinical Drug Development. https://doi.org/10.1109/PRML59573.2023.10348356

[56] Machine learning approaches to predict drug efficacy and toxicity in oncology. https://doi.org/10.1016/j.crmeth.2023.100413

[57] Machine learning in chemoinformatics and drug discovery. https://doi.org/10.1016/j.drudis.2018.05.010

[58] Machine learning approaches and their applications in drug discovery and design. https://doi.org/10.1111/cbdd.14057

[59] Deep learning tools for advancing drug discovery and development. https://doi.org/10.1007/s13205-022-03165-8

[60] Quantum Machine Learning in Drug Discovery: Current State and Challenges. https://doi.org/10.1145/3575879.3576024
