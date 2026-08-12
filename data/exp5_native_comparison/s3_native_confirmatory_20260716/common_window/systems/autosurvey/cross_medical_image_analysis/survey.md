# Registered common-window excerpt

# Deep Learning for Medical Image Analysis: A Comprehensive Survey

## 1 Introduction and Background

### 1.1 Historical Evolution of Medical Image Analysis

The field of medical image analysis has undergone a profound transformation over the past several decades, evolving from rudimentary manual inspection and rule-based processing techniques to highly sophisticated, data-driven deep learning systems. Understanding this historical trajectory is essential for appreciating both the scale of progress achieved and the challenges that remain — and for contextualizing the deep learning foundations that underpin modern medical imaging pipelines.

**Early Foundations: Manual and Rule-Based Methods**

In the early decades of medical imaging, analysis relied almost entirely on the expertise of trained clinicians who visually interpreted radiographs, histological slides, and other diagnostic images. The introduction of computed tomography (CT) in the 1970s and magnetic resonance imaging (MRI) in the 1980s dramatically expanded the clinical information available, but also increased the complexity and volume of data requiring interpretation. These advances created early demand for computational assistance.

The initial computational approaches to medical image analysis were largely rule-based and hand-engineered. Researchers developed algorithms grounded in classical signal processing and computer vision principles, including edge detection, region growing, thresholding, and morphological operations. These methods required extensive domain-specific engineering, where experts manually designed features believed to capture relevant diagnostic information. While effective in controlled environments, rule-based systems proved brittle when confronted with the natural variability of clinical imaging data, including differences in patient anatomy, imaging protocols, and scanner hardware.

**The Rise of Classical Machine Learning**

Through the 1990s and 2000s, the field transitioned toward statistical machine learning approaches. Methods such as support vector machines (SVMs), random forests, and AdaBoost began to be applied to problems of lesion detection, tissue classification, and anatomical segmentation. These approaches retained hand-crafted feature extraction pipelines but replaced rigid rule-based classifiers with learned statistical models capable of better generalizing from training examples.

Atlas-based registration and multi-atlas segmentation also emerged as prominent strategies during this period, leveraging prior anatomical knowledge encoded in labeled template images to segment new patient scans. Statistical shape models and active appearance models further enabled deformable, data-driven representations of anatomy. Despite these advances, performance remained constrained by the expressive capacity of manually designed features and the difficulty of capturing complex, nonlinear patterns in high-dimensional imaging data. These limitations set the stage for the deep learning revolution that would follow.

**The Deep Learning Revolution**

The landscape shifted dramatically following the landmark success of convolutional neural networks (CNNs) in the 2012 ImageNet Large Scale Visual Recognition Challenge, where AlexNet demonstrated the power of deep architectures trained on large-scale labeled data [1; 2]. This achievement catalyzed a rapid adoption of deep learning techniques across computer vision and, shortly thereafter, medical image analysis.

CNNs offered a fundamentally different paradigm: rather than relying on hand-crafted features, these models automatically learned hierarchical representations directly from raw image data through end-to-end training. The ability to jointly optimize feature extraction and classification, combined with the availability of increasingly powerful graphics processing units (GPUs), enabled the training of models of unprecedented depth and capacity [3]. This shift from manual feature engineering to automatic hierarchical feature learning — a concept explored in greater technical depth in the following section — proved to be the defining characteristic of the new paradigm.

The impact on medical imaging was swift and far-reaching. Deep learning methods rapidly achieved state-of-the-art performance across a broad range of tasks, including classification of diabetic retinopathy, detection of pulmonary nodules in CT, segmentation of brain tumors in MRI, and analysis of histopathological slides [4]. The introduction of the U-Net architecture in 2015, with its encoder-decoder structure and skip connections, proved particularly transformative for medical image segmentation by enabling precise localization even with limited training data.

**Architectural Evolution and Expanding Capabilities**

Subsequent years witnessed rapid architectural innovation. Residual networks (ResNets) addressed the challenge of training very deep networks by introducing skip connections that facilitate gradient flow during backpropagation. Densely connected networks (DenseNets) extended this idea by connecting each layer to all subsequent layers, encouraging feature reuse and improving parameter efficiency. These architectural advances made it possible to train significantly deeper and more expressive models, yielding substantial performance gains across medical imaging benchmarks [1].

More recently, the advent of vision transformers and self-attention mechanisms has introduced another paradigm shift, enabling models to capture long-range spatial dependencies that are challenging for local convolutional operations. Transformer-based architectures have been applied across the full spectrum of medical image analysis tasks, achieving competitive or superior performance compared to CNN-based approaches [5]. Hybrid CNN-transformer models have also emerged, combining the inductive biases of convolutional layers with the global context modeling capabilities of attention mechanisms.

**The Era of Data-Driven and Self-Supervised Learning**

Alongside architectural advances, the field has increasingly grappled with the fundamental challenge of limited annotated medical data. Unlike natural image datasets containing millions of labeled examples, medical imaging datasets are typically constrained by the time and expertise required for clinical annotation. This has motivated the development of transfer learning strategies, where models pretrained on large natural image datasets are adapted to medical tasks through fine-tuning [6]. More recently, self-supervised and contrastive pretraining approaches have demonstrated that powerful representations can be learned from large collections of unlabeled medical images, substantially reducing dependence on expert annotations [7; 8].

The recognition that medical images carry unique characteristics — including volumetric structure, specific acquisition physics, and temporal dynamics — has further driven the development of domain-specific methods. Three-dimensional CNN architectures have been designed to exploit the volumetric nature of CT and MRI data [2], while recurrent and transformer-based models have been applied to capture temporal disease progression from longitudinal imaging sequences [9; 10].

**Toward Clinical Translation**

The maturation of deep learning in medical imaging has also brought increasing attention to the requirements for clinical deployment. Early enthusiasm for raw performance metrics has given way to a more nuanced appreciation of challenges including model interpretability, robustness to domain shift across different hospitals and imaging systems, fairness across patient populations, and compliance with regulatory frameworks [11]. The field has recognized that a technically accurate model is insufficient for clinical adoption if clinicians cannot understand, trust, or reliably deploy it in real-world settings.

The historical arc of medical image analysis thus reflects a broader story in artificial intelligence: an evolution from expert-designed, knowledge-driven systems toward data-driven, end-to-end learned representations, with increasing complexity, capability, and ambition. Each paradigm has built upon its predecessors, and the current deep learning era is best understood not as a replacement of prior knowledge but as a powerful amplification of the field's long-standing goals — to assist clinicians in extracting meaningful diagnostic and prognostic information from medical images, ultimately improving patient outcomes. The technical foundations that enable this transformation, including the architectural principles, optimization strategies, and regularization techniques central to modern deep learning, are examined in detail in the following section.

### 1.2 Fundamentals of Deep Learning

Deep learning (DL) represents a class of machine learning methods built upon artificial neural networks with multiple layers of representation, enabling the automatic extraction of hierarchical features directly from raw data. As introduced in the preceding section, the shift from hand-crafted feature engineering to automatic hierarchical feature learning constitutes the defining characteristic of the deep learning paradigm — and it is precisely this capacity that has driven the rapid transformation of medical image analysis over the past decade [12; 1; 13]. The present section examines the foundational technical concepts underlying this transformation, from the architectural building blocks of convolutional networks to the optimization, regularization, and training strategies that make these systems functional in clinical contexts. These concepts form the technical substrate upon which all subsequent discussions of modality-specific methods, segmentation, detection, and clinical deployment are grounded.

**Convolutional Neural Networks (CNNs)**

At the core of most medical image analysis systems lies the convolutional neural network (CNN), an architecture specifically designed to exploit the spatial structure of image data. CNNs consist of stacked layers of learnable convolutional filters that progressively transform the input image into increasingly abstract feature representations. Early layers typically capture low-level features

[... deterministic omitted span ...]

from other parts, thereby capturing the statistical regularities of medical image distributions. Techniques such as context restoration, rotation prediction, and jigsaw puzzle solving have been applied as pretext tasks to medical imaging data, yielding representations that encode anatomical structure without requiring any expert annotation. These approaches are computationally lightweight compared to contrastive methods that require large batch sizes or memory banks of negative samples, making them particularly suitable in resource-constrained clinical settings.

### Multimodal and Domain-Specific Contrastive Learning

An important frontier in self-supervised medical imaging involves multimodal contrastive pretraining, in which image encoders are aligned with representations derived from paired radiology reports, pathology descriptions, or genomic profiles. This vision-language contrastive learning strategy, exemplified by methods analogous to CLIP adapted for medical data, allows models to develop semantically grounded representations that reflect clinical concepts described in free-text reports. The resulting models exhibit strong zero-shot and few-shot generalization to previously unseen tasks, making them especially appealing for rare disease settings where labeled data is nearly absent. This multimodal alignment also bears conceptual connections to the domain-specific pretraining strategies discussed earlier, in that both seek to close the gap between general-purpose representations and the semantic demands of clinical imaging tasks.

### Practical Benefits and Remaining Challenges

Self-supervised and contrastive learning approaches offer substantial practical benefits in the medical domain: they reduce dependence on costly expert annotation, enable leveraging large collections of routine clinical images, and produce representations that generalize across institutions and imaging protocols. These properties directly address the annotation bottleneck that motivates the broader family of label-efficient strategies surveyed in this section. However, several challenges remain. The design of appropriate augmentation pipelines that preserve clinical validity is non-trivial; aggressive augmentations used in natural image SSL (e.g., color jitter) may alter diagnostically meaningful image features in medical contexts. Furthermore, evaluation of SSL representations often relies on linear probing or fine-tuning protocols that may not fully reflect clinical deployment conditions. Standardized benchmarks and evaluation frameworks are needed to rigorously assess the clinical utility of self-supervised representations across diverse imaging modalities and disease categories. These considerations naturally set the stage for the active learning strategies discussed next, which further reduce annotation burden by strategically directing expert labeling effort toward the most informative examples within the unlabeled data pool.

### 4.6 Active Learning for Efficient Annotation

## 4.6 Active Learning

Active learning represents a powerful paradigm for addressing the annotation bottleneck in medical image analysis, where the cost of obtaining expert labels is prohibitively high. Building directly upon the self-supervised and contrastive representations discussed in the previous section — which provide rich feature spaces from unlabeled data — active learning frameworks iteratively identify the most informative or uncertain samples from an unlabeled pool and present them to expert annotators, thereby maximizing model performance gains per unit of labeling effort.

The core principle underlying active learning is the strategic selection of samples that are expected to yield the greatest improvement in model performance when labeled. Several query strategies have been proposed to guide this selection process. Uncertainty sampling, one of the most widely adopted approaches, selects samples for which the current model exhibits the lowest prediction confidence. This can be quantified through measures such as entropy, margin sampling, or least confidence scores. Diversity-based strategies, on the other hand, aim to select a batch of samples that collectively represent the breadth of the unlabeled data distribution, avoiding redundant annotations of similar instances. Hybrid approaches combine both uncertainty and diversity criteria to balance exploration and exploitation during the annotation process.

The limited availability of annotated medical data is a well-recognized challenge in the field [57; 56]. Active learning directly addresses this challenge by concentrating annotation resources on the most diagnostically ambiguous cases — those lying near decision boundaries or exhibiting high inter-class similarity — where expert judgment is most needed. This is particularly relevant in histopathology and radiology, where subtle morphological differences between benign and malignant findings demand specialized expertise [72; 89].

In practice, active learning pipelines for medical imaging typically operate in iterative cycles. An initial model is trained on a small seed set of labeled examples. The model is then applied to the remaining unlabeled pool, and an acquisition function scores each sample according to the chosen query strategy. The highest-scoring samples are selected and submitted to domain experts for labeling. The newly annotated samples are added to the training set, and the model is retrained or fine-tuned. This cycle repeats until the annotation budget is exhausted or performance plateaus.

Several acquisition functions have been explored in the context of deep learning for medical images. Bayesian uncertainty estimation, implemented through techniques such as Monte Carlo dropout, provides a principled measure of epistemic uncertainty that can guide sample selection. Ensemble-based methods aggregate predictions from multiple independently trained models to estimate disagreement as a proxy for informativeness. Core-set approaches define selection as a geometric coverage problem in the feature space, seeking to find a labeled subset whose convex hull encompasses the entire dataset distribution.

Active learning is particularly synergistic with label-efficient learning strategies such as semi-supervised and self-supervised pretraining [90; 69]. The meaningful feature spaces produced by self-supervised representations pretrained on large unlabeled corpora — as discussed in the preceding section — can be directly leveraged within active learning pipelines to compute more reliable uncertainty and diversity estimates, thereby improving the quality of sample selection before any expert annotations are obtained. Similarly, combining active learning with transfer learning from pretrained convolutional architectures can substantially reduce the number of labeled samples required to achieve competitive performance [57].

Challenges remain in scaling active learning to the complexities of medical imaging. Query strategies designed for single-label classification may not transfer straightforwardly to dense prediction tasks such as segmentation or object detection, where annotation effort per sample is substantially higher and the notion of informativeness must account for spatial heterogeneity within images [91]. Furthermore, the inherent class imbalance in medical datasets — where pathological findings are rare relative to normal cases — can bias uncertainty-based acquisition functions toward easy negative examples rather than clinically valuable positive cases.

Recent advances have explored curriculum-based active learning, where the order in which samples are presented to the model during training is jointly optimized with the selection policy to accelerate convergence. Annotator-aware active learning frameworks additionally account for variability in expert agreement, routing ambiguous cases to multiple annotators or specialists with relevant sub-domain expertise. Such multi-annotator settings are especially relevant in tasks like tumor grading or lesion classification, where inter-observer variability is clinically significant.

Overall, active learning offers a principled and practical pathway for reducing annotation burden in medical imaging, enabling the construction of high-quality labeled datasets with a fraction of the labeling effort that exhaustive annotation would require. As the complexity and heterogeneity of medical imaging tasks continue to grow, active learning frameworks integrated with modern deep learning architectures will play an increasingly vital role in enabling scalable and cost-effective clinical AI development [2]. These annotation-efficiency goals naturally extend to the federated learning paradigm discussed in the following section, which operates at an institutional scale — coordinating learning across multiple healthcare sites while preserving patient data privacy, and which can be further enhanced by integrating active learning to prioritize the most informative samples across federated participants.

### 4.7 Federated Learning for Privacy-Preserving Collaboration

Federated learning (FL) has emerged as a transformative paradigm for collaborative medical image analysis, enabling multiple healthcare institutions to jointly train deep learning models without transferring raw patient data to a central server. This approach directly addresses the stringent privacy regulations governing medical data, such as HIPAA and GDPR, as well as the practical challenge of institutional data silos that prevent centralized pooling of clinical imaging datasets. Where active learning seeks to maximize the value of each annotation within a single institution, federated learning operates at a broader scale — coordinating learning across institutions while preserving the confidentiality of each site's data.

In the federated learning framework, each participating institution trains a local model on its own data and shares only model parameters or gradients with

[... deterministic omitted span ...]

from retinal optical coherence  tomography images

[9] Temporal Context Matters  Enhancing Single Image Prediction with Disease  Progression Representations

[10] Learning Spatio-Temporal Model of Disease Progression with NeuralODEs  from Longitudinal Volumetric Data

[11] Medical Imaging and Machine Learning

[12] A Survey on Deep Learning in Medical Image Analysis

[13] Medical Image Analysis using Convolutional Neural Networks  A Review

[14] Deep Learning with ConvNET Predicts Imagery Tasks Through EEG

[15] A Robust Backpropagation-Free Framework for Images

[16] Biologically-Motivated Deep Learning Method using Hierarchical  Competitive Learning

[17] Deeper Image Quality Transfer  Training Low-Memory Neural Networks for  3D Images

[18] An Analysis on Ensemble Learning optimized Medical Image Classification  with Deep Convolutional Neural Networks

[19] A novel adversarial learning strategy for medical image classification

[20] Representation Learning with Information Theory for COVID-19 Detection

[21] MedNet  Pre-trained Convolutional Neural Network Model for the Medical  Imaging Tasks

[22] Medical Image Segmentation on MRI Images with Missing Modalities  A  Review

[23] Cross-Modality Neuroimage Synthesis  A Survey

[24] Engineering AI Tools for Systematic and Scalable Quality Assessment in  Magnetic Resonance Imaging

[25] Deep and Statistical Learning in Biomedical Imaging  State of the Art in  3D MRI Brain Tumor Segmentation

[26] Sinogram upsampling using Primal-Dual UNet for undersampled CT and  radial MRI reconstruction

[27] Progressively Volumetrized Deep Generative Models for Data-Efficient  Contextual Learning of MR Image Recovery

[28] Modern Convex Optimization to Medical Image Analysis

[29] Multi-Contrast Computed Tomography Healthy Kidney Atlas

[30] Deep PCCT  Photon Counting Computed Tomography Deep Learning  Applications Review

[31] Case Studies on X-Ray Imaging, MRI and Nuclear Imaging

[32] X-ray In-Depth Decomposition  Revealing The Latent Structures

[33] The Ultrasound Visualization Pipeline - A Survey

[34] Deep Learning for Ultrasound Beamforming

[35] Less is More  Simultaneous View Classification and Landmark Detection  for Abdominal Ultrasound Images

[36] Multimodal brain tumor classification

[37] Deep Multi-modal Fusion of Image and Non-image Data in Disease Diagnosis  and Prognosis  A Review

[38] Co-Learning Feature Fusion Maps from PET-CT Images of Lung Cancer

[39] Multi-modality imaging with structure-promoting regularisers

[40] Multi-domain improves out-of-distribution and data-limited scenarios for  medical image analysis

[41] Multi-Modality Cardiac Image Computing  A Survey

[42] Review of multimodal machine learning approaches in healthcare

[43] Robust Image Reconstruction with Misaligned Structural Information

[44] Deep Learning for Automated Medical Image Analysis

[45] Recent advances and clinical applications of deep learning in medical  image analysis

[46] A Survey on Active Learning and Human-in-the-Loop Deep Learning for  Medical Image Analysis

[47] Development and Validation of a Novel Prognostic Model for Predicting  AMD Progression Using Longitudinal Fundus Images

[48] Leveraging Deep Learning and Xception Architecture for High-Accuracy MRI  Classification in Alzheimer Diagnosis

[49] Deep Residual CNN for Multi-Class Chest Infection Diagnosis

[50] Learning Shape Features and Abstractions in 3D Convolutional Neural  Networks for Detecting Alzheimer's Disease

[51] Promises and pitfalls of deep neural networks in neuroimaging-based  psychiatric research

[52] Cross-modality Guidance-aided Multi-modal Learning with Dual Attention  for MRI Brain Tumor Grading

[53] Explainable deep learning models in medical image analysis

[54] Self-Attention Equipped Graph Convolutions for Disease Prediction

[55] A Survey on Incorporating Domain Knowledge into Deep Learning for  Medical Image Analysis

[56] A scoping review of transfer learning research on medical image analysis  using ImageNet

[57] Deep Convolutional Neural Networks for Computer-Aided Detection  CNN  Architectures, Dataset Characteristics and Transfer Learning

[58] ALLNet  A Hybrid Convolutional Neural Network to Improve Diagnosis of  Acute Lymphocytic Leukemia (ALL) in White Blood Cells

[59] Capturing Local and Global Features in Medical Images by Using Ensemble  CNN-Transformer

[60] Benchmarking CNN on 3D Anatomical Brain MRI  Architectures, Data  Augmentation and Deep Ensemble Learning

[61] Effects of Degradations on Deep Neural Network Architectures

[62] Stain Normalized Breast Histopathology Image Recognition using  Convolutional Neural Networks for Cancer Detection

[63] Dermatological Diagnosis Explainability Benchmark for Convolutional  Neural Networks

[64] Lets keep it simple, Using simple architectures to outperform deeper and  more complex architectures

[65] Malaria detection from RBC images using shallow Convolutional Neural  Networks

[66] Compact & Capable  Harnessing Graph Neural Networks and Edge Convolution  for Medical Image Classification

[67] CFPNet-M  A Light-Weight Encoder-Decoder Based Network for Multimodal  Biomedical Image Real-Time Segmentation

[68] Is it enough to optimize CNN architectures on ImageNet 

[69] LVM-Med  Learning Large-Scale Self-Supervised Vision Models for Medical  Imaging via Second-order Graph Matching

[70] Prediction of lung and colon cancer through analysis of  histopathological images by utilizing Pre-trained CNN models with  visualization of class activation and saliency maps

[71] CA-Net  Comprehensive Attention Convolutional Neural Networks for  Explainable Medical Image Segmentation

[72] Cell nuclei classification in histopathological images using hybrid  OLConvNet

[73] C2G-Net  Exploiting Morphological Properties for Image Classification

[74] Deep Learning Methods and Applications for Region of Interest Detection  in Dermoscopic Images

[75] MultiResUNet   Rethinking the U-Net Architecture for Multimodal  Biomedical Image Segmentation

[76] Harmonized Spatial and Spectral Learning for Robust and Generalized  Medical Image Segmentation

[77] Learning With Context Feedback Loop for Robust Medical Image  Segmentation

[78] Progressive Adversarial Semantic Segmentation

[79] Human Treelike Tubular Structure Segmentation  A Comprehensive Review  and Future Perspectives

[80] Boundary-aware Context Neural Network for Medical Image Segmentation

[81] Is attention all you need in medical image analysis  A review

[82] DRU-net  An Efficient Deep Convolutional Neural Network for Medical  Image Segmentation

[83] All-in-one platform for AI R&D in medical imaging, encompassing data  collection, selection, annotation, and pre-processing

[84] GaNDLF  A Generally Nuanced Deep Learning Framework for Scalable  End-to-End Clinical Workflows in Medical Imaging

[85] Deep Learning with Cinematic Rendering  Fine-Tuning Deep Neural Networks  Using Photorealistic Medical Images

[86] Structurally aware bidirectional unpaired image to image translation  between CT and MR

[87] Domain Generalization for Medical Image Analysis  A Survey

[88] Generalizing deep learning models for medical image classification

[89] Hierarchical Deep Convolutional Neural Networks for Multi-category  Diagnosis of Gastrointestinal Disorders on Histopathological Images

[90] Deeply Supervised Layer Selective Attention Network  Towards  Label-Efficient Learning for Medical Image Classification

[91] Convolutional neural networks for medical image segmentation

[92] Visual Interpretable and Explainable Deep Learning Models for Brain  Tumor MRI and COVID-19 Chest X-ray Images

[93] Sequential Interpretability  Methods, Applications, and Future Direction  for Understanding Deep Learning Models in the Context of Sequential Data

[94] Artificial intelligence technology in oncology  a new technological  paradigm

[95] COIN  Counterfactual inpainting for weakly supervised semantic  segmentation for medical images

[96] Global explainability in aligned image modalities

[97] Explainable and Lightweight Model for COVID-19 Detection Using Chest  Radiology Images

[98] Interpretable and synergistic deep learning for visual explanation and  statistical estimations of segmentation of disease features from medical  images

[99] Deep image mining for diabetic retinopathy screening

[100] Global and Local Interpretability for Cardiac MRI Classification

[101] A Trustworthy Framework for Medical Image Analysis with Deep Learning

[102] RIDGE  Reproducibility, Integrity, Dependability, Generalizability, and  Efficiency Assessment of Medical Image Segmentation Models

[103] Deep learning-based synthetic-CT generation in radiotherapy and PET  a  review

[104] A New Multimodal Medical Image Fusion based on Laplacian Autoencoder  with Channel Attention

[105] Differentiable Graph Module (DGM) for Graph Convolutional Networks

[106] MICA  Towards Explainable Skin Lesion Diagnosis via Multi-Level  Image-Concept Alignment

[107] Unified Representation Learning for Efficient Medical Image Analysis

[108] ResNet Structure Simplification with the Convolutional Kernel Redundancy  Measure

[109] IamNN  Iterative and Adaptive Mobile Neural Network for Efficient Image  Classification

[110] Deep learning for unsupervised domain adaptation in medical imaging   Recent advancements and future perspectives

[111] Medical Image Harmonization Using Deep Learning Based Canonical Mapping   Toward Robust and Generalizable Learning in Imaging

[112] Rescuing referral failures during automated diagnosis of domain-shifted  medical images

[113] View it like a radiologist  Shifted windows for deep learning  augmentation of CT images

[114] Domain Shift in Computer Vision models for MRI data analysis  An  Overview

[115] O-MedAL  Online Active Deep Learning for Medical Image Analysis

[116] Rethinking Model Prototyping through the MedMNIST+ Dataset Collection

[117] A Novel Corpus of Annotated Medical Imaging Reports and Information  Extraction Results Using BERT-based Language Models

[118] Specialty-Oriented Generalist Medical AI for Chest CT Screening

[119] Justifying Diagnosis Decisions by Deep Neural Networks

[120] High Efficient Reconstruction of Single-shot T2 Mapping from  OverLapping-Echo Detachment Planar Imaging Based on Deep Residual Network

[121] Efficient ResNets  Residual Network Design

[122] RegNet  Self-Regulated Network for Image Classification
