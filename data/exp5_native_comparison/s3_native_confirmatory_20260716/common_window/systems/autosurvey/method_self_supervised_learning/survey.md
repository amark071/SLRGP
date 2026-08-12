# Registered common-window excerpt

# Self-Supervised Representation Learning: Methods, Theories, and Applications Across Domains

## 1 Introduction and Foundations of Self-Supervised Representation Learning

### 1.1 Motivation and the Annotation Bottleneck

## 1.1 The Annotation Bottleneck and the Case for Self-Supervised Learning

The remarkable success of deep learning over the past decade has been fundamentally underpinned by the availability of large-scale, carefully annotated datasets. From image classification and object detection to speech recognition and natural language understanding, supervised deep learning systems have consistently demonstrated that performance scales with the quantity and quality of labeled training data [1]. However, this dependence on human-annotated labels introduces a significant and often prohibitive bottleneck that constrains the broader applicability of deep learning across domains and modalities.

The cost of human annotation is substantial across virtually every data type. For natural images, annotating millions of samples with class labels, bounding boxes, or pixel-wise segmentation masks demands enormous investments of time and expert labor [2]. The challenge is magnified considerably for video data, where an annotator must watch entire recordings end-to-end before providing a single label, making large-scale annotation economically infeasible for many organizations [3; 4]. In speech and audio processing, transcription and speaker labeling require specialized linguistic expertise and careful attention, further escalating costs [5; 6]. For medical imaging modalities such as ultrasound, mammography, and histopathology slides, annotation must be performed by trained clinicians, making labeled data exceptionally scarce and expensive [7; 8]. In domains such as remote sensing, biological graphs, and scientific simulations, the annotation process requires domain expertise that is difficult to scale [9; 10].

Beyond raw cost, the annotation process introduces additional challenges including label noise, annotator disagreement, and inconsistencies arising from subjective or ambiguous category definitions [11]. Large-scale datasets obtained from web sources often carry inherently noisy labels that must be carefully managed [12; 13]. Furthermore, the requirement for human-generated annotations biases learned models toward the specific label taxonomy chosen at dataset creation time, limiting generalization to novel classes, tasks, or domains [3].

In stark contrast to the scarcity of labeled data, unlabeled data is abundant and freely available across essentially every domain and modality. The internet continuously generates vast quantities of unannotated images, videos, audio recordings, text, and sensor readings. Self-supervised learning (SSL) has emerged as a compelling paradigm that exploits this wealth of unlabeled data by constructing supervisory signals directly from the intrinsic structure of the data itself, without requiring any human annotation [14; 15]. Rather than relying on externally provided labels, SSL methods define so-called pretext tasks—auxiliary objectives derived automatically from the raw data—whose solution requires the model to learn meaningful and transferable feature representations.

The practical impact of this paradigm shift is substantial. Self-supervised pre-trained representations have been shown to dramatically reduce the need for labeled examples in downstream tasks [16; 17], to improve annotation efficiency when human labeling is still required [2], and to serve as strong initializations for fine-tuning on specialized domains where only small labeled datasets exist [18; 19]. Moreover, SSL representations have demonstrated greater robustness to dataset imbalance compared to their supervised counterparts, since they are not constrained to learn only label-relevant features [20]. Collectively, these properties establish self-supervised learning as a scalable, practical, and increasingly essential alternative to purely supervised approaches—one whose historical development, diverse methodological families, and broad applications this survey aims to comprehensively document.

Having established the fundamental motivation for self-supervised learning, the following section traces the historical arc of the field, from early unsupervised feature learning and handcrafted pretext tasks to the large-scale pretraining frameworks and foundation models that define the current state of the art.

### 1.2 Historical Context and Evolution of Self-Supervised Learning

## 1.2 Historical Context and Evolution of Self-Supervised Learning

Having established the fundamental motivation for self-supervised learning—namely, the annotation bottleneck and the abundance of unlabeled data—this section traces the historical arc of the field, from early unsupervised feature learning and handcrafted pretext tasks to the large-scale pretraining frameworks and foundation models that define the current state of the art. Understanding this evolution provides essential context for appreciating the design choices and motivations behind contemporary methods, and for situating the core concepts and terminology that will be introduced in the following section.

### Early Unsupervised Feature Learning and Autoencoders

The intellectual roots of self-supervised learning lie in classical unsupervised learning, where the primary goal was to discover compact, informative representations of data without relying on human-provided labels. Autoencoders, one of the earliest neural network-based approaches to unsupervised representation learning, were designed to reconstruct their inputs through a bottleneck architecture, forcing the network to learn a compressed representation of the data. Denoising autoencoders extended this principle by training networks to recover clean inputs from corrupted versions, introducing the idea that useful representations should be robust to noise. These early approaches established the foundational principle that the structure of the data itself could serve as a source of supervisory signal—a core insight that underlies all subsequent developments in self-supervised learning and that directly addresses the annotation bottleneck described in the preceding section.

Alongside autoencoders, restricted Boltzmann machines and deep belief networks explored generative approaches to feature learning through probabilistic modeling of data distributions. While these methods demonstrated that meaningful representations could be discovered without labels, they were often computationally demanding, difficult to scale, and outperformed by supervised methods when labeled data was available. As noted in comprehensive reviews of the field, traditional unsupervised methods such as layer-wise clustering or autoencoders remained inferior to supervised pretraining in many practical settings [21].

### The Era of Handcrafted Pretext Tasks in Computer Vision

A significant shift occurred when researchers began constructing explicit pretext tasks—auxiliary prediction problems defined over unlabeled data—to guide representation learning. Rather than asking networks to generalize across all possible tasks, pretext tasks provided concrete objectives that could be optimized end-to-end, encouraging networks to extract semantically meaningful features as a byproduct. As will be formalized in the following section, these pretext tasks are defined over automatically generated pseudo-labels derived from the intrinsic structure of the data itself.

One of the earliest and most influential pretext tasks in computer vision was image colorization, where a network is trained to predict the color channels of an image from its grayscale version. The intuition was compelling: to colorize a sky, a car, or vegetation correctly, the network must first learn to recognize these objects, thereby acquiring visual semantics in a fully self-supervised manner [21]. This work demonstrated that a single, well-designed pretext task could yield representations competitive with those learned by supervised methods on downstream recognition tasks.

Building on this success, a variety of handcrafted pretext tasks were proposed throughout the mid-2010s. Rotation prediction, in which networks are trained to classify which of four canonical rotations has been applied to an input image, became a widely studied approach. The rationale was that recognizing the rotation of an object requires understanding its canonical orientation, which in turn demands a degree of semantic understanding. Empirical studies later confirmed that rotation prediction is among the most semantically meaningful of the classical pretext tasks [22]. Other pretext tasks explored during this period included jigsaw puzzle solving, where the spatial arrangement of image patches is scrambled and the network must recover the correct permutation, relative patch position prediction, and context prediction from image regions.

While these handcrafted pretext tasks succeeded in extracting more meaningful representations than pure autoencoders, they also revealed an important challenge: networks often found shortcuts—low-level statistical regularities such as color aberrations, watermarks, or chromatic aberrations—that allowed them to solve the pretext task without learning semantically meaningful features [23]. This shortcut problem motivated researchers to design more robust pretext tasks and augmentation strategies that would discourage networks from exploiting superficial cues, a challenge that—as will be seen—ultimately shaped the design of the data augmentation pipelines central to modern SSL frameworks.

### Key Milestones in Natural Language Processing

In parallel with developments in computer vision, the NLP community made transformative advances in self-supervised representation learning. The introduction of word embedding methods such as Word2Vec and GloVe established that rich semantic relationships between words

[... deterministic omitted span ...]

4 Theoretical Understanding and Analysis of Self-Supervised Learning

### 4.1 Information-Theoretic Frameworks and Mutual Information Maximization

Information-theoretic frameworks provide a principled foundation for understanding why self-supervised learning (SSL) methods work and how to design more effective objectives. At the core of these frameworks lies the principle of mutual information (MI) maximization: by maximizing the mutual information between representations derived from different views or augmentations of the same underlying data, a model is encouraged to capture the shared, semantically meaningful content while discarding view-specific noise and irrelevant variation. This information-theoretic perspective complements the probabilistic and generative viewpoints discussed later, and together they form a cohesive theoretical foundation for modern SSL.

**Mutual Information Maximization.** The mutual information maximization perspective posits that a good representation should retain as much task-relevant information as possible from the input while compressing away nuisances. Formally, given two views $v_1$ and $v_2$ generated from a common source $x$ through stochastic augmentation, the representation learning objective can be written as maximizing $I(z_1; z_2)$, the mutual information between the encoded views. This principle underlies many contrastive methods: the InfoNCE loss, used in frameworks such as SimCLR and MoCo, can be interpreted as a lower bound on the mutual information between paired representations [14]. By optimizing this bound, contrastive methods effectively learn to align representations of positive pairs (augmented views of the same instance) while separating those of negative pairs, thereby maximizing a tractable surrogate for MI.

**Multi-View Redundancy and Task-Relevant Information.** A central concept in the information-theoretic analysis of SSL is the role of multi-view redundancy. When multiple views of an input are constructed—whether through spatial cropping, color jitter, temporal sampling in video, or cross-modal pairing—the shared information across views tends to correspond to high-level semantic content that is invariant to the specific augmentation applied [14; 3]. The principle is that by training a model to predict one view from another, or to produce consistent representations across views, it is implicitly forced to extract the redundant (shared) information between them. Theoretical analyses show that when augmentations are chosen such that the shared information between views corresponds closely to the information relevant for downstream tasks, the representations learned by MI maximization will generalize well to those tasks [15].

**Partial Information Decomposition.** A more refined information-theoretic lens is provided by partial information decomposition (PID), which decomposes the total information that a set of views provides about a target variable into components: redundant information (carried by all views), unique information (carried by only one view), and synergistic information (only available when all views are considered jointly). In the context of SSL, the redundant component corresponds to what contrastive objectives capture—the information shared across all constructed views. This decomposition clarifies an important limitation of standard multi-view contrastive learning: it may fail to capture synergistic information that is only accessible by jointly considering all views, and may discard unique information present in individual views that is nonetheless relevant for downstream tasks. This motivates the design of objectives that explicitly balance the trade-off between capturing redundant and synergistic components, a direction with implications for multi-modal self-supervised learning [14].

**Unifying Contrastive and Predictive Objectives.** The information-theoretic perspective offers a powerful lens for unifying seemingly disparate SSL objectives. Contrastive methods, which maximize agreement between positive pairs and minimize it for negative pairs, and predictive methods, which train a model to forecast masked or missing portions of the input, can both be cast as instances of MI maximization between different portions or views of the data [1; 15]. Predictive coding objectives, for example, maximize the MI between a context representation and future observations; masked language modeling maximizes the MI between context tokens and the masked token distribution. By framing both families under the common umbrella of maximizing lower bounds on MI, the theory provides not only a unified conceptual account of why these methods work, but also suggests principled criteria for evaluating and improving pretext task design: a pretext task is informative to the extent that the MI it maximizes aligns with the MI relevant for the downstream task. This unification sets the stage for the subsequent discussion of how generative and probabilistic latent variable models further bridge these paradigms under a common theoretical language.

**Implications for Augmentation Design.** The information-theoretic framework also illuminates the critical role of data augmentation. The choice of augmentation directly determines the shared information between views and therefore what the model learns to capture. If augmentations are too weak, the two views remain nearly identical and the MI objective is trivially satisfied without learning meaningful abstractions. If augmentations are too aggressive, the shared information may no longer include task-relevant semantics, leading to poor downstream performance [14; 35]. This analysis explains empirical findings that the precise choice and composition of augmentation strategies profoundly influences the quality of learned representations, and motivates information-theoretic criteria for augmentation selection.

**Limitations and Extensions.** Despite its elegance, the MI maximization framework faces practical challenges. Direct computation of MI in high-dimensional spaces is intractable, necessitating the use of variational bounds (e.g., InfoNCE) that may be loose or require large batch sizes with many negatives. Moreover, maximizing MI alone does not guarantee useful representations if the views share spurious correlations or if the downstream task requires information that is not captured by any pair of views [14]. Extensions of the framework, including minimum sufficient statistics perspectives, rate-distortion theory, and the information bottleneck principle, have been proposed to address these limitations by explicitly regularizing the amount of information retained in the representation and focusing it on task-relevant content. These limitations and extensions naturally motivate a complementary probabilistic treatment, wherein SSL objectives are grounded in explicit generative models and approximate inference procedures—a perspective developed in the following subsection. Together, these information-theoretic foundations continue to guide both the theoretical understanding and practical design of self-supervised representation learning systems.

### 4.2 Probabilistic and Generative Latent Variable Models

Probabilistic and generative latent variable models offer a principled theoretical lens through which the mechanisms of self-supervised learning can be understood from first principles. Rather than treating SSL objectives as heuristically motivated engineering choices, these frameworks interpret representation learning as approximate inference in a generative model, where the encoder learns to infer a compact latent code that explains the observed data. Building directly upon the information-theoretic foundations established in the previous section—where SSL was characterized through mutual information maximization and view redundancy—the probabilistic perspective deepens this understanding by grounding diverse SSL paradigms within a unified language of probabilistic inference and generative modeling. This perspective connects contrastive, clustering-based, and masked modeling methods, setting the stage for the kernel-based and spectral analyses that follow by establishing the structural properties of the latent spaces these methods produce.

**Variational Autoencoders and Generative Foundations**

The variational autoencoder (VAE) framework provides a natural starting point for understanding generative approaches to representation learning. By optimizing the evidence lower bound (ELBO), VAEs learn a probabilistic encoder that maps inputs to distributions over latent variables, and a decoder that reconstructs the input from samples of those latent variables. This generative perspective naturally motivates many self-supervised objectives: masked autoencoders, for instance, can be understood as learning to complete partially observed data under a generative model, where the masked patches serve as missing variables to be inferred from visible context [24]. The connection to denoising autoencoders is particularly illuminating, as denoising objectives can be derived from the score function of the data distribution, providing theoretical grounding for reconstruction-based SSL methods.

**Latent Variable Models for Contrastive Learning**

Contrastive learning methods have been interpreted through the lens of latent variable models, where the data augmentation process defines a generative process over views. Under this framework, two augmented views of the same image are treated as conditionally independent observations of a shared latent semantic variable. The contrastive objective then approximates maximum likelihood estimation in this generative model, encouraging the encoder to recover the shared latent factor while discarding view-specific nuisances. This formulation connects contrastive SSL to classical latent variable models such as factor analysis and probabilistic PCA, and provides theoretical grounding for why contrastive representations transfer

[... deterministic omitted span ...]

Learning  A Survey on Image-based Generative  and Discriminative Training

[27] Self-Supervised Speech Representation Learning  A Review

[28] Self-supervised Learning on Graphs  Contrastive, Generative,or  Predictive

[29] Beyond Just Vision  A Review on Self-Supervised Representation Learning  on Multimodal and Temporal Data

[30] Survey on Self-Supervised Multimodal Representation Learning and  Foundation Models

[31] A Multi-view Perspective of Self-supervised Learning

[32] Combining Probabilistic Logic and Deep Learning for Self-Supervised  Learning

[33] Self-supervised Label Augmentation via Input Transformations

[34] Self-supervised Learning from a Multi-view Perspective

[35] Recent Advancements in Self-Supervised Paradigms for Visual Feature  Representation

[36] Contrastive learning, multi-view redundancy, and linear models

[37] Self-supervision of Feature Transformation for Further Improving  Supervised Learning

[38] Self-supervised Feature Enhancement  Applying Internal Pretext Task to  Supervised Learning

[39] Pretext Tasks selection for multitask self-supervised speech  representation learning

[40] Investigating the Benefits of Projection Head for Representation  Learning

[41] Toward a Geometrical Understanding of Self-supervised Contrastive  Learning

[42] Deciphering the Projection Head  Representation Evaluation  Self-supervised Learning

[43] Rethinking Evaluation Protocols of Visual Representations Learned via  Self-supervised Learning

[44] Speech Self-Supervised Representations Benchmarking  a Case for Larger  Probing Heads

[45] On the Transferability of Large-Scale Self-Supervision to Few-Shot Audio  Classification

[46] Boosting Few-Shot Visual Learning with Self-Supervision

[47] Improving Few-Shot Learning with Auxiliary Self-Supervised Pretext Tasks

[48] Parameter Efficient Transfer Learning for Various Speech Processing  Tasks

[49] Boosting Self-Supervised Learning via Knowledge Transfer

[50] The Role of Pre-training Data in Transfer Learning

[51] A surprisingly simple technique to control the pretraining bias for  better transfer  Expand or Narrow your representation

[52] Revisiting the Transferability of Supervised Pretraining  an MLP  Perspective

[53] SEED  Self-supervised Distillation For Visual Representation

[54] Unsupervised Representation Learning for Binary Networks by Joint  Classifier Learning

[55] Bi-tuning of Pre-trained Representations

[56] How to prepare your task head for finetuning

[57] Understanding Optimal Feature Transfer via a Fine-Grained Bias-Variance  Analysis

[58] Self-Supervised Learning in Event Sequences  A Comparative Study and  Hybrid Approach of Generative Modeling and Contrastive Learning

[59] Self-Supervised Learning for Time Series  Contrastive or Generative 

[60] Self-Supervised Learning for Time Series Analysis  Taxonomy, Progress,  and Prospects

[61] Debiased Contrastive Learning

[62] Contrastive Self-supervised Learning in Recommender Systems  A Survey

[63] Self-supervised representation learning from electroencephalography  signals

[64] Self-Distilled Representation Learning for Time Series

[65] A Bayesian Unification of Self-Supervised Clustering and Energy-Based  Models

[66] Semi-supervised learning made simple with self-supervised clustering

[67] Understand and Improve Contrastive Learning Methods for Visual  Representation  A Review

[68] GEDI  GEnerative and DIscriminative Training for Self-Supervised  Learning

[69] Learning Speech Representations from Raw Audio by Joint Audiovisual  Self-Supervision

[70] SubTab  Subsetting Features of Tabular Data for Self-Supervised  Representation Learning

[71] A Survey on Self-Supervised Learning for Non-Sequential Tabular Data

[72] Self-Supervised MultiModal Versatile Networks

[73] M5Product  Self-harmonized Contrastive Learning for E-commercial  Multi-modal Pretraining

[74] Self-Supervised Learning by Cross-Modal Audio-Video Clustering

[75] VideoBERT  A Joint Model for Video and Language Representation Learning

[76] DABS  A Domain-Agnostic Benchmark for Self-Supervised Learning

[77] A Survey on Self-supervised Pre-training for Sequential Transfer  Learning in Neural Networks

[78] Transfer of Pretrained Model Weights Substantially Improves  Semi-Supervised Image Classification

[79] Better Self-training for Image Classification through Self-supervision

[80] Statistical and Algorithmic Insights for Semi-supervised Learning with  Self-training

[81] Self-Tuning for Data-Efficient Deep Learning

[82] Transfer Learning or Self-supervised Learning  A Tale of Two Pretraining  Paradigms

[83] A Survey of the Impact of Self-Supervised Pretraining for Diagnostic  Tasks with Radiological Images

[84] An Evaluation of Self-Supervised Pre-Training for Skin-Lesion Analysis

[85] Functional Knowledge Transfer with Self-supervised Representation  Learning

[86] Composite Learning for Robust and Effective Dense Predictions

[87] Graph-Based Neural Network Models with Multiple Self-Supervised  Auxiliary Tasks

[88] A Brief Summary of Interactions Between Meta-Learning and  Self-Supervised Learning

[89] Advances and Challenges in Meta-Learning  A Technical Review

[90] Diverse Distributions of Self-Supervised Tasks for Meta-Learning in NLP

[91] Unleash Model Potential  Bootstrapped Meta Self-supervised Learning

[92] Self-Supervised Video Representation Learning with Meta-Contrastive  Network

[93] Supervised Momentum Contrastive Learning for Few-Shot Classification

[94] Self-Supervised Models are Continual Learners

[95] Revisiting Supervision for Continual Representation Learning

[96] Task Agnostic Representation Consolidation  a Self-supervised based  Continual Learning Approach

[97] Continual Self-supervised Learning  Towards Universal Multi-modal  Medical Data Representation Learning

[98] Hypernetworks for Continual Semi-Supervised Learning

[99] Heterogeneous Contrastive Learning for Foundation Models and Beyond

[100] Brief Introduction to Contrastive Learning Pretext Tasks for Visual  Representation

[101] RegCLR  A Self-Supervised Framework for Tabular Representation Learning  in the Wild

[102] Aggregative Self-Supervised Feature Learning from a Limited Sample

[103] Learning Downstream Task by Selectively Capturing Complementary  Knowledge from Multiple Self-supervisedly Learning Pretexts

[104] DMT  Comprehensive Distillation with Multiple Self-supervised Teachers

[105] Generative or Contrastive  Phrase Reconstruction for Better Sentence  Representation Learning

[106] Multi-task Self-Supervised Learning for Human Activity Detection

[107] Self-supervised Representation Learning for Ultrasound Video

[108] Coreset Sampling from Open-Set for Fine-Grained Self-Supervised Learning

[109] When Does Contrastive Visual Representation Learning Work 

[110] Relational Self-Supervised Learning

[111] Can Generative Models Improve Self-Supervised Representation Learning 

[112] Improving out-of-distribution generalization via multi-task  self-supervised pretraining

[113] An Experimental Comparison Of Multi-view Self-supervised Methods For  Music Tagging

[114] SLIP  Self-supervision meets Language-Image Pre-training

[115] MT-SLVR  Multi-Task Self-Supervised Learning for Transformation  In(Variant) Representations

[116] LeOCLR  Leveraging Original Images for Contrastive Learning of Visual  Representations

[117] Identical and Fraternal Twins  Fine-Grained Semantic Contrastive  Learning of Sentence Representations

[118] Synthetic Hard Negative Samples for Contrastive Learning

[119] Max-Margin Contrastive Learning

[120] Learning the Unlearned  Mitigating Feature Suppression in Contrastive  Learning

[121] Time-Series Contrastive Learning against False Negatives and Class  Imbalance

[122] Contrastive Learning with Negative Sampling Correction

[123] Siamese Prototypical Contrastive Learning

[124] Contrastive Attraction and Contrastive Repulsion for Representation  Learning

[125] ScoreCL  Augmentation-Adaptive Contrastive Learning via Score-Matching  Function

[126] Adaptive Multi-head Contrastive Learning

[127] Contrastive Speaker Embedding With Sequential Disentanglement

[128] An Effective Deployment of Contrastive Learning in Multi-label Text  Classification

[129] CODER  Coupled Diversity-Sensitive Momentum Contrastive Learning for  Image-Text Retrieval

[130] Dual Temperature Helps Contrastive Learning Without Many Negative  Samples  Towards Understanding and Simplifying MoCo

[131] Decoupled Contrastive Learning

[132] EqCo  Equivalent Rules for Self-supervised Contrastive Learning

[133] Unbiased and Efficient Self-Supervised Incremental Contrastive Learning

[134] On the Surrogate Gap between Contrastive and Supervised Losses

[135] On neural and dimensional collapse in supervised and unsupervised  contrastive learning with hard negative sampling

[136] A Brief Overview of Unsupervised Neural Speech Representation Learning

[137] A Probabilistic Model to explain Self-Supervised Representation Learning

[138] Unraveling Projection Heads in Contrastive Learning  Insights from  Expansion and Shrinkage

[139] Clustering augmented Self-Supervised Learning  Anapplication to Land  Cover Mapping

[140] Scaling and Benchmarking Self-Supervised Visual Representation Learning

[141] ViCE  Improving Dense Representation Learning by Superpixelization and  Contrasting Cluster Assignment

[142] A simple, efficient and scalable contrastive masked autoencoder for  learning visual representations

[143] Masked Contrastive Representation Learning

[144] Unsupervised Object-Level Representation Learning from Scene Images

[145] SegLoc  Visual Self-supervised Learning Scheme for Dense Prediction  Tasks of Security Inspection X-ray Images

[146] i-Code  An Integrative and Composable Multimodal Learning Framework

[147] TVDIM  Enhancing Image Self-Supervised Pretraining via Noisy Text Data

[148] Unsupervised Natural Language Inference via Decoupled Multimodal  Contrastive Learning

[149] TextTopicNet - Self-Supervised Learning of Visual Features Through  Embedding Images on Semantic Text Spaces

[150] Evaluating Fairness in Self-supervised and Supervised Models for  Sequential Data

[151] A Simple Framework for Contrastive Learning of Visual Representations

[152] Trip-ROMA  Self-Supervised Learning with Triplets and Random Mappings

[153] Understanding Collapse in Non-Contrastive Siamese Representation  Learning

[154] Elastic Weight Consolidation Improves the Robustness of Self-Supervised  Learning Methods under Transfer

[155] Masked Autoencoders Are Scalable Vision Learners

[156] Modality-Agnostic Self-Supervised Learning with Meta-Learned Masked  Auto-Encoder

[157] Feature Guided Masked Autoencoder for Self-supervised Learning in Remote  Sensing

[158] Multi-modal Masked Autoencoders Learn Compositional Histopathological  Representations

[159] ViC-MAE  Self-Supervised Representation Learning from Images and Video  with Contrastive Masked Autoencoders

[160] Masked Autoencoders that Listen

[161] Masked Spectrogram Modeling using Masked Autoencoders for Learning  General-purpose Audio Representation

[162] SupMAE  Supervised Masked Autoencoders Are Efficient Vision Learners

[163] iBoot  Image-bootstrapped Self-Supervised Video Representation Learning

[164] Understanding Self-Supervised Learning of Speech Representation via  Invariance and Redundancy Reduction

[165] Revisiting Self-Training with Regularized Pseudo-Labeling for Tabular  Data

[166] Discriminative, Generative and Self-Supervised Approaches for  Target-Agnostic Learning

[167] Self-reinforcing Unsupervised Matching

[168] ArCL  Enhancing Contrastive Learning with Augmentation-Robust  Representations
