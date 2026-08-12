# Registered common-window excerpt

# Self-Supervised Representation Learning: Methods, Modalities, and Applications

## Introduction and Foundations of Self-Supervised Representation Learning

### Motivation and Problem Formulation

Supervised deep learning has achieved remarkable success across diverse domains; however, its dependence on large-scale annotated datasets constitutes a fundamental bottleneck. Manual annotation is costly, time-consuming, and often impractical, particularly in specialized domains such as histopathology ref [1], medical imaging ref [16], and industrial inspection ref [39]. Self-supervised representation learning (SSRL) addresses this challenge by exploiting the intrinsic structure of unlabeled data to generate supervisory signals automatically ref [27], eliminating reliance on human-provided labels. By formulating pretext tasks from raw data, SSRL enables models to learn transferable representations applicable to diverse downstream tasks ref [29], achieving performance competitive with fully supervised alternatives across modalities including images, speech ref [23], time series ref [2], and graphs ref [19].

### Historical Context and Evolution

The evolution of self-supervised representation learning (SSRL) traces from early handcrafted pretext tasks—such as rotation prediction ref [12], jigsaw puzzle solving ref [10], colorization, and inpainting ref [7]—toward increasingly sophisticated frameworks. Contrastive learning methods subsequently emerged as dominant paradigms, leveraging instance discrimination objectives across visual ref [27], temporal ref [2], and graph-structured data ref [19]. The field then witnessed the rise of masked modeling approaches ref [40], exemplified by masked autoencoders ref [4] and masked prediction for speech ref [36]. Most recently, self-distillation and clustering-based frameworks ref [3] have unified generative and discriminative objectives, extending SSRL across diverse modalities including video ref [13], point clouds ref [15], and multimodal data ref [35].

### Taxonomy of Self-Supervised Learning Approaches

Self-supervised representation learning methods can be organized into four principal families. Generative approaches learn representations by reconstructing masked or corrupted inputs, as exemplified by masked autoencoders ref [4] and masked diffusion models ref [42]. Contrastive methods maximize agreement between augmented views of the same instance while repelling representations of distinct instances, underpinning frameworks such as TS-TCC ref [2] and graph contrastive methods ref [19]. Predictive approaches solve pretext tasks—including rotation prediction ref [12], permutation solving ref [10], and pace prediction ref [8]—to induce structurally meaningful representations. Self-distillation methods, such as BYOL and DINO-based frameworks ref [56], align student outputs with a momentum teacher without explicit negatives. Hybrid frameworks ref [44] combine multiple paradigms, leveraging complementary supervisory signals across these categories ref [27].

### Key Concepts and Terminology

A precise understanding of foundational terminology is essential for navigating the SSRL literature. A **pretext task** is a self-defined auxiliary objective—such as rotation prediction ref [12], jigsaw puzzle solving ref [10], or masked reconstruction ref [4]—that generates supervisory signals from unlabeled data. The representations learned are subsequently transferred to **downstream tasks** via linear evaluation or fine-tuning ref [27]. **Data augmentation** produces **positive pairs** (views of the same instance) and **negative pairs** (views of distinct instances) central to contrastive objectives ref [2]. **Representation transferability** measures generalization across tasks ref [32], while **evaluation protocols** including linear probing, few-shot learning, and fine-tuning benchmark learned feature quality ref [33].

### Relationship to Related Paradigms

Self-supervised representation learning (SSRL) occupies a distinct yet overlapping position among related learning paradigms. Unlike supervised learning, SSRL requires no manual annotations, instead deriving supervisory signals from data structure itself ref [27]. Compared to unsupervised representation learning, SSRL employs explicit pretext objectives rather than purely generative modeling ref [29]. It differs from semi-supervised learning, which assumes limited labeled data availability, though SSRL representations can be extended into semi-supervised frameworks ref [2]. Transfer learning leverages SSRL-pretrained models for downstream adaptation ref [35], while multi-task learning jointly optimizes multiple objectives simultaneously ref [26]. SSRL thus serves as a foundational pretraining paradigm that complements and enables these related approaches across diverse modalities.

## Core Methodological Frameworks in Self-Supervised Representation Learning

### Contrastive Learning Methods

Contrastive learning constitutes one of the most influential methodological families in self-supervised representation learning, operating through instance discrimination objectives that maximize agreement between differently augmented views of the same instance while repelling representations of distinct instances. Foundational frameworks such as SimCLR and MoCo establish the core paradigm of positive and negative pair construction, employing projection heads and memory banks to stabilize training ref [27]. Addressing learning inefficiency, ref [28] identifies under-clustering and over-clustering as fundamental obstacles and proposes a truncated triplet loss to improve discriminative efficiency. Extensions to dense prediction leverage set-wise similarity ref [22], while domain-specific adaptations demonstrate contrastive learning's versatility across astronomical imaging ref [58], metal surface inspection ref [39], and wafer defect classification ref [48].

### Masked Image and Signal Modeling

Masked image and signal modeling constitutes a prominent SSRL paradigm wherein models learn representations by reconstructing masked portions of input data. The Context Autoencoder (CAE) ref [4] advances masked image modeling (MIM) by predicting representations of masked patches in encoded space via an encoder–regressor–decoder architecture, encouraging separation between representation learning and reconstruction. Mixed Autoencoder (MixedAE) ref [9] integrates mixing augmentation with an auxiliary homologous recognition pretext task to mitigate mutual information increase, achieving superior dense prediction performance. For video, Masked Video Distillation (MVD) ref [13] employs a two-stage framework distilling from spatial and temporal teachers via masked feature modeling. Masked diffusion models ref [42] further extend this paradigm by substituting Gaussian noise with masking mechanisms for semantic segmentation. Comprehensive surveys of masked modeling ref [40] systematize masking strategies, reconstruction targets, and architectures across modalities.

### Predictive and Pretext Task-Based Methods

Predictive and pretext task-based methods constitute a foundational family of SSRL approaches, wherein models are trained to solve auxiliary tasks that require understanding the intrinsic structure of data. Canonical examples include rotation prediction ref [12], jigsaw puzzle solving and permutation prediction ref [10], colorization, and inpainting ref [7]. For video data, space-time cubic puzzles ref [7] and pace prediction ref [8] extend these ideas to spatio-temporal domains, compelling networks to capture temporal dynamics. In medical imaging, anatomy-aware pretext tasks such as video reordering and geometric transformation prediction have been employed for ultrasound data ref [16]. These tasks collectively encourage models to learn semantically meaningful, transferable representations without manual annotation.

### Self-Distillation and Knowledge Distillation Approaches

Self-distillation and knowledge distillation approaches constitute a prominent family of SSRL methods that learn representations through teacher-student frameworks without requiring explicit negative samples. BYOL and DINO pioneered this paradigm, where a student network is trained to match the outputs of a momentum-updated or exponential moving average teacher. DinoSR extends this to speech by combining self-distillation with online clustering and masked language modeling, yielding strong speech representations ref [3]. MOKD further advances this framework by enabling two heterogeneous models to learn collaboratively through self-distillation and cross-distillation modes, enhancing semantic alignment via cross-attention feature search ref [56]. Point2Vec adapts data2vec's masked student-teacher approach to point clouds, demonstrating strong transferability across shape classification and segmentation tasks ref [15].

### Clustering-Based and Prototype Learning Methods

Clustering-based and prototype learning methods leverage unsupervised cluster assignments or learned prototypes as self-supervised signals, circumventing the need for explicit negative pairs. HuBERT ref [36] employs offline k-means clustering to generate discrete pseudo-labels for masked prediction in speech, iteratively refining cluster quality across training stages. DinoSR ref [3] extends this paradigm by integrating online clustering with self-distillation, dynamically constructing a machine-discovered phoneme inventory to guide student network training. For video, deep clustering iteratively assigns pseudo-labels via k-means to enforce intra-class temporal compactness ref [38]. SlotCon ref [20] assigns pixels to learnable prototypes through attentive pooling, jointly optimizing semantic grouping and contrastive objectives for scene-centric representation learning, demonstrating strong performance on dense prediction tasks.

### Hybrid and Combined Frameworks

Hybrid and combined frameworks integrate multiple self-supervised paradigms to overcome the limitations inherent in any single approach. SiameseIM ref [21] unifies instance discrimination and masked image modeling by predicting dense representations of one augmented view from a masked sibling view, achieving superior spatial sensitivity and semantic alignment simultaneously. CS-CO ref [44] sequentially combines generative cross-stain prediction with discriminative contrastive learning for histopathological images, demonstrating that generative and discriminative objectives complement each other. TimesURL ref [17] jointly optimizes contrastive learning with time reconstruction to capture both segment-level and instance-level temporal information. Similarly, MVD ref [13] integrates masked

[... deterministic omitted span ...]

to 100 million images can match or exceed supervised pre-training on object detection, surface normal estimation, and visual navigation, while revealing that current methods insufficiently exploit large-scale data for high-level semantic representations. Complementarily, ref [32] introduces the Visual Representation Benchmark (ViRB), systematically analyzing over 700 training experiments across 30 encoders, 4 pre-training datasets, and 20 downstream tasks, providing rigorous comparative evaluation of contrastive self-supervised methods against supervised counterparts and illuminating how pre-training data characteristics influence downstream task performance.

## Self-Supervised Representation Learning for Sequential and Temporal Data

### Time Series Representation Learning

Self-supervised representation learning for time series has emerged as a critical research direction given the scarcity of labeled temporal data. TS-TCC introduces temporal and contextual contrasting modules with weak and strong augmentations to learn robust representations from unlabeled time series ref [2]. TimesURL addresses limitations in positive/negative pair construction through frequency-temporal augmentation and double Universum hard negatives, achieving state-of-the-art performance across six downstream tasks including forecasting, imputation, classification, and anomaly detection ref [17]. The Cross Reconstruction Transformer (CRT) leverages dropped temporal-spectral modeling via fast Fourier transform to capture cross-domain correlations, supplemented by instance discrimination constraints ref [25]. Mixing-based contrastive frameworks further exploit label smoothing perspectives for both univariate and multivariate time series, demonstrating strong transfer learning capabilities for clinical applications ref [11].

### Augmentation Strategies for Temporal Data

Augmentation strategies are critical to the effectiveness of self-supervised contrastive learning for temporal data. TS-TCC introduces weak and strong augmentations tailored to time series, enabling robust temporal and contextual contrasting ref [2]. TimesURL proposes frequency-temporal-based augmentation to preserve temporal properties while constructing hard negative pairs via double Universums ref [17]. CRT employs cross-domain dropping in both time and frequency domains, leveraging fast Fourier transform to capture temporal-spectral correlations ref [25]. STF-CSL applies both time-domain and frequency-domain augmentations within an STFNet-based framework for IoT sensing signals ref [6]. CARLA injects synthetic anomalies as negative samples, redefining augmentation assumptions for anomaly detection ref [47]. Collectively, these works demonstrate that domain-aware augmentation design is essential for learning high-quality temporal representations.

### Self-Supervised Speech Representation Learning

Self-supervised speech representation learning has emerged as a transformative paradigm, addressing the scarcity of labeled speech data across diverse languages and tasks. Foundational frameworks such as wav2vec 2.0 and HuBERT ref [36] employ masked prediction objectives over continuous speech inputs, with HuBERT utilizing offline clustering to generate pseudo-labels that guide BERT-like masked prediction. DinoSR ref [3] further advances this direction by integrating masked language modeling, self-distillation, and online clustering to yield machine-discovered phonetic inventories. At scale, XLS-R ref [18] extends wav2vec 2.0 to cross-lingual pretraining on 128 languages, achieving substantial improvements in ASR and speech translation. These methods collectively demonstrate that self-supervised objectives can yield universal speech representations transferable across recognition, verification, and translation tasks ref [23].

### EEG and Physiological Signal Representation Learning

EEG and physiological signals present unique challenges for self-supervised representation learning due to high noise levels, inter-subject variability, and costly clinical annotations. ref [14] demonstrates that predicting temporal context—whether time windows originate from the same temporal neighborhood—yields informative EEG representations that outperform supervised baselines in low-label regimes, with strong performance on sleep staging tasks. Building upon this, ref [34] proposes ContraWR, a contrastive framework that leverages global dataset statistics as negative references rather than explicit negative sample sets, achieving superior sleep staging performance across multiple large-scale EEG datasets while exhibiting robustness to noise and label scarcity. These approaches collectively establish self-supervised pretraining as a principled strategy for physiological signal analysis.

### IoT and Sensing Signal Representation Learning

IoT and physical sensing applications present unique challenges for self-supervised representation learning (SSRL), as underlying physical phenomena such as vibration, acceleration, and wireless signal propagation are fundamentally frequency-dependent. To address this, ref [6] proposes STF-CSL, a contrastive SSRL framework built upon Short-Time Fourier Neural Networks that incorporates both time-domain and frequency-domain augmentations, substantially outperforming time-domain-only baselines on human activity recognition tasks. Complementarily, ref [57] introduces self-supervised RF signal representation learning for automatic modulation recognition, demonstrating nearly an order-of-magnitude improvement in sample efficiency. Together, these works establish that domain-informed, frequency-aware augmentation and architecture design are critical for effective SSRL in IoT and sensing contexts.

### Trajectory and Spatial-Temporal Representation Learning

Trajectory representation learning presents unique challenges arising from the complex spatial-temporal characteristics inherent in mobility data. START (Self-supervised Trajectory Representation learning with Temporal Regularities and Travel semantics) ref [41] addresses these challenges through a two-stage framework. The first stage employs a Trajectory Pattern-Enhanced Graph Attention Network (TPE-GAT) to encode road network features and travel semantics into road segment representations. The second stage utilizes a Time-Aware Trajectory Encoder incorporating temporal regularities. Two complementary self-supervised tasks—span-masked trajectory recovery and trajectory contrastive learning—explicitly exploit spatial-temporal characteristics during training. Extensive experiments on large-scale real-world datasets across multiple downstream tasks, including trajectory classification, clustering, and similarity computation, demonstrate START's effectiveness and cross-city transferability.

### Anomaly Detection via Self-Supervised Representations

Self-supervised representations have demonstrated considerable utility in anomaly detection, where labeled anomalous samples are inherently scarce. CARLA ref [47] introduces a contrastive representation learning framework for time series anomaly detection that injects synthetic anomalies as negative samples, enabling the model to learn both normal behavior and deviation patterns simultaneously, surpassing state-of-the-art unsupervised methods across seven benchmarks. For anomalous sound detection, ref [60] proposes a unified self-supervised framework with time-frequency attention and a centre imprinting strategy for domain shift adaptation. Additionally, neighborhood-relational encoding ref [49] demonstrates applicability to video anomaly detection, while TimesURL ref [17] further validates self-supervised representations on time series anomaly detection as one of six evaluated downstream tasks.

## Self-Supervised Representation Learning for Graph and Structured Data

### Graph Contrastive Learning

Graph contrastive learning adapts contrastive self-supervised objectives to graph-structured data, addressing the scarcity of labeled graphs by exploiting structural correspondences as supervisory signals. MERIT enhances Siamese self-distillation with multi-scale contrastive learning, generating augmented views from local and global graph perspectives and employing cross-view and cross-network contrastiveness to maximize node representation agreement ref [5]. Subg-Con leverages strong correlations between central nodes and sampled subgraphs, defining contrastive losses over subgraph regions rather than complete graphs to improve scalability and capture regional structural information ref [45]. Graph Barlow Twins eliminates the need for negative samples entirely by employing a cross-correlation-based loss, achieving competitive performance while requiring substantially fewer hyperparameters and approximately 30 times faster computation than BGRL ref [19].

### Graph-Level and Substructure-Level Representation Learning

Graph-level and substructure-level representation learning addresses the limitation of methods that aggregate only node-level information, neglecting the rich semantic content encoded in local substructures such as motifs and subgraphs. CLEAR ref [59] exemplifies this direction by modeling graph semantics at two complementary granularities: global semantics, captured via graph-level augmentation and graph neural network encoders, and local semantics, obtained by partitioning graphs into subgraphs through clustering techniques and aggregating their representations via self-attention. By integrating both views into a multiview contrastive learning framework, CLEAR substantially enhances semantic-discriminative ability, demonstrating strong performance on graph classification and transfer learning benchmarks, thereby establishing the importance of jointly exploiting global and substructure-level representations.

### Node-Level Self-Supervised Learning

Node-level self-supervised learning focuses on learning discriminative representations for individual nodes within graph-structured data. A prominent approach is the Multi-scale contrastive Siamese network (MERIT), which enhances Siamese self-distillation with multi-scale contrastive learning by generating augmented views from local and global graph perspectives ref [5]. MERIT employs cross-view and cross-network contrastiveness objectives to maximize agreement between node representations across different views and networks, achieving state-of-the-art results that surpass semi-supervised counterparts. Complementarily, neighborhood-relational encoding (NRE) preserves local manifold structure among training samples, integrating relational information into encoder-decoder architectures to yield robust node-level representations applicable across classification, detection, and segmentation tasks ref [49]. Together, these approaches demonstrate that combining multi-scale perspectives with relational neighborhood information substantially enhances node representation quality.

### Scalability and Efficiency in Graph SSL

Scalability and efficiency represent critical challenges in self-supervised graph representation learning, as conventional graph neural networks operating on complete graph data face prohibitive computational and memory costs on large-scale graphs. Subg-Con addresses this limitation by learning node representations through contrastive objectives defined over sampled subgraphs rather

[... deterministic omitted span ...]

learning and downstream task adaptation constitute a critical evaluation paradigm for self-supervised representations. Learned representations are typically transferred via linear probing, full fine-tuning, or few-shot learning protocols. Linear evaluation assesses representational quality without task-specific adaptation, while fine-tuning allows deeper alignment with downstream objectives ref [27]. Self-supervised pretraining has demonstrated strong transferability across diverse tasks: HistoSSL transfers to tissue phenotyping and cancer metastasis detection ref [1], Point2Vec achieves competitive results on shape classification and part segmentation ref [15], and TimesURL supports six downstream tasks including forecasting and anomaly detection ref [17]. In low-label regimes, self-supervised pretraining consistently outperforms purely supervised baselines, as demonstrated in astronomical imaging ref [58], EEG sleep staging ref [14], and CAD geometry analysis ref [24].

### Semi-Supervised Extensions of Self-Supervised Learning

Semi-supervised extensions of self-supervised learning leverage limited labeled data alongside abundant unlabeled data to further enhance representation quality. A prominent example is CA-TCC ref [2], which extends the contrastive TS-TCC framework by incorporating pseudo-labels generated from self-supervised pretraining into a class-aware contrastive loss, yielding substantial gains in low-label regimes. Similarly, self-supervised pretraining followed by fine-tuning on scarce annotations has demonstrated effectiveness in EEG analysis ref [14], astronomical imaging ref [58], and wafer defect classification ref [48]. Masked diffusion models ref [42] also exhibit strong few-shot segmentation performance. Collectively, these approaches confirm that self-supervised representations provide robust initializations that significantly reduce labeled data requirements across diverse domains.

### Computational Efficiency and Scalability

Computational efficiency and scalability represent critical considerations in deploying self-supervised representation learning (SSRL) at scale. Graph Barlow Twins demonstrates approximately 30 times faster computation than BGRL while achieving competitive performance, highlighting that negative-sample-free objectives can substantially reduce training overhead ref [19]. MixedAE accelerates training by 2× compared to iBOT while surpassing its performance ref [9]. Scaling SSRL to 100 million images reveals that larger data and increased task difficulty can match supervised pretraining on detection and navigation tasks, though current methods struggle to fully exploit large-scale data for high-level semantic representations ref [33]. CCBERT further illustrates efficiency gains, requiring 6–10× less training time and 5–30× less inference time than comparable large pretrained models ref [30].

### Addressing Learning Inefficiency in Contrastive Learning

Contrastive learning methods suffer from pronounced learning inefficiency, requiring approximately ten times more training epochs than supervised learning to achieve comparable recognition accuracy ref [28]. Two contradictory phenomena underlie this inefficiency: under-clustering, wherein insufficient negative pairs prevent the model from effectively distinguishing inter-class samples, and over-clustering, wherein excessive negative pairs force semantically similar samples into disparate clusters ref [28]. To simultaneously address both problems, a truncated triplet loss framework has been proposed, employing a triplet objective to maximize relative distances between positive and negative pairs while selecting a negative sample deputy via a Bernoulli Distribution model to mitigate over-clustering ref [28]. Evaluations on large-scale benchmarks including ImageNet and COCO demonstrate clear superiority over state-of-the-art contrastive methods in learning efficiency.

### Domain-Specific Applications and Case Studies

Self-supervised representation learning has demonstrated substantial impact across diverse real-world domains. In healthcare, HistoSSL extracts multi-level histopathological representations ref [1], while CS-CO integrates cross-stain prediction with contrastive learning for H&E-stained images ref [44], and ContraWR advances EEG-based sleep staging ref [34]. Industrial inspection benefits from contrastive frameworks for metal surface defect detection ref [39] and wafer bin map classification ref [48]. In astronomy, contrastive pretraining enables galaxy morphology classification with fewer labels ref [58]. For wireless communications, SSL significantly improves automatic modulation recognition ref [57], while IoT sensing applications leverage frequency-domain contrastive learning ref [6]. These case studies collectively demonstrate that domain-informed augmentation and pretext task design are critical for effective SSL deployment.

### Open Challenges and Future Research Directions

Despite remarkable progress, several open challenges persist in self-supervised representation learning. Designing universal augmentation strategies applicable across diverse modalities remains unresolved, as domain-specific augmentations are often required for time series ref [17], speech ref [23], graphs ref [19], and medical imaging ref [1]. Theoretical understanding of why contrastive and masked objectives yield transferable representations is still nascent ref [28]. Evaluation beyond standard benchmarks is critical, as current protocols may not capture real-world transferability ref [32], ref [33]. Handling long-tail distributions, robustness under domain shift ref [60], and extending SSL to emerging modalities such as CAD geometry ref [24] and RF signals ref [57] represent fertile directions for future investigation.

## References

[1] HistoSSL: Self-Supervised Representation Learning for Classifying Histopathology Images. https://doi.org/10.3390/math11010110

[2] Self-Supervised Contrastive Representation Learning for Semi-Supervised Time-Series Classification. https://doi.org/10.1109/TPAMI.2023.3308189

[3] DinoSR: Self-Distillation and Online Clustering for Self-supervised Speech Representation Learning. https://doi.org/10.48550/arXiv.2305.10005

[4] Context Autoencoder for Self-supervised Representation Learning. https://doi.org/10.1007/s11263-023-01852-4

[5] Multi-Scale Contrastive Siamese Networks for Self-Supervised Graph Representation Learning. https://doi.org/10.24963/ijcai.2021/204

[6] Contrastive Self-Supervised Representation Learning for Sensing Signals from the Time-Frequency Perspective. https://doi.org/10.1109/ICCCN52240.2021.9522151

[7] Self-Supervised Video Representation Learning with Space-Time Cubic Puzzles. https://doi.org/10.1609/aaai.v33i01.33018545

[8] Self-supervised Video Representation Learning by Pace Prediction. https://doi.org/10.1007/978-3-030-58520-4_30

[9] Mixed Autoencoder for Self-Supervised Visual Representation Learning. https://doi.org/10.1109/CVPR52729.2023.02178

[10] Self-supervised representation learning by predicting visual permutations. https://doi.org/10.1016/j.knosys.2020.106534

[11] Mixing Up Contrastive Learning: Self-Supervised Representation Learning for Time Series. https://doi.org/10.1016/j.patrec.2022.02.007

[12] Self-Supervised Representation Learning by Rotation Feature Decoupling. https://doi.org/10.1109/CVPR.2019.01061

[13] Masked Video Distillation: Rethinking Masked Feature Modeling for Self-supervised Video Representation Learning. https://doi.org/10.1109/CVPR52729.2023.00611

[14] Self-Supervised Representation Learning from Electroencephalography Signals. https://doi.org/10.1109/MLSP.2019.8918693

[15] Point2Vec for Self-Supervised Representation Learning on Point Clouds. https://doi.org/10.48550/arXiv.2303.16570

[16] Self-Supervised Representation Learning for Ultrasound Video. https://doi.org/10.1109/ISBI45749.2020.9098666

[17] TimesURL: Self-supervised Contrastive Learning for Universal Time Series Representation Learning. https://doi.org/10.48550/arXiv.2312.15709

[18] XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale. https://doi.org/10.21437/interspeech.2022-143

[19] Graph Barlow Twins: A self-supervised representation learning framework for graphs. https://doi.org/10.1016/j.knosys.2022.109631

[20] Self-Supervised Visual Representation Learning with Semantic Grouping. https://doi.org/10.48550/arXiv.2205.15288

[21] Siamese Image Modeling for Self-Supervised Vision Representation Learning. https://doi.org/10.1109/CVPR52729.2023.00212

[22] Exploring Set Similarity for Dense Self-supervised Representation Learning. https://doi.org/10.1109/CVPR52688.2022.01609

[23] Self-Supervised Speech Representation Learning: A Review. https://doi.org/10.1109/JSTSP.2022.3207050

[24] Self-Supervised Representation Learning for CAD. https://doi.org/10.1109/CVPR52729.2023.02043

[25] Self-Supervised Time Series Representation Learning via Cross Reconstruction Transformer. https://doi.org/10.1109/TNNLS.2023.3292066

[26] Self-Supervised Representation Learning From Multi-Domain Data. https://doi.org/10.1109/ICCV.2019.00334

[27] Self-Supervised Representation Learning: Introduction, advances, and challenges. https://doi.org/10.1109/MSP.2021.3134634

[28] Solving Inefficiency of Self-supervised Representation Learning. https://doi.org/10.1109/ICCV48922.2021.00937

[29] A Survey on Self-Supervised Representation Learning. https://doi.org/10.48550/arXiv.2308.11455

[30] CCBERT: Self-Supervised Code Change Representation Learning. https://doi.org/10.1109/ICSME58846.2023.00028

[31] Distilling Localization for Self-Supervised Representation Learning. https://doi.org/10.1609/aaai.v35i12.17312

[32] Contrasting Contrastive Self-Supervised Representation Learning Pipelines. https://doi.org/10.1109/ICCV48922.2021.00980

[33] Scaling and Benchmarking Self-Supervised Visual Representation Learning. https://doi.org/10.1109/ICCV.2019.00649

[34] Self-Supervised Electroencephalogram Representation Learning for Automatic Sleep Staging: Model Development and Evaluation Study. https://doi.org/10.2196/46769

[35] Beyond Just Vision: A Review on Self-Supervised Representation Learning on Multimodal and Temporal Data. https://doi.org/10.48550/arXiv.2206.02353

[36] HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. https://doi.org/10.1109/taslp.2021.3122291

[37] Large-Scale Self-Supervised Speech Representation Learning for Automatic Speaker Verification. https://doi.org/10.1109/icassp43922.2022.9747814

[38] Self-Supervised Video Representation Learning Using Improved Instance-Wise Contrastive Learning and Deep Clustering. https://doi.org/10.1109/TCSVT.2022.3169469

[39] Contrastive self-supervised representation learning framework for metal surface defect detection. https://doi.org/10.1186/s40537-023-00827-z

[40] Masked Modeling for Self-supervised Representation Learning on Vision and Beyond. https://doi.org/10.48550/arXiv.2401.00897

[41] Self-supervised Trajectory Representation Learning with Temporal Regularities and Travel Semantics. https://doi.org/10.1109/ICDE55515.2023.00070

[42] Masked Diffusion as Self-supervised Representation Learner. https://doi.org/10.48550/arXiv.2308.05695

[43] Patch-level Representation Learning for Self-supervised Vision Transformers. https://doi.org/10.1109/CVPR52688.2022.00817

[44] CS-CO: A Hybrid Self-Supervised Visual Representation Learning Method for H&E-stained Histopathological Images. https://doi.org/10.1016/j.media.2022.102539

[45] Sub-graph Contrast for Scalable Self-Supervised Graph Representation Learning. https://doi.org/10.1109/ICDM50108.2020.00031

[46] Self-supervised Representation Learning From Random Data Projectors. https://doi.org/10.48550/arXiv.2310.07756

[47] CARLA: Self-supervised contrastive representation learning for time series anomaly detection. https://doi.org/10.1016/j.patcog.2024.110874

[48] Self-Supervised Representation Learning for Wafer Bin Map Defect Pattern Classification. https://doi.org/10.1109/TSM.2020.3038165

[49] Self-Supervised Representation Learning via Neighborhood-Relational Encoding. https://doi.org/10.1109/ICCV.2019.00810

[50] Self-Supervised Self-Organizing Clustering Network: A Novel Unsupervised Representation Learning Method. https://doi.org/10.1109/TNNLS.2022.3185638

[51] A Self-supervised Representation Learning of Sentence Structure for Authorship Attribution. https://doi.org/10.1145/3491203

[52] Self-Distilled Self-supervised Representation Learning. https://doi.org/10.1109/WACV56688.2023.00285

[53] Self-Supervised Representation Learning using Visual Field Expansion on Digital Pathology. https://doi.org/10.1109/ICCVW54120.2021.00077

[54] Self-supervised Contrastive Cross-Modality Representation Learning for Spoken Question Answering. https://doi.org/10.18653/v1/2021.findings-emnlp.3

[55] SelfDoc: Self-Supervised Document Representation Learning. https://doi.org/10.1109/CVPR46437.2021.00560

[56] Multi-Mode Online Knowledge Distillation for Self-Supervised Visual Representation Learning. https://doi.org/10.1109/CVPR52729.2023.01140

[57] Self-Supervised RF Signal Representation Learning for NextG Signal Classification With Deep Learning. https://doi.org/10.1109/LWC.2022.3217292

[58] Self-supervised Representation Learning for Astronomical Images. https://doi.org/10.3847/2041-8213/abf2c7

[59] CLEAR: Cluster-Enhanced Contrast for Self-Supervised Graph Representation Learning. https://doi.org/10.1109/TNNLS.2022.3177775

[60] Self-Supervised Representation Learning for Unsupervised Anomalous Sound Detection Under Domain Shift. https://doi.org/10.1109/icassp43922.2022.9747863
