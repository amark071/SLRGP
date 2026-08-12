# Registered common-window excerpt

# Vision-Language Pretraining: A Comprehensive Survey of Methods, Architectures, and Applications

## Introduction and Background

### Motivation and Significance of Vision-Language Pretraining

Vision-language pretraining (VLP) addresses fundamental limitations of unimodal models by learning joint representations that capture cross-modal semantics. Unimodal systems trained exclusively on images or text fail to model the rich correspondences between visual and linguistic content essential for tasks such as visual question answering, image captioning, and cross-modal retrieval ref [29]. VLP models like BEiT-3 ref [2] and FLAVA ref [10] demonstrate that unified multimodal pretraining achieves state-of-the-art performance across diverse benchmarks simultaneously. Furthermore, VLP enables zero-shot and few-shot transfer ref [11], domain-specific adaptation ref [8], and scalable generalization ref [27], establishing multimodal foundation models as indispensable infrastructure for modern AI systems.

### Historical Evolution: From Unimodal to Multimodal Pretraining

The trajectory of vision-language pretraining (VLP) originates from seminal advances in unimodal pretraining. In NLP, transformer-based models such as BERT established masked language modeling as a powerful self-supervised objective, while GPT demonstrated autoregressive generation at scale. Concurrently, computer vision progressed from convolutional architectures to Vision Transformers. Early VLP models, including Unicoder-VL ref [9], bridged these domains by feeding visual and linguistic inputs into shared transformers. Subsequent works such as CLIP introduced contrastive dual-stream pretraining, while BEiT-3 ref [2] exemplified the convergence toward unified multimodal foundation models supporting vision, language, and cross-modal tasks within a single backbone.

### Taxonomy of Vision-Language Tasks

Vision-language pretraining supports a diverse spectrum of downstream tasks, which can be systematically categorized into several groups. Understanding tasks include visual question answering (VQA) ref [2], visual reasoning ref [28], and visual entailment ref [24]. Retrieval tasks encompass image-text and cross-modal retrieval ref [51]. Generation tasks cover image captioning ref [27] and dense video captioning ref [7]. Grounding tasks involve visual grounding and region-level alignment ref [13]. Dense prediction tasks include object detection and semantic segmentation ref [2]. Specialized tasks extend to vision-and-language navigation ref [32], sign language translation ref [47], chart comprehension ref [26], and scene text detection ref [53].

### Overview of Survey Structure and Scope

This survey is organized into seven principal sections. Following this introduction, Section 2 examines model architectures, encompassing visual encoding strategies, single-stream ref [9], dual-stream ref [12], encoder-decoder ref [15], and unified multiway transformer designs ref [2]. Section 3 surveys pretraining objectives, including contrastive ref [6], masked modeling ref [28], and self-distillation strategies ref [4]. Section 4 addresses data collection, curation, and augmentation ref [27]. Section 5 covers downstream adaptation methodologies ref [19]. Section 6 evaluates benchmarks and analytical frameworks ref [25], ref [40]. Section 7 identifies emerging directions ref [44], ref [11]. Coverage is restricted to transformer-based VLP models published through 2023.

## Model Architectures for Vision-Language Pretraining

### Visual Encoding Strategies

Visual encoding strategies in VLP models span three principal paradigms. Region-based approaches employ object detectors such as Faster R-CNN to extract semantically rich region-of-interest features, as adopted in Unicoder-VL ref [9] and LEMON ref [27], offering strong semantic grounding at the cost of computational overhead. Grid-based CNN features provide a more efficient alternative, enabling end-to-end training as demonstrated in KD-VLP ref [42] and ref [31]. Vision Transformers (ViT) have emerged as the dominant backbone, underpinning models such as BEiT-3 ref [2], CLIP-based architectures ref [6], and RegionCLIP ref [13], offering scalable patch-level representations that facilitate both global alignment and dense prediction tasks.

### Language Encoding Strategies

Language encoding in VLP models predominantly relies on BERT-based transformer encoders, GPT-style autoregressive decoders, and domain-adapted language models. BERT-derived encoders, as employed in Unicoder-VL ref [9] and FLAVA ref [10], leverage bidirectional attention for rich contextual representations. GPT-style decoders underpin generative architectures such as VL-BEiT ref [28] and Vid2Seq ref [7]. Domain-specific adaptation is critical in specialized settings; for instance, biomedical VLP benefits substantially from radiology-specific vocabulary and pretraining objectives ref [8]. The SNARE benchmark further demonstrates that utilizing large generative language models as backbones and emphasizing syntactic and semantic knowledge significantly improves multimodal alignment ref [22].

### Single-Stream (Fusion Encoder) Architectures

Single-stream architectures jointly encode visual and textual inputs through a shared transformer, enabling deep cross-modal interaction at every layer. Unicoder-VL ref [9] pioneered this paradigm by feeding both modalities into a unified multi-layer transformer with masked language modeling, masked object classification, and visual-linguistic matching objectives. FLAVA ref [10] extended this approach to simultaneously support unimodal and multimodal tasks within one foundational model. VL-BEiT ref [28] introduced generative pretraining via masked prediction across both modalities using a shared backbone. BEiT-3 ref [2] further advanced single-stream modeling through Multiway Transformers, enabling modality-specific encoding alongside deep fusion, achieving state-of-the-art performance across diverse vision-language benchmarks.

### Dual-Stream (Contrastive) Architectures

Dual-stream architectures encode visual and textual inputs through separate encoders, aligning representations via contrastive objectives. CLIP established this paradigm, enabling efficient inference through independent modality encoding. ALIGN and COOKIE extended contrastive pretraining to larger noisy datasets. COTS enhances two-stream models with token-level and task-level interactions via momentum contrastive learning, achieving inference efficiency orders of magnitude faster than single-stream alternatives ref [20]. Chinese CLIP adapts this framework for non-English settings through two-stage pretraining ref [12]. PyramidCLIP addresses semantic misalignment through hierarchical feature alignment ref [6], while HiCLIP incorporates hierarchy-aware attention for progressive semantic aggregation ref [14], collectively advancing alignment fidelity within the dual-stream paradigm.

### Encoder-Decoder Architectures for Generative Tasks

Encoder-decoder architectures address the dual requirements of vision-language understanding and generation within a unified framework. TDEN introduces a decoupled two-stream encoder-decoder design with scheduled sampling to bridge the train-test discrepancy inherent in masked pretraining ref [15]. MV-GPT employs bidirectional generation objectives, jointly training a multimodal video encoder and sentence decoder for video captioning ref [55]. Vid2Seq augments a language model with special time tokens for dense video captioning, pretrained on large-scale narrated videos ref [7]. InstructBLIP extends BLIP-2 with an instruction-aware Query Transformer, enabling general-purpose vision-language generation through systematic instruction tuning ref [11]. UniChart adopts a chart-grounded text decoder for structured chart comprehension and reasoning ref [26].

### Unified and Multiway Transformer Architectures

Unified and multiway transformer architectures represent a significant convergence in vision-language pretraining, enabling both modality-specific encoding and deep cross-modal fusion within a single backbone. BEiT-3 ref [2] exemplifies this paradigm through its Multiway Transformer design, where shared feed-forward layers coexist with modality-specific expert modules, supporting masked modeling across images, texts, and image-text pairs uniformly. Similarly, FLAVA ref [10] pursues a foundational model that simultaneously handles unimodal vision, unimodal language, and multimodal tasks through a unified architecture. VL-BEiT ref [28] further demonstrates that a single shared transformer with generative pretraining objectives can achieve strong performance across diverse vision-language benchmarks, underscoring the viability of architecturally unified approaches.

### Scaling Laws and Model Size Considerations

Empirical investigations into scaling laws reveal consistent performance gains as model size and pretraining data volume increase. LEMON ref [27] provides the first systematic study of VLP scaling for image captioning, varying transformer parameters from 13M to 675M and pretraining data up to 200M image-text pairs (ALT200M), demonstrating log-linear improvements with scale. BEiT-3 ref [2] further advances scaling through its Multiway Transformer backbone, achieving state-of-the-art results across diverse benchmarks by scaling both model capacity and multimodal pretraining data. Chinese CLIP ref [12] similarly demonstrates that scaling from 77M to 958M parameters yields substantial gains across retrieval and classification tasks, underscoring that both architectural efficiency and data quality critically mediate the benefits of increased scale.

## Pretraining Objectives and Learning Strategies

### Contrastive Learning Objectives

Contrastive learning objectives constitute a foundational pretraining strategy in VLP, wherein image-text pairs are aligned by maximizing agreement between matched pairs while repelling mismatched ones. CLIP established the paradigm of large-scale image-text contrastive (ITC) learning, subsequently extended by ALIGN, COOKIE ref [51], and COTS ref [20] through momentum contrastive learning and token-level interactions. PyramidCLIP ref [6] addresses semantic mismatch in web-crawled data via hierarchical peer-level and cross-level alignment with softened negative loss. HiCLIP ref [14] introduces hierarchy-aware attention for progressive semantic alignment. Chinese CLIP ref [12] adapts contrastive pretraining to non-English corpora, while domain-specific extensions address biomedical imaging ref [8] and medical contrastive triage ref [21].

### Masked Language

[... deterministic omitted span ...]

Augmentation

### Large-Scale Web-Scraped Image-Text Datasets

Large-scale web-scraped image-text datasets constitute the foundational data infrastructure for vision-language pretraining. Prominent corpora include Conceptual Captions, LAION-400M, YFCC, and WIT, which provide hundreds of millions of image-text pairs harvested from the internet. LEMON demonstrates pretraining on up to 200 million automatically collected ALT-text pairs (ALT200M), revealing systematic scaling behaviors ref [27]. However, web-scraped data inherently suffers from semantic misalignment and noise, as highlighted by PyramidCLIP ref [6], which addresses imperfect one-to-one correspondence. Furthermore, such corpora may embed harmful societal biases, including sexual objectification, as documented in contrastive models trained on web-scraped data ref [1].

### Domain-Specific and Specialized Datasets

Beyond general-purpose web-scraped corpora, VLP research has increasingly relied on domain-specific datasets tailored to specialized tasks. For vision-and-language navigation, the BnB dataset provides large-scale indoor image-caption and path-instruction pairs ref [32]. Biomedical VLP leverages radiology report corpora with domain-adapted language models ref [8], while semantics-aware triage addresses noise in medical image-text pairs ref [21]. Chart comprehension is supported by a large chart corpus underpinning UniChart ref [26]. Video-language pretraining utilizes YT-Temporal-1B for dense captioning ref [7] and VIDAL-10M for multi-modal alignment across video, depth, infrared, and audio ref [44]. Echocardiography segmentation employs synthetic diffusion-generated datasets ref [34], and continual pretraining is supported by the P9D product image-text benchmark ref [16].

### Handling Noisy and Misaligned Image-Text Pairs

Web-crawled image-text pairs frequently exhibit semantic misalignment, posing significant challenges for VLP. PyramidCLIP addresses this by constructing hierarchical input pyramids and softening negative sample losses to mitigate strict constraints imposed by compatible but unpaired samples ref [6]. In the medical domain, semantics-aware triage partitions image-report pairs into positive, negative, and neutral groups to prevent false negatives from corrupting learned representations ref [21]. Complementarily, the MCD approach reframes misalignments constructively by predicting continuous misalignment scales as auxiliary supervision, converting a training liability into a source of transferable knowledge ref [58]. Collectively, these strategies substantially improve data efficiency and representation quality.

### Pseudo-Labeling and Self-Training for Data Augmentation

Pseudo-labeling and self-training strategies address data scarcity by generating synthetic annotations from unlabeled sources. Self-training VL-BERTs employ a unified conditional model to produce pseudo captions on unlabeled images, iteratively refining student models without requiring additional paired data ref [41]. Vid2Seq reformulates speech transcription boundaries as pseudo event boundaries for dense video captioning pretraining ref [7]. GTGM leverages large language models to synthesize medical-style text descriptions from 3D images, circumventing the scarcity of paired medical image-text corpora ref [38]. Similarly, Synthetic Boost employs semantic diffusion models to generate synthetic echocardiography images, improving VLSM convergence and segmentation performance ref [34].

### Unsupervised and Weakly Supervised Data Construction

Constructing VLP training data without relying on manually annotated parallel image-text pairs represents a critical research direction. UVLP demonstrates that retrieval-based alignment can construct weakly paired corpora from non-parallel image and text datasets, applying multi-granular alignment tasks spanning region-to-tag, region-to-phrase, and image-to-sentence levels ref [35]. Complementarily, COSA converts existing image-text corpora into pseudo video-paragraph data by concatenating multiple image-text pairs sequentially, simulating temporal semantic correlations without requiring actual video data ref [17]. MCD further leverages inherent image-text misalignments in web-crawled data constructively, treating augmentation-induced misalignments as continuous supervision signals rather than noise ref [58].

### Multilingual and Cross-Lingual Pretraining Data

Multilingual and cross-lingual vision-language pretraining remains significantly underexplored relative to English-centric approaches. Chinese CLIP ref [12] represents a prominent effort in this direction, constructing a large-scale Chinese image-text dataset by retrieving pairs from publicly available sources and proposing a two-stage pretraining strategy—first training with a frozen image encoder, then jointly optimizing all parameters—to achieve state-of-the-art performance on Chinese benchmarks including MUGE, Flickr30K-CN, and COCO-CN. Key challenges include constructing high-quality aligned corpora across languages, handling cultural and linguistic divergence in web-scraped data, and adapting tokenization and vocabulary design for non-Latin scripts within unified multimodal transformer frameworks ref [2].

## Downstream Task Adaptation and Transfer Learning

### Full Fine-Tuning and Task-Specific Adaptation

Full fine-tuning remains the dominant paradigm for adapting pretrained VLP models to downstream tasks, wherein all model parameters are updated using task-specific labeled data. BEiT-3 demonstrates state-of-the-art fine-tuned performance across object detection, semantic segmentation, VQA, visual reasoning, image captioning, and cross-modal retrieval ref [2]. Similarly, VL-BEiT achieves strong results on VQA, visual reasoning, and image-text retrieval through unified generative pretraining followed by task-specific fine-tuning ref [28]. ROSITA significantly outperforms existing methods on VQA and retrieval benchmarks after fine-tuning ref [18], while MAP achieves state-of-the-art results on retrieval, VQA, visual reasoning, and visual entailment ref [24], collectively demonstrating the broad effectiveness of full fine-tuning across diverse vision-language benchmarks.

### Prompt Tuning and Context Optimization

Prompt tuning has emerged as a parameter-efficient paradigm for adapting frozen vision-language pretrained models to downstream tasks using minimal labeled data. Context Optimization (CoOp) introduced learnable soft prompt vectors to replace hand-crafted text templates, enabling task-specific adaptation without full fine-tuning. Building upon this, CLIP-Adapter ref [19] demonstrated that feature-level adaptation via lightweight bottleneck modules offers a complementary alternative to textual prompt tuning. GRAM ref [36] further addressed generalization limitations by meta-learning prompt initializations and gradient-regulating functions from unlabeled pretraining data. PTP ref [56] introduced prototype-based prompting, where image and prompt prototypes jointly guide few-shot recognition, improving adaptability across diverse visual categories.

### Adapter-Based and Parameter-Efficient Fine-Tuning

Adapter-based methods offer a compelling alternative to full fine-tuning by inserting lightweight trainable modules into frozen pretrained VLP models. CLIP-Adapter ref [19] introduces bottleneck feature adapters on either the visual or language branch, performing residual-style feature blending with original pretrained representations, thereby achieving superior performance over prompt tuning while preserving model simplicity. SVL-Adapter ref [60] extends this paradigm by combining vision-language pretraining with self-supervised representation learning, yielding approximately 10% accuracy improvement in low-shot settings on challenging out-of-distribution classification tasks. These parameter-efficient strategies enable effective adaptation of large-scale VLP models under limited supervision and computational constraints.

### Zero-Shot and Few-Shot Transfer

Zero-shot and few-shot transfer represent defining capabilities of contrastive VLP models. CLIP-style dual-stream architectures enable zero-shot classification by matching image embeddings against textual class descriptors, while PyramidCLIP substantially improves zero-shot accuracy through hierarchical alignment, surpassing CLIP on ImageNet with significantly less data ref [6]. RegionCLIP extends zero-shot capabilities to object detection via region-text alignment ref [13], and SILC further advances zero-shot segmentation and classification through self-distillation ref [30]. For few-shot adaptation, methods such as CLIP-Adapter ref [19], GRAM ref [36], and PTP ref [56] efficiently leverage pretrained representations with minimal labeled examples, while Vid2Seq demonstrates strong few-shot generalization in dense video captioning ref [7].

### Dense Prediction Tasks: Detection and Segmentation

Adapting VLP models for dense prediction tasks such as object detection and semantic segmentation requires region-level understanding beyond global image-text alignment. RegionCLIP extends CLIP by aligning image regions with textual concepts, achieving significant gains in open-vocabulary object detection on COCO and LVIS ref [13]. SCLIP introduces a Correlative Self-Attention mechanism enabling training-free zero-shot semantic segmentation ref [49], while SILC augments contrastive pretraining with local-to-global self-distillation to improve dense prediction ref [30]. BEiT-3 achieves state-of-the-art segmentation on ADE20K through unified masked modeling ref [2]. In the medical domain, GTGM extends VLP to 3D segmentation ref [38], and synthetic data augmentation further enhances echocardiography segmentation ref [34].

### Video and Temporal Understanding

Video-and-language pretraining has emerged as a critical extension of image-text VLP, requiring models to capture temporal semantics alongside visual-linguistic alignment. VindLU ref [37] provides a systematic empirical study identifying temporal modeling, video-to-text fusion, and masked objectives as key design factors, achieving state-of-the-art video-text retrieval and video question answering. Vid2Seq ref [7] introduces dense video captioning pretraining on narrated videos by augmenting a language model with temporal tokens. COSA ref [17] simulates video-paragraph corpora from image-text pairs via concatenated sample pretraining, while MV-GPT ref [55] employs bidirectional generation objectives for multimodal video captioning, collectively advancing temporal understanding across diverse benchmarks.

### Specialized Domain Applications

Vision-language pretraining has been successfully extended to numerous specialized domains beyond general-purpose tasks. In biomedical imaging, domain-adapted contrastive pretraining leverages radiology reports for improved segmentation and classification ref [8], while GTGM extends VLP to 3D medical images using LLM-generated synthetic text ref [38], and semantics-aware triage

[... deterministic omitted span ...]

through contrastive learning, freezing the language encoder while training modality-specific encoders. This language-centric alignment strategy enables indirect cross-modal correspondence without requiring exhaustive pairwise training data. The accompanying VIDAL-10M dataset provides aligned video, infrared, depth, and audio pairs centered on textual descriptions. Such N-modal frameworks present unique challenges including modality-specific encoder design, scalable dataset construction, and maintaining alignment quality as the number of modalities increases.

### Instruction Tuning and General-Purpose Vision-Language Models

Instruction tuning has emerged as a pivotal paradigm for developing general-purpose vision-language models capable of flexible, task-agnostic reasoning. InstructBLIP ref [11] systematically investigates vision-language instruction tuning by transforming 26 diverse datasets into instruction-following format and introducing an instruction-aware Query Transformer that extracts features conditioned on the given instruction. Trained on held-in datasets, InstructBLIP achieves state-of-the-art zero-shot performance across held-out benchmarks, substantially outperforming BLIP-2 and Flamingo. This approach demonstrates that combining large-scale pretraining with structured instruction tuning enables models to generalize across heterogeneous visual and linguistic input distributions, representing a critical step toward truly general-purpose multimodal foundation models ref [2].

### 3D and Spatiotemporal Vision-Language Pretraining

Extending VLP beyond 2D images to 3D and spatiotemporal domains presents unique challenges, including volumetric data complexity and the scarcity of paired text annotations. GTGM addresses 3D medical VLP by leveraging large language models to synthesize medical-style text from CT, MRI, and electron microscopy volumes, enabling annotation-free pretraining across 13 datasets ref [38]. For temporal understanding, Vid2Seq augments language models with time tokens for dense video captioning using narrated video pretraining ref [7], while COSA simulates long-form video-paragraph corpora by concatenating image-text pairs to capture temporal semantics ref [17]. VindLU further systematically identifies temporal modeling and masked objectives as critical factors for video-language pretraining ref [37].

### Compositional Reasoning and Fine-Grained Alignment

Compositional reasoning and fine-grained alignment remain critical open challenges in VLP. The CREPE benchmark demonstrates that current models struggle with systematicity and productivity, with retrieval performance degrading substantially as compositional complexity increases ref [40]. SNARE further reveals that VLP models exhibit insensitivity to complex syntactic structures, negation, and ternary relational reasoning ref [22]. VL-CheckList provides complementary diagnostics by decomposing alignment capabilities into objects, attributes, and relations ref [25]. Addressing these gaps, PyramidCLIP introduces hierarchical peer-level and cross-level alignment ref [6], while SIMLA proposes patch-token and semantic-level alignment through cross-modal reconstruction ref [57], and ROSITA employs structural knowledge masking via scene graphs ref [18].

### Responsible AI: Bias Mitigation and Ethical Considerations

Responsible AI considerations in VLP encompass bias detection, fairness evaluation, and ethical deployment of large-scale multimodal models. Research has demonstrated that contrastive vision-language models pretrained on web-scraped data exhibit sexual objectification bias, reflecting harmful societal stereotypes encoded in uncurated corpora ref [1]. Studies on gender bias amplification reveal that pretraining and fine-tuning stages independently propagate group disparities, while continued pretraining on gender-neutral data reduces such disparities without significantly compromising task performance ref [43]. These findings underscore the necessity of proactive bias auditing, dataset curation, and fairness-aware training strategies to ensure equitable outcomes across diverse demographic groups in deployed VLP systems.

### Efficient and Sustainable Pretraining

The escalating computational demands of VLP necessitate research into efficient and sustainable pretraining paradigms. Free Language Modeling (FLM) decouples prediction and corruption rates in masked language modeling, achieving a 2.5× pretraining time reduction while maintaining competitive performance ref [23]. ViCHA proposes hierarchical cross-modal alignment with masked image modeling and visual concepts derived from foundation models, outperforming approaches trained on four times more data ref [39]. PyramidCLIP improves data efficiency by surpassing CLIP trained on 400M pairs using only 143M samples ref [6]. Parameter-efficient adaptation methods such as CLIP-Adapter ref [19] and continual pretraining via CTP ref [16] further reduce redundant retraining costs.

### Towards Unified Multimodal Foundation Models

A prominent trend in vision-language pretraining is the convergence toward unified multimodal foundation models capable of handling vision, language, and multimodal tasks within a single architecture. BEiT-3 ref [2] exemplifies this direction through Multiway Transformers that enable both modality-specific encoding and deep cross-modal fusion under a unified masked modeling objective. FLAVA ref [10] similarly targets all modalities simultaneously, demonstrating strong performance across 35 diverse tasks. OneR ref [48] further explores modality-agnostic representation learning within a single-tower framework. Despite these advances, challenges remain in achieving compositional reasoning ref [40], fine-grained alignment, and scalable generalization across heterogeneous tasks and domains.

## References

[1] Contrastive Language-Vision AI Models Pretrained on Web-Scraped Multimodal Data Exhibit Sexual Objectification Bias. https://doi.org/10.1145/3593013.3594072

[2] Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks. https://doi.org/10.48550/arXiv.2208.10442

[3] Image as a Foreign Language: BEIT Pretraining for Vision and Vision-Language Tasks. https://doi.org/10.1109/CVPR52729.2023.01838

[4] MaskCLIP: Masked Self-Distillation Advances Contrastive Language-Image Pretraining. https://doi.org/10.1109/CVPR52729.2023.01058

[5] Robust Navigation with Language Pretraining and Stochastic Sampling. https://doi.org/10.18653/v1/D19-1159

[6] PyramidCLIP: Hierarchical Feature Alignment for Vision-language Model Pretraining. https://doi.org/10.48550/arXiv.2204.14095

[7] Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning. https://doi.org/10.1109/CVPR52729.2023.01032

[8] Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing. https://doi.org/10.1007/978-3-031-20059-5_1

[9] Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training. https://doi.org/10.1609/AAAI.V34I07.6795

[10] FLAVA: A Foundational Language And Vision Alignment Model. https://doi.org/10.1109/CVPR52688.2022.01519

[11] InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning. https://doi.org/10.48550/arXiv.2305.06500

[12] Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese. https://doi.org/10.48550/arXiv.2211.01335

[13] RegionCLIP: Region-based Language-Image Pretraining. https://doi.org/10.1109/CVPR52688.2022.01629

[14] HiCLIP: Contrastive Language-Image Pretraining with Hierarchy-aware Attention. https://doi.org/10.48550/arXiv.2303.02995

[15] Scheduled Sampling in Vision-Language Pretraining with Decoupled Encoder-Decoder Network. https://doi.org/10.1609/aaai.v35i10.17034

[16] CTP: Towards Vision-Language Continual Pretraining via Compatible Momentum Contrast and Topology Preservation. https://doi.org/10.1109/ICCV51070.2023.02034

[17] COSA: Concatenated Sample Pretrained Vision-Language Foundation Model. https://doi.org/10.48550/arXiv.2306.09085

[18] ROSITA: Enhancing Vision-and-Language Semantic Alignments via Cross- and Intra-modal Knowledge Integration. https://doi.org/10.1145/3474085.3475251

[19] CLIP-Adapter: Better Vision-Language Models with Feature Adapters. https://doi.org/10.1007/s11263-023-01891-x

[20] COTS: Collaborative Two-Stream Vision-Language Pre-Training Model for Cross-Modal Retrieval. https://doi.org/10.1109/CVPR52688.2022.01524

[21] Improving Medical Vision-Language Contrastive Pretraining With Semantics-Aware Triage. https://doi.org/10.1109/TMI.2023.3294980

[22] Can Linguistic Knowledge Improve Multimodal Alignment in Vision-Language Pretraining?. https://doi.org/10.1145/3690640

[23] Accelerating Vision-Language Pretraining with Free Language Modeling. https://doi.org/10.1109/CVPR52729.2023.02218

[24] MAP: Multimodal Uncertainty-Aware Vision-Language Pre-training Model. https://doi.org/10.1109/CVPR52729.2023.02228

[25] VL-CheckList: Evaluating Pre-trained Vision-Language Models with Objects, Attributes and Relations. https://doi.org/10.48550/arXiv.2207.00221

[26] UniChart: A Universal Vision-language Pretrained Model for Chart Comprehension and Reasoning. https://doi.org/10.48550/arXiv.2305.14761

[27] Scaling Up Vision-Language Pretraining for Image Captioning. https://doi.org/10.1109/CVPR52688.2022.01745

[28] VL-BEiT: Generative Vision-Language Pretraining. https://doi.org/10.48550/arXiv.2206.01127

[29] Vision-and-Language Pretraining. https://doi.org/10.48550/arXiv.2207.01772

[30] SILC: Improving Vision Language Pretraining with Self-Distillation. https://doi.org/10.48550/arXiv.2310.13355

[31] Effective End-to-End Vision Language Pretraining With Semantic Visual Loss. https://doi.org/10.1109/TMM.2023.3237166

[32] Airbert: In-domain Pretraining for Vision-and-Language Navigation. https://doi.org/10.1109/ICCV48922.2021.00166

[33] HOP: History-and-Order Aware Pretraining for Vision-and-Language Navigation. https://doi.org/10.1109/CVPR52688.2022.01498

[34] Synthetic Boost: Leveraging Synthetic Data for Enhanced Vision-Language Segmentation in Echocardiography. https://doi.org/10.1007/978-3-031-44521-7_9

[35] Unsupervised Vision-and-Language Pretraining via Retrieval-based Multi-Granular Alignment. https://doi.org/10.1109/CVPR52688.2022.01599

[36] Gradient-Regulated Meta-Prompt Learning for Generalizable Vision-Language Models. https://doi.org/10.1109/ICCV51070.2023.00241

[37] VindLU: A Recipe for Effective Video-and-Language Pretraining. https://doi.org/10.1109/CVPR52729.2023.01034

[38] Generative Text-Guided 3D Vision-Language Pretraining for Unified Medical Image Segmentation. https://doi.org/10.48550/arXiv.2306.04811

[39] Efficient Vision-Language Pretraining with Visual Concepts and Hierarchical Alignment. https://doi.org/10.48550/arXiv.2208.13628

[40] @ CREPE: Can Vision-Language Foundation Models Reason Compositionally?. https://doi.org/10.1109/CVPR52729.2023.01050

[41] Self-Training Vision Language BERTs With a Unified Conditional Model. https://doi.org/10.1109/TCSVT.2023.3235704

[42] KD-VLP: Improving End-to-End Vision-and-Language Pretraining with Object Knowledge Distillation. https://doi.org/10.18653/v1/2022.findings-naacl.119

[43] Evaluating Bias and Fairness in Gender-Neutral Pretrained Vision-and-Language Models. https://doi.org/10.48550/arXiv.2310.17530

[44] LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment. https://doi.org/10.48550/arXiv.2310.01852

[45] MILAN: Masked Image Pretraining on Language Assisted Representation. https://doi.org/10.48550/arXiv.2208.06049

[46] Vision Language Transformers: A Survey. https://doi.org/10.48550/arXiv.2307.03254

[47] Gloss-free Sign Language Translation: Improving from Visual-Language Pretraining. https://doi.org/10.1109/ICCV51070.2023.01908

[48] Unifying Vision-Language Representation Space with Single-tower Transformer. https://doi.org/10.48550/arXiv.2211.11153

[49] SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference. https://doi.org/10.48550/arXiv.2312.01597

[50] Multimodal Pretraining Unmasked: A Meta-Analysis and a Unified Framework of Vision-and-Language BERTs. https://doi.org/10.1162/tacl_a_00408

[51] COOKIE: Contrastive Cross-Modal Knowledge Sharing Pre-training for Vision-Language Representation. https://doi.org/10.1109/ICCV48922.2021.00221

[52] Multimodal Adaptive Distillation for Leveraging Unimodal Encoders for Vision-Language Tasks. https://doi.org/10.48550/arXiv.2204.10496

[53] Vision-Language Pre-Training for Boosting Scene Text Detectors. https://doi.org/10.1109/CVPR52688.2022.01523

[54] CLIPSonic: Text-to-Audio Synthesis with Unlabeled Videos and Pretrained Language-Vision Models. https://doi.org/10.1109/WASPAA58266.2023.10248160

[55] End-to-end Generative Pretraining for Multimodal Video Captioning. https://doi.org/10.1109/CVPR52688.2022.01743

[56] Prompting through Prototype: A Prototype-based Prompt Learning on Pretrained Vision-Language Models. https://doi.org/10.48550/arXiv.2210.10841

[57] Single-Stream Multi-Level Alignment for Vision-Language Pretraining. https://doi.org/10.48550/arXiv.2203.14395

[58] Misalign, Contrast then Distill: Rethinking Misalignments in Language-Image Pretraining. https://doi.org/10.1109/ICCV51070.2023.00242

[59] Vision-and-Language Pretrained Models: A Survey. https://doi.org/10.48550/arXiv.2204.07356

[60] SVL-Adapter: Self-Supervised Adapter for Vision-Language Pretrained Models. https://doi.org/10.48550/arXiv.2210.03794
