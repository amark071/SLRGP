# Registered common-window excerpt

# Vision-Language Pretraining: Architectures, Objectives, Data Strategies, Applications, and Future Directions

## 1 Introduction and Background

### 1.1 Motivation and Significance of Vision-Language Pretraining

The rapid proliferation of multimodal content on the internet has created an unprecedented opportunity for learning joint visual and linguistic representations. Billions of image-text pairs, ranging from social media posts to news articles and e-commerce listings, are now freely available at web scale, providing a rich and diverse source of supervision for training multimodal models without the need for costly manual annotation [1]. This abundance of weakly aligned data has served as a primary catalyst for the emergence and growth of Vision-Language Pretraining (VLP) as a dominant paradigm in multimodal artificial intelligence.

Prior to the VLP era, the prevailing approach to vision-language tasks involved designing and training separate, task-specific models tailored to individual problems such as visual question answering, image captioning, or image-text retrieval [2]. These task-specific models required expensive, curated datasets with precise human annotations, limiting their scalability and generalizability. Each new task necessitated building a new data collection pipeline, designing custom architectures, and training from scratch—a process that was not only resource-intensive but also failed to leverage the shared knowledge across related vision-language tasks [3]. Furthermore, the performance of these models was inherently bounded by the quantity and quality of available task-specific labeled data, making it difficult to achieve robust generalization to unseen scenarios.

The fundamental motivation behind VLP is to overcome these limitations by learning universal, transferable multimodal representations from large-scale, weakly supervised data [4]. By pretraining a single model on diverse image-text corpora, VLP methods can capture rich cross-modal semantic correspondences that generalize across a broad spectrum of downstream tasks. Once pretrained, these models can be fine-tuned with relatively little task-specific data, dramatically reducing the annotation burden while achieving strong performance [5]. This transfer learning paradigm mirrors the success of large-scale language pretraining in NLP, where models such as BERT and GPT demonstrated that representations learned from massive text corpora could be effectively adapted to diverse downstream applications—a conceptual foundation that, as discussed in the following section, directly inspired the trajectory of vision-language research.

The significance of VLP extends across a wide range of practically important tasks. In visual question answering (VQA), pretrained multimodal models can leverage both visual understanding and linguistic reasoning to answer complex questions about image content [2]. In image captioning, VLP enables models to generate fluent and semantically accurate descriptions by learning fine-grained visual-textual correspondences [6]. For image-text retrieval, pretrained dual-encoder architectures can efficiently index and query large-scale multimodal databases by aligning visual and textual representations in a shared embedding space [1]. Beyond these core tasks, VLP has shown promise in visual grounding, visual reasoning, scene text understanding [7], and even domain-specific applications such as medical image analysis [8].

In summary, VLP represents a paradigm shift from isolated, task-specific learning to unified, data-driven pretraining of multimodal representations. By harnessing the scale of web-crawled image-text data and reducing dependence on expensive annotations, VLP has democratized access to powerful vision-language capabilities and established a new state of the art across a diverse array of downstream tasks [9]. The growing significance of VLP motivates a systematic and comprehensive examination of the methods, architectures, objectives, and challenges that define this rapidly evolving field—an examination that begins with a closer look at the historical developments that set the stage for the modern VLP paradigm.

### 1.2 Historical Development: From Task-Specific Models to Pretrained Multimodal Models

The development of vision-language research has undergone a remarkable transformation over the past two decades, evolving from narrowly scoped task-specific systems to broadly capable pretrained multimodal models. Understanding this trajectory is essential for appreciating the current state of the field and the motivations behind modern vision-language pretraining (VLP) approaches—motivations that, as outlined in the preceding section, are fundamentally rooted in the limitations of earlier paradigms and the opportunities afforded by web-scale data.

**Early Task-Specific Methods**

In the early era of vision-language research, the dominant paradigm involved designing and training separate, specialized models for each individual task. Methods for visual question answering, image captioning, visual grounding, and image-text retrieval were developed largely in isolation, each with its own bespoke architecture, training pipeline, and dataset. These task-specific models were effective within their target domains but suffered from a fundamental limitation: knowledge and representations learned for one task could not easily be transferred to another. Every new task required collecting task-specific labeled data, designing a new model architecture, and training from scratch—a costly and inefficient process that severely limited scalability and generalization [10].

**The Deep Learning Revolution in Computer Vision and NLP**

The landscape began shifting with the advent of deep learning, which brought about powerful feature representations that could be learned automatically from large amounts of data. In computer vision, the introduction of deep convolutional neural networks led to transformative progress on tasks such as image classification and object detection, and the paradigm of pretraining on large datasets like ImageNet followed by fine-tuning on downstream tasks became standard practice. Concurrently, in natural language processing (NLP), the transformer architecture and large-scale language pretraining models such as BERT demonstrated the tremendous value of learning general-purpose representations from massive text corpora and subsequently adapting them to a wide variety of downstream language tasks [2]. These dual revolutions—whose technical underpinnings are examined in greater depth in the following section—established the conceptual and technical foundations upon which vision-language pretraining would be built.

**The Emergence of Pretrained Vision-Language Models**

Inspired by the success of transfer learning in unimodal domains, researchers began exploring how to jointly pretrain models on large collections of image-text pairs in order to learn universal multimodal representations. Rather than training task-specific models from scratch, the new paradigm involved a two-stage process: first, pretraining on large-scale, often weakly labeled or web-crawled image-text data to learn rich cross-modal representations, and then fine-tuning these representations on specific downstream tasks [11]. Early VLP models demonstrated that pretraining on diverse image-caption data could lead to strong performance improvements across multiple tasks simultaneously, effectively replacing the fragmented ecosystem of task-specific models with a more unified and reusable framework [12].

This shift represented a fundamental reconceptualization of vision-language research. Instead of treating each task as an isolated problem, the pretraining paradigm viewed diverse VL tasks as different manifestations of underlying shared skills—such as understanding object attributes, spatial relationships, and semantic correspondences between visual and textual content. Multi-task training further reinforced this view by demonstrating that jointly learning across multiple VL tasks with a single model could improve performance compared to task-specific baselines [3].

**Scaling and Consolidation**

As the field matured, VLP models grew rapidly in scale and capability. The development of large transformer-based architectures enabled richer and deeper cross-modal interactions, while the availability of increasingly large web-crawled image-text datasets provided the data necessary to train such models effectively. Critically, the introduction of unified transformer architectures—drawing directly on the architectural convergence between NLP and computer vision discussed in the following section—allowed a single pretrained model to serve as a versatile backbone for a broad range of downstream tasks including visual question answering, image captioning, image-text retrieval, and visual grounding, often achieving state-of-the-art results across all of them simultaneously [13]. This convergence marked a definitive transition from the fragmented task-specific era to the era of large-scale pretrained multimodal foundation models [14].

In summary, the historical evolution of vision-language research reflects a broader trend in machine learning toward general-purpose, scalable pretraining paradigms. The movement from task-specific models to pretrained multimodal models was driven by the practical limitations of task-specific training, inspired by the success of transfer learning in NLP and computer vision, and enabled by the growing availability of large-scale paired image-text data and powerful transformer architectures [15]. It is precisely these two unimodal pretraining traditions—and the technical insights they contributed—that the following section examines in detail, tracing how advances in NLP and computer vision converged to shape the methodological foundations of modern VLP.

### 1.3 Roots in Transfer Learning: NLP and Computer Vision Foundations

The success of vision-language pretraining (VLP) is deeply rooted in the

[... deterministic omitted span ...]

as pretraining supervision. Rather than relying solely on original alt-text, these approaches leverage the generative capabilities of multimodal large language models (MLLMs) to produce comprehensive descriptions that articulate fine-grained attributes, object relationships, spatial arrangements, and contextual scene details. Such re-captioning strategies can substantially enrich the textual side of image-text pairs, providing pretraining models with denser and more informative supervision signals—effectively complementing the noise-filtering and tag-augmentation strategies that form part of the broader data quality toolkit in VLP research.

One prominent example of this direction is DreamLIP, which employs multimodal large language models to re-caption images with lengthy, descriptive text. A key design insight in DreamLIP is the use of sub-caption sampling strategies, which decompose lengthy descriptions into multiple shorter sub-captions. These sub-captions are then used to construct multiple positive image-text pairs during pretraining, encouraging the model to learn diverse and complementary aspects of the visual content from a single image. This multi-positive training scheme effectively increases the granularity of supervision without requiring additional annotated data.

The benefits of richer textual supervision are multifaceted. First, detailed captions explicitly describe visual attributes, object identities, and inter-object relationships that are often absent from short alt-text, enabling more fine-grained alignment between visual regions and linguistic concepts. Second, longer descriptions reduce the semantic ambiguity inherent in brief captions, helping models learn more precise correspondences. Third, by covering diverse visual aspects in a single training example, rich captions improve the model's ability to generalize to tasks requiring nuanced visual understanding, such as visual question answering, compositional reasoning, and attribute-level retrieval. These advantages are particularly consequential at scale, where the compounding effect of richer supervision across billions of training examples can yield substantial improvements over models trained on equivalently large but textually impoverished corpora.

This paradigm aligns with the broader trend observed across VLP research, where the quality and richness of textual supervision—rather than sheer data scale alone—plays a decisive role in determining downstream performance [2]. As multimodal foundation models become increasingly capable of generating high-fidelity image descriptions [14], re-captioning pipelines offer a scalable and cost-effective means of upgrading existing noisy datasets into richer pretraining corpora. This strategy directly informs the scalability considerations examined in the following section, where the interplay between data volume, annotation quality, and training efficiency emerges as a central theme in pushing the frontiers of large-scale vision-language pretraining.

### 4.8 Scalability of Pretraining Data and Models

## 4.8 Scalability and Large-Scale Training Strategies

Scalability stands as one of the most fundamental and consequential dimensions of vision-language pretraining research. Building directly on the data quality considerations raised in the preceding discussion—where richer, re-captioned supervision was shown to unlock finer-grained cross-modal alignment—the present section examines how scale itself interacts with data quality, model architecture, and training efficiency to determine downstream performance. The empirical relationship between pretraining data scale, model size, and downstream task performance has been a central focus of the VLP community, motivating ever-larger training runs and increasingly sophisticated strategies for harvesting and organizing web-scale multimodal corpora.

**Scaling Laws in VLP Contexts**

The vision-language pretraining paradigm has broadly inherited scaling intuitions from large language models, where performance tends to improve predictably as a function of data volume, model parameter count, and compute budget. In VLP settings, scaling manifests somewhat differently due to the added complexity of cross-modal alignment. Increasing data scale generally improves downstream performance on tasks such as visual question answering, image-text retrieval, and image captioning, but the nature of the data—its alignment quality, conceptual diversity, and label richness—matters enormously alongside raw volume. Models trained on hundreds of millions to billions of image-text pairs have consistently demonstrated stronger zero-shot generalization compared to those trained on smaller curated datasets, suggesting that breadth of coverage compensates partially for imperfect alignment. Crucially, this observation reinforces the insight from the previous section: when textual supervision is enriched through re-captioning, scaling the resulting corpus yields compounding benefits that go well beyond what equivalent volumes of noisy alt-text can deliver.

Unified transformer architectures designed for scalability have played a critical role in enabling this trajectory. Approaches such as [67] demonstrate how stagewise pretraining strategies—first leveraging large-scale image-only and text-only corpora before incorporating image-text pairs—allow models to benefit from the vastly larger unimodal data pools that exist on the web. This strategy effectively decouples the data requirements for unimodal and cross-modal learning, enabling scale without proportionally increasing the need for expensive aligned pairs. Similarly, [13] shows that treating image and text tokens under a unified masked modeling framework across monomodal and multimodal data at massive scale leads to strong generalization across both vision and vision-language benchmarks, illustrating that architectural unification can be a scalability enabler.

**Benefits and Diminishing Returns of Scaling Web Data**

While scaling web-crawled data generally yields performance gains, the benefits are not uniformly distributed across tasks or model sizes, and diminishing returns become apparent at extreme scales. Contrastive models such as those following the CLIP paradigm benefit enormously from scale due to the implicit denoising effect of large in-batch negative sampling, but gains on fine-grained tasks such as compositional reasoning and spatial understanding plateau or stagnate even as data volume grows. This is partly because web-crawled alt-text descriptions are short, incomplete, and often fail to capture the fine-grained visual semantics needed for compositional understanding tasks—precisely the quality bottleneck that re-captioning strategies aim to address. The quality and diversity of the text associated with images—not merely the count of pairs—determines how effectively scaling translates to downstream improvement.

Efficient end-to-end architectures such as [82] address scalability from a computational efficiency angle, demonstrating that a unified pretraining task—masked signal modeling across image pixels and text tokens—can accelerate training by approximately 3.5× compared to models that combine image-text contrastive and matching losses, while maintaining competitive downstream performance. This efficiency gain is critical at scale, where training costs otherwise become prohibitive.

**Weakly Supervised Annotation Derivation from Heterogeneous Text Signals**

A key strategy for enabling large-scale training without proportional annotation costs is the derivation of weakly supervised labels from heterogeneous textual signals co-occurring with images on the web. Rather than relying solely on curated captions or expensive human annotations, models can leverage alt-text, surrounding document text, metadata, hashtags, and other naturally occurring textual signals as approximate supervision. This paradigm allows the training corpus to scale to billions of examples while keeping annotation costs near zero—representing a complementary pole to the re-captioning approach, which invests in per-example quality rather than aggregate volume.

Models like [64] demonstrate end-to-end pretraining on large-scale image-text pairs collected from diverse web sources, achieving strong performance across understanding and generation tasks. The diversity of textual signals—even when individually noisy—provides a rich supervision landscape that encourages the model to learn broad visual-semantic associations. However, the heterogeneity of these signals also introduces inconsistency, requiring careful architectural design or training strategies to prevent the model from overfitting to superficial correlations.

**Clustering Techniques for Long-Tailed Label Distributions**

Web-crawled image-text corpora exhibit pronounced long-tailed distributions, where a small fraction of concepts, objects, and scenes are overwhelmingly frequent while the vast majority appear rarely. This imbalance poses significant challenges for learning robust representations across the full conceptual spectrum, as gradient updates are dominated by frequent concepts and rare but important visual categories receive insufficient training signal.

Clustering-based approaches offer a principled mechanism for addressing long-tail challenges at scale. By grouping semantically similar examples together and ensuring balanced sampling during training, clustering techniques mitigate the overrepresentation of common concepts and improve coverage of rare categories. This connects to tag augmentation strategies—such as those using online multi-label recognition to derive richer semantic supervision—which help make the training signal more informative for diverse and less common visual concepts. Unified architectures that handle diverse concept vocabularies, as seen in the broad multi-task pretraining of models like [83], benefit from such balancing strategies by ensuring that the model is exposed to sufficient examples of each concept type during optimization.

**Efficient Large-Scale Training Infrastructure**

Beyond algorithmic strategies, efficient large-scale VLP training requires careful engineering of data pipelines, training objectives, and model architectures. Encoder designs that can process visual and textual inputs with reduced computational overhead—such

[... deterministic omitted span ...]

remains underrepresented in multilingual and cross-cultural settings, as well as in modalities such as 3D point clouds, infrared, depth, and physiological signals. Closing these gaps is directly connected to the evaluation standardization challenge raised previously: without culturally diverse and geographically representative benchmarks, it is impossible to objectively measure progress in multilingual and cross-modal generalization. Future directions include extending multilingual cross-modal pretraining frameworks [105] to cover low-resource and typologically diverse languages, leveraging shared visual grounding as a universal semantic pivot. Simultaneously, expanding modality coverage through multi-modal binding strategies that treat language as a semantic anchor for aligning diverse sensory inputs will broaden the applicability of VLP to domains such as robotics, healthcare, and autonomous systems.

**Safer, Fairer, and More Interpretable Systems.** As VLP models are increasingly deployed in high-stakes real-world applications—including healthcare, hiring, and content moderation—the imperative for safety, fairness, and interpretability becomes paramount. The evaluation limitations discussed earlier, particularly the lack of culturally diverse benchmarks and out-of-distribution evaluation, directly impede our ability to detect and address such biases. Future research must develop principled methods for auditing and mitigating social biases absorbed from web-scale training corpora, building on emerging debiasing and counterfactual evaluation frameworks. Beyond bias, adversarial robustness remains a critical concern, as multimodal models inheriting vulnerabilities from pretrained visual encoders can be exploited through carefully crafted input perturbations. Developing training strategies and architectural safeguards that improve robustness without sacrificing generalization will be essential. Finally, interpretability research that reveals which visual and textual features drive model predictions—especially in safety-critical domains—will be indispensable for building trustworthy vision-language systems that align with human values and societal norms.

**Continual and Lifelong Multimodal Learning.** Real-world deployment requires models that can continuously assimilate new knowledge from non-stationary data streams without catastrophic forgetting of previously acquired capabilities. Continual VLP frameworks that combine compatibility-preserving momentum contrast with topology preservation offer early solutions to this challenge, but significant research is needed to handle heterogeneous, domain-shifted, and privacy-sensitive data streams at scale. Integrating federated learning paradigms with efficient continual pretraining strategies will enable privacy-preserving collaborative training across distributed data sources, an increasingly important consideration as regulatory requirements around data governance grow more stringent.

In summary, the future of vision-language pretraining lies at the intersection of scalability, efficiency, safety, and inclusivity. Progress along these directions will not only address the evaluation and benchmarking gaps identified throughout this survey, but will also advance benchmark performance and broaden the societal impact and trustworthiness of multimodal AI systems.


## References

[1] Scaling Up Visual and Vision-Language Representation Learning With Noisy  Text Supervision

[2] VLP  A Survey on Vision-Language Pre-training

[3] 12-in-1  Multi-Task Vision and Language Representation Learning

[4] SimVLM  Simple Visual Language Model Pretraining with Weak Supervision

[5] BLIP  Bootstrapping Language-Image Pre-training for Unified  Vision-Language Understanding and Generation

[6] Unified Vision-Language Pre-Training for Image Captioning and VQA

[7] Towards Models that Can See and Read

[8] Multi-modal Pre-training for Medical Vision-language Understanding and  Generation  An Empirical Study with A New Benchmark

[9] Vision-Language Pre-training  Basics, Recent Advances, and Future Trends

[10] Vision-Language Intelligence  Tasks, Representation Learning, and Large  Models

[11] A Survey of Vision-Language Pre-Trained Models

[12] Vision-and-Language Pretraining

[13] Image as a Foreign Language  BEiT Pretraining for All Vision and  Vision-Language Tasks

[14] Multimodal Foundation Models  From Specialists to General-Purpose  Assistants

[15] Vision Language Transformers  A Survey

[16] Anatomy of Neural Language Models

[17] Evolution of transfer learning in natural language processing

[18] The Lottery Ticket Hypothesis for Pre-trained BERT Networks

[19] DistilBERT, a distilled version of BERT  smaller, faster, cheaper and  lighter

[20] Robust Transfer Learning with Pretrained Language Models through  Adapters

[21] ViLT  Vision-and-Language Transformer Without Convolution or Region  Supervision

[22] Foundation Transformers

[23] ViLBERT  Pretraining Task-Agnostic Visiolinguistic Representations for  Vision-and-Language Tasks

[24] Revisiting Multimodal Representation in Contrastive Learning  From Patch  and Token Embeddings to Finite Discrete Tokens

[25] FILIP  Fine-grained Interactive Language-Image Pre-Training

[26] KD-VLP  Improving End-to-End Vision-and-Language Pretraining with Object  Knowledge Distillation

[27] SemVLP  Vision-Language Pre-training by Aligning Semantics at Multiple  Levels

[28] UNIMO-3  Multi-granularity Interaction for Vision-Language  Representation Learning

[29] MVPTR  Multi-Level Semantic Alignment for Vision-Language Pre-Training  via Multi-Stage Learning

[30] Seeing What You Miss  Vision-Language Pre-training with Semantic  Completion Learning

[31] Sieve  Multimodal Dataset Pruning Using Image Captioning Models

[32] VLMAE  Vision-Language Masked Autoencoder

[33] Probing Cross-modal Semantics Alignment Capability from the Textual  Perspective

[34] E2E-VLP  End-to-End Vision-Language Pre-training Enhanced by Visual  Learning

[35] Do Vision-and-Language Transformers Learn Grounded Predicate-Noun  Dependencies 

[36] Grid-VLP  Revisiting Grid Features for Vision-Language Pre-training

[37] Silence

[38] An Empirical Study of Training End-to-End Vision-and-Language  Transformers

[39] Self-Imitation Learning

[40] Roget's Thesaurus  a Lexical Resource to Treasure

[41] Leveraging per Image-Token Consistency for Vision-Language Pre-training

[42] E2E Refined Dataset

[43] B-Splines

[44] Accelerating Vision-Language Pretraining with Free Language Modeling

[45] GLIPv2  Unifying Localization and Vision-Language Understanding

[46] ROSITA  Enhancing Vision-and-Language Semantic Alignments via Cross- and  Intra-modal Knowledge Integration

[47] IDEA  Increasing Text Diversity via Online Multi-Label Recognition for  Vision-Language Pre-training

[48] Retrieval-based Knowledge Augmented Vision Language Pre-training

[49] Generalizing Multimodal Pre-training into Multilingual via Language  Acquisition

[50] CCMB  A Large-scale Chinese Cross-modal Benchmark

[51] STOA-VLP  Spatial-Temporal Modeling of Object and Action for  Video-Language Pre-training

[52] Medical Vision Language Pretraining  A survey

[53] IMITATE  Clinical Prior Guided Hierarchical Vision-Language Pre-training

[54] CTP  Towards Vision-Language Continual Pretraining via Compatible  Momentum Contrast and Topology Preservation

[55] VL-CheckList  Evaluating Pre-trained Vision-Language Models with  Objects, Attributes and Relations

[56] Can Linguistic Knowledge Improve Multimodal Alignment in Vision-Language  Pretraining 

[57] VLUE  A Multi-Task Benchmark for Evaluating Vision-Language Models

[58] RC3  Regularized Contrastive Cross-lingual Cross-modal Pre-training

[59] VALSE  A Task-Independent Benchmark for Vision and Language Models  Centered on Linguistic Phenomena

[60] MMBench  Is Your Multi-modal Model an All-around Player 

[61] Are we pretraining it right  Digging deeper into visio-linguistic  pretraining

[62] Foundational Models Defining a New Era in Vision  A Survey and Outlook

[63] Efficient Vision-and-Language Pre-training with Text-Relevant Image  Patch Selection

[64] mPLUG  Effective and Efficient Vision-Language Learning by Cross-modal  Skip-connections

[65] VALOR  Vision-Audio-Language Omni-Perception Pretraining Model and  Dataset

[66] BridgeTower  Building Bridges Between Encoders in Vision-Language  Representation Learning

[67] VLMo  Unified Vision-Language Pre-Training with  Mixture-of-Modality-Experts

[68] Scheduled Sampling in Vision-Language Pretraining with Decoupled  Encoder-Decoder Network

[69] Masked Faces with Faced Masks

[70] VL-BEiT  Generative Vision-Language Pretraining

[71] Oscar  Object-Semantics Aligned Pre-training for Vision-Language Tasks

[72] SyncMask  Synchronized Attentional Masking for Fashion-centric  Vision-Language Pretraining

[73] Contrastive Visual-Linguistic Pretraining

[74] All or Nothing at All

[75] CLIP-Event  Connecting Text and Images with Event Structures

[76] Align before Fuse  Vision and Language Representation Learning with  Momentum Distillation

[77] UFO  A UniFied TransfOrmer for Vision-Language Representation Learning

[78] PiTL  Cross-modal Retrieval with Weakly-supervised Vision-language  Pre-training via Prompting

[79] Seeing Out of tHe bOx  End-to-End Pre-training for Vision-Language  Representation Learning

[80] VoLTA  Vision-Language Transformer with Weakly-Supervised Local-Feature  Alignment

[81] Unsupervised Vision-and-Language Pre-training via Retrieval-based  Multi-Granular Alignment

[82] EVE  Efficient Vision-Language Pre-training with Masked Prediction and  Modality-Aware MoE

[83] 4M  Massively Multimodal Masked Modeling

[84] MURAL  Multimodal, Multitask Retrieval Across Languages

[85] Masked Contrastive Reconstruction for Cross-modal Medical Image-Report  Retrieval

[86] UNIMO-2  End-to-End Unified Vision-Language Grounded Learning

[87] Single-Stream Multi-Level Alignment for Vision-Language Pretraining

[88] Localization vs. Semantics  Visual Representations in Unimodal and  Multimodal Models

[89] Position-guided Text Prompt for Vision-Language Pre-training

[90] Toward Interactive Regional Understanding in Vision-Large Language  Models

[91] UNITER  UNiversal Image-TExt Representation Learning

[92] LXMERT  Learning Cross-Modality Encoder Representations from  Transformers

[93] Global and Local Semantic Completion Learning for Vision-Language  Pre-training

[94] MAGMA -- Multimodal Augmentation of Generative Models through  Adapter-based Finetuning

[95] Towards Unifying Medical Vision-and-Language Pre-training via Soft  Prompts

[96] Mitigating Heterogeneity in Federated Multimodal Learning with  Biomedical Vision-Language Pre-training

[97] Context-Aware Prompt Tuning for Vision-Language Model with  Dual-Alignment

[98] Learning Hierarchical Prompt with Structured Linguistic Knowledge for  Vision-Language Models

[99] APLe  Token-Wise Adaptive for Multi-Modal Prompt Learning

[100] HiVLP  Hierarchical Vision-Language Pre-Training for Fast Image-Text  Retrieval

[101] DLIP  Distilling Language-Image Pre-training

[102] VQA and Visual Reasoning  An Overview of Recent Datasets, Methods and  Challenges

[103] Learning Visual Representation from Modality-Shared Contrastive  Language-Image Pre-training

[104] Does my multimodal model learn cross-modal interactions  It's harder to  tell than you might think!

[105] ERNIE-UniX2  A Unified Cross-lingual Cross-modal Framework for  Understanding and Generation
