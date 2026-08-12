# Registered common-window excerpt

# Vision-Language Pretraining: Foundations, Architectures, Objectives, Adaptation, and Future Horizons

## 1 Introduction

The convergence of visual perception and natural language understanding represents one of the most intellectually fertile frontiers in modern artificial intelligence. Early deep learning research advanced unimodal capabilities independently—convolutional networks achieving breakthrough object recognition on [1] while recurrent architectures and word embeddings [2] transformed natural language processing—yet the fundamental human capacity to reason jointly across sight and language demanded unified representations. Vision-language pretraining (VLP) emerged as the paradigm bridging this divide, enabling models pretrained on large-scale image-text corpora to transfer rich multimodal knowledge to diverse downstream tasks [3; 4; 5]. The dramatic scaling of noisy web-collected supervision [6] catalyzed contrastive pretraining paradigms, while generative objectives enabled flexible understanding and generation within unified frameworks [7; 8; 9]. This survey comprehensively organizes these advances across data foundations, architectures, pretraining objectives, adaptation strategies, and emerging frontiers [10; 11; 12].

## 2 Data Foundations for Vision-Language Pretraining

### 2.1 Large-Scale Web-Crawled Image-Text and Video-Text Corpora

Large-scale web-crawled corpora constitute the indispensable data foundation upon which modern vision-language pretraining rests. Landmark datasets such as Conceptual Captions and the billion-scale noisy corpus underlying [6] demonstrated that alt-text harvested from Common Crawl, despite imperfect semantic alignment, could drive powerful contrastive representations when paired with scalable dual-encoder architectures. Curated counterparts including [4] and [5] leveraged MSCOCO and Visual Genome to establish high-quality fine-tuning benchmarks. For video, corpora exploiting automatic speech recognition transcripts introduce inherent temporal misalignment challenges distinct from static image-text settings, as addressed by unified frameworks like [13] and contrastive video pretraining approaches [14]. Critically, naive scaling without quality control yields diminishing returns [8], underscoring that domain diversity and distributional proximity to downstream tasks are equally decisive as raw corpus magnitude.

### 2.2 Noise, Semantic Misalignment, and Data Quality in Web-Collected Corpora

Web-crawled corpora inherently contain pervasive semantic misalignment between images and accompanying text, ranging from topically irrelevant alt-text to boilerplate metadata [6; 15]. This challenge, already identified as a central limitation of large-scale data collection, demands principled curation strategies that can distinguish genuine semantic correspondence from incidental co-occurrence. To mitigate this, CLIP-score-based filtering has emerged as the dominant quality gate [16], providing a scalable mechanism for retaining only those image-text pairs whose cross-modal similarity exceeds a learned threshold. Complementing this filtering paradigm, BLIP's caption bootstrapping—generating synthetic captions and filtering noise-contaminated pairs—demonstrates that noise-aware training objectives can reclaim value from imperfect data [8], effectively transforming noisy web-crawled supervision into a cleaner training signal. Collectively, these strategies reveal a fundamental quality-quantity trade-off: raw scale alone yields diminishing returns without principled curation [17]. This insight naturally motivates the richer portfolio of data enrichment techniques explored in the following section, which move beyond filtering to actively augment and restructure the supervisory signal available from raw image-text corpora.

### 2.3 Data Augmentation and Semantic Enrichment Strategies

Raw image-text pairs harvested from the web frequently suffer from semantic imprecision and sparse conceptual coverage, motivating a diverse portfolio of enrichment strategies that augment supervision without human re-annotation. Automated captioning and caption bootstrapping represent a particularly influential direction: [8] demonstrates that iteratively generating synthetic captions with a learned captioner and filtering them with a trained noise detector yields measurably cleaner supervision than the original web-crawled text, breaking the dependence on static noisy corpora. Complementary to caption-level enrichment, object detection and visual concept annotation pipelines inject structured region-level semantics by grounding textual tokens to specific visual entities, a strategy that underpins cross-modal alignment in grounded pretraining frameworks. OCR-based extraction further expands the supervisory signal by recovering scene text embedded within images. On the linguistic side, cross-modal CutMix augmentation [18] replaces visually grounded words with semantically similar image patches, enabling token-level cross-modal alignment learning even from unpaired corpora. Hierarchical semantic alignment [19] and scene-graph-guided structural masking [20] further enrich pretraining by enforcing multi-level correspondences between visual regions and linguistic structures, collectively establishing that systematic enrichment of raw data yields more semantically grounded representations than naive scaling alone.

### 2.4 Synthetic and Automatically Generated Multimodal Pretraining Data

Synthetic and automatically generated multimodal data has emerged as a compelling complement to web-crawled corpora, addressing data scarcity and alignment quality limitations that enrichment strategies alone cannot fully resolve. Caption bootstrapping, as exemplified by [8], demonstrates that iteratively generating and filtering synthetic captions meaningfully improves pretraining signal quality over raw noisy web data, extending the quality-oriented philosophy of automated enrichment into a fully generative regime. Meanwhile, automated annotation pipelines leveraging large pretrained models—such as using CLIP-guided pseudo-labeling for object detection [21]—illustrate how synthetic supervision can scale vocabulary coverage without costly human effort, complementing the structured region-level semantics introduced by detection-based enrichment approaches. However, critical trade-offs persist: synthetically generated images from diffusion models exhibit distributional artifacts relative to natural photographs [22], and retrieved real images frequently match or outperform purely synthetic alternatives. Collectively, these findings suggest that the utility of synthetic data depends heavily on its domain proximity to downstream tasks and the fidelity of the generative pipeline employed—a dependency that becomes particularly acute when pretraining shifts from general web-crawled domains to specialized technical or clinical domains with distinct visual grammars and restricted data availability.

### 2.5 Specialized and Domain-Specific Pretraining Datasets

General-domain web-crawled corpora prove insufficient for domains requiring specialized visual grammars and technical linguistic registers. In radiology, datasets such as MIMIC-CXR and OpenI pair chest radiographs with free-text clinical reports, demanding preprocessing pipelines that extract structured clinical entities; [23] addresses this through triplet extraction and knowledge-base entity translation, enabling fine-grained patch-level alignment between visual signals and medical concepts. Similarly, [24] demonstrates that BERT-based architectures with domain-adapted attention masking substantially improve both report generation and medical visual question answering over general-purpose baselines. Document-centric domains present orthogonal challenges, where tight spatial coupling between rendered text and visual layout necessitates OCR-augmented preprocessing; [25] illustrates how incorporating text-rich images from LAION through OCR pipelines markedly improves text-aware visual understanding. Across all specialized domains, the unifying challenge remains reconciling extreme data scarcity—imposed by privacy constraints, expert annotation costs, or rare condition prevalence—with the scale demands of effective pretraining, motivating hybrid strategies that combine synthetic augmentation with carefully curated real paired examples.

### 2.6 Dataset Design Principles and Their Impact on Downstream Generalization

Dataset construction choices fundamentally shape the generalization profile of pretrained vision-language models, and the evidence accumulated across both general and specialized domains converges on a set of unifying principles. Empirical evidence consistently demonstrates that domain proximity between pretraining corpora and downstream evaluation benchmarks is a stronger determinant of transfer quality than raw dataset scale [26; 8], while principled filtering—such as metadata-curated balancing [27] and quality-aware subset selection—can yield models that outperform those trained on order-of-magnitude larger but noisier corpora. Multi-source combination does not universally improve robustness and can dilute the benefits of the strongest individual source, underscoring the importance of source-aware sampling and mixture-ratio optimization [28]. Together, these insights reinforce the theme running through the preceding discussion of specialized domains: that the tension between data quantity and domain fidelity is best resolved not by indiscriminate scaling, but by task-aware, theoretically grounded curation as the central principle guiding future dataset design.

## 3 Model Architectures for Vision-Language Pretraining

### 3.1 Visual Feature Extraction Backbones

I need to check each citation in the subsection to verify it supports the corresponding claim.

1. "[29] demonstrated compellingly that upgrading the detection backbone substantially improves cross-modal alignment across all downstream tasks" - VinVL paper is about improving visual representations via better object detection model for VL tasks. ✓ Correct.

2. "Models including [3], [4], and [5] inherited this paradigm" - All three use region-based features from detectors. ✓ Correct.

3. "[7] and [30][31] demonstrated that ViT-based encoders trained end-to-end with language models capture long-range visual dependencies more effectively" - SimVLM uses patch-based encoding end-to-end. BEiT-3 uses ViT-based backbone. ✓ Correct.

4. "[32] scaled visual encoding to 4B parameters, revealing that joint vision-language scaling yields disproportionate gains" - PaLI trains a 4B parameter ViT and discusses joint scaling benefits. ✓ Correct.

5. "Discrete visual tokenization, as explored in [33], further enables unified autoregressive pretraining" - Emu (Generative Pretraining

[... deterministic omitted span ...]

Building naturally on the hybrid supervision strategies discussed above, which combine structured knowledge, distributional representations, and perception-derived signals, a key question becomes how to organize these diverse signal sources into coherent training procedures that maximize their collective benefit. Multi-stage pretraining has emerged as a principled solution: [57] demonstrates that first pretraining on large weakly supervised web data before fine-tuning on curated corpora consistently outperforms single-stage training, while [52] employs stagewise curricula that leverage unimodal image and text corpora before introducing paired data, effectively bootstrapping cross-modal alignment from stronger unimodal representations. Noise-robust training is equally critical given the prevalence of misaligned web pairs; [8] addresses this through caption bootstrapping, wherein a learned captioner generates synthetic captions subsequently filtered by a trained noise detector, progressively improving supervision quality during training itself. Taken together, these curriculum and data handling strategies represent an essential complement to enriched objective designs, ensuring that the full potential of structured, probabilistic, and perception-derived supervision signals is realized through disciplined and efficient optimization procedures.

## 5 Downstream Tasks, Evaluation, and Benchmarking

### 5.1 Visual Understanding and Reasoning Benchmarks

I need to check each citation in the subsection to verify it's supported by the corresponding paper content.

1. "[4], [3], and [7] have demonstrated substantial accuracy gains through pretraining" - All three papers show accuracy gains on VQA benchmarks. ✓

2. "Higher-order reasoning benchmarks including NLVR2 and VCR expose limitations of pattern-matching approaches, as evidenced by evaluations of [54] and [30][31]" - FLAVA mentions 35 tasks including multimodal ones. BEiT-3 explicitly mentions NLVR2. FLAVA does evaluate on reasoning tasks. ✓ BEiT-3 ✓

3. "Visual entailment datasets like SNLI-VE probe logical consistency across modalities [7]" - SimVLM explicitly mentions SNLI-VE (+1.37% accuracy). ✓

4. "grounding benchmarks RefCOCO and Flickr30k Entities demand precise spatial localization [58; 59]" - CogVLM mentions RefCOCO, RefCOCO+, RefCOCOg. ✓ X-VLM (Multi-Grained) focuses on alignment/localization. ✓

5. "[60] demonstrates that aggregate benchmark scores systematically obscure fine-grained capability differences" - VL-CheckList explicitly states this. ✓

All citations appear correct. Here is the subsection:

Visual understanding and reasoning benchmarks constitute the primary evaluation substrate for vision-language pretrained models, spanning tasks from perceptual question answering to multi-step logical inference. Foundational VQA benchmarks including VQAv2, GQA, TextVQA, and OK-VQA assess progressively deeper cross-modal competencies, while models such as [4], [3], and [7] have demonstrated substantial accuracy gains through pretraining, simultaneously revealing dataset biases exploitable without genuine visual grounding. Higher-order reasoning benchmarks including NLVR2 and VCR expose limitations of pattern-matching approaches, as evidenced by evaluations of [54] and [31]. Visual entailment datasets like SNLI-VE probe logical consistency across modalities [7], whereas grounding benchmarks RefCOCO and Flickr30k Entities demand precise spatial localization [58; 59]. Crucially, [60] demonstrates that aggregate benchmark scores systematically obscure fine-grained capability differences, motivating diagnostic evaluation beyond headline metrics.

### 5.2 Image-Text Generation Benchmarks and Evaluation Metrics

Image-text generation evaluation provides a critical complement to the reasoning and grounding benchmarks discussed above, extending assessment to the generative capabilities of vision-language pretrained models. Canonical benchmarks including MSCOCO, NoCaps, and Flickr30K captioning tracks serve as the primary evaluation substrate in this domain, where models such as [47], [34], and [8] are assessed using reference-based metrics including BLEU, CIDEr, METEOR, SPICE, and ROUGE. While CIDEr correlates reasonably with human consensus, these reference-based metrics remain fundamentally insensitive to semantically valid paraphrases, echoing the fine-grained evaluation concerns raised by diagnostic benchmarks such as VL-CheckList. Emerging model-based evaluators leveraging vision-language models offer stronger alignment with human judgment, pointing toward hybrid automatic-human evaluation as the field's trajectory—a direction that becomes particularly salient as evaluation demands shift from static image-text generation toward the retrieval and zero-shot recognition paradigms examined in the following section.

### 5.3 Vision-Language Retrieval and Zero-Shot Recognition Tasks

Cross-modal retrieval and zero-shot recognition represent the most direct evaluation paradigms for contrastively pretrained vision-language models, with MSCOCO and Flickr30k serving as canonical retrieval benchmarks measured via Recall@K metrics [6]. Models such as CLIP demonstrate remarkable zero-shot image classification by matching image embeddings against textual class-name descriptors, with prompt engineering and ensemble strategies yielding substantial gains across ImageNet and distribution-shifted variants [61; 62]. Hierarchical alignment strategies further improve data efficiency in zero-shot settings [19], while open-vocabulary detection on COCO and LVIS benchmarks reveals the generalization boundaries of contrastive representations toward novel categories [63]. Critically, false-negative contamination in standard retrieval benchmarks systematically underestimates model capability, motivating corrected evaluation protocols with denser positive associations [64].

### 5.4 Video-Language Understanding and Generation Benchmarks

Video-language benchmarks extend the evaluation landscape beyond the static image-text alignment paradigms discussed above, introducing temporal reasoning demands that retrieval and zero-shot recognition tasks cannot capture. Tasks spanning video question answering (MSRVTT-QA, ActivityNet-QA), text-video retrieval, and dense video captioning collectively probe whether models can ground language in dynamic visual sequences. [65] demonstrates that bidirectional generation objectives over multimodal video encoders yield strong performance across captioning and retrieval simultaneously, while [66] shows that pretraining on narrated videos with pseudo event boundaries substantially advances dense video description. Notably, [8] reveals that image-language models transfer surprisingly well to video tasks zero-shot, suggesting that temporal modeling and cross-modal alignment share foundational mechanisms worth exploiting jointly. While these benchmarks meaningfully broaden the scope of vision-language evaluation, they nonetheless remain organized around individual tasks or modalities. As vision-language models continue to scale and diversify in capability, this task-specific framing proves increasingly insufficient for holistic assessment, motivating the comprehensive evaluation frameworks discussed in the following section.

### 5.5 Holistic and Integrated Capability Benchmarks for Large Vision-Language Models

I need to check each citation in the subsection to verify they support the claims made.

1. "large vision-language models [67; 51; 68]" - These are indeed large vision-language models. ✓

2. "Comprehensive evaluation frameworks such as MM-Vet and LAMM [69]" - LAMM is indeed an evaluation benchmark/framework. ✓ (MM-Vet is not in the provided papers list, so that reference is fine as it's not cited with brackets)

3. "leveraging GPT-4-style judges [70; 71]" - "Aligning Large Multimodal Models with Factually Augmented RLHF" uses RLHF with human annotators, not specifically GPT-4-style judges. "To See is to Believe" uses GPT-4V for generating instruction data, not as a judge for evaluation. These citations are questionable. The LAMM paper mentions using LLM-assisted evaluation. I'll remove these citations as they don't directly support "GPT-4-style judges" for evaluation scoring.

4. "Models including [25; 72] are systematically compared" - LLaVA-Plus and BLIVA are indeed models that are compared across benchmarks. BLIVA mentions MME benchmark comparisons. ✓

As large vision-language models [67; 51; 68] have grown increasingly capable across diverse tasks, single-task benchmarks prove insufficient for holistic assessment. Comprehensive evaluation frameworks such as MM-Vet and LAMM [69] organize evaluation across perception, reasoning, and generation using LLM-assisted automated scoring—leveraging GPT-4-style judges to assess free-form outputs beyond rigid reference matching. Models including [25; 72] are systematically compared across integrated capability dimensions, revealing that holistic benchmarks expose complementary strengths and failure modes that narrow benchmarks obscure entirely.

### 5.6 Fine-Grained Diagnostic and Compositional Evaluation Frameworks

Motivated by the limitations of holistic benchmarks described above, specialized diagnostic frameworks have emerged that decompose capability into fine-grained dimensions, revealing mechanistic insights that aggregate accuracy conceals. [73] demonstrates that even high-performing VLP models struggle systematically with structured visio-linguistic concepts—attributes, relations, and object states—that aggregate scores obscure, motivating data-driven evaluation and training interventions targeting these specific phenomena. [74] further reveals that multimodal models remain subject to reporting bias across color, shape, material, and co-occurrence properties, with scale providing surprisingly little remedy—underscoring that diagnostic granularity, rather than leaderboard ranking, exposes genuine representational gaps and guides targeted model improvement. Together, these finer-grained approaches highlight that advancing vision-language pretraining requires not only broader benchmarks but also principled decomposition of the specific visio-linguistic axes along which models continue to fall short.

## 6 Adaptation, Transfer Learning, and Applications

### 6.1 Full Fine-Tuning and Task-Specific Adaptation Protocols

Full fine-tuning remains the dominant adaptation paradigm for transferring pretrained vision-language representations to downstream tasks, wherein all model parameters are updated jointly on task-specific data. Models such as [3], [4], and [5] established foundational protocols whereby task-specific classification heads or decoder modules are appended

[... deterministic omitted span ...]

paper describes jointly modeling vision, audio, and language through unified alignment (MGA) and captioning (MGC) objectives, yielding strong generalization across retrieval and generation tasks. ✓ This citation is correct.

2. **[85]** - The paper describes leveraging ImageBind's joint embedding space to extend language-conditioned reasoning to audio, 3D point clouds, video, without modality-specific retraining (only image-text alignment training). ✓ This citation is correct.

3. **[69]** - The paper mentions covering "a wide range of vision tasks for 2D and 3D vision" and mentions AI agents. However, it's described as an instruction-tuning dataset/framework/benchmark, and it does mention 3D vision tasks. The claim about "embodied settings" and "robotic manipulation and navigation" and "agent grounding" is a stretch - the paper mentions AI agents and 3D vision but doesn't specifically focus on robotic manipulation/navigation. However, it does mention "bridging the gap between ideas and execution" for AI agents and covers 3D vision. The citation is partially supportable for 3D perception aspects.

The citation for LAMM is the closest available paper for the embodied/3D perception claim, and it does cover 3D vision tasks. I'll keep it as it's the best available citation from the provided list.

Beyond static image-text pairs, the vision-language pretraining paradigm is increasingly extended to incorporate additional sensory modalities and embodied settings. Frameworks such as [84] demonstrate that jointly modeling vision, audio, and language through unified alignment and captioning objectives yields strong generalization across retrieval and generation tasks. Similarly, [85] leverages a joint embedding space to extend language-conditioned reasoning to audio, depth, and 3D point clouds without modality-specific retraining. In embodied settings, pretrained vision-language representations inform robotic manipulation and navigation policies, with works like [69] explicitly addressing 3D perception for agent grounding. A critical open challenge remains bridging the distributional gap between internet-scale pretraining data and the continuous, action-conditioned observations encountered in interactive environments.

### 7.6 Continual Learning, Multilingual Pretraining, and Grounded Compositional Reasoning

Three interconnected frontier challenges—continual visual concept acquisition, multilingual grounded pretraining, and compositional reasoning—collectively define the next frontier for vision-language models. Building directly on the multimodal and embodied extensions discussed above, these challenges highlight fundamental limitations that persist even in the most capable current systems. Standard pretraining assumes static data distributions, yet real-world deployment demands incremental adaptation without catastrophic forgetting. Multilingual frameworks reveal that jointly grounding visual perception across diverse linguistic communities improves low-resource transfer. Meanwhile, contrastive models trained on web-scale corpora exhibit persistent bag-of-words tendencies [26], failing on structured compositional queries requiring attribute binding and relational reasoning [73]. Addressing these challenges simultaneously—through composition-aware objectives, multilingual alignment signals, and incremental learning curricula—represents the most promising path toward vision-language systems capable of genuine causal understanding rather than statistical co-occurrence exploitation.

## 8 Conclusion

Vision-language pretraining has undergone a remarkable transformation, progressing from region-based fusion architectures [3; 4; 5] through contrastive large-scale alignment [6] to unified generative foundations [9; 31; 33]. Collectively, advances in data curation [8], parameter-efficient adaptation [78], and holistic evaluation [60; 86] have reshaped multimodal AI. Persistent challenges—compositional generalization, fairness, computational sustainability, and cross-lingual grounding [32]—remain critical frontiers, compelling future research toward principled data curation, interpretable architectures, and rigorous evaluation frameworks that transcend aggregate benchmark performance [11; 87].

## References

[1] ImageNet Large Scale Visual Recognition Challenge

[2] Efficient Estimation of Word Representations in Vector Space

[3] ViLBERT  Pretraining Task-Agnostic Visiolinguistic Representations for  Vision-and-Language Tasks

[4] UNITER  UNiversal Image-TExt Representation Learning

[5] VisualBERT  A Simple and Performant Baseline for Vision and Language

[6] Scaling Up Visual and Vision-Language Representation Learning With Noisy  Text Supervision

[7] SimVLM  Simple Visual Language Model Pretraining with Weak Supervision

[8] BLIP  Bootstrapping Language-Image Pre-training for Unified  Vision-Language Understanding and Generation

[9] OFA  Unifying Architectures, Tasks, and Modalities Through a Simple  Sequence-to-Sequence Learning Framework

[10] VLP  A Survey on Vision-Language Pre-training

[11] Vision-Language Pre-training  Basics, Recent Advances, and Future Trends

[12] A Survey on Multimodal Large Language Models

[13] VindLU  A Recipe for Effective Video-and-Language Pretraining

[14] VideoCLIP  Contrastive Pre-training for Zero-shot Video-Text  Understanding

[15] LAION-400M  Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs

[16] LAION-5B  An open large-scale dataset for training next generation  image-text models

[17] Conceptual 12M  Pushing Web-Scale Image-Text Pre-Training To Recognize  Long-Tail Visual Concepts

[18] VLMixer  Unpaired Vision-Language Pre-training via Cross-Modal CutMix

[19] PyramidCLIP  Hierarchical Feature Alignment for Vision-language Model  Pretraining

[20] ROSITA  Enhancing Vision-and-Language Semantic Alignments via Cross- and  Intra-modal Knowledge Integration

[21] Exploiting Unlabeled Data with Vision and Language Models for Object  Detection

[22] Diversify Your Vision Datasets with Automatic Diffusion-Based  Augmentation

[23] MedKLIP  Medical Knowledge Enhanced Language-Image Pre-Training in  Radiology

[24] Multi-modal Understanding and Generation for Medical Images and Text via  Vision-Language Pre-Training

[25] LLaVA-Plus  Learning to Use Tools for Creating Multimodal Agents

[26] Learning Transferable Visual Models From Natural Language Supervision

[27] Demystifying CLIP Data

[28] Democratizing Contrastive Language-Image Pre-training  A CLIP Benchmark  of Data, Model, and Supervision

[29] VinVL  Revisiting Visual Representations in Vision-Language Models

[30] DeiT III  Revenge of the ViT

[31] Image as a Foreign Language  BEiT Pretraining for All Vision and  Vision-Language Tasks

[32] PaLI  A Jointly-Scaled Multilingual Language-Image Model

[33] Generative Pretraining in Multimodality

[34] CoCa  Contrastive Captioners are Image-Text Foundation Models

[35] MiniGPT-4  Enhancing Vision-Language Understanding with Advanced Large  Language Models

[36] ClipCap  CLIP Prefix for Image Captioning

[37] Scaling Language-Image Pre-training via Masking

[38] When and why vision-language models behave like bags-of-words, and what  to do about it 

[39] Why is Winoground Hard  Investigating Failures in Visuolinguistic  Compositionality

[40] SoftCLIP  Softer Cross-modal Alignment Makes CLIP Stronger

[41] Non-Contrastive Learning Meets Language-Image Pre-Training

[42] Equivariant Similarity for Vision-Language Foundation Models

[43] Unicoder-VL  A Universal Encoder for Vision and Language by Cross-modal  Pre-training

[44] Oscar  Object-Semantics Aligned Pre-training for Vision-Language Tasks

[45] LayoutLMv2  Multi-modal Pre-training for Visually-Rich Document  Understanding

[46] BLIP-2  Bootstrapping Language-Image Pre-training with Frozen Image  Encoders and Large Language Models

[47] GIT  A Generative Image-to-text Transformer for Vision and Language

[48] Scaling Autoregressive Multi-Modal Models  Pretraining and Instruction  Tuning

[49] Planting a SEED of Vision in Large Language Model

[50] DreamLLM  Synergistic Multimodal Comprehension and Creation

[51] InstructBLIP  Towards General-purpose Vision-Language Models with  Instruction Tuning

[52] VLMo  Unified Vision-Language Pre-Training with  Mixture-of-Modality-Experts

[53] LayoutLMv3  Pre-training for Document AI with Unified Text and Image  Masking

[54] FLAVA  A Foundational Language And Vision Alignment Model

[55] Learning Language-Visual Embedding for Movie Understanding with  Natural-Language

[56] Tag2Text  Guiding Vision-Language Model via Image Tagging

[57] ImageBERT  Cross-modal Pre-training with Large-scale Weak-supervised  Image-Text Data

[58] CogVLM  Visual Expert for Pretrained Language Models

[59] Multi-Grained Vision Language Pre-Training  Aligning Texts with Visual  Concepts

[60] VL-CheckList  Evaluating Pre-trained Vision-Language Models with  Objects, Attributes and Relations

[61] Data Determines Distributional Robustness in Contrastive Language Image  Pre-training (CLIP)

[62] Filtering, Distillation, and Hard Negatives for Vision-Language  Pre-Training

[63] Learning Object-Language Alignments for Open-Vocabulary Object Detection

[64] ECCV Caption  Correcting False Negatives by Collecting  Machine-and-Human-verified Image-Caption Associations for MS-COCO

[65] End-to-end Generative Pretraining for Multimodal Video Captioning

[66] Vid2Seq  Large-Scale Pretraining of a Visual Language Model for Dense  Video Captioning

[67] Visual Instruction Tuning

[68] mPLUG-Owl  Modularization Empowers Large Language Models with  Multimodality

[69] LAMM  Language-Assisted Multi-Modal Instruction-Tuning Dataset,  Framework, and Benchmark

[70] Aligning Large Multimodal Models with Factually Augmented RLHF

[71] To See is to Believe  Prompting GPT-4V for Better Visual Instruction  Tuning

[72] BLIVA  A Simple Multimodal LLM for Better Handling of Text-Rich Visual  Questions

[73] Teaching Structured Vision&Language Concepts to Vision&Language Models

[74] Visual Commonsense in Pretrained Unimodal and Multimodal Models

[75] VL-Adapter  Parameter-Efficient Transfer Learning for  Vision-and-Language Tasks

[76] Exploring CLIP for Assessing the Look and Feel of Images

[77] Generalized Decoding for Pixel, Image, and Language

[78] LLaMA-Adapter  Efficient Fine-tuning of Language Models with Zero-init  Attention

[79] A Survey of Current Datasets for Vision and Language Research

[80] OpenFlamingo  An Open-Source Framework for Training Large Autoregressive  Vision-Language Models

[81] VILA  On Pre-training for Visual Language Models

[82] FOIL it! Find One mismatch between Image and Language caption

[83] Scaling Up Vision-Language Pre-training for Image Captioning

[84] VALOR  Vision-Audio-Language Omni-Perception Pretraining Model and  Dataset

[85] ImageBind-LLM  Multi-modality Instruction Tuning

[86] Winoground  Probing Vision and Language Models for Visio-Linguistic  Compositionality

[87] Multimodal Foundation Models  From Specialists to General-Purpose  Assistants
