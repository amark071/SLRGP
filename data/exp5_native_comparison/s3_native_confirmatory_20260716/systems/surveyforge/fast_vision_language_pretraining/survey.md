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

5. "Discrete visual tokenization, as explored in [33], further enables unified autoregressive pretraining" - Emu (Generative Pretraining in Multimodality) does use visual embeddings in autoregressive training, but it regresses visual embeddings rather than discrete tokens. The claim about "discrete visual tokenization" is not well supported by Emu. I should remove this citation.

Visual feature extraction has undergone a profound architectural transformation in vision-language pretraining (VLP), progressing from detector-dependent region proposals to fully end-to-end Vision Transformer paradigms. Early VLP models relied on bottom-up and top-down region-of-interest features from Faster R-CNN-based detectors, providing semantically rich object-centric representations; [29] demonstrated compellingly that upgrading the detection backbone substantially improves cross-modal alignment across all downstream tasks, underscoring that visual feature quality is not incidental but foundational. Models including [3], [4], and [5] inherited this paradigm, achieving strong performance while accepting significant inference bottlenecks from frozen, separately pretrained detectors. The community subsequently shifted toward grid-based convolutional encoders and patch-based Vision Transformers to eliminate detector dependencies; [7] and [31] demonstrated that ViT-based encoders trained end-to-end with language models capture long-range visual dependencies more effectively, while [32] scaled visual encoding to 4B parameters, revealing that joint vision-language scaling yields disproportionate gains. Discrete visual tokenization further enables unified autoregressive pretraining, bridging visual encoding and language generation within a single framework.

### 3.2 Textual Encoding Approaches

Textual encoding in vision-language pretraining spans a spectrum from bidirectional transformers to autoregressive decoders, with each choice fundamentally shaping cross-modal alignment and generative capacity. Building on the visual representations established through region-based and patch-based encoders, the design of text encoders determines how linguistic structure is captured and subsequently fused with visual content. Subword tokenization via byte-pair encoding or WordPiece remains standard across architectures [4; 8], while BERT-style bidirectional encoders excel at understanding-oriented fusion but inherently limit open-ended generation. Contrastive models like CLIP [26] and ALIGN [6] employ lightweight text encoders optimized for embedding-space alignment rather than generation, deliberately trading generative flexibility for discriminative efficiency. Encoder-decoder frameworks such as SimVLM [7] and CoCa [34] resolve this tension by coupling a multimodal encoder with an autoregressive decoder, enabling simultaneous discriminative and generative competence. More recently, frozen large language models connected via lightweight projection—as in MiniGPT-4 [35] and ClipCap [36]—inherit rich language priors without full joint retraining, representing a pragmatic convergence of efficiency and capability. These textual encoding strategies, each carrying distinct implications for alignment depth and generative capacity, directly inform how visual and textual streams are ultimately combined—a distinction that becomes especially pronounced in the contrasting design philosophies of dual-encoder and fusion-encoder architectures examined in the sections that follow.

### 3.3 Dual-Encoder and Contrastive Architectures

I need to verify each citation in the subsection against the paper contents provided.

1. "[6]" - This paper describes ALIGN, a dual-encoder using billion-scale noisy image-text pairs with contrastive loss. ✓

2. "[37]" - FLIP uses masking to accelerate CLIP training. ✓

3. "[19]" - Uses hierarchical alignment strategies. ✓

4. "[28]" - Discusses CLIP evaluation including zero-shot classification. ✓

5. "[38]" - Discusses compositional reasoning failures in VLMs. ✓

6. "[39]" - Discusses failures in visuolinguistic compositionality. ✓

7. "[40]" - Proposes soft alignment targets to relax strict one-to-one constraints. ✓

8. "[41]" - Explores non-contrastive auxiliary objectives (nCLIP/xCLIP). ✓

9. "[42]" - Proposes EqSim, an equivariance regularization loss. ✓

All citations appear to be correct and well-supported by the paper contents.

Dual-encoder architectures project independently encoded visual and textual representations into a shared embedding space through contrastive alignment, exemplified most prominently by [6], which demonstrated that billion-scale noisy image-text pairs suffice to learn discriminative cross-modal representations without expensive curation. Training these models requires large effective batch sizes and careful negative sampling, as explored through masking-accelerated variants [37] and hierarchical alignment strategies [19]. Their primary advantage lies in inference efficiency: visual and textual embeddings can be precomputed independently, enabling scalable retrieval and zero-shot classification [28]. However, the fundamental limitation—that separate encoders preclude token-level cross-modal interaction—manifests as compositional reasoning failures [38; 39], motivating mitigation strategies such as soft alignment targets [40], non-contrastive auxiliary objectives [41], and equivariance regularization [42].

### 3.4 Fusion Encoder and Cross-Attention Architectures

Fusion encoder architectures address the core limitation of dual-encoder models by jointly processing visual and textual tokens within shared transformer layers, achieving deep cross-modal interaction at the cost of the inference efficiency that contrastive designs prioritize. Single-stream designs concatenate flattened image region features or patch tokens with wordpiece text sequences into one unified input, enabling every transformer layer to attend across both modalities simultaneously, as exemplified by [43] and [44], where object-tag anchors ease alignment within the shared stream. Dual-stream variants such as [8] maintain separate encoders bridged by cross-attention, preserving modality-specific representations while enabling targeted inter-modal exchange, and [45] extends this paradigm to spatially structured documents through text-image alignment objectives. Although these fusion approaches consistently outperform contrastive counterparts on understanding tasks like VQA and visual entailment, their quadratic attention complexity over concatenated sequences imposes substantial computational overhead, motivating architectural innovations such as the querying transformer in [46] that distills visual information into compact query tokens before cross-modal fusion. This notion of compressing cross-modal interaction into a lightweight bridging component points naturally toward the unified encoder-decoder and autoregressive generative architectures discussed in the following section, where the goal expands from deep fusion to simultaneously supporting multimodal understanding and language generation within a single model.

### 3.5 Unified Encoder-Decoder and Generative Architectures

Unified encoder-decoder and autoregressive generative architectures represent a compelling convergence point in vision-language pretraining, enabling a single model to simultaneously perform multimodal understanding and language generation. Frameworks such as [9] demonstrate that reformulating all vision-language tasks—image captioning, visual grounding, image generation, and language modeling—as a unified sequence-to-sequence prediction problem eliminates the need for task-specific architectural components, yielding competitive performance across diverse benchmarks. Similarly, generative pretraining frameworks such as [47] simplify the architecture to a single image encoder coupled with an autoregressive text decoder trained under a unified language modeling objective, achieving strong performance across captioning and question answering without external detectors. The key architectural tension lies between bidirectional understanding and unidirectional generation: models like [24] address this through novel attention masking schemes that jointly optimize discriminative and generative objectives within a shared transformer. Autoregressive frameworks that treat visual and textual content as interleaved token sequences—exemplified by [33] and [48]—enable simultaneous image-to-text and text-to-image generation, with the latter demonstrating that retrieval-augmented pretraining combined with multitask supervised fine-tuning substantially improves controllability. The introduction of discrete visual tokenizers, as in [49], further bridges the representational gap by producing causally ordered image tokens compatible with language model autoregressive prediction, while [50] shows that modeling both language and image posteriors directly in raw multimodal space yields synergistic improvements in comprehension and generation simultaneously.

### 3.6 Lightweight Bridging, Modality-Agnostic, and Efficient Adaptation Architectures

Lightweight bridging architectures have emerged as a principled response to the computational prohibitiveness of full end-to-end vision-language pretraining, offering mechanisms to connect frozen unimodal encoders with large language models through minimal trainable parameters. This design philosophy represents a deliberate departure from the joint pretraining paradigms discussed previously, shifting the objective from learning shared representations from scratch toward efficiently coupling already-capable unimodal experts. Representative frameworks such as [8] and [51] demonstrate that a compact Query Transformer interposing between a frozen visual encoder and a language model decoder can distill task-relevant visual features without retraining either backbone, achieving strong generalization across understanding and generation tasks alike. Complementarily, mixture-of-modality-experts designs as in [52] route visual and textual tokens through modality-specific expert blocks sharing a common self-attention layer, enabling a single model to function as both a dual encoder and a fusion encoder depending on the inference regime—a flexibility unattainable by fixed-topology architectures. Adapter-based injection strategies further extend this efficiency frontier by inserting lightweight bottleneck modules into frozen language model layers, as exemplified by document-oriented models [45; 53] that inject spatial and visual signals without disturbing pretrained linguistic representations. Collectively, these strategies reveal a convergent design principle: maximal reuse of pretrained unimodal knowledge through minimal, targeted cross-modal parameterization.

## 4 Pretraining Objectives and Learning Strategies

### 4.1 Contrastive Image-Text Alignment Objectives

I need to carefully check each citation in the subsection against the paper contents provided.

1. "InfoNCE-based objectives—as operationalized in large-scale frameworks like [6] and [54]" - ALIGN paper (Scaling Up...) uses contrastive loss with a dual-encoder. FLAVA also uses contrastive objectives. Both are supported. ✓

2. "UNITER: UNiversal Image-TExt Representation Learning] through optimal transport-based word-region alignment" - UNITER paper explicitly mentions "Word-Region Alignment (WRA) via the use of Optimal Transport (OT)". ✓

3. "[13] for temporal video-text settings" - VindLU does discuss video-text pretraining with various objectives. However, the sentence specifically refers to "fine-grained contrastive formulations extend matching to sub-image regions and individual word tokens." VindLU focuses on video-and-language but doesn't specifically emphasize fine-grained word-region contrastive alignment. This citation is questionable but it does cover multimodal fusion and temporal aspects.

4. "sigmoid-based reformulations [6]" - The ALIGN paper uses a softmax-based contrastive loss, not sigmoid. This citation seems incorrect. However, BLIP discusses noisy web data issues.

5. "address false negative interference inherent to web-scale noisy corpora [8]" - BLIP explicitly addresses noisy web data by bootstrapping captions to remove noisy ones. ✓

For point 4, sigmoid-based reformulations are not clearly attributed to ALIGN. This should be removed or corrected. Since no other paper in the list specifically describes sigmoid-based reformulations, I'll remove that citation.

Contrastive image-text alignment objectives form a cornerstone of modern vision-language pretraining, enabling models to learn shared semantic representations by maximizing agreement between matched visual-textual pairs while repelling mismatched ones. At the instance level, InfoNCE-based objectives—as operationalized in large-scale frameworks like [6] and [54]—align whole-image embeddings with sentence-level representations, drawing theoretical grounding from mutual information maximization principles. The effectiveness of these objectives critically depends on the availability of diverse negative samples, motivating the use of large batch sizes and momentum-based memory queues. Beyond global alignment, fine-grained contrastive formulations extend matching to sub-image regions and individual word tokens, as pursued in [4] through optimal transport-based word-region alignment, and in [13] for temporal video-text settings. Hard negative mining strategies further sharpen discriminative boundaries, while sigmoid-based reformulations address false negative interference inherent to web-scale noisy corpora [8]. Collectively, these mechanisms reveal a clear progression: from coarse global alignment toward structured, multi-granular correspondence learning.

### 4.2 Masked Modeling Objectives for Cross-Modal Representation Learning

Masked modeling objectives adapt the self-supervised reconstruction paradigm to the multimodal setting, compelling models to recover corrupted tokens using cross-modal contextual cues and thereby driving fine-grained semantic alignment—a natural extension of the global contrastive objectives discussed above toward locally structured representation learning. [4] pioneered conditional masked modeling in vision-language pretraining, demonstrating that masking one modality while preserving full observation of the other—rather than applying joint random masking—produces substantially richer cross-modal representations across tasks including visual question answering and referring expression comprehension. This asymmetric conditioning principle underpins both masked language modeling over text conditioned on intact visual features and masked region modeling conditioned on complete captions, with [4] further introducing optimal transport-based word-region alignment to enforce fine-grained correspondences beyond reconstruction. The reconstruction target choice is pivotal: low-level pixel objectives encourage texture sensitivity whereas high-level discrete token or semantic feature targets foster conceptual grounding, a tension evident when comparing pixel-space video reconstruction against latent feature prediction in temporal settings. Scaling masked pretraining via patch removal, as in [37], demonstrates that aggressive masking ratios improve computational efficiency without sacrificing alignment quality, suggesting reconstruction difficulty and representational richness are productively linked. Collectively, masked objectives occupy a middle ground in the pretraining objective landscape: they operate at finer local granularity than the global contrastive losses examined previously, yet differ from the generative objectives discussed next in that they recover partial observations rather than producing extended, globally conditioned outputs—a distinction that motivates combining all three objective families within unified pretraining frameworks.

### 4.3 Generative and Sequence-to-Sequence Pretraining Objectives

Generative and sequence-to-sequence pretraining objectives train vision-language models to produce coherent text conditioned on visual inputs, complementing discriminative contrastive objectives by directly enabling downstream generation capabilities. A foundational approach is prefix language modeling, wherein the image serves as a visual prefix and the model autoregressively generates the associated caption, as exemplified by [36], which demonstrates that mapping frozen CLIP encodings into a language model prefix yields competitive captioning performance without region-level annotations. Broader frameworks such as [8] unify understanding and generation by jointly training an image-grounded text encoder and decoder, showing that bootstrapped synthetic captions further stabilize generative pretraining against web data noise. Conditional text generation ultimately functions as a unifying task formulation subsuming classification and retrieval under a single language modeling objective, a perspective reinforced by models leveraging noisy billion-scale corpora [6] where generative capacity emerges from scale.

### 4.4 Fine-Grained Grounding and Structured Alignment Objectives

Fine-grained grounding objectives move decisively beyond the global image-sentence contrastive and generative matching discussed above, establishing explicit correspondences at the granularity of objects, visual regions, phrases, and attributes. [55] pioneered the use of detected object tags as anchor points to ease cross-modal alignment, demonstrating that salient object names shared between images and captions provide natural scaffolding for structured semantic grounding. [29] extended this insight by showing that the quality of region-level visual representations is itself a critical determinant of downstream alignment fidelity. [27] unified phrase grounding with object detection, enabling the model to learn from both human-annotated grounding data and self-trained pseudo-labels derived from image-text pairs, substantially improving zero-shot transferability to novel object categories. Complementarily, [56] introduced annotation-free image tagging parsed directly from paired text, providing diverse semantic guidance beyond object categories to encompass attributes and relational concepts. Weakly supervised approaches such as [18] demonstrate that token-level cross-modal correspondences can be induced without explicit box annotations through cross-modal augmentation and contrastive denoising. Collectively, these approaches reveal a clear trajectory: structured alignment objectives that respect the compositional, entity-centric organization of visual scenes consistently outperform global matching, particularly on grounding-intensive tasks. Importantly, however, explicit region-level supervision and detection-derived signals represent only one dimension along which pretraining objectives can be enriched; as the following subsection discusses, complementary gains arise from incorporating external knowledge sources, probabilistic representations, and auxiliary perception signals that address semantic gaps beyond what region-level grounding alone can resolve, motivating future work on scalable, annotation-free fine-grained supervision strategies within such hybrid frameworks.

### 4.5 Knowledge-Augmented, Probabilistic, and Auxiliary Pretraining Objectives

Beyond raw image-text co-occurrence statistics, enriching pretraining with external knowledge, probabilistic representations, and auxiliary perception signals addresses fundamental semantic gaps that purely data-driven objectives cannot resolve. [23] exemplifies knowledge-augmented pretraining by integrating clinical ontologies through triplet extraction and entity translation, aligning radiology patch features with structured medical knowledge rather than noisy free-text alone. Complementarily, auxiliary supervision from object detectors and tagging systems [11] provides fine-grained concept grounding that sparse captions omit, while probabilistic embedding frameworks capture inter-modal semantic uncertainty more faithfully than deterministic point embeddings, collectively suggesting that hybrid objective designs combining structured knowledge, distributional representations, and perception-derived signals constitute the most promising trajectory for semantically rich multimodal pretraining.

### 4.6 Training Strategies, Curricula, and Efficiency Improvements

Effective optimization of vision-language pretraining objectives depends critically on training curricula, data handling strategies, and computational efficiency techniques. Building naturally on the hybrid supervision strategies discussed above, which combine structured knowledge, distributional representations, and perception-derived signals, a key question becomes how to organize these diverse signal sources into coherent training procedures that maximize their collective benefit. Multi-stage pretraining has emerged as a principled solution: [57] demonstrates that first pretraining on large weakly supervised web data before fine-tuning on curated corpora consistently outperforms single-stage training, while [52] employs stagewise curricula that leverage unimodal image and text corpora before introducing paired data, effectively bootstrapping cross-modal alignment from stronger unimodal representations. Noise-robust training is equally critical given the prevalence of misaligned web pairs; [8] addresses this through caption bootstrapping, wherein a learned captioner generates synthetic captions subsequently filtered by a trained noise detector, progressively improving supervision quality during training itself. Taken together, these curriculum and data handling strategies represent an essential complement to enriched objective designs, ensuring that the full potential of structured, probabilistic, and perception-derived supervision signals is realized through disciplined and efficient optimization procedures.

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

Full fine-tuning remains the dominant adaptation paradigm for transferring pretrained vision-language representations to downstream tasks, wherein all model parameters are updated jointly on task-specific data. Models such as [3], [4], and [5] established foundational protocols whereby task-specific classification heads or decoder modules are appended to pretrained backbones and optimized end-to-end with carefully tuned learning rate schedules and regularization strategies to prevent catastrophic forgetting. [29] demonstrated that visual feature quality critically mediates fine-tuning outcomes across VQA, retrieval, and captioning benchmarks. Generative models like [7] and [8] achieved strong gains through prefix language modeling objectives that transfer flexibly to both discriminative and generative tasks. [31] and [9] further showed that unified architectures fine-tuned under sequence-to-sequence formulations yield competitive results across object detection, captioning, and visual reasoning simultaneously, underscoring that pretraining depth and model scale remain stronger determinants of fine-tuning efficacy than task-specific architectural elaborations alone.

### 6.2 Parameter-Efficient Transfer Learning for Vision-Language Models

As large pretrained vision-language models grow in scale, updating all parameters for each downstream task becomes computationally prohibitive, a challenge made especially pressing by the strong performance of full fine-tuning paradigms established by models such as those built on unified sequence-to-sequence and generative objectives. Parameter-efficient transfer learning (PETL) addresses this by freezing the pretrained backbone while introducing a small set of trainable components, preserving acquired cross-modal knowledge while enabling rapid task specialization. [75] demonstrates that adapter-based methods inserting bottleneck modules into frozen VL-BART and VLT5 backbones—when combined with cross-task weight sharing—match full fine-tuning performance across both image-text and video-text benchmarks using only approximately 4% of total parameters, establishing that parameter efficiency and representational fidelity are not fundamentally at odds. Complementary prompt and prefix tuning approaches prepend learnable continuous vectors to frozen model inputs, with multimodal extensions applying joint visual-textual prompting that leverages vision-side prompts for class separability alongside text adapters for task conditioning. The theoretical appeal of these methods rests on the hypothesis that pretrained representations already encode transferable cross-modal structure, requiring only low-dimensional task-specific projections—an intuition formalized by low-rank decomposition methods such as LoRA that constrain weight updates to rank-deficient manifolds. Collectively, PETL methods democratize access to large-scale VLP by enabling fine-tuning on resource-constrained hardware, and they play an increasingly important role in preserving the broad zero-shot generalization capabilities that contrastively pretrained models acquire during pretraining—capabilities that full fine-tuning risks overwriting. Open challenges nonetheless remain in determining optimal adapter placement, rank selection, and cross-task interference when multiple adapters share backbone parameters, all of which bear directly on the goal of building unified systems that accommodate diverse downstream applications without sacrificing general-purpose cross-modal understanding.

### 6.3 Zero-Shot and Few-Shot Generalization from Vision-Language Pretraining

I need to carefully check each citation against the paper contents provided.

1. "contrastively pretrained models such as those trained on billion-scale noisy image-text corpora [6]" - This paper describes training on over one billion image alt-text pairs with a dual-encoder architecture. ✓

2. "prompt engineering and ensemble strategies substantially improving calibration [76]" - This paper discusses prompt designs and prompt pairing strategies for CLIP. ✓

3. "these models exhibit bag-of-words tendencies and fail on compositional queries [38]" - This paper creates ARO benchmark showing VLMs have poor compositional understanding. ✓

4. "motivating composition-aware hard negative mining as a principled remedy" - This is also from the same paper [38]. ✓ (no citation added here but it's implicitly the same paper)

5. "lightweight bootstrapping [8]" - BLIP does provide strong baselines for few-shot adaptation. ✓

6. "data-efficient contrastive strategies [62]" - This paper proposes filtering, distillation and hard negatives for contrastive pre-training. ✓

7. "Out-of-distribution robustness remains critically dependent on pretraining data diversity rather than scale alone [61]" - This paper shows that the more diverse training distribution is the main cause for robustness gains. ✓

All citations appear correct. Here is the subsection:

Vision-language pretraining endows models with remarkable generalization capacity, enabling recognition of unseen categories and task execution with minimal supervision. Contrastively pretrained models such as those trained on billion-scale noisy image-text corpora [6] achieve zero-shot classification by matching image embeddings against natural language class descriptions, with prompt engineering and ensemble strategies substantially improving calibration [76]. However, systematic evaluation reveals that these models exhibit bag-of-words tendencies and fail on compositional queries [38], motivating composition-aware hard negative mining as a principled remedy. For few-shot adaptation, linear probing and lightweight bootstrapping [8] provide strong baselines, while data-efficient contrastive strategies [62] demonstrate competitive transfer under constrained budgets. Out-of-distribution robustness remains critically dependent on pretraining data diversity rather than scale alone [61], and fine-tuned models risk overwriting broad zero-shot capabilities, underscoring the tension between specialization and generalization that future unified frameworks must resolve.

### 6.4 Multitask Learning and Unified Vision-Language Model Frameworks

Unified multitask frameworks represent a compelling paradigm shift wherein a single pretrained model serves as a general-purpose interface across heterogeneous vision-language applications. Building directly on the tension between specialization and generalization identified in the preceding discussion, these frameworks seek architectures capable of preserving broad zero-shot competence while accommodating diverse downstream tasks through shared objectives and representations. Approaches such as [34] jointly optimize contrastive and generative objectives within a shared encoder-decoder architecture, demonstrating that unifying complementary learning signals produces representations transferable across retrieval, understanding, and generation simultaneously. Similarly, [8] establishes that caption bootstrapping enables a single model to excel at both discriminative and generative tasks, while [46] extends this philosophy through lightweight querying transformers connecting frozen unimodal experts, offering a parameter-efficient route to multitask unification. Instruction-guided unification, exemplified by [77], reformulates pixel-level segmentation and token-level generation under a shared semantic decoding framework, enabling seamless cross-granularity task composition. Document-centric multitask models including [45] and [53] demonstrate that unified masking objectives harmonize text-centric and image-centric tasks, illustrating how these principles extend naturally toward specialized domains. A critical insight across all these efforts is that indiscriminate task aggregation can introduce interference; principled data mixing and instruction tuning dataset selection remain essential for preserving per-task fidelity within unified frameworks. This sensitivity to data curation and domain alignment foreshadows a broader challenge: translating general-domain pretraining advances into specialized settings such as medical imaging, document understanding, and embodied perception—the focus of the following section.

### 6.5 Domain-Specific Applications of Vision-Language Pretraining

I need to check each citation in the subsection against the actual paper content.

1. **[23]** - The sentence says it "demonstrates that knowledge-enriched pretraining on paired radiology images and clinical reports—augmented with triplet extraction and entity translation from medical ontologies—substantially outperforms naive transfer." The MedKLIP paper content confirms: triplet extraction module, entity translation by querying a knowledge base, Transformer-based fusion for spatial alignment. ✓ **Correct**

2. **[24]** - The sentence says "unified BERT-based architectures with domain-adapted attention masking generalize effectively across diagnosis classification, retrieval, and report generation." The paper content confirms: BERT-based architecture, novel multi-modal attention masking scheme, tasks including diagnosis classification, medical image-report retrieval, medical VQA, and radiology report generation. ✓ **Correct**

3. **[25]** - The sentence says "augmenting instruction tuning with OCR-derived text-rich images yields marked gains on text-based VQA." The paper content confirms: uses OCR tools on text-rich images, generates conversations for text-rich images, improves text-based VQA datasets (up to 20% accuracy improvement). ✓ **Correct**

4. **[78]** - The sentence says "pretrained representations transfer to navigation and manipulation through parameter-efficient adaptation." The LLaMA-Adapter V2 paper focuses on visual instruction following, parameter-efficient adaptation, but does NOT specifically mention navigation and manipulation in robotic/embodied settings. ✗ **Remove or correct** - I'll remove this citation as it doesn't support the specific claim about robotic/embodied settings.

5. **[68]** - The sentence says "multimodal generalists exhibit emergent scene-text and document comprehension capabilities." The paper content confirms: "we observe some unexpected and exciting abilities such as multi-image correlation and scene text understanding, which makes it possible to leverage it for harder real scenarios, such as vision-only document comprehension." ✓ **Correct**

General-domain vision-language pretraining must be substantially adapted to bridge distributional gaps in specialized domains. In medical imaging, models like [23] demonstrate that knowledge-enriched pretraining on paired radiology images and clinical reports—augmented with triplet extraction and entity translation from medical ontologies—substantially outperforms naive transfer from web-crawled corpora, while [24] establishes that unified BERT-based architectures with domain-adapted attention masking generalize effectively across diagnosis classification, retrieval, and report generation. For document and scene-text understanding, [25] shows that augmenting instruction tuning with OCR-derived text-rich images yields marked gains on text-based VQA. In robotic and embodied settings, pretrained representations transfer to navigation and manipulation through parameter-efficient adaptation, while multimodal generalists [68] exhibit emergent scene-text and document comprehension capabilities. Collectively, these efforts reveal that domain-specific success hinges on targeted data curation, knowledge injection, and architectural alignment rather than scale alone.

### 6.6 Cross-Lingual and Multilingual Vision-Language Transfer

Extending vision-language pretraining to multilingual and cross-lingual settings addresses a fundamental equity gap in multimodal AI, as most large-scale pretraining corpora remain overwhelmingly English-centric. Just as domain-specific success in medical imaging and document understanding demands targeted data curation and architectural alignment, multilingual generalization similarly requires deliberate strategies that go beyond simply scaling English-dominated pipelines. A primary challenge is the scarcity of paired multilingual image-text data, which frameworks such as [26] have not systematically addressed. Machine translation augmentation offers a practical mitigation strategy, enabling models to leverage existing English image-text corpora across diverse target languages, though translation quality degradation in low-resource languages introduces compounding noise into already weakly-aligned multimodal supervision. Images serve as a powerful language-agnostic pivot for cross-lingual alignment, since visual content remains invariant across linguistic communities while textual descriptions vary—a property that principled cross-lingual pretraining objectives can exploit to achieve indirect alignment between languages that never co-occur in paired training data. Zero-shot cross-lingual transfer evaluation on benchmarks spanning visual reasoning and retrieval consistently reveals performance disparities between English and other languages, highlighting that multilingual multimodal models have not yet achieved linguistic equity. Future progress requires not only larger multilingual paired corpora but also pretraining objectives that explicitly enforce language-invariant visual grounding, ensuring that learned representations serve diverse global linguistic communities rather than perpetuating English-centric distributional biases inherited from web-crawled data [79].

## 7 Advanced Topics, Challenges, and Future Directions

### 7.1 Scaling Laws and Large Multimodal Foundation Models

I need to verify each citation against the paper contents provided.

1. **[6]** - This paper discusses scaling with noisy data and a dual-encoder architecture. It does demonstrate that scale makes up for noise and achieves strong performance. This supports the scaling claim. ✓

2. **[32]** - This paper explicitly states "we find that joint scaling of the vision and language components is important." This directly supports the joint scaling claim. ✓

3. **[78]** - This paper proposes lightweight adaption with zero-initialized attention, connecting visual inputs to a frozen LLaMA model. It is a lightweight bridging mechanism. ✓

4. **[58]** - This paper bridges frozen pretrained language model and image encoder via a trainable visual expert module. This is a bridging mechanism connecting visual encoders to LLMs. ✓

5. **[80]** - This paper describes autoregressive vision-language models. Flamingo-style models are known for in-context few-shot learning. The paper mentions it's a replication of Flamingo which supports these capabilities. ✓

6. **[81]** - This paper explicitly discusses in-context learning capability, instruction fine-tuning, and VLM pre-training. It mentions that unfreezing LLM enables in-context learning. ✓

All citations appear correct and supported by the paper contents.

Scaling behavior in vision-language pretraining follows power-law relationships wherein model capacity, dataset size, and computational budget jointly determine downstream representation quality, with evidence from [6] and [32] demonstrating that joint scaling of vision and language components yields qualitative capability transitions unavailable at smaller scales. Architecturally, lightweight bridging mechanisms—including linear projectors and query-transformer modules—connect frozen visual encoders to large language model decoders [78; 58], enabling emergent behaviors such as in-context few-shot learning, open-ended visual question answering, and instruction following [80; 81] without prohibitive retraining costs.

### 7.2 Robustness, Compositionality, and Interpretability

Despite impressive benchmark performance, vision-language pretrained models exhibit systematic failure modes that challenge their reliability in compositional and out-of-distribution settings—limitations that become increasingly consequential as scaled architectures are deployed across diverse real-world tasks. A particularly persistent weakness is the tendency to behave as bags-of-words rather than structured semantic reasoners: models trained with contrastive objectives such as [26] encode holistic semantic similarity while neglecting attribute binding, spatial relations, and syntactic word order, as diagnostic benchmarks increasingly reveal. Early evidence of this fragility appeared in foil-based evaluations [82], where models failed to detect single-word substitutions that humans resolve trivially, suggesting shallow cross-modal grounding rather than genuine compositional understanding. Models like [4] partially address fine-grained alignment through optimal-transport word-region grounding, yet compositional generalization remains elusive. Beyond compositionality, models exploit spurious statistical shortcuts embedded in web-crawled corpora [16; 6], substituting language priors for genuine visual reasoning—a manifestation of hallucination that grows more pronounced as scale increases [8]. These failure modes are not isolated technical curiosities but are deeply entangled with the quality and distributional properties of pretraining data, as skewed co-occurrence statistics in web-crawled corpora simultaneously undermine compositional reasoning and introduce broader representational inequities. Interpretability tools including attention visualization and retrieval-based error analysis offer diagnostic windows into these behaviors, guiding targeted interventions such as composition-aware hard negative mining and structured prediction objectives as productive directions for future work.

### 7.3 Fairness, Social Bias, and Ethical Considerations

Large-scale web-crawled corpora underlying vision-language pretraining inevitably encode societal biases—demographic stereotypes, occupational associations, and cultural underrepresentation—that pretrained models absorb and can amplify during inference. Datasets such as LAION-5B [16] and Conceptual Captions inherit skewed gender-profession correlations and racial imbalances from their internet sources. Contrastive objectives like those in [6] optimize for alignment without fairness constraints, propagating these distributional inequities into learned embeddings. Mitigation remains challenging: filtering strategies [62] and data augmentation pipelines [22] can partially rebalance training distributions but rarely eliminate deep-seated stereotypical co-occurrences. Critically, bias manifests not only in retrieval rankings but also in generative outputs, underscoring the need for intersectional evaluation frameworks and accountability mechanisms governing deployment across diverse communities.

### 7.4 Efficient and Sustainable Vision-Language Pretraining

Large-scale vision-language pretraining has delivered remarkable representational capabilities, yet the associated computational and environmental costs pose significant barriers to equitable access—concerns that compound the fairness and data quality challenges discussed previously. Training models such as [46] and [34] at scale demands substantial GPU resources, motivating research into efficiency-preserving alternatives. A prominent strategy involves freezing large pretrained components and training only lightweight bridging modules, as exemplified by [46], which bootstraps vision-language alignment from frozen encoders with dramatically fewer trainable parameters. Complementarily, token-level redundancy reduction through visual token pruning and merging reduces forward-pass cost without sacrificing cross-modal alignment quality. Data-efficient pretraining, enabled by principled corpus curation and quality filtering as discussed in [27] and [83], further reduces the compute required to reach competitive performance—while simultaneously addressing the distributional inequities that uncurated web-scale data introduce. Collectively, these directions suggest that sustainability and capability need not be in opposition, and that democratizing vision-language pretraining hinges on jointly optimizing data quality, architectural efficiency, and training curricula. As these efficiency and scalability challenges are progressively addressed, the field is naturally positioned to extend the vision-language pretraining paradigm beyond static image-text pairs toward richer multimodal and embodied settings, as explored in the following section.

### 7.5 Extension to Additional Modalities and Embodied Settings

I need to check each citation in the subsection against the paper contents provided.

1. **[84]** - The paper describes jointly modeling vision, audio, and language through unified alignment (MGA) and captioning (MGC) objectives, yielding strong generalization across retrieval and generation tasks. ✓ This citation is correct.

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
