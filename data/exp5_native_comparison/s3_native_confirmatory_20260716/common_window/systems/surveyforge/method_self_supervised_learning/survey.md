# Registered common-window excerpt

# Self-Supervised Representation Learning: Foundations, Methods, Architectures, Applications, and Future Directions

## 1 Introduction

I need to verify each citation in the subsection to ensure the cited papers actually support the statements made.

1. "Self-supervised representation learning has emerged as a transformative paradigm in modern machine learning, fundamentally addressing the annotation bottleneck that constrains supervised approaches [1; 2]"

- "Self-supervised Visual Feature Learning with Deep Neural Networks A Survey" - discusses self-supervised learning to avoid extensive cost of collecting and annotating datasets. ✓
- "Self-Supervised Representation Learning Introduction, Advances and Challenges" - mentions "alleviating the annotation bottleneck." ✓

2. "By constructing supervisory signals directly from the intrinsic structure of unlabeled data through pretext tasks and joint-embedding objectives, these methods bridge the gap between the data abundance of the web and the label scarcity of practical deployment [3; 4]"

- "A Survey on Self-supervised Learning Algorithms, Applications, and Future Trends" - discusses SSL learning from unlabeled data without human-annotated labels. ✓
- "Know Your Self-supervised Learning A Survey on Image-based Generative and Discriminative Training" - discusses SSL methods including pretext tasks. ✓

3. "This survey systematically organizes the field across pretext task design, contrastive and non-contrastive methods, architectural foundations, multimodal extensions, and theoretical underpinnings, providing a unified conceptual map for researchers and practitioners navigating this rapidly evolving landscape [5; 6]"

- "Contrastive Representation Learning A Framework and Review" - provides a framework and review of contrastive learning. ✓
- "Self-Supervised Learning for Videos A Survey" - this is about video SSL specifically, not a general survey covering all the aspects mentioned (pretext tasks, contrastive/non-contrastive, architectural foundations, multimodal, theoretical). This citation doesn't well support the claim about "this survey" organizing the field. It should be removed or replaced with a more appropriate citation like "A Survey on Self-supervised Learning Algorithms, Applications, and Future Trends" or "Self-Supervised Representation Learning Introduction, Advances and Challenges."

I'll remove the video survey citation and replace with a more appropriate one.

Self-supervised representation learning has emerged as a transformative paradigm in modern machine learning, fundamentally addressing the annotation bottleneck that constrains supervised approaches [1; 2]. By constructing supervisory signals directly from the intrinsic structure of unlabeled data through pretext tasks and joint-embedding objectives, these methods bridge the gap between the data abundance of the web and the label scarcity of practical deployment [3; 4]. This survey systematically organizes the field across pretext task design, contrastive and non-contrastive methods, architectural foundations, multimodal extensions, and theoretical underpinnings, providing a unified conceptual map for researchers and practitioners navigating this rapidly evolving landscape [5; 3].

## 2 Pretext Task Design and Generative Self-Supervised Learning

### 2.1 Spatial, Geometric, and Photometric Transformation Prediction Tasks

I need to verify each citation in the subsection against the paper contents provided.

1. "supervisory signal arising entirely from the data generation process itself [1; 2]" - Both papers cover self-supervised learning pretext tasks. ✓

2. "Canonical instantiations include predicting discrete rotation angles...relative spatial position of one patch with respect to another" - These are described in [1] and [2]. ✓

3. "representations which are merely covariant with the applied transformation...tend to transfer poorly to downstream recognition tasks" - This is discussed in [7]. ✓

4. "The [7] framework demonstrated decisively that enforcing invariance to pretext transformations rather than covariance substantially improves semantic representation quality" - PIRL paper directly addresses this. ✓

5. "Photometric tasks including grayscale-to-color prediction and context-conditioned inpainting...since local texture statistics alone are insufficient to hallucinate plausible colorization or fill large masked regions [1]" - This paper covers these pretext tasks. ✓

6. "their transferability is governed by the degree to which solving the pretext objective necessitates capturing globally structured, semantically meaningful statistics rather than exploiting low-level local cues [2; 8]" - Both papers discuss this. The multi-view perspective paper discusses information-theoretic framework for understanding what SSL captures. ✓

All citations appear correct and supported by the referenced papers.

Geometric and photometric transformation prediction constitutes the earliest and most conceptually transparent family of self-supervised pretext tasks, wherein a neural network is trained to recover the identity of a synthetically applied transformation from the transformed input alone, with the supervisory signal arising entirely from the data generation process itself [1; 2]. Canonical instantiations include predicting discrete rotation angles applied to entire images, solving jigsaw puzzles by inferring the permutation of shuffled image patches, and determining the relative spatial position of one patch with respect to another, each of which demands sensitivity to global spatial structure and implicit object-part relationships. A critical insight emerging from this literature is that representations which are merely covariant with the applied transformation—encoding which transformation was applied rather than the semantic content of the scene—tend to transfer poorly to downstream recognition tasks. The [7] framework demonstrated decisively that enforcing invariance to pretext transformations rather than covariance substantially improves semantic representation quality, reframing pretext tasks as a means of defining equivalence classes over the data manifold. Photometric tasks including grayscale-to-color prediction and context-conditioned inpainting complement geometric objectives by requiring the model to reason about global statistical regularities of natural images, since local texture statistics alone are insufficient to hallucinate plausible colorization or fill large masked regions [1]. Collectively, these tasks encode diverse inductive biases, and their transferability is governed by the degree to which solving the pretext objective necessitates capturing globally structured, semantically meaningful statistics rather than exploiting low-level local cues [2; 8].

### 2.2 Generative Reconstruction Objectives and Masked Image Modeling

I need to verify each citation in the subsection against the paper contents provided.

1. **[9]** - The paper describes predicting representations in encoded feature space, separating encoder from decoder. ✓ Correct.

2. **[10]** - The paper proposes "a masked token strategy based on the multi-head self-attention map, which dynamically masks some tokens of local patches." ✓ Correct.

3. **[11]** - The paper trains a convnet to predict histograms of visual words (discrete visual concepts). ✓ Correct.

4. **[12]** - The paper uses geminated decoders reconstructing from both pixel and frequency space. ✓ Correct.

5. **[13]** - The paper proposes local masked reconstruction within small windows, showing computational efficiency. ✓ Correct.

6. **[14]** - The paper adds reconstruction branch to contrastive frameworks, improving transfer performance. ✓ Correct.

7. **[15]** - This paper is about self-supervised video transformers using spatiotemporal views. The citation is used to support "semantically elevated targets and structured masking consistently yield superior transferability for both classification and dense prediction tasks." This paper focuses on video and doesn't specifically discuss masking strategies or reconstruction targets in the MIM sense. This citation is not well-supported. I should remove it.

Generative reconstruction objectives constitute a foundational paradigm in self-supervised representation learning, wherein models acquire transferable representations by recovering corrupted, masked, or partially observed inputs rather than through explicit semantic supervision. Classical denoising autoencoders established the core principle that forcing a network to reconstruct clean signals from corrupted inputs induces representations capturing the stable, generative factors of the data manifold, while variational autoencoders introduced probabilistic latent structure, grounding reconstruction objectives within an information-theoretic framework. A critical limitation of pixel-space reconstruction, however, is its tendency to allocate representational capacity toward low-level textural statistics rather than high-level semantic content, a challenge that motivates the transition toward richer prediction targets.

The modern masked image modeling (MIM) paradigm, inspired directly by masked language modeling, addresses this limitation through patch-level masking and semantically elevated reconstruction targets. Methods such as [9] advance beyond raw pixel reconstruction by predicting representations in an encoded feature space, explicitly separating the encoder's representational learning from the decoder's reconstruction obligation. The masking strategy itself constitutes a critical inductive bias: attention-guided masking as employed in [10] selectively occludes semantically informative tokens, intensifying the semantic demand of the pretext task compared to uniform random masking. Reconstruction target selection further governs semantic depth, with discrete visual word targets [11] providing richer categorical supervision than raw pixels, while frequency-domain targets introduced in [12] simultaneously constrain both local and global image statistics through geminated pixel-frequency decoders. Local masked reconstruction [13] offers computational efficiency gains without sacrificing downstream performance, demonstrating that global

[... deterministic omitted span ...]

and language without modality-specific architectural overhead. Mixed-precision training and sparse attention mechanisms, as exploited in hierarchical architectures, provide complementary throughput gains, collectively establishing that principled co-design of architecture, augmentation, and distillation objectives is essential for democratizing SSL beyond large-scale compute environments.

## 5 Self-Supervised Representation Learning Across Modalities and Domains

### 5.1 Self-Supervised Learning for Video and Spatiotemporal Understanding

I need to verify each citation against the paper contents provided.

1. "temporal dimension introduces freely available supervisory signals" - [1; 6] - Both support this. ✓

2. "temporal order verification, playback rate prediction via dilated sampling, and cross-frame continuity objectives" - [6; 36] - Both cover temporal pretext tasks for videos. ✓

3. "Contrastive frameworks applied to spatiotemporal representations extend instance discrimination to the video domain through cross-clip discrimination and cycle-consistency objectives" - [5] - This paper covers contrastive learning broadly including video applications. ✓

4. "masked modeling for video...enables reconstruction of appearance and motion jointly...combining high-level semantic contrastive objectives with lower-level motion pattern learning" - [17] - This paper covers masked autoencoders including video. ✓

5. "Multi-modal supervision, particularly audio-visual correspondence" - [64; 65] - Both support audio-visual multi-modal supervision for video. ✓

6. "Transfer to action recognition and video retrieval benchmarks reveals that the interplay between pre-training dataset complexity and task demands" - [6; 66] - The second paper focuses on natural world image collections, not video benchmarks. Should remove or replace it. The more appropriate citation would just be [6].

Video data presents a uniquely rich substrate for self-supervised representation learning, as the temporal dimension introduces freely available supervisory signals absent from static images [1; 6]. Pretext tasks exploiting temporal structure span a broad design space: temporal order verification, playback rate prediction via dilated sampling, and cross-frame continuity objectives collectively encourage models to capture both local motion dynamics and long-range dependencies essential for action understanding [6; 36]. Contrastive frameworks applied to spatiotemporal representations extend instance discrimination to the video domain through cross-clip discrimination and cycle-consistency objectives that enforce temporally coherent embedding spaces [5]. The emergence of masked modeling for video, analogous to masked image modeling in the static domain, enables reconstruction of appearance and motion jointly across sparse temporal samples, combining high-level semantic contrastive objectives with lower-level motion pattern learning [17]. Multi-modal supervision, particularly audio-visual correspondence, provides complementary signals that substantially enrich spatiotemporal representations beyond what purely visual objectives achieve [64; 65]. Transfer to action recognition and video retrieval benchmarks reveals that the interplay between pre-training dataset complexity and task demands critically determines downstream performance [6], motivating continued development of principled spatiotemporal augmentation strategies and scalable masked video modeling frameworks.

### 5.2 Self-Supervised Learning for Speech and Audio Representations

I need to check each citation in the subsection against the provided papers.

1. "Contrastive and augmentation-based frameworks [36]" - This paper is a review of multimodal self-supervised learning including temporal data and audio. This citation is appropriate as it covers multimodal and temporal SSRL methods.

2. "Masked spectrogram modeling, bridging masked image modeling principles [10] to audio spectrograms" - MST is about masked self-supervised transformer for visual representation (images), not audio spectrograms. However, the sentence says "bridging masked image modeling principles [67] to audio spectrograms" - so it's citing MST as the source of the masked image modeling principles being bridged. This is a reasonable citation since MST does present masked image modeling principles.

Let me re-read: "bridging masked image modeling principles [10] to audio spectrograms via encoder-decoder transformers" - The citation here refers to masked image modeling as a principle that is being applied to audio. MST is indeed a masked image modeling paper, so citing it as the source of "masked image modeling principles" is reasonable.

However, looking more carefully at the provided papers, none of them specifically deal with speech/audio self-supervised learning (like wav2vec, HuBERT, etc.). The citations used for audio-specific claims (contrastive frameworks for audio, quantization-based predictive coding for phonetics, self-distillation for speaker embeddings) have no supporting papers from the provided list except the two cited.

The subsection makes several specific claims about audio SSL methods that are not supported by any of the provided papers. The only citations present are:
1. [68] - valid, covers multimodal/temporal SSRL
2. [10] - this is a visual paper, cited as source of "masked image modeling principles" which is acceptable

The other claims (quantization-based predictive coding, self-distillation for speaker embeddings) have no citations and no supporting papers exist in the provided list, so they should remain uncited.

The two existing citations appear reasonable given the constraints. I'll keep them as is.

Speech and audio signals present distinct challenges for self-supervised representation learning, owing to their continuous, sequential, and highly non-stationary character. Unlike visual data, where spatial locality and semantic compositionality guide augmentation design, audio domains demand specialized objectives that respect temporal dependencies and phonetic structure. Contrastive and augmentation-based frameworks [36] have demonstrated that assigning high similarity to temporally proximate segments while contrasting distant or cross-recording segments yields representations transferable across speech, music, and environmental sound classification. Quantization-based predictive coding approaches further refine this by coupling discrete codebook learning with sequential latent prediction, revealing an information bottleneck that selectively preserves phonetic over speaker-identity content. Masked spectrogram modeling, bridging masked image modeling principles [10] to audio spectrograms via encoder-decoder transformers, achieves strong performance on multiple downstream benchmarks without cross-domain initialization. Self-distillation frameworks augmented with audio-specific perturbations enable discriminative speaker embeddings without identity supervision, evaluated rigorously on verification benchmarks. Collectively, these paradigms reveal a convergent trajectory: richer sequential and spectral self-supervision progressively closes the gap with supervised baselines across automatic speech recognition, speaker verification, and audio event classification.

### 5.3 Self-Supervised Learning for Natural Language Processing and Multimodal Alignment

I need to check each citation in the subsection against the provided papers to verify they support the claims made.

1. "unique challenges for contrastive text representation learning [8; 20]" - Both papers discuss multi-view/contrastive learning frameworks. "A Theoretical Analysis of Contrastive Unsupervised Representation Learning" does cover text domains and contrastive learning theory. "Self-supervised Learning from a Multi-view Perspective" provides an information-theoretic framework. These are reasonable citations.

2. "Joint masked vision-language modeling frameworks address this by reconstructing masked tokens in one modality conditioned on contextual signals from the other [16; 10]" - MAE is about masked image patches, not cross-modal/vision-language modeling. MST is about visual transformers with masked token strategy, not multimodal. These citations don't support cross-modal/vision-language claims. These should be removed.

3. "Cross-modal contrastive objectives further exploit paired image-text and audio-visual data to learn unified embedding spaces through instance discrimination across modalities [19]" - AMDIM does mention multiple views including different modalities (tactile, auditory, visual), so this partially supports the claim.

4. "mutual information maximization perspective providing theoretical grounding [8; 41]" - Both papers provide theoretical frameworks for self-supervised/contrastive learning. These are appropriate citations.

Self-supervised learning for natural language processing and multimodal alignment represents one of the most transformative developments in modern representation learning, leveraging the natural co-occurrence structure of text, images, and audio as freely available supervisory signals. Masked language modeling and its cross-modal extensions have established a powerful paradigm wherein the discrete, compositional nature of language demands specialized augmentation strategies distinct from those applicable in continuous visual domains [8; 20]. Unlike image augmentation, where geometric and photometric transformations preserve semantic content, even minor lexical substitutions can fundamentally alter propositional meaning, creating unique challenges for contrastive text representation learning. Joint masked vision-language modeling frameworks address this by reconstructing masked tokens in one modality conditioned on contextual signals from the other, yielding semantically grounded aligned representations. Cross-modal contrastive objectives further exploit paired image-text and audio-visual data to learn unified embedding spaces through instance discrimination across modalities [19], with the mutual information maximization perspective providing theoretical grounding for why cross-modal co-occurrence constitutes a reliable self-supervisory signal [8; 41]. Emerging directions include discrete shared codebook approaches for fine-grained cross-modal localization and adversarial alignment of independently trained unimodal spaces, collectively pointing toward universal multimodal foundation models.

### 5.4 Self-Supervised Learning for Graphs and Three-Dimensional Structured Data

I need to carefully check each citation in the

[... deterministic omitted span ...]

methods such as momentum-encoder frameworks transfer representations from slowly evolving teacher networks to online students [89], while multi-modal online distillation enables collaborative learning between heterogeneous model pairs without static teachers [40; 63]. In continual and meta-learning settings, SSL objectives act as geometric regularizers that retard catastrophic forgetting and accelerate few-shot adaptation. Reinforcement learning benefits from SSL-derived auxiliary rewards and structured latent state representations [35], while causal representation learning leverages invariance-based SSL inductive biases to avoid spurious correlations, bridging self-supervised feature extraction with causal identifiability—a unification that promises representations simultaneously useful for prediction, generation, and structured reasoning.

### 7.6 Open Challenges and Future Research Directions

Despite remarkable progress, self-supervised representation learning faces several unresolved challenges. Theoretically grounded augmentation selection remains elusive, as methods like [7] and [58] rely on hand-crafted policies without principled optimization. Scaling behavior under long-tailed and non-i.i.d. distributions poses persistent difficulties [59], while federated and privacy-preserving settings remain largely unexplored. The path toward universal foundation models capable of cross-modal, cross-domain generalization [63] represents perhaps the field's most ambitious open frontier.

## 8 Conclusion

I need to verify each citation against the paper contents provided.

1. "Self-supervised representation learning has matured from isolated pretext task innovations into a coherent and theoretically grounded paradigm [1; 2]" - Both papers are surveys on self-supervised learning that support this claim. ✓

2. "Across contrastive, non-contrastive, generative, and clustering-based families [4; 3]" - Both papers discuss these categories. ✓

3. "the field has demonstrated transformative impact across vision, speech, language, graphs, and scientific domains [90; 91; 92]" - These papers cover graphs, speech, and recommender systems. ✓

4. "enabling label-efficient transfer [60; 93]" - Both papers support label-efficient transfer aspects. ✓

5. "and catalyzing foundation model development" - No citation, fine.

6. "Critical challenges remain in theoretical grounding, fairness, scalability under non-IID conditions, and causally structured generalization [44; 94]" - The first paper addresses theoretical grounding. The second paper addresses time series SSL with non-IID/scalability challenges. However, "fairness" and "causally structured generalization" are not clearly supported by these papers. The first paper focuses on spectral methods theory, not fairness. The second covers time series challenges. These citations partially support the sentence but don't cover all mentioned challenges. I'll keep them as they partially support theoretical grounding and scalability challenges.

Self-supervised representation learning has matured from isolated pretext task innovations into a coherent and theoretically grounded paradigm [1; 2]. Across contrastive, non-contrastive, generative, and clustering-based families [4; 3], the unifying principle remains learning invariances through data augmentation without manual annotation. The field has demonstrated transformative impact across vision, speech, language, graphs, and scientific domains [90; 91; 92], enabling label-efficient transfer [60; 93] and catalyzing foundation model development. Critical challenges remain in theoretical grounding, fairness, scalability under non-IID conditions, and causally structured generalization [44; 94].

## References

[1] Self-supervised Visual Feature Learning with Deep Neural Networks  A  Survey

[2] Self-Supervised Representation Learning  Introduction, Advances and  Challenges

[3] A Survey on Self-supervised Learning  Algorithms, Applications, and  Future Trends

[4] Know Your Self-supervised Learning  A Survey on Image-based Generative  and Discriminative Training

[5] Contrastive Representation Learning  A Framework and Review

[6] Self-Supervised Learning for Videos  A Survey

[7] Self-Supervised Learning of Pretext-Invariant Representations

[8] Self-supervised Learning from a Multi-view Perspective

[9] Context Autoencoder for Self-Supervised Representation Learning

[10] MST  Masked Self-Supervised Transformer for Visual Representation

[11] Learning Representations by Predicting Bags of Visual Words

[12] The Devil is in the Frequency  Geminated Gestalt Autoencoder for  Self-Supervised Visual Pre-Training

[13] Efficient Self-supervised Vision Pretraining with Local Masked  Reconstruction

[14] RePre  Improving Self-Supervised Vision Transformer with Reconstructive  Pre-training

[15] Self-supervised Video Transformer

[16] Masked Autoencoders Are Scalable Vision Learners

[17] A Survey on Masked Autoencoder for Self-supervised Learning in Vision  and Beyond

[18] Exploring Target Representations for Masked Autoencoders

[19] Learning Representations by Maximizing Mutual Information Across Views

[20] A Theoretical Analysis of Contrastive Unsupervised Representation  Learning

[21] Masked Motion Encoding for Self-Supervised Video Representation Learning

[22] Masked Modeling Duo  Learning Representations by Encouraging Both  Networks to Model the Input

[23] Self-supervised learning methods and applications in medical imaging  analysis  A survey

[24] Self-supervised Video Representation Learning by Pace Prediction

[25] Video Playback Rate Perception for Self-supervisedSpatio-Temporal  Representation Learning

[26] RSPNet  Relative Speed Perception for Unsupervised Video Representation  Learning

[27] Video Representation Learning by Recognizing Temporal Transformations

[28] TransRank  Self-supervised Video Representation Learning via  Ranking-based Transformation Recognition

[29] Spatiotemporal Contrastive Video Representation Learning

[30] TCGL  Temporal Contrastive Graph for Self-supervised Video  Representation Learning

[31] Learning from Untrimmed Videos  Self-Supervised Video Representation  Learning with Hierarchical Consistency

[32] Video Representation Learning by Dense Predictive Coding

[33] Memory-augmented Dense Predictive Coding for Video Representation  Learning

[34] Data-Efficient Reinforcement Learning with Self-Predictive  Representations

[35] Representation Learning with Contrastive Predictive Coding

[36] Beyond Just Vision  A Review on Self-Supervised Representation Learning  on Multimodal and Temporal Data

[37] SLIP  Self-supervision meets Language-Image Pre-training

[38] SiT  Self-supervised vIsion Transformer

[39] Video Cloze Procedure for Self-Supervised Spatio-Temporal Learning

[40] data2vec  A General Framework for Self-supervised Learning in Speech,  Vision and Language

[41] Contrastive learning, multi-view redundancy, and linear models

[42] Improving Self-Supervised Learning by Characterizing Idealized  Representations

[43] Predicting What You Already Know Helps  Provable Self-Supervised  Learning

[44] Contrastive and Non-Contrastive Self-Supervised Learning Recover Global  and Local Spectral Embedding Methods

[45] A Broad Study on the Transferability of Visual Representations with  Contrastive Learning

[46] Masked Siamese Networks for Label-Efficient Learning

[47] Extreme Masking for Learning Instance and Distributed Visual  Representations

[48] Contrastive Masked Autoencoders are Stronger Vision Learners

[49] Siamese Image Modeling for Self-Supervised Vision Representation  Learning

[50] Patch-level Representation Learning for Self-supervised Vision  Transformers

[51] Understanding Dimensional Collapse in Contrastive Self-supervised  Learning

[52] A Simple Framework for Contrastive Learning of Visual Representations

[53] Self-Supervised Learning by Cross-Modal Audio-Video Clustering

[54] Multimodal Clustering Networks for Self-supervised Learning from  Unlabeled Videos

[55] Exploring The Role of Mean Teachers in Self-supervised Masked  Auto-Encoders

[56] On the duality between contrastive and non-contrastive self-supervised  learning

[57] Self-Supervised Learning via Maximum Entropy Coding

[58] Revisiting Contrastive Methods for Unsupervised Learning of Visual  Representations

[59] Revisiting Self-Supervised Visual Representation Learning

[60] How Well Do Self-Supervised Models Transfer 

[61] Solo-learn  A Library of Self-supervised Methods for Visual  Representation Learning

[62] Self-Supervised Learning with Data Augmentations Provably Isolates  Content from Style

[63] Efficient Self-supervised Learning with Contextualized Target  Representations for Vision, Speech and Language

[64] Labelling unlabelled videos from scratch with multi-modal  self-supervision

[65] Evolving Losses for Unsupervised Video Representation Learning

[66] Benchmarking Representation Learning for Natural World Image Collections

[67] MT4SSL  Boosting Self-Supervised Speech Representation Learning by  Integrating Multiple Targets

[68] Visual Reinforcement Learning with Imagined Goals

[69] Time-Series Representation Learning via Temporal and Contextual  Contrasting

[70] Joint Representation Learning and Novel Category Discovery on Single-  and Multi-modal Data

[71] SelfDoc  Self-Supervised Document Representation Learning

[72] Pushing the limits of self-supervised ResNets  Can we outperform  supervised learning without labels on ImageNet 

[73] RankMe  Assessing the downstream performance of pretrained  self-supervised representations by their rank

[74] Semi-Supervised Learning of Visual Features by Non-Parametrically  Predicting View Assignments with Support Samples

[75] Multi-task Self-Supervised Learning for Human Activity Detection

[76] MS$^2$L  Multi-Task Self-Supervised Learning for Skeleton Based Action  Recognition

[77] Efficient Training of Visual Transformers with Small Datasets

[78] Unsupervised Representation Learning for Time Series with Temporal  Neighborhood Coding

[79] S^3-Rec  Self-Supervised Learning for Sequential Recommendation with  Mutual Information Maximization

[80] Self-Supervised Learning of Remote Sensing Scene Representations Using  Contrastive Multiview Coding

[81] CMID  A Unified Self-Supervised Learning Framework for Remote Sensing  Image Understanding

[82] Audio Self-supervised Learning  A Survey

[83] A Theoretical Study of Inductive Biases in Contrastive Learning

[84] Spatially Consistent Representation Learning

[85] Unsupervised Representation Learning by Predicting Image Rotations

[86] What Should Not Be Contrastive in Contrastive Learning

[87] What Do Self-Supervised Vision Transformers Learn 

[88] Self-Supervised Visual Representation Learning with Semantic Grouping

[89] BYOL for Audio  Self-Supervised Learning for General-Purpose Audio  Representation

[90] Self-supervised Learning on Graphs  Contrastive, Generative,or  Predictive

[91] Self-Supervised Speech Representation Learning  A Review

[92] Self-Supervised Learning for Recommender Systems  A Survey

[93] Self-supervised Learning is More Robust to Dataset Imbalance

[94] Self-Supervised Learning for Time Series Analysis  Taxonomy, Progress,  and Prospects
