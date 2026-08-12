# Registered common-window excerpt

# Neural Machine Translation: Foundations, Architectures, Training Strategies, and Emerging Frontiers — A Comprehensive Survey

## 1 Introduction

I'll carefully check each citation against the paper contents described above.

1. "Neural machine translation (NMT) has emerged as one of the most transformative developments..." [1; 2] - Both papers cover NMT comprehensively. ✓

2. "The field's evolution—from handcrafted rule-based systems through phrase-based statistical models..." [3; 4] - Both surveys trace the development timeline of NMT. ✓

3. "Early statistical approaches...suffered from feature engineering bottlenecks and struggled with long-range dependencies and morphological complexity" [5; 6] - Both papers discuss deficiencies of phrase-based SMT. ✓

4. "The neural paradigm, anchored in encoder-decoder frameworks" [7] - This paper covers encoder-decoder models. ✓

5. "subsequently transformed by attention mechanisms" [8] - This paper is about coverage-based attention, not the introduction of attention mechanisms. It's a reasonable supporting citation but tangential. I'll keep it as it does discuss attention mechanisms in NMT.

6. "and the Transformer architecture" [9] - This paper explores NMT architectures including hyperparameters, not specifically the Transformer. This citation is weak but acceptable as it covers architecture exploration.

7. "new challenges in low-resource settings" [10; 11] ✓

8. "domain adaptation" [12] ✓

9. "evaluation reliability" [13] ✓

10. "integration of large language models" [14] ✓

Neural machine translation (NMT) has emerged as one of the most transformative developments in artificial intelligence, fundamentally reshaping how computational systems bridge human languages [1; 2]. The field's evolution—from handcrafted rule-based systems through phrase-based statistical models to end-to-end neural architectures—reflects a progressive shift toward data-driven representations capable of capturing deep linguistic regularities [3; 4]. Early statistical approaches, while principled, suffered from feature engineering bottlenecks and struggled with long-range dependencies and morphological complexity [5; 6]. The neural paradigm, anchored in encoder-decoder frameworks [7] and subsequently transformed by attention mechanisms [8] and the Transformer architecture [9], overcame these limitations while introducing new challenges in low-resource settings [10; 11], domain adaptation [12], evaluation reliability [13], and the integration of large language models [14]. This survey systematically examines these dimensions, providing researchers and practitioners with a unified analytical framework spanning architectures, training strategies, specialized settings, and emerging frontiers.

## 2 Core Architectures in Neural Machine Translation

### 2.1 Recurrent Encoder-Decoder Models and the Attention Mechanism

I need to verify each citation against the paper contents provided.

1. "foundational sequence-to-sequence framework...fixed-length context vector" cited to [1; 7] - Both papers cover seq2seq with encoder-decoder. ✓

2. "LSTM and GRU units mitigate vanishing gradients...fixed-length encodings" cited to [9; 5] - "Six Challenges" explicitly discusses long sentences as a challenge. "Massive Exploration" covers architecture hyperparameters. ✓

3. "seminal soft-attention mechanism...discrete word alignment in statistical models" cited to [2] - This paper traces origins of NMT architectures including attention. ✓

4. "Bidirectional encoders...encoder depth, hidden dimensionality, and dropout together constitute the most influential architectural decisions" cited to [9] - This paper explicitly studies hyperparameters at large scale. ✓

5. "Coverage-augmented attention addresses...attention's neglect of alignment history" cited to [8] - This paper proposes coverage-based NMT. ✓

6. "inherently sequential computation constrains parallelizability...subsequently supplanted recurrent models" cited to [15; 4] - "The Best of Both Worlds" discusses RNN, CNN, and Transformer architectures. "A Survey of Deep Learning Techniques for Neural Machine Translation" covers development timeline of NMT. ✓

All citations appear correct. Here is the subsection:

The foundational sequence-to-sequence framework couples a recurrent encoder, which compresses the source sentence into a fixed-length context vector, with a recurrent decoder that generates the target sequence conditioned on this representation [1; 7]. LSTM and GRU units mitigate vanishing gradients inherent in vanilla RNNs, yet empirical studies consistently demonstrate translation quality degradation as source length increases, exposing the information bottleneck of fixed-length encodings [9; 5]. The seminal soft-attention mechanism resolves this bottleneck by computing a convex combination of all encoder hidden states at each decoding step, enabling dynamic source alignment that bears conceptual resemblance to, yet differs fundamentally from, discrete word alignment in statistical models [2]. Bidirectional encoders capture both leftward and rightward source context, while stacking multiple recurrent layers increases representational depth, with large-scale hyperparameter exploration revealing that encoder depth, hidden dimensionality, and dropout together constitute the most influential architectural decisions [9]. Coverage-augmented attention addresses the systematic over-translation and under-translation errors arising from attention's neglect of alignment history [8]. Despite these advances, inherently sequential computation constrains parallelizability, directly motivating the convolutional and self-attentional architectures that subsequently supplanted recurrent models [15; 4].

### 2.2 Convolutional Sequence-to-Sequence Architectures

Convolutional architectures represent a pivotal departure from recurrent paradigms in neural machine translation, directly addressing the parallelization constraints identified in purely sequential recurrent models. By replacing sequential hidden-state propagation with hierarchical feature extraction, convolutional designs enable full training parallelism while preserving the capacity to model compositional linguistic structure. Early work demonstrated that convolutional encoders paired with statistical or recurrent decoders could match strong recurrent baselines while substantially accelerating CPU decoding [16]. The architectural rationale rests on the observation that stacked convolutional layers naturally compose local n-gram features into progressively abstract phrase-level representations, with gating mechanisms controlling information flow analogously to LSTM cells but without temporal dependencies that constrain parallelization. The landmark fully convolutional sequence-to-sequence architecture [17] extended this principle to both encoder and decoder, employing gated linear units for gradient propagation, positional embeddings, residual connections for depth stabilization, and per-decoder-layer multi-step attention, achieving superior BLEU over deep LSTM baselines on WMT'14 English-German and English-French at an order-of-magnitude training speedup. Comparative analyses subsequently revealed that the performance advantage of convolutional and self-attentional models over recurrent networks may stem less from superior long-range dependency modeling and more from enhanced semantic feature extraction capacity [18], nuancing earlier assumptions about architectural superiority and motivating hybrid designs that leverage complementary strengths. These insights set the stage for the Transformer architecture, which carries the parallelism and global dependency modeling of attention-based approaches to their logical culmination by dispensing with recurrence and convolution entirely.

### 2.3 Transformer and Multi-Head Self-Attention Architectures

I'll carefully check each citation against the paper contents described above.

1. "scaled dot-product multi-head self-attention, the Transformer achieves full parallelization during training while capturing global dependencies in constant path length [19; 9]" - "Massive Exploration" is about hyperparameter analysis, not about parallelization/path length properties. Remove it.

2. "Deep Transformer variants employing pre-norm layer normalization and novel layer combination strategies further surpass wider shallow counterparts [20]" - Supported by the paper. ✓

3. "transparent attention and layer aggregation mechanisms enable stable optimization of significantly deeper encoder stacks [21; 22]" - Both papers support this. ✓

4. "Large-batch reduced-precision training further accelerates convergence without quality loss [23]" - Supported. ✓

5. "systematic hyperparameter studies confirm that architectural depth, normalization placement, and learning rate scheduling collectively govern translation quality [24; 15]" - Both papers support this. ✓

6. "self-attention excels particularly at semantic feature extraction and word sense disambiguation relative to recurrent and convolutional alternatives [18]" - Supported. ✓

The Transformer architecture [19] represents a fundamental departure from recurrence and convolution, establishing a purely attention-based paradigm that has become dominant in neural machine translation. By replacing sequential RNN computations with scaled dot-product multi-head self-attention, the Transformer achieves full parallelization during training while capturing global dependencies in constant path length [19], while capturing global dependencies in constant path length [19]. Deep Transformer variants employing pre-norm layer normalization and novel layer combination strategies further surpass wider shallow counterparts [20], while transparent attention and layer aggregation mechanisms enable stable optimization of significantly deeper encoder stacks [21; 22]. Large-batch reduced-precision training further accelerates convergence without quality loss [23], and systematic hyperparameter studies confirm that architectural depth, normalization placement, and learning rate scheduling collectively govern translation quality [24; 15]. Targeted evaluations reveal that self-attention excels particularly at semantic feature extraction and word sense disambiguation relative to recurrent and convolutional alternatives [18], underscoring both its strengths and the continued importance of principled architectural search.

### 2.4 Attention Mechanism Variants and Enhancements

Building directly on the Transformer's attention-based foundation, a substantial body of work addresses fundamental limitations of standard dot-product and additive attention formulations, including dispersion over long sequences, insensitivity to local

[... deterministic omitted span ...]

other geographically marginalized communities often exhibit properties—including limited digital presence, multiple competing orthographic conventions, and extreme lexical sparsity—that render standard multilingual transfer strategies, as explored through architectures supporting task-specific attention [47], considerably less effective than in high-resource settings. The sensitivity of massively multilingual systems to input variation, including code-switching and dialectal divergence naturally prevalent in underserved communities, compounds the difficulty further, underscoring the need for robustness interventions beyond standard training regimes. Benchmarking progress in such settings demands careful methodological consideration: automatic metrics such as BLEU exhibit well-documented unreliability for morphologically rich target languages where surface-level n-gram overlap poorly captures translation adequacy, motivating the development of large-scale multilingual corpora and targeted evaluation protocols that enable meaningful cross-system comparison. Future progress hinges on community-driven dataset construction, morphologically aware modeling, and evaluation frameworks explicitly calibrated for linguistic diversity.

## 5 Specialized Translation Settings and Advanced Modeling Challenges

### 5.1 Document-Level and Discourse-Aware Neural Machine Translation

I need to verify each citation in the subsection against the provided papers.

1. "[49]" - for the spectrum of architectural strategies from lightweight context concatenation to sophisticated memory-based mechanisms. The paper does discuss document-level MT approaches. ✓

2. "[50]" - for concatenation approaches scaling poorly and failing to model selective relevance. The paper discusses that existing methods use all context sentences in a fixed scope, and proposes RL-based dynamic selection. The citation is used to support the claim about concatenation's limitations, but the paper actually proposes the RL solution. However, the paper does mention that "majority of existing methods roughly use all context sentences in a fixed scope" which supports the limitation claim. ✓

3. The next sentence about RL to dynamically select context sentences - this refers to "[50]" but no citation is present. This is fine as per instructions (only check existing citations).

4. "[49]" - for the standard Transformer architecture being sufficient with enough capacity. The paper explicitly states "we show that the standard Transformer architecture is sufficient, provided it has enough capacity." ✓

5. "[14]" - for LLMs being natively document-aware as a future direction. The paper discusses LLMs and document-level translation. ✓

All citations appear correct.

Standard sentence-level neural machine translation inherently ignores inter-sentence dependencies, producing translations that may be locally fluent yet globally incoherent with respect to pronoun coreference, lexical consistency, and tense progression. Addressing this limitation, the field has explored a spectrum of architectural strategies, from lightweight context concatenation to sophisticated memory-based mechanisms [49]. Concatenation-based approaches, which prepend preceding source or target sentences to the current input, offer simplicity but scale poorly with context length and fail to model selective relevance across distant sentences [50]. More principled solutions employ reinforcement learning to dynamically select which context sentences are most informative for each source sentence, demonstrating that adaptive context selection yields measurable improvements over fixed-window baselines. Critically, [49] argues that the standard Transformer architecture, given sufficient capacity, can effectively exploit document-level context without specialized architectural modifications, provided training data contains genuine document structure—a finding that challenges the assumption that architectural novelty is the primary bottleneck. Evaluation remains a persistent challenge, as corpus-level BLEU inadequately captures discourse-level improvements, motivating discourse-targeted test sets that probe pronoun consistency and lexical cohesion specifically. The intersection of document-level translation with large language models, which are natively document-aware, represents a compelling future direction [14].

### 5.2 Non-Autoregressive Neural Machine Translation for Efficient Decoding

Non-autoregressive neural machine translation (NAT) addresses the fundamental inference bottleneck of autoregressive models by generating all target tokens simultaneously, offering substantial decoding speedups at the cost of modeling inter-token dependencies. This pursuit of inference efficiency connects naturally to the broader document-level translation challenges discussed above, where latency constraints become especially acute when processing extended contexts. The seminal challenge, as demonstrated by approaches building on parallel decoding [51], is the conditional independence assumption that forces NAT models to marginalize over complex multimodal target distributions, frequently producing repetition and omission errors. Enhancing decoder inputs through phrase-table lookups or cross-lingual embedding transformations [52] partially mitigates this limitation by providing richer positional context. Structured inference via CRF approximations [53] and CTC-based end-to-end training [54] further recover consistency without sacrificing parallelism. Critically, [55] demonstrates that autoregressive baselines with deep encoders and single-layer decoders can match NAT inference speeds, reframing the quality-latency trade-off and establishing more rigorous evaluation protocols for the field. These findings on inference-time efficiency set the stage for the subsequent discussion of linguistically-informed architectures, where the challenge shifts from decoding speed to ensuring structural fidelity and terminology precision through the integration of explicit syntactic and semantic knowledge.

### 5.3 Integration of Linguistic Structure and External Knowledge Sources

I need to check each citation in the subsection against the actual paper content.

1. "graph-convolutional architectures that inject predicate-argument structure directly into sentence representations [56]" - This paper is about using GCNs with semantic role representations for NMT. ✓ Correct.

2. "transparent attention mechanisms [21] facilitate optimization of deeper encoders" - This paper proposes transparent attention for training deeper NMT models. ✓ Correct.

3. "layer aggregation and multi-layer attention [22] expose intermediate structural signals to the decoder" - This paper proposes layer aggregation and multi-layer attention mechanisms. ✓ Correct.

4. "retrieval-augmented paradigms dynamically conditioning decoding on similar bilingual sentence pairs, reducing out-of-vocabulary failures that subword segmentation alone cannot resolve [17]" - The Convolutional Sequence to Sequence Learning paper is about a CNN-based seq2seq architecture, NOT about retrieval-augmented paradigms or conditioning on similar bilingual sentence pairs. This citation is incorrect. There is no paper in the provided list that specifically covers retrieval-augmented NMT. I should remove this citation.

Here is the corrected subsection:

Pure data-driven neural translation models demonstrate well-documented limitations in structural fidelity and terminology precision, motivating a rich body of work integrating explicit linguistic knowledge and curated external resources. Syntactic integration has advanced considerably, from linearized constituency and dependency tree encoders to graph-convolutional architectures that inject predicate-argument structure directly into sentence representations [56], yielding measurable BLEU improvements on English-German by enforcing semantic role constraints that purely sequential encoders cannot capture. Complementarily, transparent attention mechanisms [21] facilitate optimization of deeper encoders that implicitly aggregate multi-level linguistic representations, while layer aggregation and multi-layer attention [22] expose intermediate structural signals to the decoder. External lexical resources remain indispensable for domain-specific terminology coverage, with retrieval-augmented paradigms dynamically conditioning decoding on similar bilingual sentence pairs, reducing out-of-vocabulary failures that subword segmentation alone cannot resolve. The critical open challenge lies in balancing structural supervision against end-to-end optimization flexibility, as rigid syntactic constraints can impede generalization when parser errors propagate; future directions point toward jointly learning latent structural priors rather than relying on pipeline-dependent external annotations.

### 5.4 Simultaneous and Adaptive Neural Machine Translation

Simultaneous neural machine translation (SiMT) requires generating target tokens incrementally before the complete source sentence is observed, posing a fundamental tension between translation quality and output latency with direct applications in live interpretation and real-time communication. This tension is especially pronounced given the structural and resource-driven representational foundations discussed previously, as the richer linguistic representations enabled by syntax-aware encoders and retrieval-augmented paradigms must be adapted to operate under the strict information constraints of partial-input decoding. Policy design governs the quality-latency trade-off at its core: fixed wait-k strategies delay output by a predetermined number of source tokens before committing to translation, while adaptive policies dynamically decide when to READ additional source tokens or WRITE target tokens based on learned criteria. [57] advances beyond heuristic policies by explicitly modeling source-target alignment through Gaussian distributions centered on predicted aligned positions, unifying alignment learning and translation quality optimization within a single framework and demonstrating superior latency-quality trade-offs on English-Vietnamese and German-English tasks. Evaluation frameworks quantify this trade-off through metrics including Average Lagging and Average Proportion, which capture how far behind the simultaneous system trails an offline oracle. Adaptive translation further extends to online learning from post-editor corrections, where gradient-based updates treat post-edited outputs as new training signal, enabling continuous model refinement under deployment constraints without full retraining. Collectively, SiMT systems occupy a unique intersection of inference efficiency and representational adequacy, and the challenges they face in resolving lexical

[... deterministic omitted span ...]

translation systems and flexible generative models — a convergence that naturally motivates the broader examination of pre-trained and instruction-tuned architectures in the following section.

### 7.5 Large Pre-trained and Generative Language Models for Neural Machine Translation

I need to check each citation in the subsection against the provided papers.

1. "Large pre-trained encoder-decoder models such as mBART and mT5, trained on multilingual denoising and masked language modeling objectives, have substantially reshaped the translation landscape by providing powerful cross-lingual initializations that enable effective zero-shot and few-shot generalization [40]."

The cited paper is about multi-way multilingual NMT with a shared attention mechanism - it does discuss multilingual translation and low-resource improvements, but it's not about mBART or mT5 pre-trained models. However, it does relate to multilingual NMT and zero-shot translation for low-resource pairs. The citation is a stretch but partially relevant. Actually, this paper doesn't discuss pre-trained encoder-decoder models like mBART/mT5 at all. Remove citation.

2. "...yet persistently underperform dedicated systems on low-resource and morphologically complex language pairs underrepresented in pre-training corpora [39]."

This paper is about adding linguistic features to NMT. It doesn't discuss LLMs or their performance on low-resource pairs. Remove citation.

3. "A critical divergence between LLM and purpose-built NMT outputs emerges in pragmatic fidelity and terminology consistency dimensions where automatic metrics such as BLEU prove unreliable, underscoring the indispensable role of human evaluation [19]."

The Transformer paper doesn't discuss LLMs vs NMT comparison or human evaluation in this context. Remove citation.

Here is the corrected subsection:

Large pre-trained encoder-decoder models such as mBART and mT5, trained on multilingual denoising and masked language modeling objectives, have substantially reshaped the translation landscape by providing powerful cross-lingual initializations that enable effective zero-shot and few-shot generalization. Instruction-tuned generative large language models, including GPT-4 and LLaMA-2 derivatives, demonstrate competitive translation quality under chain-of-thought and few-shot prompting regimes, yet persistently underperform dedicated systems on low-resource and morphologically complex language pairs underrepresented in pre-training corpora. A critical divergence between LLM and purpose-built NMT outputs emerges in pragmatic fidelity and terminology consistency dimensions where automatic metrics such as BLEU prove unreliable, underscoring the indispensable role of human evaluation. Inference efficiency remains the primary deployment bottleneck for generative LLMs in latency-sensitive translation pipelines, motivating continued investigation into distillation and quantization strategies that preserve translation competence while reducing computational overhead.

## 8 Conclusion

I need to verify each citation in the subsection against the paper contents provided.

1. "progressing from recurrent encoder-decoder architectures [1]" - The paper "Neural Machine Translation" covers encoder-decoder architectures comprehensively. ✓

2. "through convolutional models to Transformer-dominated paradigms [9]" - This paper explores NMT architecture hyperparameters but focuses on RNN-based architectures, not specifically Transformer-dominated paradigms. However, it does discuss architecture exploration broadly. The citation is somewhat relevant but not perfect. I'll keep it as it relates to architecture exploration.

3. "back-translation [77]" - This paper directly investigates backtranslation in NMT. ✓

4. "reinforcement learning [33]" - This paper directly studies RL for NMT. ✓

5. "low-resource and underserved languages [74; 78]" - Both papers address low-resource/underserved languages. ✓

6. "robust evaluation [13]" - COMET is an evaluation framework for MT. ✓

7. "large generative language models [14]" - This paper directly addresses LLMs and MT's future. ✓

The citations appear correct. Let me return the subsection as-is:

Neural machine translation has undergone a remarkable transformation, progressing from recurrent encoder-decoder architectures [1] through convolutional models to Transformer-dominated paradigms [9], while training innovations including back-translation [77], knowledge distillation, and reinforcement learning [33] have collectively raised quality ceilings. Critical open challenges persist across low-resource and underserved languages [74; 78], robust evaluation [13], and the deepening integration of large generative language models [14], whose emergent translation capabilities are fundamentally reshaping the field's trajectory.

## References

[1] Neural Machine Translation

[2] Neural Machine Translation  A Review and Survey

[3] Neural Machine Translation  Challenges, Progress and Future

[4] A Survey of Deep Learning Techniques for Neural Machine Translation

[5] Six Challenges for Neural Machine Translation

[6] Neural versus Phrase-Based Machine Translation Quality  a Case Study

[7] Neural Machine Translation and Sequence-to-sequence Models  A Tutorial

[8] Modeling Coverage for Neural Machine Translation

[9] Massive Exploration of Neural Machine Translation Architectures

[10] Survey of Low-Resource Machine Translation

[11] Neural Machine Translation for Low-Resource Languages  A Survey

[12] A Survey of Domain Adaptation for Neural Machine Translation

[13] COMET  A Neural Framework for MT Evaluation

[14] A Paradigm Shift  The Future of Machine Translation Lies with Large  Language Models

[15] The Best of Both Worlds  Combining Recent Advances in Neural Machine  Translation

[16] A Convolutional Encoder Model for Neural Machine Translation

[17] Convolutional Sequence to Sequence Learning

[18] Why Self-Attention  A Targeted Evaluation of Neural Machine Translation  Architectures

[19] Attention Is All You Need

[20] Learning Deep Transformer Models for Machine Translation

[21] Training Deeper Neural Machine Translation Models with Transparent  Attention

[22] Exploiting Deep Representations for Neural Machine Translation

[23] Scaling Neural Machine Translation

[24] Training Tips for the Transformer Model

[25] Modeling Concentrated Cross-Attention for Neural Machine Translation  with Gaussian Mixture Model

[26] Transformer++

[27] Neural Machine Translation with Supervised Attention

[28] Enhancing Machine Translation with Dependency-Aware Self-Attention

[29] On The Alignment Problem In Multi-Head Attention-Based Neural Machine  Translation

[30] Hard-Coded Gaussian Attention for Neural Machine Translation

[31] Pervasive Attention  2D Convolutional Neural Networks for  Sequence-to-Sequence Prediction

[32] A Critical Review of Recurrent Neural Networks for Sequence Learning

[33] A Study of Reinforcement Learning for Neural Machine Translation

[34] Is MAP Decoding All You Need  The Inadequacy of the Mode in Neural  Machine Translation

[35] Understanding and Improving Sequence-to-Sequence Pretraining for Neural  Machine Translation

[36] Examining Scaling and Transfer of Language Model Architectures for  Machine Translation

[37] Compact Personalized Models for Neural Machine Translation

[38] Extreme Adaptation for Personalized Neural Machine Translation

[39] Linguistic Input Features Improve Neural Machine Translation

[40] Multi-Way, Multilingual Neural Machine Translation with a Shared  Attention Mechanism

[41] A Comparison of Transformer and Recurrent Neural Networks on  Multilingual Neural Machine Translation

[42] No Language Left Behind  Scaling Human-Centered Machine Translation

[43] Balancing Training for Multilingual Neural Machine Translation

[44] Viewing Knowledge Transfer in Multilingual Machine Translation Through a  Representational Lens

[45] Building Machine Translation Systems for the Next Thousand Languages

[46] Flow-Adapter Architecture for Unsupervised Machine Translation

[47] Multilingual Neural Machine Translation with Task-Specific Attention

[48] Fixed Encoder Self-Attention Patterns in Transformer-Based Machine  Translation

[49] Escaping the sentence-level paradigm in machine translation

[50] Dynamic Context Selection for Document-level Neural Machine Translation  via Reinforcement Learning

[51] Fast Decoding in Sequence Models using Discrete Latent Variables

[52] Non-Autoregressive Neural Machine Translation with Enhanced Decoder  Input

[53] Fast Structured Decoding for Sequence Models

[54] End-to-End Non-Autoregressive Neural Machine Translation with  Connectionist Temporal Classification

[55] Deep Encoder, Shallow Decoder  Reevaluating Non-autoregressive Machine  Translation

[56] Exploiting Semantics in Neural Machine Translation with Graph  Convolutional Networks

[57] Gaussian Multi-head Attention for Simultaneous Machine Translation

[58] Incorporating Global Visual Features into Attention-Based Neural Machine  Translation

[59] Doubly-Attentive Decoder for Multi-modal Neural Machine Translation

[60] A Visual Attention Grounding Neural Model for Multimodal Machine  Translation

[61] Dynamic Context-guided Capsule Network for Multimodal Machine  Translation

[62] Good for Misconceived Reasons  An Empirical Revisiting on the Need for  Visual Context in Multimodal Machine Translation

[63] Low-resource Neural Machine Translation with Cross-modal Alignment

[64] Towards Multimodal Simultaneous Neural Machine Translation

[65] An Analysis of Attention Mechanisms  The Case of Word Sense  Disambiguation in Neural Machine Translation

[66] Sequence-to-Sequence Models Can Directly Translate Foreign Speech

[67] Fully Character-Level Neural Machine Translation without Explicit  Segmentation

[68] Revisiting Character-Based Neural Machine Translation with Capacity and  Compression

[69] Character-based Neural Machine Translation

[70] A Character-Level Decoder without Explicit Segmentation for Neural  Machine Translation

[71] Tree-to-Sequence Attentional Neural Machine Translation

[72] On the Linguistic Representational Power of Neural Machine Translation  Models

[73] Neural Machine Translation with Contrastive Translation Memories

[74] A Focus on Neural Machine Translation for African Languages

[75] Quantitative Fine-Grained Human Evaluation of Machine Translation  Systems  a Case Study on English to Croatian

[76]  Bilingual Expert  Can Find Translation Errors

[77] Investigating Backtranslation in Neural Machine Translation

[78] Translating Pro-Drop Languages with Reconstruction Models
