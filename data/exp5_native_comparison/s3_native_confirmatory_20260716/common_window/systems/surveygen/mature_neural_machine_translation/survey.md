# Registered common-window excerpt

# Neural Machine Translation: A Comprehensive Survey of Architectures, Training Strategies, and Applications

## Foundations and Historical Background of Neural Machine Translation

### From Statistical to Neural Machine Translation

Phrase-based statistical machine translation (SMT) dominated the field for decades, decomposing translation into learned phrase mappings, language models, and log-linear reranking. Despite strong empirical performance, SMT systems suffered from error propagation across independently trained components, limited generalization to unseen phrases, and an inability to capture long-range syntactic dependencies. The transition to neural approaches was motivated by the promise of end-to-end learning over continuous representations. Sutskever et al. ref [6] demonstrated that a multilayered LSTM sequence-to-sequence model could surpass a competitive phrase-based SMT baseline on WMT'14 English-French, while Cho et al. ref [3] introduced the RNN Encoder-Decoder framework that would become foundational to subsequent NMT research, collectively establishing the neural paradigm.

### Early Neural Language Models and Continuous Representations

A pivotal precursor to modern NMT was the introduction of continuous representations for words, phrases, and sentences. Kalchbrenner and Blunsom's Recurrent Continuous Translation Models (RCTMs) ref [19] demonstrated that purely continuous, alignment-free representations could model translation probabilistically, achieving substantially lower perplexity than alignment-based systems while remaining sensitive to source word order and syntax. Concurrently, Cho et al. ref [3] proposed the RNN Encoder-Decoder framework, learning continuous phrase representations that significantly improved phrase-based SMT scoring. These works collectively established that dense, distributed representations could capture linguistic structure more flexibly than discrete symbolic approaches, directly motivating the fully neural sequence-to-sequence paradigms that followed.

### The Encoder-Decoder Paradigm

The encoder-decoder paradigm constitutes the foundational architecture of modern NMT. Cho et al. introduced the RNN Encoder-Decoder framework ref [3], wherein an encoder recurrent network compresses a variable-length source sentence into a fixed-dimensional context vector, from which a decoder network autoregressively generates the target sequence. Concurrently, Sutskever et al. ref [6] proposed sequence-to-sequence learning using multilayered LSTMs, achieving competitive performance against phrase-based SMT on WMT'14 English-French translation. A critical limitation of this paradigm—the information bottleneck imposed by the fixed-length vector—was subsequently identified ref [10], motivating the development of attention mechanisms. Empirical analysis further confirmed performance degradation on longer sentences ref [16], underscoring the necessity of architectural extensions beyond the basic encoder-decoder formulation.

### Long Short-Term Memory Networks in NMT

Long Short-Term Memory (LSTM) networks, originally introduced by Hochreiter and Schmidhuber ref [8], addressed the vanishing gradient problem inherent in standard recurrent networks by enforcing constant error flow through gated memory cells. This architectural innovation proved pivotal for NMT, enabling the modeling of long-range dependencies across extended translation sequences. Sutskever et al. ref [6] demonstrated that a multilayered LSTM encoder-decoder achieved a BLEU score of 34.8 on WMT'14 English-French, surpassing phrase-based SMT baselines. Notably, reversing source sentence word order substantially improved performance by shortening temporal dependencies during optimization. Google's production NMT system subsequently scaled this approach to deep eight-layer LSTM networks with residual connections ref [17], achieving near-human translation quality.

### Properties and Early Analysis of NMT Systems

Early empirical analysis of NMT systems revealed critical behavioral properties that shaped subsequent research directions. Cho et al. ref [16] systematically examined encoder-decoder models, demonstrating that NMT performs competitively on short sentences but degrades rapidly as sentence length and unknown word count increase, attributing this to the bottleneck of compressing variable-length inputs into fixed-dimensional vectors. Sutskever et al. ref [6] corroborated strong performance on longer sequences when using LSTMs, notably finding that reversing source word order substantially improved translation quality by reducing temporal dependencies. These analyses collectively identified vocabulary coverage and long-sentence modeling as fundamental challenges, directly motivating subsequent advances in attention mechanisms ref [10] and subword segmentation ref [49].

## Core Architectures for Neural Machine Translation

### Recurrent Neural Network-Based Models

Recurrent Neural Network-based encoder-decoder architectures constitute the foundational paradigm of early NMT systems. Cho et al. introduced the RNN Encoder-Decoder framework employing Gated Recurrent Units (GRUs), compressing source sentences into fixed-length vectors for decoding ref [3]. Sutskever et al. extended this with multilayered LSTMs ref [6], achieving competitive performance against phrase-based SMT on WMT'14 English-French. A critical limitation of fixed-length context vectors was identified by Cho et al. ref [16], demonstrating performance degradation on longer sentences. Bahdanau's attention mechanism ref [10] and Luong's global and local attention variants ref [21] substantially mitigated this bottleneck. Google's GNMT system ref [17] further advanced RNN-based NMT through deep stacked LSTMs with residual connections and wordpiece segmentation.

### Attention Mechanisms

Attention mechanisms represent a pivotal advancement in NMT, addressing the bottleneck imposed by fixed-length context vectors in vanilla encoder-decoder architectures. Bahdanau et al. introduced soft-alignment attention, enabling the decoder to dynamically attend to relevant source positions when generating each target word ref [10]. Luong et al. subsequently proposed global and local attention variants: the global approach attends to all source words, while local attention restricts focus to a subset, yielding a 5.0 BLEU improvement over non-attentional baselines ref [21]. To mitigate over- and under-translation arising from neglecting alignment history, Tu et al. proposed coverage-based attention, maintaining a coverage vector that tracks previously attended source words ref [25].

### Convolutional Sequence-to-Sequence Models

Convolutional sequence-to-sequence models represent a significant architectural departure from recurrent approaches by replacing sequential computation with fully convolutional operations. The ConvS2S architecture ref [12] introduced an encoder-decoder framework built entirely on convolutional neural networks, enabling complete parallelization of computations over all input elements during training. This design yields substantially faster training compared to recurrent models, as the number of non-linearities remains fixed and independent of sequence length. ConvS2S employs gated linear units to facilitate gradient propagation and equips each decoder layer with a dedicated attention module. Empirically, ConvS2S surpassed the deep LSTM system of Wu et al. on WMT'14 English-German and English-French benchmarks at an order of magnitude faster speed on both GPU and CPU ref [35].

### The Transformer Architecture

The Transformer architecture, introduced by Vaswani et al., represents a paradigm shift in NMT by dispensing entirely with recurrence and convolutions in favor of self-attention mechanisms. Its encoder-decoder structure employs multi-head attention, enabling the model to jointly attend to information from different representation subspaces at different positions. Positional encodings substitute for sequential processing, preserving word-order information. Feed-forward sublayers with residual connections and layer normalization ensure stable training of deep networks. The Transformer achieves superior parallelization compared to RNN-based ref [3] and convolutional models ref [12], yielding state-of-the-art results on WMT benchmarks. Its architecture has since become the dominant foundation for NMT research, including multilingual ref [32] and pre-trained models ref [27].

### Deep and Enhanced Attention Models

Beyond standard attention mechanisms, researchers have proposed deep and enhanced attention architectures to further improve translation fidelity. The DeepAtt model ref [56] extends attention to multiple layers, enabling the model to automatically determine what information should be passed or suppressed from corresponding encoder layers, yielding more accurate attention weights and improved faithfulness in translation. Complementarily, the GRU-gated attention model (GAtt) ref [58] addresses the problem of non-discriminatory context vectors by dynamically updating source representations with the previous decoder state via a GRU, producing translation-sensitive representations that mitigate over-translation. Together, these approaches demonstrate that deepening and dynamically conditioning attention mechanisms substantially enhances NMT performance across multiple language pairs.

### Hybrid and Combined Architectures

Hybrid and combined architectures seek to integrate complementary strengths of RNN, CNN, and Transformer paradigms to achieve superior translation performance. Chen et al. ref [20] systematically investigated combining advances from multiple NMT architectures, demonstrating that selectively merging components—such as convolutional source encoding with Transformer-style attention—yields consistent improvements over individual architectures. Google's GNMT ref [17] exemplifies practical hybridization, employing deep LSTM layers augmented with attention and residual connections to balance accuracy and computational efficiency. Such combinations leverage the local feature extraction of CNNs ref [12], the sequential modeling capacity of RNNs ref [3], and the global dependency modeling of Transformers, collectively advancing translation fidelity beyond what any single architecture achieves independently.

## Training Strategies and Optimization for NMT

### Maximum Likelihood Estimation and Its Limitations

Maximum likelihood estimation (MLE) serves as the standard training objective for NMT, where models are trained to maximize the conditional probability

[... deterministic omitted span ...]

strategies generating synthetic sentence pairs containing rare words in novel contexts further improve translation quality ref [53]. Collectively, these methods address vulnerability to noisy perturbations and data scarcity, substantially improving model generalization across diverse translation conditions.

## Data Augmentation and the Use of Monolingual Data

### Back-Translation

Back-translation is a pivotal data augmentation technique that leverages target-side monolingual data to improve NMT performance. Sennrich et al. introduced the foundational approach of pairing monolingual target sentences with automatic back-translations to generate synthetic parallel data, yielding substantial BLEU improvements on WMT English-German and low-resource settings ref [57]. Subsequent work by Edunov et al. broadened understanding of back-translation at scale, demonstrating that synthetic source sentences generated via sampling or noised beam search provide stronger training signals than deterministic beam or greedy decoding ref [37]. Scaling to hundreds of millions of monolingual sentences, this approach achieved state-of-the-art results on WMT'14 English-German, establishing back-translation as an indispensable component of modern NMT pipelines.

### Dual Learning for Machine Translation

Dual learning exploits the inherent duality between translation directions—e.g., English-to-French and French-to-English—to extract informative training signals from unlabeled monolingual data ref [9]. In the dual-learning framework, two translation agents engage in a reinforcement learning game: one agent translates a source sentence into the target language, while the other reconstructs the original, generating reward signals based on language model likelihood and reconstruction fidelity ref [9]. These feedback signals enable iterative model updates via policy gradient methods without human annotation. Empirically, dual-NMT trained with only 10% bilingual data achieves performance comparable to fully supervised NMT on English↔French translation, demonstrating substantial data efficiency gains ref [9].

### Unsupervised Neural Machine Translation

Unsupervised Neural Machine Translation (UNMT) addresses translation without any parallel data, relying solely on monolingual corpora. Foundational approaches leverage denoising autoencoders and iterative back-translation to bootstrap cross-lingual representations ref [31]. Lample et al. demonstrated that careful parameter initialization, language model denoising, and iterative back-translation yield competitive performance, achieving 28.1 BLEU on WMT'14 English-French without parallel sentences ref [31]. Cross-lingual language model pretraining, as in XLM, further strengthens unsupervised translation by initializing models with shared multilingual representations ref [38]. Similarly, MASS employs masked sequence-to-sequence pre-training to jointly train encoder-decoder models, achieving state-of-the-art unsupervised translation results ref [42]. mBART extends these gains through multilingual denoising pre-training across many languages ref [27].

### Pre-trained Language Models for NMT

Pre-trained language models have substantially advanced NMT, particularly in low-resource and unsupervised settings. BERT ref [46] introduced deep bidirectional representations that can initialize NMT encoders, improving translation quality. XLM ref [38] extended cross-lingual pretraining using both unsupervised monolingual and supervised parallel objectives, achieving state-of-the-art results on unsupervised translation benchmarks. MASS ref [42] proposed masked sequence-to-sequence pretraining that jointly trains encoder and decoder components, yielding significant gains on low-resource and zero-resource translation tasks. mBART ref [27] further advanced this paradigm by pretraining a complete sequence-to-sequence denoising autoencoder on large multilingual monolingual corpora, enabling direct fine-tuning for supervised, document-level, and unsupervised translation with gains of up to 12 BLEU points in low-resource conditions.

### Data Augmentation for Low-Resource NMT

Low-resource NMT presents a fundamental challenge due to the scarcity of parallel corpora. Fadaee et al. ref [53] proposed a targeted data augmentation strategy that generates synthetic sentence pairs by substituting low-frequency words into new contexts, yielding improvements of up to 2.9 BLEU over baseline and 3.2 BLEU over back-translation ref [57] in simulated low-resource settings. This approach complements back-translation by specifically addressing rare word coverage rather than general data volume. Universal NMT models ref [11] further extend augmentation principles through shared cross-lingual representations, enabling knowledge transfer to extremely low-resource languages. Together, these targeted augmentation strategies form a critical toolkit for improving translation quality when parallel data remains severely limited.

### Pre-trained Word Embeddings

Pre-trained word embeddings offer a means of injecting external lexical knowledge into NMT models, particularly when parallel training data is scarce. Qi et al. ref [48] systematically investigate when and why pre-trained embeddings benefit NMT, demonstrating that their utility is most pronounced in low-resource settings and for cross-lingual transfer scenarios, where the model cannot acquire robust representations from limited bilingual data alone. In high-resource conditions, NMT models trained on large parallel corpora tend to learn competitive representations independently, diminishing the marginal benefit of pre-initialization. These findings underscore the importance of resource availability as a determining factor in embedding transfer strategies, informing practical decisions about model initialization in diverse translation settings.

## Multilingual and Cross-Lingual Neural Machine Translation

### Multi-Task and Multi-Language Training

Multi-task and multi-language training represents an early and influential paradigm for building NMT systems capable of handling multiple language pairs within shared model components. Dong et al. ref [26] pioneered this direction by proposing a multi-task learning framework that trains a single encoder with multiple decoders, enabling one-to-many translation. This approach demonstrated that shared encoder representations could effectively capture cross-lingual features, yielding mutual improvements across language pairs. Building upon sequence-to-sequence foundations ref [6], such architectures exploit positive transfer between related languages, reducing per-language data requirements. These multi-task formulations established critical groundwork for subsequent unified multilingual systems, motivating the development of more scalable approaches that handle many-to-many translation within a single model ref [32].

### Single-Model Multilingual NMT

Google's multilingual NMT system ref [32] demonstrated that a single model could translate between multiple language pairs by prepending an artificial language token to each source sentence, requiring no architectural modifications. Using a shared wordpiece vocabulary, this approach enables one-to-many, many-to-one, and many-to-many translation within a unified framework. The system achieved competitive performance on WMT'14 benchmarks and, notably, exhibited emergent zero-shot translation between language pairs never seen during training, suggesting the formation of a universal interlingua representation. Subsequent work on massively multilingual NMT ref [23] scaled this paradigm to 100+ languages, revealing capacity bottlenecks and motivating language-specific components and architectural deepening ref [55] to narrow performance gaps with bilingual models.

### Zero-Shot Translation

Zero-shot translation refers to the ability of a multilingual NMT model to translate between language pairs never explicitly seen during training. Google's multilingual NMT system demonstrated this emergent capability by introducing a target language token, enabling implicit bridging through a shared interlingua representation ref [32]. However, naïve multilingual training frequently fails in zero-shot settings due to spurious correlations, where the model defaults to generating output in a seen language rather than the intended target ref [34]. Subsequent work identified off-target translation as the primary failure mode and proposed random online back-translation to enforce unseen language pair translation, achieving improvements of up to 10 BLEU points ref [55]. Language tag strategies have also been shown to critically influence zero-shot performance ref [44].

### Massively Multilingual Models

Scaling multilingual NMT to 100+ languages presents significant capacity bottlenecks, as a single model must accommodate highly diverse linguistic structures. Aharoni et al. ref [23] demonstrated that massively multilingual models trained on numerous language pairs exhibit positive transfer, particularly for low-resource languages, while revealing capacity limitations under extreme scaling. Subsequent work ref [55] identified off-target translation as a critical failure mode in zero-shot settings and proposed language-specific components alongside deeper architectures to overcome capacity constraints. Tan et al. ref [29] further introduced language clustering strategies, grouping typologically similar languages to improve parameter sharing efficiency, demonstrating that informed architectural design substantially narrows the performance gap between multilingual and dedicated bilingual models.

### Transfer Learning for Low-Resource Language Pairs

Transfer learning offers a principled solution to the data scarcity problem in low-resource NMT by leveraging knowledge acquired from high-resource language pairs. Zoph et al. ref [45] introduced a parent-child framework wherein a parent model trained on a high-resource language pair transfers its learned parameters to initialize a child model for a low-resource pair, yielding average improvements of 5.6 BLEU points across four low-resource languages. This approach effectively bootstraps translation capabilities by sharing encoder and decoder representations across linguistically related or unrelated pairs. Complementarily, multilingual models ref [32] implicitly facilitate transfer through shared representations, while universal NMT architectures ref [11] extend this paradigm to extremely low-resource scenarios through cross-lingual generalization.

### Universal NMT for Extremely Low-Resource Languages

Universal NMT for extremely low-resource

[... deterministic omitted span ...]

mismatch that is substantially amplified under domain shift. This exposure bias manifests as hallucination—the generation of fluent but semantically unfaithful translations—when models encounter out-of-domain inputs. Crucially, ref [28] shows that Minimum Risk Training ref [36], by directly optimizing evaluation metrics and avoiding teacher-forcing, mitigates hallucination and also alleviates the beam search degradation problem, providing new justification for bias-reduction methods even when in-domain performance gains are marginal.

## Evaluation, Benchmarks, and NMT Toolkits

### Automatic Evaluation Metrics

Automatic evaluation metrics constitute the primary means of assessing NMT system quality in a scalable and reproducible manner. The BLEU score, which measures n-gram precision between system outputs and reference translations with a brevity penalty, has been the dominant metric throughout NMT research, employed consistently across landmark studies including ref [6], ref [10], ref [21], and ref [12]. Despite its widespread adoption, BLEU exhibits well-documented limitations, including insensitivity to semantic equivalence and paraphrase variation. Minimum risk training directly optimizes such metrics during model training ref [36], while Microsoft's human parity study ref [13] critically examined BLEU's reliability relative to human judgments, motivating ongoing efforts toward more robust automatic evaluation frameworks.

### Human Evaluation and Human Parity

Human evaluation represents the gold standard for assessing NMT quality, complementing automatic metrics such as BLEU. A landmark achievement in this domain is Microsoft's demonstration of human parity on Chinese-to-English news translation ref [13]. Using the WMT 2017 news translation task ref [41], Hassan et al. established a rigorous methodology for defining and measuring human parity, employing side-by-side bilingual evaluations with professional translators. Their system, leveraging advanced neural architectures and training strategies, achieved translation quality statistically indistinguishable from professional human translations, while significantly surpassing crowd-sourced non-professional translations. This milestone underscored both the remarkable progress of NMT and the critical importance of robust human evaluation protocols in validating system performance.

### Standard Benchmarks and Shared Tasks

Standard benchmarks and shared tasks have been instrumental in driving progress in NMT research by providing common evaluation frameworks. The WMT shared tasks, including WMT'14 and WMT'17, have served as primary benchmarks, with English-German and English-French translation pairs being the most widely adopted testbeds ref [41]. Foundational NMT models such as sequence-to-sequence LSTMs ref [6], attention-based models ref [10], convolutional architectures ref [12], and back-translation methods ref [37] have all reported results on WMT datasets. The IWSLT benchmark complements WMT by focusing on spoken language translation in lower-resource conditions ref [57]. Microsoft's human parity achievement was also evaluated on WMT'17 Chinese-English ref [13], underscoring these benchmarks' role in measuring translation quality milestones.

### Open-Source NMT Toolkits

Open-source toolkits have been instrumental in democratizing NMT research and enabling reproducible experimentation. OpenNMT ref [50] prioritizes efficiency, modularity, and extensibility, supporting diverse model architectures and source modalities while maintaining competitive performance. Nematus ref [14], developed by leading NMT researchers, became widely adopted for attention-based RNN models and supported key innovations including BPE-based subword segmentation. fairseq ref [2], developed by Facebook AI Research, provides a fast and extensible framework supporting convolutional, recurrent, and Transformer architectures with efficient GPU utilization. Additionally, the tensor2tensor framework, referenced in large-scale architectural explorations ref [30], facilitated systematic hyperparameter studies. Collectively, these toolkits have substantially accelerated progress across the NMT research community.

### Decoding Strategies and Beam Search

Decoding in NMT typically employs beam search, which maintains a fixed number of candidate hypotheses at each generation step. Sutskever et al. ref [6] demonstrated that even a modest beam size substantially improves translation quality over greedy decoding. Google's GNMT system ref [17] introduced critical refinements including length normalization and coverage penalties, preventing the decoder from favoring shorter outputs and encouraging full source coverage. Adaptive beam search strategies ref [52] improve decoding efficiency by dynamically varying candidate set sizes, achieving up to 43% speedup without quality loss. Furthermore, exposure bias has been linked to beam search degradation under domain shift ref [28], motivating continued research into more robust decoding algorithms.

### Architectural Sensitivity and Hyperparameter Analysis

Systematic empirical investigation of NMT architectural sensitivity was conducted by Britz et al. ref [30], who performed a massive exploration spanning several hundred experimental configurations and over 250,000 GPU hours on WMT English-to-German translation. Their large-scale analysis quantified the relative importance of embedding size, network depth, RNN cell type, residual connections, attention mechanism design, and decoding heuristics. Key findings revealed that deeper architectures with residual connections consistently outperform shallow counterparts, while LSTM and GRU cells yield comparable performance ref [6]. Attention mechanism choice and beam search configuration also demonstrated measurable impact on translation quality, providing practitioners with principled guidance for architecture selection and hyperparameter tuning in production NMT systems.

## References

[1] On Using Very Large Target Vocabulary for Neural Machine Translation. https://doi.org/https://doi.org/10.3115/v1/p15-1001

[2] fairseq: A Fast, Extensible Toolkit for Sequence Modeling. https://doi.org/https://doi.org/10.18653/v1/n19-4009

[3] Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. https://doi.org/https://doi.org/10.3115/v1/d14-1179

[4] Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. https://doi.org/10.3115/v1/D14-1179

[5] Competence-based Curriculum Learning for Neural Machine Translation. https://doi.org/10.18653/v1/N19-1119

[6] Sequence to Sequence Learning with Neural Networks. https://doi.org/https://doi.org/10.48550/arxiv.1409.3215

[7] Sequence to Sequence Learning with Neural Networks.

[8] Long Short-Term Memory. https://doi.org/https://doi.org/10.1162/neco.1997.9.8.1735

[9] Dual Learning for Machine Translation. https://doi.org/https://doi.org/10.48550/arxiv.1611.00179

[10] Neural Machine Translation by Jointly Learning to Align and Translate. https://doi.org/https://doi.org/10.48550/arxiv.1409.0473

[11] Universal Neural Machine Translation for Extremely Low Resource Languages. https://doi.org/https://doi.org/10.18653/v1/n18-1032

[12] Convolutional Sequence to Sequence Learning. https://doi.org/https://doi.org/10.48550/arxiv.1705.03122

[13] Achieving Human Parity on Automatic Chinese to English News Translation. https://doi.org/https://doi.org/10.48550/arxiv.1803.05567

[14] Nematus: a Toolkit for Neural Machine Translation. https://doi.org/10.18653/V1/E17-3017

[15] Simple, Scalable Adaptation for Neural Machine Translation. https://doi.org/10.18653/v1/D19-1165

[16] On the Properties of Neural Machine Translation: Encoder–Decoder Approaches. https://doi.org/10.3115/v1/W14-4012

[17] Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation. https://doi.org/https://doi.org/10.48550/arxiv.1609.08144

[18] Sequence-Level Knowledge Distillation. https://doi.org/https://doi.org/10.18653/v1/d16-1139

[19] Recurrent Continuous Translation Models. https://doi.org/https://doi.org/10.18653/v1/d13-1176

[20] The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation. https://doi.org/10.18653/v1/P18-1008

[21] Effective Approaches to Attention-based Neural Machine Translation. https://doi.org/https://doi.org/10.18653/v1/d15-1166

[22] Effective Approaches to Attention-based Neural Machine Translation. https://doi.org/10.18653/v1/D15-1166

[23] Massively Multilingual Neural Machine Translation. https://doi.org/https://doi.org/10.18653/v1/n19-1388

[24] Massively Multilingual Neural Machine Translation. https://doi.org/10.18653/v1/N19-1388

[25] Modeling Coverage for Neural Machine Translation. https://doi.org/10.18653/v1/P16-1008

[26] Multi-Task Learning for Multiple Language Translation. https://doi.org/https://doi.org/10.3115/v1/p15-1166

[27] Multilingual Denoising Pre-training for Neural Machine Translation. https://doi.org/10.1162/tacl_a_00343

[28] On Exposure Bias, Hallucination and Domain Shift in Neural Machine Translation. https://doi.org/10.18653/v1/2020.acl-main.326

[29] Multilingual Neural Machine Translation with Language Clustering. https://doi.org/10.18653/v1/D19-1089

[30] Massive Exploration of Neural Machine Translation Architectures. https://doi.org/10.18653/v1/D17-1151

[31] Phrase-Based & Neural Unsupervised Machine Translation. https://doi.org/10.18653/v1/D18-1549

[32] Google’s Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation. https://doi.org/https://doi.org/10.1162/tacl_a_00065

[33] Google’s Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation. https://doi.org/10.1162/tacl_a_00065

[34] Improved Zero-shot Neural Machine Translation via Ignoring Spurious Correlations. https://doi.org/10.18653/v1/P19-1121

[35] Convolutional Sequence to Sequence Learning.

[36] Minimum Risk Training for Neural Machine Translation. https://doi.org/https://doi.org/10.18653/v1/p16-1159

[37] Understanding Back-Translation at Scale. https://doi.org/https://doi.org/10.18653/v1/d18-1045

[38] Cross-lingual Language Model Pretraining. https://doi.org/https://doi.org/10.48550/arxiv.1901.07291

[39] Overcoming Catastrophic Forgetting During Domain Adaptation of Neural Machine Translation. https://doi.org/10.18653/v1/N19-1209

[40] Contrastive Learning for Many-to-many Multilingual Neural Machine Translation. https://doi.org/10.18653/v1/2021.acl-long.21

[41] Findings of the 2017 Conference on Machine Translation (WMT17). https://doi.org/https://doi.org/10.18653/v1/w17-4717

[42] MASS: Masked Sequence to Sequence Pre-training for Language Generation. https://doi.org/https://doi.org/10.48550/arxiv.1905.02450

[43] Bridging the Gap between Training and Inference for Neural Machine Translation. https://doi.org/10.18653/V1/P19-1426

[44] Language Tags Matter for Zero-Shot Neural Machine Translation. https://doi.org/10.18653/v1/2021.findings-acl.264

[45] Transfer Learning for Low-Resource Neural Machine Translation. https://doi.org/10.18653/v1/D16-1163

[46] . https://doi.org/https://doi.org/10.18653/v1/n19-1423

[47] Six Challenges for Neural Machine Translation. https://doi.org/10.18653/v1/W17-3204

[48] When and Why Are Pre-Trained Word Embeddings Useful for Neural Machine Translation?. https://doi.org/10.18653/v1/N18-2084

[49] Neural Machine Translation of Rare Words with Subword Units. https://doi.org/10.18653/v1/P16-1162

[50] OpenNMT: Open-Source Toolkit for Neural Machine Translation. https://doi.org/10.18653/V1/P17-4012

[51] Robust Neural Machine Translation with Doubly Adversarial Inputs. https://doi.org/10.18653/v1/P19-1425

[52] Beam Search Strategies for Neural Machine Translation. https://doi.org/10.18653/v1/W17-3207

[53] Data Augmentation for Low-Resource Neural Machine Translation. https://doi.org/10.18653/v1/P17-2090

[54] Leveraging Monolingual Data with Self-Supervision for Multilingual Neural Machine Translation. https://doi.org/10.18653/V1/2020.ACL-MAIN.252

[55] Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation. https://doi.org/10.18653/v1/2020.acl-main.148

[56] Neural Machine Translation with Deep Attention. https://doi.org/10.1109/TPAMI.2018.2876404

[57] Improving Neural Machine Translation Models with Monolingual Data. https://doi.org/10.18653/v1/P16-1009

[58] Neural Machine Translation With GRU-Gated Attention Model. https://doi.org/10.1109/TNNLS.2019.2957276

[59] AdvAug: Robust Adversarial Augmentation for Neural Machine Translation. https://doi.org/10.18653/v1/2020.acl-main.529

[60] Curriculum Learning for Domain Adaptation in Neural Machine Translation. https://doi.org/10.18653/v1/N19-1189
