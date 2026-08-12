# Registered common-window excerpt

# Neural Machine Translation: A Comprehensive Survey of Architectures, Methods, and Applications

## 1 Introduction and Historical Background

### 1.1 The Origins of Machine Translation: Early Visions and Rule-Based Systems

The dream of automated translation between human languages predates the digital computer itself. As early as the 1940s and 1950s, visionary thinkers began to articulate the possibility that machines could one day serve as linguistic intermediaries, bridging communication gaps across the world's many tongues. The famous Georgetown-IBM experiment of 1954, which demonstrated the automatic translation of over sixty Russian sentences into English using a small set of vocabulary and grammar rules, generated considerable optimism—and perhaps excessive confidence—about the prospects for rapid progress in the field. This early enthusiasm reflected a broader belief that language could be reduced to a finite set of transferable rules, an assumption that would prove far more complex to realize in practice [1].

Rule-based machine translation (RBMT) emerged as the dominant paradigm throughout the 1960s, 1970s, and well into the 1980s. These systems encoded linguistic knowledge explicitly, relying on hand-crafted grammars, bilingual dictionaries, and morphological analyzers developed by expert linguists to mediate between source and target languages [2]. The underlying philosophy was straightforward: if the syntactic and semantic structures of two languages could be formally described, and if systematic correspondences between them could be articulated, then translation could be reduced to the application of deterministic transformation rules. This linguistically motivated approach gave researchers and practitioners a high degree of control over the system's output, a feature that remains attractive even in the modern era [3].

Within the RBMT paradigm, researchers developed several distinct architectural approaches reflecting different theoretical assumptions about the nature of translation. Direct translation systems operated by mapping source language words and phrases directly to target language equivalents through dictionaries and simple reordering rules, with minimal intermediate representation. While computationally straightforward, direct systems struggled with languages exhibiting divergent syntactic structures or rich morphology, since surface-level word correspondences often fail to capture deeper grammatical relationships.

Transfer-based architectures addressed these limitations by introducing an intermediate level of representation. In a transfer system, the source sentence is first analyzed into an abstract linguistic representation—typically a syntactic or shallow semantic parse—which is then transformed into a corresponding target-language representation through transfer rules, before being regenerated as fluent target-language text. This three-stage analysis-transfer-generation pipeline allowed linguists to modularize their knowledge and handle structural divergences more elegantly [4]. Prominent projects such as the SYSTRAN system and the European Community's EUROTRA project were grounded in transfer-based principles, and the approach demonstrated genuine practical utility for constrained domains and language pairs with sufficient structural similarity.

At the far end of the abstraction spectrum lay interlingua-based systems, which sought to translate through a universal, language-neutral semantic representation. The theoretical appeal of the interlingua approach was considerable: if all source language sentences could be analyzed into a common deep representation capturing their meaning, and if target language sentences could be generated from this representation, then a single system could in principle handle any pair of languages by sharing the same intermediate layer. In practice, however, constructing a sufficiently rich and general interlingua proved extraordinarily difficult. The diversity of human languages, the ambiguity inherent in natural language, and the context-dependence of meaning conspired to make robust interlingua-based translation elusive outside narrowly defined domains [2].

Despite their theoretical elegance, RBMT systems faced fundamental practical limitations that became increasingly apparent over time. The construction and maintenance of comprehensive linguistic resources—grammars, lexicons, morphological analyzers, and transfer rules—required enormous investment of expert human effort and was not easily scalable to new language pairs or domains [5]. Handling the sheer variety and ambiguity of real-world text proved extremely challenging, as rules designed for canonical sentences often failed on the irregular, idiomatic, or domain-specific language encountered in practice. Coverage gaps were pervasive: words or constructions not anticipated by the rule writers produced errors or untranslated fragments. Furthermore, the rigid linguistic assumptions embedded in RBMT systems made them brittle in the face of language change, informal registers, or unexpected input [3].

The treatment of morphologically rich languages exposed particular weaknesses in RBMT frameworks. Languages such as Arabic, Turkish, and the Indic languages pose significant challenges for rule-based analysis and generation due to their complex inflectional and derivational morphology, which can produce enormous vocabularies from relatively small sets of roots [6]. Similarly, pairs of languages with radically different word orders—such as English and Japanese—demanded elaborate reordering rules that were difficult to specify correctly and comprehensively. These difficulties with morphology and reordering would later become recurring themes in the statistical and neural translation paradigms that followed.

Despite these shortcomings, the RBMT era was far from unproductive. The careful linguistic analysis it demanded produced deep insights into the structure of human languages and the nature of translation equivalence. The modular architectures developed—analysis, transfer, and generation components—provided a conceptual framework that influenced subsequent research. Many of the linguistic resources created for RBMT systems, including morphological analyzers and bilingual lexicons, continued to prove useful as components in later statistical and neural translation pipelines. Indeed, some researchers have revisited RBMT resources specifically to support low-resource neural machine translation, exploiting the terminological and morphological knowledge encoded in rule-based systems to augment data-driven approaches [3; 7].

The limitations of RBMT ultimately spurred the search for data-driven alternatives, setting the stage for the rise of statistical machine translation in the late 1980s and 1990s. Rather than encoding linguistic knowledge manually, the emerging statistical paradigm would instead learn translation patterns directly from large bilingual corpora—a fundamental shift in philosophy that addressed many of the scalability and coverage shortcomings of rule-based approaches. Yet the fundamental challenges that RBMT sought to address—structural divergence between languages, lexical ambiguity, morphological complexity, and the need for linguistically faithful translations—remain central concerns in contemporary neural machine translation research. The history of rule-based approaches thus forms an essential foundation for understanding both the motivations behind and the achievements of subsequent translation paradigms [5; 8].

### 1.2 Statistical Machine Translation: Foundations and Advancements

Statistical machine translation (SMT) emerged as the dominant paradigm in machine translation research from the late 1980s through the 2000s, fundamentally reshaping how translation systems were conceived and built. Arising directly from the frustrations with rule-based approaches described above, SMT systems learned translation patterns directly from large collections of bilingual text corpora rather than encoding linguistic knowledge manually, offering a data-driven alternative to the brittle rule-based systems that had previously dominated the field. This shift in philosophy — from hand-crafted rules to learned statistical regularities — addressed many of the scalability and coverage shortcomings of RBMT, though it would eventually reveal its own set of fundamental limitations that would in turn motivate the neural approaches that followed.

**IBM Word Alignment Models.** The foundations of SMT were laid by the IBM word alignment models, which formalized the translation problem in terms of probabilistic generative processes. These models, ranging from IBM Model 1 through Model 5, introduced the notion of word-level alignment between source and target sentences and provided a principled statistical framework for estimating translation probabilities from parallel corpora. A central challenge in these models was parameter estimation in the presence of data sparsity, and researchers developed smoothing techniques to address overfitting [9]. The IBM models established core concepts such as lexical translation probabilities and distortion models that remained influential throughout subsequent SMT development.

**Phrase-Based SMT.** Building upon word alignment models, phrase-based statistical machine translation (PBSMT) became the workhorse of practical MT systems during the 2000s. Rather than operating on individual words, phrase-based systems learned translations of contiguous sequences of words, capturing local lexical and syntactic regularities that word-based models could not adequately represent. These systems were parameterized within a log-linear modeling framework that allowed multiple feature functions — including phrase translation probabilities, language model scores, reordering models, and word penalty features — to be combined and optimized discriminatively on development data. The Moses toolkit became the standard open-source implementation for building such systems [10; 11], enabling widespread adoption across language pairs and domains.

**Log-Linear Frameworks and Feature Engineering.** The log-linear

[... deterministic omitted span ...]

the training steps required to train a model from scratch. This dramatic reduction in training time is particularly significant for low-resource scenarios where computational resources may also be constrained.

### Cross-Lingual Pre-Training

Cross-lingual pre-training extends the idea of monolingual pre-trained language models such as BERT and GPT to multilingual settings, enabling representations that are shared across languages. Models pre-trained on large multilingual corpora with objectives such as masked language modeling, denoising autoencoding, or translation language modeling learn language-agnostic representations that can be fine-tuned for specific low-resource translation directions [27]. The key advantage of cross-lingual pre-training is that it does not require parallel data for every language pair; instead, monolingual corpora in many languages are sufficient to build shared representations. When fine-tuned on even a small amount of parallel data for a low-resource pair, such pre-trained models generalize far better than models trained from scratch — a property that makes them a natural complement to back-translation and other data augmentation techniques described previously.

Large-scale multilingual pre-trained models such as mBART and multilingual BERT have demonstrated strong transfer to low-resource language pairs, providing a powerful initialization that encodes cross-lingual semantic and syntactic correspondences. The effectiveness of these models depends on whether the low-resource language is represented in the pre-training corpus, and on vocabulary coverage. When a target language has limited representation in the pre-training data, vocabulary transfer techniques that adapt the tokenizer and embedding layer become necessary.

### Vocabulary Transfer for Unrelated Language Pairs

A persistent challenge in transfer learning for NMT is that vocabularies are often built using byte-pair encoding or wordpiece segmentation applied to the parent language pair, making them suboptimal or even incompatible with morphologically or orthographically distinct target languages. Several techniques have been developed to bridge this gap. One approach involves constructing a shared multilingual vocabulary that covers all languages involved in both parent and child training [131]. Another approach reuses the parent's subword vocabulary but augments it with new tokens for the child language, adapting only the embedding parameters while keeping deeper encoder-decoder weights fixed.

[132] takes a complementary perspective, proposing to replace the static source embedding layer with a bi-directional recurrent network that generates compositional representations from character n-grams, enabling robust handling of morphologically rich and low-resource languages. By composing word representations from character-level features, this approach consistently outperforms NMT systems relying on statistically generated subword units, with BLEU improvements ranging from 1.71 to 2.48 points across five typologically diverse languages in low-resource conditions. Such character-level compositional encoders can naturally complement transfer learning pipelines by providing a language-agnostic input representation that reduces vocabulary mismatch between parent and child models.

### Cross-Lingual Word Embeddings

Cross-lingual word embeddings provide a complementary mechanism for bridging linguistic gaps in low-resource NMT. By mapping word representations from different languages into a shared embedding space, cross-lingual embeddings allow a model trained primarily on high-resource data to leverage lexical correspondences across languages. These embeddings can be constructed using bilingual lexicons, parallel corpora, or fully unsupervised alignment of independently trained monolingual embedding spaces. When used to initialize NMT encoder or decoder embedding layers, cross-lingual embeddings reduce the cold-start problem inherent in low-resource transfer and improve convergence speed.

[133] proposes a shared-private embedding architecture where source and target language embeddings are partially shared for similar words while retaining language-specific components, reducing the total number of parameters while improving cross-lingual generalization. Experiments across five language pairs from six different language families demonstrate significant BLEU improvements over strong baselines with substantially fewer model parameters, underscoring the value of structured sharing in multilingual embedding spaces.

Pointer-based fusion of bilingual lexicons into NMT provides yet another avenue for leveraging lexical transfer [134]. By incorporating a bilingual lexicon as an external knowledge source accessible during decoding, this approach improves translation quality in low-resource scenarios while also reducing out-of-vocabulary problems — an issue that, as noted in the previous section, can alternatively be addressed through phrase table augmentation and pre-translation strategies.

### Summary

Transfer learning and cross-lingual pre-training collectively address the most critical bottleneck in low-resource NMT: the lack of sufficient parallel data. Through parent-child model transfer, dynamic vocabulary adaptation [130], cross-lingual pre-training, compositional character-level encoders [132], and cross-lingual embedding sharing [133], researchers have demonstrated consistent and often dramatic improvements over models trained from scratch. Together with the data augmentation techniques discussed in Section 4.2, these transfer learning strategies form a comprehensive toolkit for low-resource NMT practitioners. The field continues to evolve rapidly as larger and more diverse multilingual pre-trained models become available, raising new research questions about optimal fine-tuning strategies, vocabulary design, and the transfer of knowledge to extremely low-resource and endangered languages — questions that directly inform the design of the multilingual NMT systems examined in the following section.

### 4.4 Multilingual NMT Systems and Knowledge Transfer

## 4.4 Multilingual Neural Machine Translation

Building on the transfer learning and cross-lingual pre-training strategies discussed in the previous section, multilingual neural machine translation (NMT) represents a natural architectural extension of these ideas: rather than transferring knowledge from a pre-trained model to a specific low-resource pair in a sequential pipeline, multilingual systems aim to handle multiple language pairs within a single unified model simultaneously. By sharing model parameters across languages, multilingual systems enable positive knowledge transfer, where information learned from high-resource language pairs can benefit the translation of lower-resource ones — a goal closely aligned with the cross-lingual pre-training objectives reviewed previously. This shared representation space is particularly valuable when certain language pairs lack sufficient parallel training data on their own.

A central motivation for multilingual NMT is the observation that languages share underlying structural and semantic regularities, and a single model with sufficient capacity can learn to exploit these commonalities. When encoder and decoder parameters are jointly trained across multiple languages, the resulting representations tend to be more language-agnostic, facilitating cross-lingual transfer. This is especially evident in settings where related languages — sharing morphology, syntax, or vocabulary — are trained together, as the model can generalize patterns from one language to another. This property also connects directly to the cross-lingual word embeddings and shared vocabulary techniques discussed in Section 4.3, where bridging linguistic gaps was a central concern.

However, multilingual training introduces a fundamental capacity bottleneck: a single model of fixed size must simultaneously represent all language pairs, and as the number of languages grows, the available capacity per language diminishes. This often results in a trade-off where high-resource languages see a performance drop compared to dedicated bilingual models, while low-resource languages benefit from the shared representations. This tension has been described as the "curse of multilinguality," where gains for low-resource settings come at the expense of high-resource performance. Architectures that deepen the network or incorporate language-specific components — such as language-specific encoder or decoder layers, adapters, or gating mechanisms — have been proposed as a means to alleviate this bottleneck, allowing the model to allocate additional capacity to specific languages without abandoning the shared multilingual framework.

The design of multilingual NMT systems also intersects with the question of how best to represent language identity within the model. A common approach involves prepending a special language token to the source or target sequence to signal the desired output language, enabling a single model to produce output in multiple target languages. This simple but effective strategy has been shown to work well in practice, supporting both many-to-one and one-to-many translation configurations. Importantly, this language tag conditioning mechanism also plays a role in the zero-shot translation setting discussed in Section 4.5, where the decoder must be guided toward a target language it has never directly been supervised on.

Beyond architectural considerations, the training data distribution across languages plays a critical role. Naive concatenation of all available parallel corpora leads to imbalanced training, where high-resource languages dominate gradient updates. Temperature-based data sampling — where low-resource language pairs are oversampled — has emerged as a standard technique to mitigate this imbalance and improve performance across the full set of supported languages, complementing the data augmentation and back-translation strategies discussed in Section 4.2.

The multilingual setting also naturally motivates

[... deterministic omitted span ...]

on English to Croatian

[19] Neural machine translation for low-resource languages

[20] Neural Machine Translation Advised by Statistical Machine Translation

[21] Neural versus Phrase-Based Machine Translation Quality  a Case Study

[22] SMT vs NMT  A Comparison over Hindi & Bengali Simple Sentences

[23] Neural System Combination for Machine Translation

[24] A Hybrid Model for Enhancing Lexical Statistical Machine Translation  (SMT)

[25] Six Challenges for Neural Machine Translation

[26] Salute the Classic  Revisiting Challenges of Machine Translation in the  Age of Large Language Models

[27] Neural Machine Translation  A Review and Survey

[28] Learning Phrase Representations using RNN Encoder-Decoder for  Statistical Machine Translation

[29] On the Properties of Neural Machine Translation  Encoder-Decoder  Approaches

[30] Listen, Attend, and Walk  Neural Mapping of Navigational Instructions to  Action Sequences

[31] Neural Machine Translation with Recurrent Highway Networks

[32] Asynchronous Bidirectional Decoding for Neural Machine Translation

[33] Cseq2seq  Cyclic Sequence-to-Sequence Learning

[34] Learning to Refine Source Representations for Neural Machine Translation

[35] Understanding Neural Machine Translation by Simplification  The Case of  Encoder-free Models

[36] Is Encoder-Decoder Redundant for Neural Machine Translation 

[37] Effective Approaches to Attention-based Neural Machine Translation

[38] Modeling Coverage for Neural Machine Translation

[39] Neural Machine Translation with Key-Value Memory-Augmented Attention

[40] Interactive Attention for Neural Machine Translation

[41] Neural Machine Translation with Supervised Attention

[42] Temporal Attention Model for Neural Machine Translation

[43] Syntax-Directed Attention for Neural Machine Translation

[44] Fine-Grained Attention Mechanism for Neural Machine Translation

[45] Attention Weights in Transformer NMT Fail Aligning Words Between  Sequences but Largely Explain Model Predictions

[46] An Analysis of Attention Mechanisms  The Case of Word Sense  Disambiguation in Neural Machine Translation

[47] Document-Level Neural Machine Translation with Hierarchical Attention  Networks

[48] Multimodal Attention for Neural Machine Translation

[49] Attention Is All You Need

[50] Weighted Transformer Network for Machine Translation

[51] Modeling Recurrence for Transformer

[52] DTMT  A Novel Deep Transition Architecture for Neural Machine  Translation

[53] Parallel Attention Mechanisms in Neural Machine Translation

[54] Exploring Transformers in Natural Language Generation  GPT, BERT, and  XLNet

[55] Transformers   The End of History  for NLP 

[56] Generative Pre-trained Transformer  A Comprehensive Review on Enabling  Technologies, Potential Applications, Emerging Challenges, and Future  Directions

[57] Leveraging Pre-trained Checkpoints for Sequence Generation Tasks

[58] AMMUS   A Survey of Transformer-based Pretrained Models in Natural  Language Processing

[59] Q8BERT  Quantized 8Bit BERT

[60] Accelerating Training of Transformer-Based Language Models with  Progressive Layer Dropping

[61] Foundation Transformers

[62] How Good Are GPT Models at Machine Translation  A Comprehensive  Evaluation

[63] Low Resource Summarization using Pre-trained Language Models

[64] A Survey on Document-level Neural Machine Translation  Methods and  Evaluation

[65] Improving Long Context Document-Level Machine Translation

[66] IITP at WAT 2021  System description for English-Hindi Multimodal  Translation Task

[67] Zero-Resource Neural Machine Translation with Multi-Agent Communication  Game

[68] Low-Latency Neural Speech Translation

[69] Exploiting Linguistic Resources for Neural Machine Translation Using  Multi-task Learning

[70] A Comparison of Transformer and Recurrent Neural Networks on  Multilingual Neural Machine Translation

[71] A Study of Multilingual Neural Machine Translation

[72] Neural Machine Translation For Low Resource Languages

[73] Language Models are Good Translators

[74] Systematic Inequalities in Language Technology Performance across the  World's Languages

[75] A Survey of Deep Learning Techniques for Neural Machine Translation

[76] Improving Neural Machine Translation Models with Monolingual Data

[77] Bi-Directional Neural Machine Translation with Synthetic Parallel Data

[78] Learning to Generalize to More  Continuous Semantic Augmentation for  Neural Machine Translation

[79] A Survey on Low-Resource Neural Machine Translation

[80] Neural Machine Translation for Low-Resource Languages  A Survey

[81] Context in Neural Machine Translation  A Review of Models and  Evaluations

[82] Diving Deep into Context-Aware Neural Machine Translation

[83] Toward a full-scale neural machine translation in production  the  Booking.com use case

[84] Neural Machine Translation  Challenges, Progress and Future

[85] Self-attention based end-to-end Hindi-English Neural Machine Translation

[86] Override and update

[87] Overt choice

[88] A Survey of Methods to Leverage Monolingual Data in Low-resource Neural  Machine Translation

[89] Investigating Neural Machine Translation for Low-Resource Languages   Using Bavarian as a Case Study

[90] Jointly Learning to Align and Translate with Transformer Models

[91] Context- and Sequence-Aware Convolutional Recurrent Encoder for Neural  Machine Translation

[92] Convolutional Sequence to Sequence Learning

[93] Universal Transformers

[94] Joint Source-Target Self Attention with Locality Constraints

[95] Decoding-History-Based Adaptive Control of Attention for Neural Machine  Translation

[96] Learning When to Concentrate or Divert Attention  Self-Adaptive  Attention Temperature for Neural Machine Translation

[97] Transformer++

[98] Fixed Encoder Self-Attention Patterns in Transformer-Based Machine  Translation

[99] Only 5\% Attention Is All You Need  Efficient Long-range Document-level  Neural Machine Translation

[100] Efficient Attention using a Fixed-Size Memory Representation

[101] Monotonic Multihead Attention

[102] Local Monotonic Attention Mechanism for End-to-End Speech and Language  Processing

[103] Self-Attentive Residual Decoder for Neural Machine Translation

[104] Dense Information Flow for Neural Machine Translation

[105] Syntax-Aware Complex-Valued Neural Machine Translation

[106] Guided Alignment Training for Topic-Aware Neural Machine Translation

[107] GroupBERT  Enhanced Transformer Architecture with Efficient Grouped  Structures

[108] Memory Transformer

[109] Cached Transformers  Improving Transformers with Differentiable Memory  Cache

[110] BP-Transformer  Modelling Long-Range Context via Binary Partitioning

[111] Using Monolingual Data in Neural Machine Translation  a Systematic Study

[112] Combining SMT and NMT Back-Translated Data for Efficient NMT

[113] Investigating Backtranslation in Neural Machine Translation

[114] An Effective Approach to Unsupervised Machine Translation

[115] Language-Independent Representor for Neural Machine Translation

[116] Structured-based Curriculum Learning for End-to-end English-Japanese  Speech Translation

[117] Better Neural Machine Translation by Extracting Linguistic Information  from BERT

[118] Improving Neural Machine Translation with Pre-trained Representation

[119] Leveraging Monolingual Data with Self-Supervision for Multilingual  Neural Machine Translation

[120] Application of Low-resource Machine Translation Techniques to  Russian-Tatar Language Pair

[121] Urdu-English Machine Transliteration using Neural Networks

[122] Towards Machine Translation for the Kurdish Language

[123] Machine Translation for Ge'ez Language

[124] A Framework for Hierarchical Multilingual Machine Translation

[125] Hindi to English  Transformer-Based Neural Machine Translation

[126] Towards Neural Machine Translation with Partially Aligned Corpora

[127] Neural Machine Translation Model with a Large Vocabulary Selected by  Branching Entropy

[128] Pre-Translation for Neural Machine Translation

[129] Revisiting Low-Resource Neural Machine Translation  A Case Study

[130] Transfer Learning in Multilingual Neural Machine Translation with  Dynamic Vocabulary

[131] A Comprehensive Survey of Multilingual Neural Machine Translation

[132] Compositional Representation of Morphologically-Rich Input for Neural  Machine Translation

[133] Shared-Private Bilingual Word Embeddings for Neural Machine Translation

[134] Pointer-based Fusion of Bilingual Lexicons into Neural Machine  Translation

[135] Multilingual Neural Machine Translation with Task-Specific Attention

[136] Improving Neural Machine Translation by Multi-Knowledge Integration with  Prompting

[137] Reference Network for Neural Machine Translation

[138] An Overview on Machine Translation Evaluation

[139] A Paradigm Shift  The Future of Machine Translation Lies with Large  Language Models

[140] Simul-LLM  A Framework for Exploring High-Quality Simultaneous  Translation with Large Language Models

[141] Google's Neural Machine Translation System  Bridging the Gap between  Human and Machine Translation

[142] Sequence-to-Sequence Models Can Directly Translate Foreign Speech

[143] Findings of the Third Workshop on Neural Generation and Translation

[144] Multi-channel Encoder for Neural Machine Translation

[145] AlignAtt  Using Attention-based Audio-Translation Alignments as a Guide  for Simultaneous Speech Translation

[146] Sharp Models on Dull Hardware  Fast and Accurate Neural Machine  Translation Decoding on the CPU

[147] LEPOR  An Augmented Machine Translation Evaluation Metric

[148] Good, but not always Fair  An Evaluation of Gender Bias for three  commercial Machine Translation Systems

[149] Identifying and Controlling Important Neurons in Neural Machine  Translation

[150] Evaluating Layers of Representation in Neural Machine Translation on  Part-of-Speech and Semantic Tagging Tasks

[151] Coverage Embedding Models for Neural Machine Translation

[152] An Empirical Analysis of NMT-Derived Interlingual Embeddings and their  Use in Parallel Sentence Identification

[153] CODET  A Benchmark for Contrastive Dialectal Evaluation of Machine  Translation

[154] Language Modeling, Lexical Translation, Reordering  The Training Process  of NMT through the Lens of Classical SMT

[155] Information-Propogation-Enhanced Neural Machine Translation by Relation  Model

[156] Calibration of Encoder Decoder Models for Neural Machine Translation

[157] Improving Context-aware Neural Machine Translation with Target-side  Context

[158] BiVert  Bidirectional Vocabulary Evaluation using Relations for Machine  Translation

[159] Handling Syntactic Divergence in Low-resource Machine Translation

[160] A Large-Scale Study of Machine Translation in the Turkic Languages
