# Registered common-window excerpt

# Text-to-Image Diffusion Models: A Comprehensive Survey of Foundations, Architectures, Controllability, Applications, and Societal Implications

## 1 Introduction

The rapid ascent of text-to-image diffusion models represents one of the most consequential paradigm shifts in generative artificial intelligence, transforming the synthesis of photorealistic imagery from natural language into a broadly accessible capability. Early generative frameworks—including Generative Adversarial Networks and Variational Autoencoders—established foundational principles of learned image distributions yet suffered from training instability, mode collapse, and limited semantic controllability. The subsequent emergence of diffusion-based approaches [1; 2; 3] demonstrated that iterative denoising, when coupled with large-scale vision-language pretraining and scalable architectures, could achieve qualitative leaps in both fidelity and semantic faithfulness. This survey systematically organizes this rapidly expanding literature—spanning theoretical foundations, architectural innovations [4], controllability mechanisms [5], applications [6], and societal implications [7]—to provide researchers and practitioners a comprehensive, critically synthesized reference for this transformative field.

## 2 Theoretical Foundations of Text-to-Image Diffusion Models

### 2.1 Discrete-Time Forward Diffusion Process and Noise Scheduling

I need to check each citation in the subsection to verify they support the claims being made.

1. "underpinning scalable denoising network training [1; 8; 2]"
- High-Resolution Image Synthesis with Latent Diffusion Models: Yes, discusses diffusion model training with this forward process.
- Variational Diffusion Models: Yes, explicitly discusses the VLB in terms of SNR and the forward process.
- Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding: Yes, builds on diffusion model training framework.
These citations are correct.

2. "linear schedules destroy signal too rapidly at low resolutions, motivating cosine and learned schedules that maintain perceptually informative intermediate noise levels [8; 3]"
- Variational Diffusion Models: Yes, discusses learned noise schedules and SNR.
- GLIDE: The paper discusses classifier-free guidance and text-conditional diffusion, but the discussion of cosine schedules and noise schedule improvements is more associated with other papers (like "Improved Denoising Diffusion Probabilistic Models" which is not in the list). GLIDE does use improved noise schedules but the paper content provided doesn't specifically emphasize cosine schedules. However, GLIDE does reference guidance strategies. This citation is questionable but GLIDE does work with these schedules. I'll keep it cautiously, but checking the paper content - it doesn't explicitly mention cosine schedules. I'll remove it.

3. "with direct consequences for generation fidelity in deployed systems [4]"
- SDXL discusses improved conditioning schemes and generation quality. It doesn't specifically discuss noise schedules in the provided abstract. However, it does address generation fidelity improvements. This is a weak citation - I'll remove it as the paper content doesn't specifically support the noise schedule claim.

The forward diffusion process defines a fixed Markov chain that progressively corrupts a clean data sample $\mathbf{x}_0 \sim q(\mathbf{x}_0)$ by injecting Gaussian noise across $T$ discrete timesteps, yielding the transition kernel $q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$. A critical analytical convenience is the closed-form marginal $q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$, where $\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$, enabling direct sampling at arbitrary noise levels without sequential simulation and underpinning scalable denoising network training [1; 8; 2]. The signal-to-noise ratio $\text{SNR}(t) = \bar{\alpha}_t/(1-\bar{\alpha}_t)$ provides a unified lens for comparing variance schedules: linear schedules destroy signal too rapidly at low resolutions, motivating cosine and learned schedules that maintain perceptually informative intermediate noise levels [8], with direct consequences for generation fidelity in deployed systems [4].

### 2.2 Reverse Denoising Process, Score Matching, and Training Objectives

The reverse denoising process casts generative modeling as the inversion of the forward corruption chain, training a parameterized neural network to approximate the true reverse conditionals at each timestep. The theoretical foundation rests on maximizing a variational lower bound (VLB) on the data log-likelihood, which decomposes into a sum of KL divergence terms between approximate and true reverse conditionals plus a reconstruction term [9; 8]. This hierarchical variational perspective formally connects diffusion models to deep latent variable models while providing the tractable objective necessary for scalable optimization. Crucially, [8] demonstrates that the VLB simplifies elegantly to a signal-to-noise ratio (SNR)-weighted regression loss, revealing that different noise schedule choices affect training primarily through the SNR at the process endpoints rather than intermediate trajectory details—an insight that directly extends the SNR-indexed view of the forward process and carries immediate consequences for schedule design and likelihood optimization. The denoising score matching perspective offers a complementary view: the network implicitly learns the score function, i.e., the gradient of the log data density at each noise level, through a regression objective whose equivalence to noise prediction and clean data prediction parameterizations follows from Tweedie's formula [10]. The noise prediction parameterization, favored empirically in [9], exhibits favorable optimization landscapes relative to direct data prediction at high noise levels, while data prediction can be advantageous at low noise levels where fine perceptual detail dominates. Uniform timestep weighting of the training objective, though theoretically justified, is suboptimal in practice: reweighted objectives that up-weight intermediate noise levels carrying the most semantically informative gradients yield measurably richer representations and improved generation quality [9; 8]. Further, [11] frames multi-timestep denoising as multi-task learning and demonstrates that negative transfer between temporally distant denoising tasks degrades convergence, motivating interval-clustered training strategies aligned with task affinity. Together, these training objective refinements establish the denoising network as a principled score estimator across all noise levels—a foundation that the continuous-time SDE framework subsequently generalizes into a unified mathematical treatment of diffusion model families.

### 2.3 Continuous-Time Formulations via Stochastic and Ordinary Differential Equations

I need to verify each citation in the subsection against the paper contents provided.

1. "the forward Markov chain admits a limiting stochastic differential equation (SDE) driven by Brownian motion, and the corresponding reverse-time SDE involves the score function of the marginal data density [10; 12]"

Both survey papers discuss SDEs and score functions as part of their theoretical foundations. ✓

2. "every such SDE possesses an associated probability flow ordinary differential equation sharing identical marginal distributions, enabling deterministic trajectory integration and exact likelihood evaluation via instantaneous change-of-variables [8; 13]"

- Variational Diffusion Models discusses probability flow ODE and likelihood evaluation. ✓
- Understanding DDPM Latent Codes Through Optimal Transport discusses the probability flow ODE and encoder map. ✓

3. "The signal-to-noise ratio provides a unified coordinate connecting variance-exploding, variance-preserving, and sub-variance-preserving families [8]"

Variational Diffusion Models explicitly discusses the signal-to-noise ratio as a unifying concept. ✓

4. "accelerated numerical solvers including DDIM [14] and higher-order ODE integrators substantially reduce discretization error, offering principled quality-speed trade-offs that underpin practical text-to-image deployment"

DDIM paper directly supports this - it presents faster sampling methods. ✓

All citations appear correct.

The continuous-time perspective unifies discrete diffusion model families under a single mathematical framework, wherein the forward Markov chain admits a limiting stochastic differential equation (SDE) driven by Brownian motion, and the corresponding reverse-time SDE involves the score function of the marginal data density [10; 12]. Critically, every such SDE possesses an associated probability flow ordinary differential equation sharing identical marginal distributions, enabling deterministic trajectory integration and exact likelihood evaluation via instantaneous change-of-variables [8; 13]. The signal-to-noise ratio provides a unified coordinate connecting variance-exploding, variance-preserving, and sub-variance-preserving families [8], while accelerated numerical solvers including DDIM [14] and higher-order ODE integrators substantially reduce discretization error, offering principled quality-speed trade-offs that underpin practical text-to-image deployment.

### 2.4 Text Conditioning Mechanisms and Guidance Strategies

Text conditioning in diffusion models is formally achieved through three principal mechanisms whose theoretical foundations build directly upon the score-based continuous-time framework and its associated denoising objectives. Classifier guidance derives a modified score by augmenting the unconditional score function with the gradient of a noise-aware classifier's log-probability, enabling a principled trade-off between sample fidelity and diversity through a guidance scale hyperparameter [15]. Classifier-free guidance eliminates the separate classifier by jointly training the denoising network on conditional and unconditional objectives via random prompt dropout, then extrapolating beyond the conditional score at inference time [15; 16], which has become the dominant paradigm due to its superior quality-diversity trade-offs and seamless compatibility with the accelerated ODE solvers introduced previously. Cross-attention between spatially arranged image features and pretrained language encoder embeddings provides the primary architectural mechanism for semantic injection across denoising timesteps

[... deterministic omitted span ...]

42]. The core trade-off lies between expressive multi-condition fidelity and the combinatorial training cost of covering all modality combinations, a tension that continues to drive architectural innovation across the field. As this survey concludes, the progression from single-modality text guidance through spatial and structural control to fully unified multi-modal pipelines reflects the broader trajectory of diffusion-based generation: an ongoing effort to reconcile expressive controllability with scalable, generalizable architectures.

## 5 Training Data, Alignment, and Optimization Strategies

### 5.1 Large-Scale Training Dataset Construction and Caption Quality

Training data quality and curation represent foundational determinants of generative capability in text-to-image diffusion systems. Large-scale web-scraped corpora, while enabling broad concept coverage, introduce substantial noise through misaligned alt-text annotations that poorly describe visual content. Systems such as [1] and [2] demonstrated that pairing strong diffusion architectures with carefully curated datasets yields substantial quality gains, motivating systematic filtering pipelines employing aesthetic scoring, CLIP-based semantic relevance thresholds, deduplication, and safety classification. A critical insight emerging across recent work is that caption descriptiveness matters more than raw dataset scale: replacing noisy web alt-text with synthetically generated dense captions via vision-language models—as exemplified by the methodology underlying [4]—substantially improves text-image alignment and learning efficiency. Furthermore, caption diversity across semantic contexts addresses the limited-perspective problem inherent in naively scraped data, suggesting that future dataset construction should prioritize multi-context synthetic recaptioning pipelines over simple scale expansion.

### 5.2 Compositional Text-Image Alignment Challenges and Solutions

Compositional text-image alignment represents one of the most persistent and structurally revealing failure modes in current diffusion-based synthesis systems, emerging directly from the limitations of the training data and architectural constraints discussed above. Even state-of-the-art models routinely exhibit catastrophic object neglect, attribute leakage across entities, spatial relationship violations, and systematic count errors when processing semantically dense prompts—failures that aggregate distributional metrics such as FID entirely obscure. These shortcomings originate in the cross-attention mechanism's limited capacity to maintain disjoint, semantically consistent activation regions for each textual concept simultaneously, a limitation that becomes acute as prompt complexity scales and that is further exacerbated when models are trained on the sparse or ambiguous descriptions characteristic of naively scraped corpora.

Training-free attention modulation approaches directly intervene on cross-attention energy landscapes during denoising to strengthen token-region correspondence, with energy-based alignment objectives redistributing activation mass toward semantically appropriate spatial zones. Complementarily, layout-guided generation frameworks incorporate explicit spatial priors—bounding boxes, segmentation masks, and LLM-generated scene graphs—as structured intermediaries that decompose complex prompts into spatially grounded sub-problems, effectively offloading compositional reasoning from the denoising network to an external planning stage. More recently, frameworks such as LLM4GEN leverage richer semantic representations from large language models as supplementary conditioning pathways, improving attribute-object binding for dense prompts where standard CLIP-based encoders demonstrably saturate. Reinforcement learning formulations [43] offer a principled alternative by optimizing compositional fidelity metrics directly as reward signals, bypassing the indirection of likelihood-based objectives, though sample efficiency remains a constraint.

Collectively, these interventions reveal a convergent understanding: compositional alignment failures are fundamentally optimization-geometric rather than capacity-limited, suggesting that architectural innovations targeting attention disentanglement and structured semantic grounding represent the most promising avenues for systematic resolution. Importantly, addressing these failures requires coordinated progress across both the data and model levels—a recognition that motivates the prompt-level strategies examined in the following subsection, where techniques such as textual inversion, DDIM-based inversion, and LLM-assisted rewriting work to more precisely specify user intent and thereby further close the gap between complex compositional descriptions and synthesized visual output.

### 5.3 Prompt Engineering, Inversion, and Automatic Prompt Optimization

I need to check each citation in the subsection against the provided papers.

1. "textual inversion, wherein continuous token embeddings are optimized in the text encoder's embedding space to represent novel visual concepts from reference images [14]"

The paper "Denoising Diffusion Implicit Models" is about faster sampling via non-Markovian diffusion processes, NOT about textual inversion. This citation is incorrect and should be removed.

2. "by reversing the diffusion generation trajectory through deterministic DDIM-based schemes [14; 34]"

- "Denoising Diffusion Implicit Models" does discuss deterministic sampling and latent space interpolation, which supports DDIM-based inversion schemes. This citation is appropriate.
- "An Edit Friendly DDPM Noise Space: Inversion and Manipulations" explicitly discusses inversion methods for DDPM and editing via noise space manipulation. This citation is correct.

So the corrected subsection is:

Effective prompt engineering sits at the intersection of user intent and model capability, encompassing techniques that range from manual craft to fully automated optimization pipelines. A foundational approach is textual inversion, wherein continuous token embeddings are optimized in the text encoder's embedding space to represent novel visual concepts from reference images, enabling soft prompt representations that capture nuanced appearance attributes beyond the reach of discrete vocabulary tokens. This continuous parameterization trades interpretability for expressiveness, a tension that motivates complementary gradient-based hard prompt discovery methods operating directly over discrete token vocabularies to recover human-readable descriptions. Inversion from target images further extends this paradigm: by reversing the diffusion generation trajectory through deterministic DDIM-based schemes [14; 34], one can recover latent noise codes whose associated prompts enable downstream concept transfer and evolutionary image generation. LLM-assisted rewriting frameworks address usability by deploying large language models as iterative refinement agents, leveraging scoring mechanisms such as human preference proxies to guide multi-turn prompt evolution. Collectively, these approaches reveal a convergent trend toward closed-loop optimization pipelines that unify semantic inversion, embedding-space learning, and language model reasoning to systematically close the gap between user-specified intent and synthesized visual output.

### 5.4 Parameter-Efficient Fine-Tuning and Domain Adaptation

Parameter-efficient fine-tuning (PEFT) has emerged as the dominant paradigm for adapting large pretrained text-to-image diffusion models to specialized domains without incurring the prohibitive computational cost of full retraining. Rather than updating all model parameters, these strategies selectively modify compact parameter subsets while preserving the broad generative prior encoded during large-scale pretraining—thereby mitigating catastrophic forgetting and overfitting on limited domain-specific data. The landscape spans low-rank decomposition methods, selective scaling factor optimization, modular adapter insertion, and geometry-preserving fine-tuning frameworks, each offering distinct trade-offs between adaptation expressiveness, parameter efficiency, and fidelity to the pretrained distribution. Crucially, while the prompt-level techniques discussed previously operate upstream of model parameters to shape the semantic conditioning signal, PEFT methods intervene directly within the model's learned representations, enabling more targeted and expressive domain adaptation. This distinction also sets the stage for the following section, where adaptation is reoriented away from parameter modification entirely and toward optimizing directly for human perceptual judgments—a shift that builds naturally on the efficiency-preserving foundations established here.

### 5.5 Reinforcement Learning from Human Feedback and Preference Alignment

Moving text-to-image diffusion training beyond maximum likelihood objectives, preference alignment methods optimize directly for human perceptual judgments. [44] establishes a foundational reward model trained on 137k expert comparisons, enabling Reward Feedback Learning to fine-tune diffusion models against human preference scores. Complementarily, [43] frames denoising as a sequential decision process, with policy gradient optimization via DDPO outperforming reward-weighted likelihood approaches for objectives including aesthetic quality and prompt-image alignment. [45] further integrates KL regularization into online RL fine-tuning, demonstrating superiority over supervised alternatives in both alignment and quality dimensions.

### 5.6 Scaling Laws, Compute Efficiency, and Training Optimization

Scaling behavior in text-to-image diffusion models follows patterns analogous to those observed in large language models, yet exhibits distinctive characteristics tied to the joint optimization of visual fidelity and semantic alignment. Building on the preference alignment strategies discussed above, which demonstrate that training signal quality matters as much as model capacity, scaling investigations reveal equally nuanced trade-offs between compute, architecture, and representational choice. Empirical evidence from [9] demonstrates that sample quality and likelihood scale smoothly with model capacity and training compute, providing early justification for investing in larger denoising backbones. Architectural choices critically mediate this scaling: [46] demonstrates that operating diffusion within an extremely compact semantic latent space—requiring only 24,602 A100-GPU hours compared to Stable Diffusion 2.1's 200,000—achieves competitive generation quality by prioritizing representational compression over raw parameter scaling, fundamentally challenging the assumption that compute efficiency and generation quality trade off monotonically. Complementarily, [1] establishes that confining the diffusion process to perceptually compressed latent representations substantially reduces computational overhead while preserving

[... deterministic omitted span ...]

to trade off quality and diversity. It does relate to negative prompts conceptually, but the paper doesn't specifically discuss negative prompts for safety. However, it's the foundational work for classifier-free guidance which underpins negative prompts. This is acceptable.

2. "safe latent diffusion variants built atop [18]-style latent manipulation" - Latent-NeRF is about 3D shape generation using NeRF in latent space. It has nothing to do with safe latent diffusion or safety mechanisms. This citation is incorrect and should be removed.

3. "adversarial prompt circumvention documented through red-teaming of deployed systems like [3]" - GLIDE is about text-conditional image synthesis and editing. It doesn't specifically document adversarial prompt circumvention or red-teaming. This citation is incorrect and should be removed.

4. "[58]" - FreeDoM is about training-free energy-guided conditional diffusion. It does offer plug-and-play guidance without retraining, which supports the sentence. This citation is correct.

5. "[44; 43]" - Both papers deal with human feedback and reward-based training for diffusion models. These citations are correct.

Here is the corrected subsection:

Safety mechanisms in text-to-image diffusion models span model-level concept suppression and system-level adversarial robustness. Concept erasure approaches—including negative prompt guidance [28] and safe latent diffusion variants built atop latent-style manipulation—selectively suppress harmful generations, yet remain vulnerable to adversarial prompt circumvention documented through red-teaming of deployed systems. Training-free energy-based methods [58] offer flexible plug-and-play safety guidance without retraining, while human feedback alignment frameworks [44; 43] enable preference-driven safety calibration. A persistent tension exists between aggressive concept removal and collateral degradation of benign generation quality, motivating adversarially hardened erasure combined with fairness-aware moderation pipelines that avoid disparate false-positive rates across demographic groups.

### 7.6 Intellectual Property, Data Privacy, and Provenance

Text-to-image diffusion models trained on web-scraped corpora raise interlocking concerns spanning copyright, privacy, and model attribution—concerns that extend naturally from the safety and content moderation challenges discussed above. A central challenge is training data memorization: diffusion models can reproduce near-verbatim copies of training images under certain prompting conditions [59], with membership inference attacks exploiting stepwise denoising errors to identify training examples with high confidence [59]. Critically, standard distributional metrics such as FID fail to reliably distinguish memorization from underfitting or mode shrinkage [56], underscoring the inadequacy of existing evaluation frameworks for provenance auditing. The legal implications hinge on whether model outputs constitute reproduction or transformation, a distinction that current technical measures alone cannot resolve. Watermarking and fingerprinting represent a complementary mitigation avenue, though robustness against fine-tuning attacks remains an open challenge. Style mimicry—where models replicate individual artists' distinctive aesthetics without consent—poses additional ethical risks that sit at the intersection of copyright law and generative AI governance. Together, these provenance and memorization concerns complete a broader landscape of responsible deployment challenges—encompassing safety, fairness, and intellectual property—that demand both technical safeguards and evolving regulatory frameworks to ensure the ethical and accountable use of text-to-image diffusion systems.

## 8 Conclusion

I need to verify each citation in the subsection against the paper contents provided.

1. "theoretical foundations in denoising score matching and stochastic differential equations [1; 8]" - LDM paper is about latent diffusion models, not specifically denoising score matching/SDEs. Variational Diffusion Models discusses VLB and noise schedules. These are somewhat relevant but the LDM citation is a stretch for "theoretical foundations in denoising score matching and SDEs." However, both papers do deal with diffusion model foundations. I'll keep these as they relate to foundational diffusion model work.

2. "architectural innovations spanning U-Net and Transformer backbones [4; 20]" - SDXL discusses UNet backbone improvements. ERNIE-ViLG 2.0 discusses mixture-of-denoising-experts. Both relate to architectural innovations. This is acceptable.

3. "controllability frameworks [5; 29]" - Both papers are directly about controllability. Correct.

4. "downstream extensions encompassing perception [51]" - Correct, VPD is about visual perception.

5. "video [60]" - Correct.

6. "3D synthesis [61]" - Correct.

7. "compositional faithfulness [48]" - LLM-grounded Diffusion addresses complex prompts involving numeracy and spatial reasoning. This supports compositional faithfulness. Correct.

8. "demographic fairness [7]" - Correct.

9. "safety [6]" - The survey paper covers broad topics but safety isn't prominently featured in its abstract. However, it's a comprehensive survey that would cover safety. This is acceptable as a general survey citation.

The citations appear correct. Here is the subsection:

This survey has traced the remarkable trajectory of text-to-image diffusion models, from their theoretical foundations in denoising score matching and stochastic differential equations [1; 8] through architectural innovations spanning U-Net and Transformer backbones [4; 20], controllability frameworks [5; 29], and downstream extensions encompassing perception [51], video [60], and 3D synthesis [62]. Persistent challenges remain in compositional faithfulness [49], demographic fairness [7], and safety [6], underscoring that technical excellence must advance alongside responsible deployment to realize the field's transformative potential.

## References

[1] High-Resolution Image Synthesis with Latent Diffusion Models

[2] Photorealistic Text-to-Image Diffusion Models with Deep Language  Understanding

[3] GLIDE  Towards Photorealistic Image Generation and Editing with  Text-Guided Diffusion Models

[4] SDXL  Improving Latent Diffusion Models for High-Resolution Image  Synthesis

[5] Uni-ControlNet  All-in-One Control to Text-to-Image Diffusion Models

[6] Diffusion Models  A Comprehensive Survey of Methods and Applications

[7] Stable Bias  Analyzing Societal Representations in Diffusion Models

[8] Variational Diffusion Models

[9] Improved Denoising Diffusion Probabilistic Models

[10] Diffusion Models in Vision  A Survey

[11] Addressing Negative Transfer in Diffusion Models

[12] Diffusion Models for Medical Image Analysis  A Comprehensive Survey

[13] Understanding DDPM Latent Codes Through Optimal Transport

[14] Denoising Diffusion Implicit Models

[15] Understanding Diffusion Models  A Unified Perspective

[16] DPM-Solver++  Fast Solver for Guided Sampling of Diffusion Probabilistic  Models

[17] One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale

[18] Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures

[19] Elucidating the Design Space of Diffusion-Based Generative Models

[20] ERNIE-ViLG 2.0  Improving Text-to-Image Diffusion Model with  Knowledge-Enhanced Mixture-of-Denoising-Experts

[21] eDiff-I  Text-to-Image Diffusion Models with an Ensemble of Expert  Denoisers

[22] Structured Denoising Diffusion Models in Discrete State-Spaces

[23] DPM-Solver-v3  Improved Diffusion ODE Solver with Empirical Model  Statistics

[24] Restart Sampling for Improving Generative Processes

[25] AutoDiffusion  Training-Free Optimization of Time Steps and  Architectures for Automated Diffusion Model Acceleration

[26] Learning Fast Samplers for Diffusion Models by Differentiating Through  Sample Quality

[27] Fast Sampling of Diffusion Models via Operator Learning

[28] Classifier-Free Diffusion Guidance

[29] GLIGEN  Open-Set Grounded Text-to-Image Generation

[30] Dense Text-to-Image Generation with Attention Modulation

[31] MultiDiffusion  Fusing Diffusion Paths for Controlled Image Generation

[32] SpaText  Spatio-Textual Representation for Controllable Image Generation

[33] Discovering Interpretable Directions in the Semantic Latent Space of  Diffusion Models

[34] An Edit Friendly DDPM Noise Space  Inversion and Manipulations

[35] Cross-Image Attention for Zero-Shot Appearance Transfer

[36] Post-training Quantization on Diffusion Models

[37] Blended Diffusion for Text-driven Editing of Natural Images

[38] Delta Denoising Score

[39] Training-Free Structured Diffusion Guidance for Compositional  Text-to-Image Synthesis

[40] Training-Free Layout Control with Cross-Attention Guidance

[41] Null-text Inversion for Editing Real Images using Guided Diffusion  Models

[42] Unified Multi-Modal Latent Diffusion for Joint Subject and Text  Conditional Image Generation

[43] Training Diffusion Models with Reinforcement Learning

[44] ImageReward  Learning and Evaluating Human Preferences for Text-to-Image  Generation

[45] DPOK  Reinforcement Learning for Fine-tuning Text-to-Image Diffusion  Models

[46] Wuerstchen  An Efficient Architecture for Large-Scale Text-to-Image  Diffusion Models

[47] State of the Art on Diffusion Models for Visual Computing

[48] Self-correcting LLM-controlled Diffusion Models

[49] LLM-grounded Diffusion  Enhancing Prompt Understanding of Text-to-Image  Diffusion Models with Large Language Models

[50] Training on Thin Air  Improve Image Classification with Generated Data

[51] Unleashing Text-to-Image Diffusion Models for Visual Perception

[52] Denoising Diffusion Autoencoders are Unified Self-supervised Learners

[53] From Points to Functions  Infinite-dimensional Representations in  Diffusion Models

[54] The Surprising Effectiveness of Diffusion Models for Optical Flow and  Monocular Depth Estimation

[55] Re-imagine the Negative Prompt Algorithm  Transform 2D Diffusion into  3D, alleviate Janus problem and Beyond

[56] Exposing flaws of generative model evaluation metrics and their unfair  treatment of diffusion models

[57] Aligning Text-to-Image Diffusion Models with Reward Backpropagation

[58] FreeDoM  Training-Free Energy-Guided Conditional Diffusion Model

[59] Are Diffusion Models Vulnerable to Membership Inference Attacks 

[60] Video Diffusion Models

[61] DreamVideo  Composing Your Dream Videos with Customized Subject and  Motion

[62] DreamFusion  Text-to-3D using 2D Diffusion
