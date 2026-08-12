# Registered common-window excerpt

# Text-to-Image Diffusion Models: A Comprehensive Survey of Architectures, Methods, and Applications

## 1 Introduction and Background

### 1.1 The Evolution of Text-to-Image Generation

The field of text-to-image generation has undergone a remarkable transformation over the past decade, driven by successive waves of architectural innovation and the availability of large-scale multimodal datasets. Understanding this evolution is essential for appreciating the capabilities and limitations of modern diffusion-based systems, as well as the theoretical foundations described in the sections that follow.

**Early GAN-Based Approaches**

The earliest neural approaches to text-to-image generation were rooted in Generative Adversarial Networks (GANs), a framework in which a generator and discriminator engage in adversarial training to produce realistic images [1]. Landmark models in this era include StackGAN, which introduced a multi-stage generation strategy that progressively refined images from low to high resolution by conditioning on text descriptions at each stage, and AttnGAN, which incorporated cross-modal attention mechanisms to focus generation on relevant image regions corresponding to specific words in the input description [2]. These attention-guided architectures represented a significant step forward in aligning visual content with textual semantics, and subsequent work such as GR-GAN further explored gradual refinement strategies that decomposed the generation process into coarse-to-fine stages with complementary sentence-level and word-level text constraints [3].

Despite these advances, GAN-based text-to-image models suffered from several fundamental limitations. Training instability, mode collapse, and the tendency to produce blurry or artifact-laden images at high resolutions constrained their practical utility [4]. Moreover, the expressivity of GAN generators was inherently limited by the adversarial objective, making it difficult to capture the full diversity of complex textual descriptions involving multiple objects, spatial relationships, and fine-grained attributes. Attempts to scale GANs to larger datasets and more expressive architectures, such as GigaGAN [5] and StyleGAN-T [6], partially addressed these issues but struggled to match the generation quality achievable by emerging alternative paradigms.

**Autoregressive Models**

The advent of large-scale transformer architectures opened a new chapter in text-to-image generation. Autoregressive models reframed image generation as a sequence prediction problem by representing images as sequences of discrete tokens produced by vector-quantized autoencoders. DALL-E, developed by OpenAI, was among the first to demonstrate that a single large transformer trained on billions of image-text pairs could generate coherent and creative images from arbitrary textual prompts. Shortly thereafter, CogView [7] introduced a four-billion-parameter transformer with a VQ-VAE tokenizer, achieving state-of-the-art FID scores on MS-COCO and demonstrating the feasibility of training competitive text-to-image models using a unified autoregressive framework. These models demonstrated a remarkable ability to generalize to novel concept combinations, benefiting from the rich linguistic representations encoded in large-scale language model pretraining.

However, autoregressive models also faced significant challenges. Their sequential token generation process incurred substantial computational costs at inference time, as generating a single high-resolution image could require thousands of autoregressive steps. Furthermore, errors introduced early in the generation sequence tended to propagate and compound through subsequent steps, limiting spatial coherence and image fidelity. Models such as EMAGE explored non-autoregressive generation strategies that produce image tokens in parallel to address this latency bottleneck [8], but these alternatives often sacrificed generation quality for speed.

**The Rise of Diffusion-Based Paradigms**

The limitations of both GAN-based and autoregressive approaches created fertile ground for a new class of generative models based on diffusion processes. Rather than relying on adversarial objectives or sequential token prediction, diffusion models formulate image generation as the iterative reversal of a noise corruption process, learning to progressively denoise a Gaussian noise signal into a high-fidelity image over many denoising steps. The introduction of Denoising Diffusion Probabilistic Models (DDPMs) demonstrated that this framework could achieve image quality surpassing GANs on established benchmarks while avoiding training instabilities. The subsequent development of latent diffusion models, which perform the diffusion process in the compressed latent space of a pretrained autoencoder rather than in pixel space, dramatically improved computational efficiency without sacrificing visual quality [9]. The precise mathematical mechanisms underlying these processes—including the forward corruption process, the learned reverse denoising network, and the variational training objective—are described in detail in the following section.

The application of diffusion models to text-to-image synthesis quickly produced transformative results. Imagen [10] demonstrated that combining large pretrained language models, specifically T5, with cascaded diffusion architectures could achieve unprecedented levels of photorealism and language understanding. DALL-E 2 extended this paradigm by conditioning a diffusion decoder on CLIP image embeddings derived from text prompts. Stable Diffusion made high-quality text-to-image generation broadly accessible through its open-source release and efficient latent diffusion architecture. Models such as Kandinsky [11] further refined this approach by combining image prior models with latent diffusion techniques, while RAPHAEL [12] introduced mixture-of-experts designs to enhance text-image alignment. CogView3 [13] explored relay diffusion frameworks to reduce both training and inference costs while maintaining competitive generation quality.

The rapid ascendance of diffusion models can be attributed to several key advantages over their predecessors. Unlike GANs, diffusion models do not require adversarial training and thus benefit from greater training stability and coverage of the data distribution. Unlike autoregressive models, the iterative denoising process can be guided by rich conditioning signals including text embeddings, spatial layouts, and reference images, enabling fine-grained semantic control. These properties, combined with the growing scale of training data and model parameters, have established diffusion models as the dominant paradigm for state-of-the-art text-to-image generation [14], setting the stage for the formal theoretical treatment of denoising diffusion probabilistic models that follows.

### 1.2 Fundamentals of Denoising Diffusion Probabilistic Models

## 1.2 Denoising Diffusion Probabilistic Models

Building on the historical trajectory described in Section 1.1—where the limitations of GAN-based and autoregressive approaches motivated the search for more stable and expressive generative frameworks—Denoising Diffusion Probabilistic Models (DDPMs) have emerged as a powerful class of deep generative models that achieve state-of-the-art performance across a wide range of synthesis tasks [15]. The theoretical framework of DDPMs rests on two coupled stochastic processes: a forward diffusion process that systematically corrupts data with Gaussian noise, and a learned reverse process that reconstructs clean data from noise through iterative denoising. Understanding these processes in detail is essential both for appreciating the advantages of diffusion models over their predecessors and for following the score-based and SDE-based theoretical extensions presented in Section 1.3.

**Forward Diffusion Process.** Given a data sample $\mathbf{x}_0 \sim q(\mathbf{x}_0)$, the forward process defines a Markov chain that gradually adds Gaussian noise over $T$ timesteps:

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t}\, \mathbf{x}_{t-1},\, \beta_t \mathbf{I}),$$

where $\{\beta_t\}_{t=1}^T$ is a variance schedule controlling the rate of noise injection. A key convenience of this formulation is that the marginal distribution at any timestep $t$ admits a closed-form expression [16]:

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\, \mathbf{x}_0,\, (1 - \bar{\alpha}_t) \mathbf{I}),$$

where $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$. As $T \to \infty$ and the schedule is appropriately designed, $\mathbf{x}_T$ converges to an isotropic Gaussian distribution, providing a tractable starting point for generation [15].

**Reverse Denoising Process.** The generative process inverts the forward diffusion by learning a parameterized reverse Markov chain:

$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\, \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)),$$

where a neural network predicts the mean $\boldsymbol{\mu}_\theta$ (and optionally the variance $\boldsymbol{\Sigma}_\theta$) of the denoising transition. In practice, the network is often reparameterized to predict the added noise $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$, leading to a simplified mean estimate [16]:

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right).$$

**Variational Lower Bound Training Objective.** Training is performed by maximizing a variational lower bound (ELBO) on the log-likelihood of the data. The ELBO decomposes into a sum of KL divergence terms at each timestep, measuring how closely the learned reverse transitions match the true posterior $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ [16; 17]. In the noise-prediction parameterization, this objective reduces to a weighted mean-squared error loss:

$$\mathcal{L}_\text{simple} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[18],$$

which is both simple to optimize and empirically effective for high-quality sample generation. As will be discussed in Section 1.3, this noise-prediction objective is also directly interpretable as a form of denoising score matching, underscoring the deep theoretical connections between the DDPM framework and score-based generative models.

**Learned Variance Schedules.** Early formulations fixed the reverse process

[... deterministic omitted span ...]

researchers have proposed more sophisticated adapter-based architectures that dynamically couple LLM representations with the internal conditioning mechanisms of pretrained diffusion models. One prominent paradigm involves training lightweight connector modules that translate LLM-derived text embeddings into conditioning signals compatible with cross-attention layers inside the denoising network. These adapters can be designed to be timestep-aware, recognizing that the informational requirements of the denoising process vary across diffusion timesteps — coarse semantic structure is primarily resolved at high noise levels, while fine-grained details such as attribute assignment and object placement are refined at lower noise levels. A timestep-dependent semantic connector extracts dynamically varying conditioning features from a frozen LLM according to the current denoising timestep, enabling the model to leverage different aspects of the language understanding at different stages of generation. This approach has demonstrated notable improvements in handling dense prompts with multiple co-occurring entities, complex attribute-object bindings, and relational descriptions that routinely challenge standard CLIP-conditioned models.

The broader trend of using LLMs as prompt adapters also intersects with prompt enrichment strategies. Systems such as DiffusionGPT employ LLMs to parse and interpret input prompts, routing them to domain-appropriate generative models and augmenting the prompt representation with structured semantic information [84]. Similarly, interactive systems inspired by DALL-E 3's integration of conversational LLMs explore how natural language dialogue can be used to iteratively refine and expand prompts before passing them to the image generation backend [85]. These approaches collectively highlight a shift toward treating the text encoding stage not as a static lookup but as an active semantic reasoning process.

Furthermore, the integration of LLM-based encoders aligns with findings that emphasize the importance of deep language understanding for compositional generation. Imagen's experiments confirm that a generic large language model pretrained purely on text corpora can serve as a highly effective image-conditioning encoder, even without any image-text paired pretraining [10]. This suggests that the richness of linguistic structure encoded in LLMs transfers beneficially to the visual domain, enabling more accurate rendering of complex scene descriptions, multi-attribute objects, and spatially grounded relationships. As text-to-image systems continue to scale, enhanced text encoders and LLM-based prompt adapters represent a foundational direction for narrowing the semantic alignment gap between user intent and generated visual content. This improved semantic grounding, in turn, sets the stage for more targeted interventions within the diffusion model itself — most notably through direct manipulation of cross-attention maps, which serve as the primary mechanism by which textual conditioning is spatially distributed across the generated image.

### 5.2 Attention Map Manipulation for Semantic Alignment

Cross-attention maps in text-to-image diffusion models encode the spatial correspondence between textual tokens and image regions, making them a natural target for direct manipulation to improve semantic alignment. As established in the previous section, enhanced text encoders and LLM-based adapters provide richer linguistic representations to the diffusion pipeline; however, even with improved textual conditioning, the internal mechanism by which these representations are spatially distributed across the generated image remains a critical bottleneck. When attention maps are poorly calibrated—for instance, when an attribute intended for one object bleeds into another region—the resulting image fails to faithfully reflect the text prompt regardless of the quality of the upstream text encoding. Researchers have therefore developed a variety of techniques that intervene directly in the attention mechanism at inference time or during fine-tuning to steer generation toward more accurate text-image correspondence.

One prominent line of work frames attention manipulation through an energy-based perspective, where desired attention configurations are encoded as energy functions whose gradients guide the denoising process. Such frameworks enable Bayesian-style context updating, progressively refining the model's internal representation of each textual concept as generation proceeds. By penalizing attention activations that deviate from target spatial assignments, these methods encourage individual tokens to attend to their semantically appropriate image regions rather than diffusing their influence across irrelevant areas.

Object-conditioned energy-based alignment has proven particularly valuable for addressing the attribute binding problem, where attributes such as color, shape, or texture are incorrectly associated with unintended objects. By constructing per-object energy terms that enforce localized attention peaks corresponding to each entity and its attributes, these approaches reduce the cross-contamination between distinct semantic concepts, yielding images where "a red cube next to a blue sphere" renders with correct color-object pairings rather than blended or swapped attributes. This class of failure modes is precisely what the richer compositional understanding afforded by LLM-based encoders begins to address at the representation level, but which attention-level interventions tackle more directly within the spatial generation process itself.

Syntactic structure provides another powerful signal for focusing cross-attention. By parsing the input prompt to extract grammatical dependencies—such as adjective-noun pairs or prepositional phrase attachments—methods guided by syntactic constraints can define expected attention topologies, penalizing deviations from linguistically motivated spatial groupings. This allows the model to leverage the compositional structure of language more explicitly, improving multi-object scene understanding and relational accuracy beyond what is achievable through unstructured prompt conditioning alone. In this sense, syntactic guidance at the attention level complements the semantic richness provided by upstream LLM encoders, forming a coherent two-stage approach to semantic alignment.

A complementary challenge arises in image editing scenarios, where the goal is to modify specific aspects of an existing image while preserving unrelated content. In this context, cross-attention leakage—where editing tokens inadvertently affect regions that should remain unchanged—can degrade editing quality and introduce unwanted artifacts. Dynamic prompt learning strategies have been proposed to address this issue by learning lightweight token adjustments that repair leaking attention patterns during the editing process, ensuring that modification signals are spatially contained and do not propagate to unrelated image regions. Together, these attention manipulation techniques represent a coherent and effective toolkit for improving the fidelity of text-image correspondence across both generation and editing tasks in diffusion-based systems. While these methods operate at the level of internal model activations, a complementary and practically significant direction involves intervening even earlier in the pipeline—at the level of the textual input itself—through prompt engineering strategies, which are explored in the following section.

### 5.3 Prompt Engineering and Automatic Prompt Optimization

Prompt engineering has emerged as a practical and impactful strategy for improving the semantic alignment between user intent and the outputs of text-to-image diffusion models. While the previous approaches discussed in this survey—such as cross-attention manipulation and energy-based guidance—intervene directly in model internals, prompt engineering operates at the input level, focusing on crafting, refining, and structuring textual inputs themselves to elicit more accurate and coherent visual outputs from pretrained generative systems without modifying model weights or architectural components.

A fundamental observation motivating this line of work is that text-to-image models are highly sensitive to the precise wording, structure, and content of input prompts. Small variations in phrasing can lead to dramatically different generated images, suggesting that the space of effective prompts is complex and non-trivial to navigate. This sensitivity, closely related to the attention-level misalignments discussed in the preceding sections, has motivated a variety of strategies ranging from manual best-practice guidelines to fully automated optimization frameworks.

Interactive prompt exploration systems represent one category of approaches, where large language models are leveraged as intelligent assistants to help users iteratively refine and expand their prompts. By engaging in a dialogue-style interaction, users can progressively disambiguate their intent, add relevant descriptive details, and receive suggestions for more expressive or specific language. Such systems democratize access to effective prompt crafting by reducing the expertise required from end users, enabling individuals without specialized knowledge of model behavior to achieve high-quality semantic alignment.

Automatic prompt optimization frameworks constitute a more systematic approach, wherein an optimization loop iteratively revises prompts to maximize quantitative alignment scores between generated images and intended semantics. These frameworks typically employ metrics such as CLIP-based consistency scores or vision-language model evaluations as proxy objectives, and use language models to propose candidate prompt modifications. The iterative revision process can correct common failure modes such as missing objects, incorrect attribute bindings, or ambiguous spatial descriptions—precisely the categories of errors that attention manipulation and discriminative guidance methods aim to address at the model level—thereby improving the overall fidelity of generated

[... deterministic omitted span ...]

intellectual property violations—escalate correspondingly. Achieving the right balance between generative expressiveness and safety requires robust content filtering, watermarking, concept erasure [111], and model-level suppression mechanisms that are resistant to adversarial circumvention. The challenge of concept ablation and machine unlearning discussed previously illustrates precisely this tension: overly aggressive removal degrades model utility, while insufficiently targeted removal leaves residual vulnerabilities. Moreover, reinforcement learning from human feedback [73] and reward-based alignment techniques must be extended to incorporate not only aesthetic quality but also fairness, safety, and ethical constraints. The co-evolution of evaluation, safety mechanisms, and generative capability is perhaps the most complex long-term challenge facing the field, requiring interdisciplinary collaboration across machine learning, ethics, law, and social science to ensure that these powerful tools are developed and deployed responsibly.

**Improving Compositional and Semantic Understanding.** Despite notable advances in cross-attention manipulation [79; 112; 113] and layout-guided generation, accurately rendering complex compositional scenes with multiple interacting objects, correct attribute bindings, and precise spatial relationships remains a persistent challenge. Future work should integrate richer structural reasoning—drawing on scene graphs, structured language representations, and planning-based approaches—to bridge the gap between what users describe and what models reliably produce.

In summary, the future of text-to-image diffusion research lies not only in scaling model capacity and data, but in developing architectures and training paradigms that are efficient, controllable, compositionally faithful, culturally inclusive, and aligned with human values and safety requirements. The challenges outlined above—spanning generation quality, computational accessibility, cultural representation, evaluation rigor, safety, and compositional understanding—are deeply interconnected, and progress on any one front will often depend on simultaneous advances across the others. Addressing these challenges collectively will determine whether these models can fulfill their transformative potential across creative, scientific, and societal applications.


## References

[1] Image Synthesis with Adversarial Networks  a Comprehensive Survey and  Case Studies

[2] A Simple and Effective Baseline for Attentional Generative Adversarial  Networks

[3] GR-GAN  Gradual Refinement Text-to-image Generation

[4] Evolving GAN Formulations for Higher Quality Image Synthesis

[5] Scaling up GANs for Text-to-Image Synthesis

[6] StyleGAN-T  Unlocking the Power of GANs for Fast Large-Scale  Text-to-Image Synthesis

[7] CogView  Mastering Text-to-Image Generation via Transformers

[8] Emage  Non-Autoregressive Text-to-Image Generation

[9] A Survey of Diffusion Based Image Generation Models  Issues and Their  Solutions

[10] Photorealistic Text-to-Image Diffusion Models with Deep Language  Understanding

[11] Kandinsky  an Improved Text-to-Image Synthesis with Image Prior and  Latent Diffusion

[12] RAPHAEL  Text-to-Image Generation via Large Mixture of Diffusion Paths

[13] CogView3  Finer and Faster Text-to-Image Generation via Relay Diffusion

[14] RenAIssance  A Survey into AI Text-to-Image Generation in the Era of  Large Model

[15] Diffusion Models in Vision  A Survey

[16] Denoising Diffusion Probabilistic Models in Six Simple Steps

[17] Learning to Efficiently Sample from Diffusion Probabilistic Models

[18] The Theta Number of Simplicial Complexes

[19] Improved Denoising Diffusion Probabilistic Models

[20] Denoising Diffusion Implicit Models

[21] Pseudo Numerical Methods for Diffusion Models on Manifolds

[22] Fast Diffusion Probabilistic Model Sampling through the lens of Backward  Error Analysis

[23] Bilateral Denoising Diffusion Models

[24] A Variational Perspective on Diffusion-Based Generative Models and Score  Matching

[25] Score-Based Generative Modeling through Stochastic Differential  Equations

[26] The probability flow ODE is provably fast

[27] Convergence for score-based generative modeling with polynomial  complexity

[28] Sampling is as easy as learning the score  theory for diffusion models  with minimal data assumptions

[29] Convergence of score-based generative modeling for general data  distributions

[30] FP-Diffusion  Improving Score-based Diffusion Models by Enforcing the  Underlying Score Fokker-Planck Equation

[31] Closing the ODE-SDE gap in score-based diffusion models through the  Fokker-Planck equation

[32] Understanding Diffusion Models  A Unified Perspective

[33] Score-Based Generative Modeling with Critically-Damped Langevin  Diffusion

[34] Score-based Diffusion Models in Function Space

[35] High-Resolution Image Synthesis with Latent Diffusion Models

[36] Wuerstchen  An Efficient Architecture for Large-Scale Text-to-Image  Diffusion Models

[37] Differentially Private Latent Diffusion Models

[38] Align your Latents  High-Resolution Video Synthesis with Latent  Diffusion Models

[39] Efficient Video Diffusion Models via Content-Frame Motion-Latent  Decomposition

[40] LD-Pruner  Efficient Pruning of Latent Diffusion Models using  Task-Agnostic Insights

[41] Hierarchical Text-Conditional Image Generation with CLIP Latents

[42] clip2latent  Text driven sampling of a pre-trained StyleGAN using  denoising diffusion and CLIP

[43] Object-Attribute Binding in Text-to-Image Generation  Evaluation and  Control

[44] eDiff-I  Text-to-Image Diffusion Models with an Ensemble of Expert  Denoisers

[45] SceneGenie  Scene Graph Guided Diffusion Models for Image Synthesis

[46] Augmenting CLIP with Improved Visio-Linguistic Reasoning

[47] BERT  Pre-training of Deep Bidirectional Transformers for Language  Understanding

[48] On the Difference of BERT-style and CLIP-style Text Encoders

[49] ELLA  Equip Diffusion Models with LLM for Enhanced Semantic Alignment

[50] SUR-adapter  Enhancing Text-to-Image Pre-trained Diffusion Models with  Large Language Models

[51] Diffusion Models Beat GANs on Image Synthesis

[52] Elucidating The Design Space of Classifier-Guided Diffusion Generation

[53] Towards Practical Plug-and-Play Diffusion Models

[54] FreeDoM  Training-Free Energy-Guided Conditional Diffusion Model

[55] Universal Guidance for Diffusion Models

[56] Understanding Training-free Diffusion Guidance  Mechanisms and  Limitations

[57] Classifier-Free Diffusion Guidance

[58] GLIDE  Towards Photorealistic Image Generation and Editing with  Text-Guided Diffusion Models

[59] Theoretical Insights for Diffusion Guidance  A Case Study for Gaussian  Mixture Models

[60] Rethinking the Spatial Inconsistency in Classifier-Free Diffusion  Guidance

[61] Bridging the Gap  Addressing Discrepancies in Diffusion Model Training  for Classifier-Free Guidance

[62] Analysis of Classifier-Free Guidance Weight Schedulers

[63] Adaptive Guidance  Training-free Acceleration of Conditional Diffusion  Models

[64] On Distillation of Guided Diffusion Models

[65] Conditioning Diffusion Models via Attributes and Semantic Masks for Face  Generation

[66] Improving Sample Quality of Diffusion Models Using Self-Attention  Guidance

[67] Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance

[68] CADS  Unleashing the Diversity of Diffusion Models through  Condition-Annealed Sampling

[69] Towards Controllable Diffusion Models via Reward-Guided Exploration

[70] Gradient Guidance for Diffusion Models  An Optimization Perspective

[71] End-to-End Diffusion Latent Optimization Improves Classifier Guidance

[72] Fine-Tuning of Continuous-Time Diffusion Models as Entropy-Regularized  Control

[73] TextCraftor  Your Text Encoder Can be Image Quality Controller

[74] Scalable Diffusion Models with Transformers

[75] Nested Diffusion Processes for Anytime Image Generation

[76] A Comprehensive Survey on Knowledge Distillation of Diffusion Models

[77] GLIGEN  Open-Set Grounded Text-to-Image Generation

[78] Unraveling the Temporal Dynamics of the Unet in Diffusion Models

[79] Training-Free Structured Diffusion Guidance for Compositional  Text-to-Image Synthesis

[80] Uni-ControlNet  All-in-One Control to Text-to-Image Diffusion Models

[81] Adding Conditional Control to Text-to-Image Diffusion Models

[82] Layout-to-Image Generation with Localized Descriptions using ControlNet  with Cross-Attention Control

[83] PanGu-Draw  Advancing Resource-Efficient Text-to-Image Synthesis with  Time-Decoupled Training and Reusable Coop-Diffusion

[84] DiffusionGPT  LLM-Driven Text-to-Image Generation System

[85] Mini-DALLE3  Interactive Text to Image by Prompting Large Language  Models

[86] MaskDiffusion  Boosting Text-to-Image Consistency with Conditional Mask

[87] StyleDiffusion  Prompt-Embedding Inversion for Text-Based Editing

[88] SCEdit  Efficient and Controllable Image Diffusion Generation via Skip  Connection Editing

[89] Continual Diffusion  Continual Customization of Text-to-Image Diffusion  with C-LoRA

[90] All are Worth Words  A ViT Backbone for Diffusion Models

[91] ET3D  Efficient Text-to-3D Generation via Multi-View Distillation

[92] Free-ATM  Exploring Unsupervised Learning on Diffusion-Generated Images  with Free Attention Masks

[93] Scaling Vision Transformers

[94] Diffusion Models Trained with Large Data Are Transferable Visual Models

[95] Your ViT is Secretly a Hybrid Discriminative-Generative Diffusion Model

[96] GenTron  Delving Deep into Diffusion Transformers for Image and Video  Generation

[97] Lafite2  Few-shot Text-to-Image Generation

[98] DALL-Eval  Probing the Reasoning Skills and Social Biases of  Text-to-Image Generation Models

[99] Human Evaluation of Text-to-Image Models on a Multi-Task Benchmark

[100] What is in a Text-to-Image Prompt  The Potential of Stable Diffusion in  Visual Arts Education

[101] Does CLIP Bind Concepts  Probing Compositionality in Large Image Models

[102] Diffusion Model with Perceptual Loss

[103] Latent Diffusion Counterfactual Explanations

[104] Fair Sampling in Diffusion Models through Switching Mechanism

[105] ToDo  Token Downsampling for Efficient Generation of High-Resolution  Images

[106] Scalable Adaptive Computation for Iterative Generation

[107] BK-SDM  A Lightweight, Fast, and Cheap Version of Stable Diffusion

[108] DeepCache  Accelerating Diffusion Models for Free

[109] Cache Me if You Can  Accelerating Diffusion Models through Block Caching

[110] SnapFusion  Text-to-Image Diffusion Model on Mobile Devices within Two  Seconds

[111] Localizing and Editing Knowledge in Text-to-Image Generative Models

[112] Directed Diffusion  Direct Control of Object Placement through Attention  Guidance

[113] Grounded Text-to-Image Synthesis with Attention Refocusing
