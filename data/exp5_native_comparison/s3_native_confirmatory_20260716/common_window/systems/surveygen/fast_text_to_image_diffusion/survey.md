# Registered common-window excerpt

# Text-to-Image Diffusion Models: A Comprehensive Survey

## Foundations and Architectures of Text-to-Image Diffusion Models

### Denoising Diffusion Probabilistic Models (DDPMs)

Denoising Diffusion Probabilistic Models (DDPMs) constitute the mathematical foundation of modern text-to-image synthesis. The forward process incrementally corrupts data by adding Gaussian noise over T timesteps, while the reverse process learns to denoise iteratively, recovering the original data distribution. Score matching provides the theoretical basis for training the denoising network, which predicts the noise added at each step. This framework underpins prominent text-to-image systems such as Imagen ref [20] and VQ-Diffusion ref [12], which extend DDPMs to conditional generation. Spatial guidance methods such as Sketch-Guided Diffusion ref [5] further exploit DDPM's intermediate latent features, demonstrating the versatility of the DDPM framework across diverse generation and control tasks.

### Latent Diffusion Models and VQ-VAE Frameworks

Latent diffusion models (LDMs) address the computational intractability of pixel-space diffusion by operating within compressed latent representations produced by a variational autoencoder. Stable Diffusion, a prominent instantiation, performs the denoising process in a low-dimensional latent space, substantially reducing memory and computational requirements while preserving high image fidelity. Complementarily, the Vector Quantized Diffusion (VQ-Diffusion) model ref [12] combines a VQ-VAE encoder with a conditional DDPM over discrete latent tokens, eliminating unidirectional autoregressive bias and enabling a mask-and-replace diffusion strategy that mitigates error accumulation. Crucially, VQ-Diffusion supports reparameterization-based parallel decoding, achieving a fifteen-fold inference speedup over conventional autoregressive methods ref [12]. These latent-space frameworks collectively establish the efficiency-quality foundation upon which subsequent controllable and personalized generation methods are built.

### Text Conditioning Mechanisms and Language Encoders

Textual conditioning is central to text-to-image diffusion models, with the choice of language encoder critically influencing semantic fidelity and image-text alignment. Imagen ref [20] demonstrated that large pretrained text-only encoders such as T5 yield superior conditioning compared to CLIP-based encoders, with encoder scale proving more impactful than diffusion model scale. ERNIE-ViLG 2.0 ref [7] incorporates fine-grained textual and visual knowledge to enhance semantic grounding. eDiff-I ref [30] exploits multiple embeddings—T5 text, CLIP text, and CLIP image—revealing distinct behavioral contributions from each. Swinv2-Imagen ref [22] further augments T5-based conditioning with scene graph representations to capture relational semantics, collectively demonstrating that richer language encoders substantially improve compositional understanding and generation quality.

### Transformer-Based Diffusion Architectures

Transformer-based architectures have emerged as compelling alternatives to conventional UNet backbones in text-to-image diffusion models. PixArt-α ref [33] introduces a Diffusion Transformer (DiT) that incorporates cross-attention modules to inject textual conditioning, achieving competitive image quality with substantially reduced training costs compared to UNet-based counterparts. Swinv2-Imagen ref [22] proposes a hierarchical vision transformer architecture, Swinv2-UNet, built upon the Swin Transformer to address limitations of CNN-based convolution operations, integrating scene graph representations to enhance semantic fidelity. These transformer-based designs offer improved scalability, global receptive fields, and more effective modeling of long-range spatial dependencies, collectively advancing the expressiveness and efficiency of text-to-image diffusion frameworks.

### Cross-Attention as the Bridge Between Text and Image

Cross-attention layers serve as the primary mechanism linking textual tokens to spatial image features in text-to-image diffusion models. In models such as Stable Diffusion and PixArt-α, cross-attention modules inject text conditions into the denoising network by computing attention between visual feature queries and text-derived keys and values ref [33]. Empirical analyses reveal that these keys and values carry strong semantic meanings associated with object layouts and content, making their manipulation effective for compositional control ref [16]. Exploiting this, Attend-and-Excite guides cross-attention activations to mitigate catastrophic neglect of subject tokens ref [1], while TIME directly edits cross-attention projection matrices to update implicit model assumptions ref [14]. Custom Diffusion further demonstrates that fine-tuning only cross-attention parameters suffices for efficient multi-concept personalization ref [11].

### Ensemble and Mixture-of-Experts Denoising Strategies

A key insight motivating ensemble and mixture-of-experts denoising strategies is that the synthesis behavior of diffusion models qualitatively changes across denoising stages: early steps establish coarse semantic structure guided strongly by text, while later steps refine fine-grained visual details with diminishing textual dependence. eDiff-I exploits this observation by training an ensemble of specialized expert denoisers, each optimized for distinct stages of the iterative generation process, thereby improving text alignment without increasing inference cost ref [30]. Similarly, ERNIE-ViLG 2.0 employs a mixture-of-denoising-experts mechanism, assigning different expert networks to different denoising stages alongside knowledge-enhanced conditioning, achieving state-of-the-art text-image alignment and image fidelity ref [7].

## Controllable and Spatially-Guided Text-to-Image Generation

### Adapter-Based Spatial Control Frameworks

Adapter-based spatial control frameworks address the limitation of text-only conditioning by injecting structured spatial signals into frozen pretrained diffusion models via lightweight auxiliary modules. ControlNet ref [4] introduces zero-initialized convolutional adapters that clone the encoder layers of Stable Diffusion, enabling conditioning on edges, depth maps, segmentation masks, and human poses without corrupting pretrained weights. T2I-Adapter ref [48] proposes even more lightweight adapters that align external control signals with the model's internal knowledge, supporting composable multi-condition control. ControlNet-XS ref [36] reframes spatial control as a feedback-control system, increasing communication bandwidth between the controlling network and the generation backbone, yielding improved fidelity with fewer parameters and faster inference.

### Training-Free Spatial Guidance Approaches

Training-free spatial guidance approaches enable precise spatial control over pretrained text-to-image diffusion models without requiring additional training or fine-tuning. ZestGuide ref [21] exploits implicit segmentation maps extracted from cross-attention layers to align generated content with user-provided segmentation masks in a zero-shot manner, outperforming prior supervised methods. BoxDiff ref [23] introduces three spatial constraints—Inner-Box, Outer-Box, and Corner Constraints—seamlessly integrated into the denoising process to enforce bounding-box-level object placement. Sketch-Guided Diffusion ref [5] trains a lightweight per-pixel Latent Guidance Predictor on limited data to propagate spatial gradients from sketch inputs back through the diffusion network, accommodating out-of-domain free-hand drawings without dedicated model retraining.

### Layout- and Bounding-Box-Conditioned Generation

Layout- and bounding-box-conditioned generation extends text-to-image diffusion models by incorporating explicit spatial grounding signals such as bounding boxes and scene layouts. GLIGEN ref [17] introduces trainable gated layers into frozen pretrained models, enabling open-world grounded generation conditioned on caption-bounding box pairs, achieving strong zero-shot performance on COCO and LVIS. LLM-grounded Diffusion ref [32] adopts a two-stage pipeline wherein a large language model first generates scene layouts as captioned bounding boxes, which subsequently guide an off-the-shelf diffusion model, significantly improving spatial reasoning and numeracy. BoxDiff ref [23] proposes a training-free alternative using inner-box, outer-box, and corner spatial constraints integrated directly into the denoising process, requiring no additional annotated layout data.

### Compositional and Attribute-Binding Control

Compositional and attribute-binding control addresses fundamental failure modes in text-to-image diffusion models, including catastrophic neglect of subjects and incorrect attribute-to-object binding. Attend-and-Excite ref [1] introduces Generative Semantic Nursing, intervening during inference to strengthen cross-attention activations for all subject tokens, ensuring complete subject generation. Training-Free Structured Diffusion Guidance ref [16] incorporates linguistic structures into cross-attention manipulation, demonstrating that keys and values encode semantic layouts, achieving 5–8% improvement in compositional accuracy. Ranni ref [51] employs a semantic panel as middleware, parsing visual concepts via large language models to inject structured control signals, substantially improving multi-object and attribute-binding fidelity in complex prompts.

### Multi-Modal and Image-Prompt Conditioning

Beyond purely textual conditioning, incorporating image prompts alongside text enables richer multimodal control over the generation process. IP-Adapter ref [15] introduces a decoupled cross-attention mechanism that maintains separate attention layers for text and image features, enabling a lightweight 22M-parameter adapter to achieve image-prompt capability compatible with existing controllable tools. BLIP-Diffusion ref [27] pre-trains a multimodal encoder following BLIP-2 to produce visually aligned subject representations, enabling zero-shot subject-driven generation and efficient fine-tuning with up to 20× speedup. Furthermore, eDiff-I ref [30] demonstrates that CLIP image embeddings facilitate intuitive style transfer from reference images, highlighting how multimodal conditioning enriches semantic controllability beyond what text prompts alone can specify.

### Unified Multi-Condition Control Frameworks

Unified multi-condition control frameworks address the limitation of single-condition adapters by enabling simultaneous, composable use of heterogeneous control signals within a single model. Uni-ControlNet ref [55] introduces a unified architecture that accommodates both local controls (edge maps, depth maps, segmentation masks) and global controls (CLIP image embeddings) through only two additional adapters atop frozen pretrained diffusion models, maintaining a constant parameter overhead regardless of condition

[... deterministic omitted span ...]

## Text-Guided Image Editing with Diffusion Models

### Real Image Inversion and Reconstruction

Real image inversion is a foundational prerequisite for text-guided editing of natural images using pretrained diffusion models. DDIM inversion maps a real image into the model's latent space by reversing the deterministic sampling trajectory, enabling subsequent edits while preserving the original image's structure and appearance. Methods such as Imagic ref [6] leverage this inversion paradigm, first optimizing a text embedding to align with the input image and then fine-tuning the diffusion model to faithfully reconstruct it before applying semantic edits. Similarly, SINE ref [19] addresses single-image editing by combining model-based guidance with patch-based fine-tuning to mitigate overfitting during inversion, ensuring content fidelity alongside flexible text-driven modifications.

### Text-Based Semantic Image Editing

Text-based semantic image editing leverages pretrained text-to-image diffusion models to apply complex, non-rigid edits to real images guided solely by target text prompts. Imagic ref [6] pioneered this direction by optimizing a text embedding to align with both the source image and target description, followed by fine-tuning the diffusion model to capture image-specific appearance, enabling edits such as altering object posture or composition. SINE ref [19] addresses the challenging single-image setting by introducing model-based guidance built upon classifier-free guidance, distilling knowledge from a single-image-trained model into the pretrained diffusion model, while employing patch-based fine-tuning to support arbitrary-resolution generation and diverse editing operations including style transfer and object manipulation.

### Attention-Based and Training-Free Editing Methods

Attention-based and training-free editing methods exploit the internal cross-attention and self-attention mechanisms of pretrained diffusion models to achieve localized or global image edits without additional training. Training-Free Structured Diffusion Guidance ref [16] manipulates cross-attention keys and values guided by linguistic structures to improve attribute binding and compositional accuracy. Attend-and-Excite ref [1] intervenes during inference by strengthening cross-attention activations for neglected subject tokens, mitigating catastrophic neglect. For spatial-temporal control, ref [44] explicitly regulates cross-attention across both spatial and temporal dimensions. Similarly, ref [57] demonstrates that partially modifying text embeddings while fixing Gaussian noise enables disentangled style editing without fine-tuning, underscoring the inherent editability latent within diffusion model attention structures.

### Model-Level Implicit Assumption Editing

Beyond inference-time interventions, a complementary paradigm directly modifies pretrained diffusion model weights to correct implicit assumptions encoded during training. TIME (Text-to-Image Model Editing) ref [14] exemplifies this approach by accepting source and destination prompt pairs and updating cross-attention projection matrices so that under-specified source prompts are projected toward desired destination representations. This targeted intervention modifies only 2.2% of model parameters in under one second, enabling efficient correction of outdated knowledge, factual errors, and social biases without full retraining. Evaluated on the introduced TIMED benchmark of 147 prompt pairs, TIME demonstrates strong generalization to unseen related prompts while minimally affecting unrelated generations.

### Object-Level Shape and Appearance Variation

Generating controlled variations in object shape and appearance within synthesized images represents a nuanced challenge in text-guided editing. Ref [53] introduces a prompt-mixing technique that alternates between prompts during the denoising process to produce diverse shape choices for a specified object, complemented by localization mechanisms leveraging self-attention and cross-attention layers to confine manipulations to the target object region. Complementarily, ref [57] demonstrates that partially interpolating text embeddings between neutral and stylized descriptions, while fixing Gaussian noise, enables disentangled appearance modification without semantic disruption. Together, these approaches establish principled frameworks for object-level exploration, balancing shape diversity with semantic coherence through attention-guided localization and embedding-space manipulation.

### Video Editing via Text-to-Image Diffusion Models

Extending text-to-image diffusion models to video editing presents unique challenges, particularly maintaining temporal consistency across frames while applying semantically meaningful edits. Text2Video-Zero ref [54] addresses zero-shot video generation and editing by enriching latent codes with motion dynamics and employing cross-frame attention mechanisms anchored to the first frame, enabling instruction-guided video editing without additional training. Ground-A-Video ref [45] introduces a grounding-guided framework for multi-attribute video editing, incorporating Cross-Frame Gated Attention with optical flow-guided latent smoothing to achieve temporally consistent edits while preserving unmodified regions, outperforming baseline methods in edit accuracy and frame consistency across complex multi-attribute scenarios.

## Accelerating Inference and Improving Efficiency

### Knowledge Distillation and Progressive Distillation

Knowledge distillation and progressive distillation have emerged as principal strategies for compressing the multi-step sampling process inherent to text-to-image diffusion models. Progressive distillation iteratively halves the number of required denoising steps by training student models to replicate two-step teacher trajectories in a single step, enabling substantial inference acceleration. SnapFusion ref [38] advances this paradigm by combining efficient UNet architecture design with enhanced step distillation, incorporating classifier-free guidance regularization to achieve mobile deployment within two seconds. HiPA ref [43] further demonstrates that progressive distillation underperforms in one-step generation, proposing parameter-efficient low-rank adaptors targeting high-frequency information recovery, achieving superior FID scores with dramatically reduced training overhead compared to conventional progressive distillation approaches.

### Rectified Flow and Trajectory Straightening

Rectified Flow offers a principled approach to accelerating diffusion-based generation by learning straight-line probability flow trajectories between noise and data distributions. InstaFlow ref [26] applies this framework to text-to-image synthesis by introducing a text-conditioned reflow pipeline that iteratively straightens the trajectories of Stable Diffusion, refining noise-image couplings to facilitate subsequent distillation into student models. This reflow procedure proves critical for enabling functional one-step generation, achieving an FID of 23.3 on MS-COCO 2017-5k—substantially surpassing progressive distillation baselines—while requiring only 199 A100 GPU days of training, demonstrating that trajectory straightening constitutes an effective and computationally tractable pathway toward ultra-fast text-to-image generation.

### One-Step and Few-Step Generation Methods

Achieving high-quality text-to-image synthesis in one or very few denoising steps represents a critical frontier in diffusion model efficiency. InstaFlow ref [26] pioneers one-step generation by applying Rectified Flow's reflow procedure to Stable Diffusion, straightening probability flow trajectories and achieving an FID of 23.3 on MS-COCO 2017-5k, substantially outperforming progressive distillation. Complementarily, HiPA ref [43] introduces High-frequency-Promoting Adaptation, training parameter-efficient low-rank adaptors that specifically recover high-frequency details absent in one-step diffusion, attaining an FID of 23.8 with only 0.04% of trainable parameters and 28.6× training speedup. SnapFusion ref [38] further demonstrates few-step feasibility on mobile devices, achieving competitive FID and CLIP scores with merely eight denoising steps through efficient UNet design and enhanced step distillation.

### Efficient Network Architecture Design

Efficient network architecture design is critical for deploying text-to-image diffusion models in resource-constrained environments. SnapFusion ref [38] demonstrates that architectural redundancy in standard UNet designs can be systematically identified and eliminated, enabling mobile deployment within two seconds through efficient UNet redesign combined with data distillation for the image decoder. PixArt-α ref [33] introduces a computationally streamlined Diffusion Transformer incorporating cross-attention modules for text conditioning, achieving competitive image quality at merely 10.8% of Stable Diffusion's training cost. Similarly, Swinv2-Imagen ref [22] replaces conventional CNN-based UNet components with hierarchical Swin Transformer architectures, addressing limitations of convolution operations while improving semantic fidelity in generated images.

### Reparameterization and Latent-Space Efficiency

Reparameterization techniques offer a compelling avenue for improving the speed-quality trade-off in text-to-image diffusion models without sacrificing generation fidelity. VQ-Diffusion ref [12] exemplifies this approach by operating within a vector-quantized latent space modeled by a conditional DDPM variant. A key contribution of VQ-Diffusion is its reparameterization strategy that enables parallel decoding of latent tokens, circumventing the sequential dependencies inherent in autoregressive methods. This yields generation speeds approximately fifteen times faster than conventional autoregressive approaches while achieving superior image quality. By confining the diffusion process to a compact discrete latent space, such frameworks simultaneously reduce computational overhead and mitigate error accumulation, establishing latent-space reparameterization as a foundational efficiency mechanism in modern text-to-image synthesis.

### Training Efficiency and Cost Reduction

Training efficiency represents a critical bottleneck in scaling text-to-image diffusion models, given the substantial computational resources required. PixArt-α ref [33] exemplifies a principled approach to cost reduction through a decomposed three-stage training pipeline that separately optimizes pixel dependency, text-image alignment, and aesthetic quality, achieving competitive image quality at merely 10.8% of Stable Diffusion v1.5's training cost, reducing CO₂ emissions by approximately 90%. Complementarily, parameter-efficient fine-tuning strategies such as Custom Diffusion ref [11] demonstrate that optimizing only a small subset of conditioning parameters suffices for concept learning, requiring approximately six minutes

[... deterministic omitted span ...]

precise instruction following for multi-object and attribute-binding scenarios.

## Safety, Ethics, and Responsible Deployment

### Harmful and NSFW Content Generation and Mitigation

Text-to-image diffusion models trained on large-scale internet data are susceptible to generating harmful, explicit, or Not-Safe-for-Work (NSFW) content, posing significant ethical and societal risks. To mitigate such risks, Safe Self-Distillation (SDD) ref [49] proposes guiding noise estimates conditioned on target removal concepts toward unconditional predictions, enabling simultaneous removal of multiple harmful concepts without degrading overall image quality. However, safety filters deployed in systems such as DALL·E 2 and Stable Diffusion remain vulnerable to adversarial circumvention; SneakyPrompt ref [60] demonstrates that reinforcement learning-guided token perturbations can systematically jailbreak these filters, exposing critical limitations in current content moderation pipelines and underscoring the need for more robust safety mechanisms.

### Concept Erasure and Unlearning in Diffusion Models

Concept erasure and unlearning address the critical challenge of removing specific concepts, styles, or memorized content from pretrained text-to-image diffusion models without full retraining. Ablating Concepts ref [13] proposes matching the image distribution of a target concept to an anchor concept, effectively preventing generation of copyrighted styles, specific instances, or undesired content while preserving closely related concepts. Complementing this, Receler ref [10] introduces lightweight erasers trained with concept-localized regularization and adversarial prompt learning, ensuring robustness against paraphrased or adversarially crafted prompts while maintaining locality for non-target concepts. Together, these approaches establish principled frameworks for selective concept removal, balancing safety requirements with the preservation of general generative capabilities.

### Copyright Protection and Style Mimicry Prevention

The proliferation of text-to-image diffusion models has raised significant concerns regarding unauthorized replication of artists' distinctive styles. Glaze ref [29] addresses this challenge by enabling artists to apply imperceptible adversarial perturbations, termed 'style cloaks,' to their artwork prior to online publication. These perturbations systematically mislead diffusion models attempting to fine-tune on the protected images, effectively disrupting style mimicry while remaining visually tolerable to human observers. Empirical evaluations involving over 1,000 professional artists demonstrated disruption rates exceeding 92% under standard conditions and above 85% against adaptive countermeasures, establishing adversarial cloaking as a viable mechanism for intellectual property protection in the era of generative AI.

### Adversarial Attacks and Jailbreaking Safety Filters

Despite the deployment of safety filters in text-to-image diffusion models, adversarial attacks have demonstrated significant vulnerabilities in these protective mechanisms. SneakyPrompt ref [60] represents the first automated jailbreaking framework, employing reinforcement learning to strategically perturb input tokens, successfully bypassing safety filters in systems such as DALL·E 2 and Stable Diffusion to generate Not-Safe-for-Work (NSFW) content. By iteratively querying the model and refining adversarial prompts, SneakyPrompt outperforms conventional text adversarial attacks in both query efficiency and output quality. These findings expose fundamental limitations in current content moderation pipelines, underscoring the urgent need for more robust, semantically-aware safety mechanisms resistant to token-level perturbation strategies.

### Anti-Customization and Privacy Protection

The proliferation of diffusion-based personalization techniques raises significant privacy concerns, particularly regarding unauthorized customization using individuals' facial images. To address this, anti-customization methods have been developed to protect personal privacy by disrupting the fine-tuning process of diffusion models. SimAC ref [59] proposes a principled adversarial framework that exploits intrinsic properties of diffusion models, including frequency-domain perception across time steps and layer-wise feature roles during denoising, to generate protective perturbations that effectively disrupt identity learning. By employing adaptive greedy time-step selection and feature-based optimization, SimAC substantially increases identity disruption compared to naive end-to-end adversarial approaches, offering robust protection against unauthorized personalization pipelines such as DreamBooth ref [8].

### Bias, Fairness, and Implicit Assumption Editing

Large-scale text-to-image diffusion models encode social biases and implicit assumptions from their training data, producing outputs that reflect stereotypes or outdated world knowledge. The TIME framework ref [14] directly addresses this by editing cross-attention projection matrices to redirect implicit assumptions—such as default color or demographic attributes—toward desired alternatives, modifying only 2.2% of model parameters in under one second. Complementarily, reward-based fine-tuning approaches such as DPOK ref [9] and AlignProp ref [52] optimize models against human preference signals, indirectly mitigating bias by aligning generation with broader human values. Auditing tools and structured evaluation benchmarks remain essential for systematically identifying and correcting fairness violations in deployed models.

## References

[1] Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models. https://doi.org/10.1145/3592116

[2] Encoder-based Domain Tuning for Fast Personalization of Text-to-Image Models. https://doi.org/10.1145/3592133

[3] ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation. https://doi.org/10.48550/arXiv.2304.05977

[4] Adding Conditional Control to Text-to-Image Diffusion Models. https://doi.org/10.1109/ICCV51070.2023.00355

[5] Sketch-Guided Text-to-Image Diffusion Models. https://doi.org/10.1145/3588432.3591560

[6] Imagic: Text-Based Real Image Editing with Diffusion Models. https://doi.org/10.1109/CVPR52729.2023.00582

[7] ERNIE-ViLG 2.0: Improving Text-to-Image Diffusion Model with Knowledge-Enhanced Mixture-of-Denoising-Experts. https://doi.org/10.1109/CVPR52729.2023.00977

[8] DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. https://doi.org/10.1109/CVPR52729.2023.02155

[9] DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models. https://doi.org/10.48550/arXiv.2305.16381

[10] Receler: Reliable Concept Erasing of Text-to-Image Diffusion Models via Lightweight Erasers. https://doi.org/10.48550/arXiv.2311.17717

[11] Multi-Concept Customization of Text-to-Image Diffusion. https://doi.org/10.1109/CVPR52729.2023.00192

[12] Vector Quantized Diffusion Model for Text-to-Image Synthesis. https://doi.org/10.1109/CVPR52688.2022.01043

[13] Ablating Concepts in Text-to-Image Diffusion Models. https://doi.org/10.1109/ICCV51070.2023.02074

[14] Editing Implicit Assumptions in Text-to-Image Diffusion Models. https://doi.org/10.1109/ICCV51070.2023.00649

[15] IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models. https://doi.org/10.48550/arXiv.2308.06721

[16] Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis. https://doi.org/10.48550/arXiv.2212.05032

[17] GLIGEN: Open-Set Grounded Text-to-Image Generation. https://doi.org/10.1109/CVPR52729.2023.02156

[18] SUR-adapter: Enhancing Text-to-Image Pre-trained Diffusion Models with Large Language Models. https://doi.org/10.1145/3581783.3611863

[19] SINE: SINgle Image Editing with Text-to-Image Diffusion Models. https://doi.org/10.1109/CVPR52729.2023.00584

[20] Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. https://doi.org/10.48550/arXiv.2205.11487

[21] Zero-shot spatial layout conditioning for text-to-image diffusion models. https://doi.org/10.1109/ICCV51070.2023.00207

[22] Swinv2-Imagen: hierarchical vision transformer diffusion models for text-to-image generation. https://doi.org/10.1007/s00521-023-09021-x

[23] BoxDiff: Text-to-Image Synthesis with Training-Free Box-Constrained Diffusion. https://doi.org/10.1109/ICCV51070.2023.00685

[24] Lego: Learning to Disentangle and Invert Personalized Concepts Beyond Object Appearance in Text-to-Image Diffusion Models. https://doi.org/10.1007/978-3-031-72633-0_7

[25] PhotoVerse: Tuning-Free Image Customization with Text-to-Image Diffusion Models. https://doi.org/10.48550/arXiv.2309.05793

[26] InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation. https://doi.org/10.48550/arXiv.2309.06380

[27] BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing. https://doi.org/10.48550/arXiv.2305.14720

[28] Human Preference Score: Better Aligning Text-to-image Models with Human Preference. https://doi.org/10.1109/ICCV51070.2023.00200

[29] GLAZE: Protecting Artists from Style Mimicry by Text-to-Image Models. https://doi.org/10.48550/arXiv.2302.04222

[30] eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers. https://doi.org/10.48550/arXiv.2211.01324

[31] Controlling Text-to-Image Diffusion by Orthogonal Finetuning. https://doi.org/10.48550/arXiv.2306.07280

[32] LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models. https://doi.org/10.48550/arXiv.2305.13655

[33] PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis. https://doi.org/10.48550/arXiv.2310.00426

[34] Taming Encoder for Zero Fine-tuning Image Customization with Text-to-Image Diffusion Models. https://doi.org/10.48550/arXiv.2304.02642

[35] Create Your World: Lifelong Text-to-Image Diffusion. https://doi.org/10.1109/TPAMI.2024.3382753

[36] ControlNet-XS: Rethinking the Control of Text-to-Image Diffusion Models as Feedback-Control Systems. https://doi.org/10.1007/978-3-031-73223-2_20

[37] Continual Diffusion: Continual Customization of Text-to-Image Diffusion with C-LoRA. https://doi.org/10.48550/arXiv.2304.06027

[38] SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds. https://doi.org/10.48550/arXiv.2306.00980

[39] ConceptBed: Evaluating Concept Learning Abilities of Text-to-Image Diffusion Models. https://doi.org/10.48550/arXiv.2306.04695

[40] FreeControl: Training-Free Spatial Control of Any Text-to-Image Diffusion Model with Any Condition. https://doi.org/10.1109/CVPR52733.2024.00713

[41] TexFusion: Synthesizing 3D Textures with Text-Guided Image Diffusion Models. https://doi.org/10.1109/ICCV51070.2023.00385

[42] Specialist Diffusion: Plug-and-Play Sample-Efficient Fine-Tuning of Text-to-Image Diffusion Models to Learn Any Unseen Style. https://doi.org/10.1109/CVPR52729.2023.01371

[43] HiPA: Enabling One-Step Text-to-Image Diffusion Models via High-Frequency-Promoting Adaptation. https://doi.org/10.48550/arXiv.2311.18158

[44] Harnessing the Spatial-Temporal Attention of Diffusion Models for High-Fidelity Text-to-Image Synthesis. https://doi.org/10.1109/ICCV51070.2023.00714

[45] Ground-A-Video: Zero-shot Grounded Video Editing using Text-to-image Diffusion Models. https://doi.org/10.48550/arXiv.2310.01107

[46] Text-to-image Diffusion Models in Generative AI: A Survey. https://doi.org/10.48550/arXiv.2303.07909

[47] AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. https://doi.org/10.48550/arXiv.2307.04725

[48] T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models. https://doi.org/10.48550/arXiv.2302.08453

[49] Towards Safe Self-Distillation of Internet-Scale Text-to-Image Diffusion Models. https://doi.org/10.48550/arXiv.2307.05977

[50] UPainting: Unified Text-to-Image Diffusion Generation with Cross-modal Guidance. https://doi.org/10.48550/arXiv.2210.16031

[51] Ranni: Taming Text-to-Image Diffusion for Accurate Instruction Following. https://doi.org/10.1109/CVPR52733.2024.00454

[52] Aligning Text-to-Image Diffusion Models with Reward Backpropagation. https://doi.org/10.48550/arXiv.2310.03739

[53] Localizing Object-level Shape Variations with Text-to-Image Diffusion Models. https://doi.org/10.1109/ICCV51070.2023.02107

[54] Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators. https://doi.org/10.1109/ICCV51070.2023.01462

[55] Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models. https://doi.org/10.48550/arXiv.2305.16322

[56] DreamFusion: Text-to-3D using 2D Diffusion. https://doi.org/10.48550/arXiv.2209.14988

[57] Uncovering the Disentanglement Capability in Text-to-Image Diffusion Models. https://doi.org/10.1109/CVPR52729.2023.00189

[58] Unleashing Text-to-Image Diffusion Models for Visual Perception. https://doi.org/10.1109/ICCV51070.2023.00527

[59] SimAC: A Simple Anti-Customization Method for Protecting Face Privacy Against Text-to-Image Synthesis of Diffusion Models. https://doi.org/10.1109/CVPR52733.2024.01145

[60] SneakyPrompt: Jailbreaking Text-to-image Generative Models. https://doi.org/10.1109/SP54263.2024.00123
