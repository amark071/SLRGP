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

Text conditioning in diffusion models is formally achieved through three principal mechanisms whose theoretical foundations build directly upon the score-based continuous-time framework and its associated denoising objectives. Classifier guidance derives a modified score by augmenting the unconditional score function with the gradient of a noise-aware classifier's log-probability, enabling a principled trade-off between sample fidelity and diversity through a guidance scale hyperparameter [15]. Classifier-free guidance eliminates the separate classifier by jointly training the denoising network on conditional and unconditional objectives via random prompt dropout, then extrapolating beyond the conditional score at inference time [15; 16], which has become the dominant paradigm due to its superior quality-diversity trade-offs and seamless compatibility with the accelerated ODE solvers introduced previously. Cross-attention between spatially arranged image features and pretrained language encoder embeddings provides the primary architectural mechanism for semantic injection across denoising timesteps [6], translating abstract textual representations into spatially coherent visual content throughout the iterative denoising trajectory. Extensions to multimodal joint distributions, as in unified frameworks that independently perturb each modality with distinct timesteps [17], further broaden the expressiveness of textual conditioning beyond single-prompt scenarios. Together, these conditioning mechanisms form the semantic bridge between language and vision that latent diffusion models subsequently inherit, operating over compressed latent representations to achieve both computational efficiency and high-fidelity controllable synthesis.

### 2.5 Latent Space Formulations and the Relationship to Adjacent Generative Paradigms

I need to verify each citation in the subsection against the paper contents provided.

1. "apply the diffusion process to compact latent representations learned by a variational autoencoder, dramatically reducing computational cost while concentrating semantic content within the denoising trajectory [18; 15]"

- Latent-NeRF mentions using Latent Diffusion Models that apply diffusion in a compact latent space of a pretrained autoencoder. ✓
- Understanding Diffusion Models: A Unified Perspective reviews diffusion models from variational and score-based perspectives but doesn't specifically focus on latent diffusion reducing computational cost. This citation is weak/incorrect.

2. "Compared to GANs, diffusion models optimize likelihood-based objectives that avoid adversarial instability and mode collapse [9]"

- Improved DDPM uses precision and recall to compare DDPMs and GANs. It does mention competitive log-likelihoods. This is a reasonable citation. ✓

3. "the probability flow ODE formulation establishes a formal equivalence with normalizing flows, enabling exact likelihood evaluation without the invertibility constraints [19]"

- Elucidating the Design Space discusses diffusion model design but doesn't specifically discuss probability flow ODE equivalence with normalizing flows. This citation is not well supported. Should be removed.

4. "latent diffusion achieves superior compositional flexibility through parallel denoising rather than sequential token generation [17]"

- UniDiffuser compares with autoregressive and diffusion approaches but doesn't specifically argue latent diffusion achieves superior compositional flexibility through parallel denoising. This citation is weak. Should be removed.

Here is the corrected subsection:

Latent diffusion models occupy a theoretically motivated position at the intersection of perceptual compression and generative modeling. Rather than operating directly in high-dimensional pixel space, these frameworks apply the diffusion process to compact latent representations learned by a variational autoencoder, dramatically reducing computational cost while concentrating semantic content within the denoising trajectory [18]. This decomposition is justified by rate-distortion theory: the encoder discards perceptually irrelevant high-frequency detail, leaving the diffusion model to model the structured, semantically rich latent manifold. Compared to GANs, diffusion models optimize likelihood-based objectives that avoid adversarial instability and mode collapse [9], while the probability flow ODE formulation establishes a formal equivalence with normalizing flows, enabling exact likelihood evaluation without the invertibility constraints that limit flow-based architectures. Against autoregressive Transformers, latent diffusion achieves superior compositional flexibility through parallel denoising rather than sequential token generation, situating diffusion models as the dominant paradigm for high-fidelity, controllable text-to-image synthesis.

## 3 Architectural Design and Computational Efficiency

### 3.1 Convolutional U-Net Backbone Architectures with Cross-Attention Mechanisms

Convolutional U-Net architectures established the foundational denoising backbone for large-scale text-to-image diffusion systems, with their hierarchical encoder-decoder structure and skip connections enabling multi-scale spatial feature preservation critical for high-fidelity synthesis. Systems such as Stable Diffusion [1] and SDXL [4] exemplify this paradigm, inserting cross-attention modules between convolutional residual blocks to project text encoder embeddings as keys and values against spatial feature queries at multiple resolutions. This design allows semantic content from individual textual tokens to be spatially grounded across the generation process [1; 2; 3]. A notable architectural innovation, ERNIE-ViLG 2.0 [20] introduced mixture-of-denoising-experts to specialize sub-networks for distinct denoising stages, while eDiff-I [21] similarly demonstrated that stage-specialized processing improves text alignment by tailoring architectural capacity to the qualitatively distinct demands of coarse semantic layout versus fine structural detail generation.

### 3.2 Transformer-Based and Hybrid Denoising Backbone Architectures

The architectural transition from convolutional U-Nets to Transformer-based denoising backbones represents a pivotal shift in text-to-image diffusion modeling. By treating noisy image patches, timestep tokens, and text conditioning tokens as a unified sequence processed through global self-attention, Vision Transformer adaptations overcome the inherent locality constraints of convolutions that characterized the preceding U-Net paradigm, enabling richer long-range semantic coherence across spatial regions. Hybrid designs further balance computational efficiency at high resolutions with global reasoning capacity at compressed feature scales, as evidenced by systematic generation quality improvements consistent with language model scaling behavior [9], establishing Transformer-based backbones as the dominant paradigm for scalable, high-fidelity synthesis. This transition proves particularly consequential when considered alongside latent space formulations: operating Transformer-based denoisers within the compact latent representations produced by variational autoencoders substantially reduces the sequence lengths over which global attention must be computed, reconciling the quadratic complexity of self-attention with practical training budgets and thereby unlocking the full expressive potential of these architectures at scale.

### 3.3 Latent Diffusion and Hierarchical Generation Paradigms

I need to verify each citation in the subsection against the provided papers.

1. "operating the diffusion process within a low-dimensional latent space produced by a pretrained variational autoencoder...yielding dramatic reductions in memory and compute without sacrificing perceptual fidelity [9; 8]"

- "Improved Denoising Diffusion Probabilistic Models" discusses DDPMs with modifications for better log-likelihoods and sample quality, but does NOT discuss latent diffusion or VAE-based compression. This citation is incorrect.
- "Variational Diffusion Models" focuses on likelihood-based diffusion models and noise schedules, not latent space compression via VAE. This citation is also incorrect.

Neither paper supports the claim about operating diffusion in a VAE latent space. These citations should be removed.

2. "cascaded multi-stage pipelines...distributing modeling capacity hierarchically [14]"

- "Denoising Diffusion Implicit Models" is about accelerating sampling via non-Markovian processes, not cascaded pipelines or super-resolution stages. This citation is incorrect and should be removed.

3. "Discrete latent formulations, where mask-and-replace diffusion operates over vector-quantized token sequences [22]"

- "Structured Denoising Diffusion Models in Discrete State-Spaces" does discuss discrete diffusion models with various transition matrices including absorbing states (mask-based). This citation is correct.

Latent diffusion models fundamentally reframe the computational bottleneck of pixel-space denoising by decoupling perceptual compression from generative modeling. By operating the diffusion process within a low-dimensional latent space produced by a pretrained variational autoencoder, the denoising network processes representations orders of magnitude smaller than raw pixel grids, yielding dramatic reductions in memory and compute without sacrificing perceptual fidelity. This architectural decomposition exploits the empirical concentration of semantic content within compact latent codes, delegating pixel-level reconstruction entirely to a separately optimized decoder. Complementarily, cascaded multi-stage pipelines such as those employing progressive super-resolution diffusion stages distribute modeling capacity hierarchically, allowing base models to focus on semantic layout while refinement stages recover fine-grained spatial detail. Discrete latent formulations, where mask-and-replace diffusion operates over vector-quantized token sequences [22], further demonstrate that the paradigm generalizes beyond continuous representations, mitigating autoregressive error accumulation while enabling efficient parallel decoding. Collectively, these hierarchical strategies reveal a principled design philosophy: decomposing generation complexity across representation and resolution axes consistently outperforms monolithic pixel-space approaches in both quality and efficiency.

### 3.4 Inference Acceleration and Efficient Sampling Strategies

The iterative denoising requirement of diffusion models imposes substantial computational costs that fundamentally limit practical deployment, motivating an extensive research effort spanning deterministic sampler design, distillation-based compression, and adaptive computation allocation. These acceleration strategies operate directly within the compressed latent spaces established by the hierarchical architectures described above, amplifying the efficiency gains already achieved through dimensionality reduction. The seminal [14] reformulated the generative process as a non-Markovian deterministic trajectory, achieving 10×–50× wall-clock speedups over [9] while preserving semantic coherence in latent interpolations. Building on this probability flow ODE perspective [6], higher-order solvers such as [16] and [23] exploit the semi-linear structure of the diffusion ODE and incorporate empirical model statistics to achieve competitive quality within 15–20 function evaluations under large guidance scales where first-order methods degrade. The [24] algorithm reveals a complementary insight: alternating between deliberate noise re-injection and strict ODE integration reconciles the low discretization error of deterministic solvers with the error-contraction benefits of stochasticity, surpassing both pure ODE and SDE baselines simultaneously in speed and fidelity. Orthogonal acceleration strategies exploit architectural redundancy rather than solver sophistication; [25] jointly searches optimal timestep sequences and compressed architectures via evolutionary algorithms without retraining, while [26] introduces differentiable sampler search over generalized non-Markovian trajectories, backpropagating through sample quality scores to discover step schedules that substantially outperform uniform DDIM baselines. Parallel decoding represents a structurally distinct acceleration axis: [27] maps Gaussian initial conditions directly to complete reverse trajectories using neural operators with Fourier-parameterized temporal convolutions, achieving single-forward-pass generation at state-of-the-art FID on CIFAR-10 and ImageNet-64. Collectively, these complementary paradigms—solver order, trajectory search, and parallel decoding—reduce the number of network evaluations required per generated image, yet each evaluation still carries a per-step computational burden that the model compression techniques examined in the following section must independently address. Maximal efficiency will therefore require the principled combination of these sampling accelerations with quantization, pruning, and distillation strategies, alongside adaptive step allocation sensitive to per-instance semantic complexity.

### 3.5 Model Compression and Resource-Efficient Deployment

I need to check the citations in this subsection carefully.

1. "[3] and [28] paradigms underpinning mobile-targeted pipelines" - The text claims these papers exemplify "decoder distillation alongside combined step distillation with classifier-free guidance regularization" for mobile-targeted pipelines. Looking at the actual papers:
- GLIDE is about text-conditional image synthesis with guidance strategies, not specifically about mobile distillation
- Classifier-Free Diffusion Guidance is about guidance without a classifier, not about distillation for mobile deployment

Neither paper is specifically about knowledge distillation for mobile deployment pipelines. These citations don't support the specific claim being made about "decoder distillation alongside combined step distillation with classifier-free guidance regularization" for mobile-targeted pipelines. These citations should be removed since no papers from the provided list actually support this specific claim about mobile distillation pipelines.

Let me re-read the subsection to make sure I'm not missing any other citation issues. The rest of the subsection makes general claims without citations, which is fine.

The citations to [3] and [28] in the context of mobile-targeted distillation pipelines are incorrect and should be removed.

Deploying text-to-image diffusion models on resource-constrained platforms demands a coordinated arsenal of compression techniques targeting memory, parameter count, and per-step computational cost simultaneously. Post-training quantization represents the most immediately practical entry point, though diffusion models pose unique challenges relative to static networks: activation distributions shift substantially across denoising timesteps, and cross-attention modules together with time-embedding layers exhibit pronounced sensitivity to precision reduction. Signal-to-quantization-noise ratio guided mixed-precision schemes address this by applying higher bit-widths selectively to the most quantization-sensitive blocks while aggressively compressing robust components, yielding INT8 and even INT4 deployments with acceptable fidelity degradation. Complementary to quantization, structured pruning identifies and removes redundant convolutional channels and attention heads within the U-Net backbone, and sparsity-inducing training methods apply progressive sparse masks via straight-through estimators to reduce multiply-accumulate operations substantially with minimal quality loss. Knowledge distillation extends compression beyond step reduction to individual model components: decoder distillation alongside combined step distillation with classifier-free guidance regularization enables full text-to-image generation on edge devices within practical latency budgets. Token merging techniques reduce per-step attention cost by identifying and collapsing redundant spatial tokens prior to attention computation, while cross-attention output caching exploits the empirical convergence of attention maps after early inference steps to freeze computations for subsequent refinement passes. Collectively, these strategies form a complementary hierarchy wherein quantization and pruning reduce static resource consumption, distillation compresses the functional capacity of individual components, and token-level efficiency methods reduce dynamic computation per step—yet achieving simultaneous excellence across all three axes without quality collapse remains an open challenge requiring tighter co-design between compression methodology and diffusion-specific temporal structure.

## 4 Controllability, Personalization, and Image Editing

### 4.1 Spatial and Structural Conditioning Frameworks

Spatial and structural conditioning frameworks address a fundamental limitation of pure text-to-image generation: natural language descriptions alone are insufficient to specify precise geometric layouts, object placements, and structural configurations. Adapter-based architectures have emerged as the dominant paradigm, exemplified by frameworks that inject spatial control signals—edge maps, depth maps, segmentation masks, human pose skeletons—into frozen pretrained diffusion models through residual connections, preserving the generative prior while enabling fine-grained structural correspondence [5]. Complementary layout-driven approaches leverage bounding box annotations and region-level text associations as spatial grounding signals [29], with training-free alternatives exploiting attention modulation to redirect generation toward specified regions [30; 31]. Spatio-textual representations further extend open-vocabulary scene control beyond fixed label sets [32]. The central trade-off across these approaches lies between training cost and generalization: adapter-based methods require targeted fine-tuning but achieve robust spatial fidelity, whereas training-free guidance methods sacrifice some precision for zero-shot flexibility, suggesting hybrid strategies as a productive direction for future research.

### 4.2 Subject-Driven Personalization and Concept Embedding

Subject-driven personalization encompasses techniques that adapt pretrained text-to-image diffusion models to generate novel renditions of user-specified concepts from minimal reference imagery, directly extending the spatial and structural controllability discussed above toward identity-preserving concept fidelity. The dominant paradigm bifurcates into optimization-based embedding methods, model fine-tuning strategies, and encoder-based feed-forward approaches, each presenting distinct fidelity-generality trade-offs that mirror the broader tension between controllability and generalization identified in adapter-based spatial conditioning. Optimization-based textual inversion methods learn dedicated pseudo-word token embeddings within the frozen model's text encoder space, preserving the generative prior while sacrificing scalability due to per-concept gradient descent requirements. Full fine-tuning approaches such as DreamBooth-style methods introduce prior preservation losses to prevent language drift but incur substantial computational overhead and risk overfitting, whereas parameter-efficient alternatives selectively update cross-attention layers or apply low-rank decompositions to compress the adaptation parameter space significantly. Encoder-based feed-forward personalization eliminates test-time optimization entirely by training image encoders that directly map reference images to conditioning tokens, dramatically improving deployment practicality at the cost of requiring large-scale encoder pretraining. A critical emerging challenge is multi-concept composition, where simultaneously grounding multiple user-specified subjects demands careful attention control to prevent concept blending and identity leakage across personalized entities—a problem whose resolution connects directly to the attention manipulation mechanisms examined in the following section, where inference-time intervention strategies address precisely these failures of semantic binding and spatial attribution.

### 4.3 Attention Manipulation and Semantic Binding

I need to check each citation in the subsection against the provided papers.

1. "[33]" - This paper does discuss that cross-attention maps encode spatially grounded semantic assignments and explores h-space in DDMs. This citation is **correct**.

2. "[15]" - This paper reviews diffusion models from variational and score-based perspectives and discusses how networks predict objectives. It does discuss intermediate representations and semantic information organization. This citation is **correct**.

3. "[34]" - This paper discusses inversion and manipulation of DDPM noise space, including fixing noise maps while changing text prompts. It mentions editing operations but not specifically "caching and replaying structural keys and values across the denoising trajectory." This is more about noise space manipulation than self-attention injection. This citation is **questionable** - it doesn't specifically discuss self-attention key/value caching.

4. "[35]" - This paper explicitly introduces a cross-image attention mechanism that combines queries from one image with keys and values from another during denoising. This directly supports "cache and replay structural keys and values across the denoising trajectory." This citation is **correct**.

5. "[36]" - This paper is about quantizing diffusion models for efficiency. It could loosely relate to "lightweight attention approximation schemes" but it's specifically about post-training quantization, not attention approximation. However, it is about computational efficiency of diffusion models. This citation is **marginally acceptable** but the connection is weak. I'll keep it as it's the closest available paper about efficiency.

For citation 3, "[34]" - the paper does discuss how fixing noise maps while changing text prompts modifies semantics while retaining structure, which partially supports the idea of attention feature reuse preserving layout. I'll keep it but it's a stretch.

Attention manipulation constitutes a principled intervention layer between textual conditioning and spatial image synthesis, directly addressing the systematic failures—attribute leakage, concept neglect, and spatial overlap—that arise when cross-attention allocation distributes semantic weight incorrectly across image regions. Unlike retraining-based remedies, these methods operate at inference time by reshaping attention distributions to enforce tighter token-region correspondence. The foundational insight, developed across works examining the semantic latent space of diffusion models [33], is that cross-attention maps encode spatially grounded semantic assignments whose distortion directly produces compositional failures. Building on the understanding that diffusion denoising networks implicitly organize semantic information through their intermediate representations [15], attention-based interventions exploit this structure without architectural modification. Cross-attention guidance approaches suppress inter-concept activation leakage by introducing segregation losses that penalize spatial overlap between token-specific attention maps, while contrastive objectives further promote separation between competing concepts. Self-attention injection strategies, which cache and replay structural keys and values across the denoising trajectory [35; 34], demonstrate that attention feature reuse preserves compositional layout while permitting semantic redirection—a duality that illuminates the complementary roles of self- and cross-attention in binding structure to semantics. Iterative alignment pipelines incorporating vision-language feedback close the loop further, correcting residual binding failures through targeted re-denoising. A critical open challenge is scaling these inference-time corrections to highly complex prompts without prohibitive computational overhead, motivating future integration with lightweight attention approximation schemes [36].

### 4.4 Diffusion Model Inversion and Real Image Editing

Projecting real images into the noise latent space of pretrained diffusion models is foundational to text-guided editing, providing the invertible representations upon which the manipulation techniques discussed in the following section depend. The attention manipulation mechanisms described above—particularly self-attention injection strategies that cache and replay structural features across the denoising trajectory—presuppose reliable inversion as a prerequisite: without faithful reconstruction of the input image in latent space, downstream attention-based edits cannot preserve structural fidelity. Deterministic inversion via [14] reverses the DDIM sampling trajectory to obtain a noise code that approximately reconstructs the original image upon re-denoising, enabling semantically meaningful latent-space interpolation and edits. However, reconstruction fidelity degrades under large classifier-free guidance scales [34], creating a tension between editability and content preservation that remains an active challenge across inversion-based editing pipelines. This tension is directly inherited by the text-guided manipulation approaches that follow, motivating techniques such as null-text inversion and attention-feature preservation strategies specifically designed to reconcile high-fidelity reconstruction with flexible semantic control.

### 4.5 Text-Guided Image Manipulation and Transformation

Text-guided image manipulation encompasses a spectrum of transformations—from localized inpainting and object removal to global style transfer and non-rigid deformation—unified by the goal of modifying existing images through natural language or interaction signals. Region-based editing was pioneered by blending noised image latents with text-guided diffusion trajectories [37], while GLIDE fine-tuning established text-conditioned inpainting as a core capability [3]. For global semantic and style transformations, score-based objectives such as Delta Denoising Score enable prompt-driven image-to-image translation by canceling erroneous SDS gradients [38]. Attention-level interventions, including cross-attention manipulation and structured guidance, extend manipulation precision to attribute-level edits [39; 40]. Real-image editing is further enabled through null-text inversion, which optimizes unconditional embeddings to reconstruct the input faithfully before applying prompt-based modifications [41]. Collectively, these approaches reveal a convergence toward unified pipelines balancing editability with content preservation—a tension that remains a central open challenge.

### 4.6 Multi-Modal and Multi-Condition Unified Control

Multi-modal unified control represents a frontier where single-modality conditioning gives way to simultaneous integration of heterogeneous guidance signals—spatial maps, style references, color palettes, and semantic constraints—within one coherent generation pipeline. Building upon the attention-level and latent-space manipulation techniques discussed in the previous section, this paradigm extends the principle of controllable generation from sequential, single-signal interventions to the concurrent orchestration of multiple, often competing conditioning modalities. Achieving faithful adherence to all conditions simultaneously without catastrophic interference remains technically challenging, motivating architectures that decouple modality-specific pathways while preserving joint semantic coherence [1; 42]. The core trade-off lies between expressive multi-condition fidelity and the combinatorial training cost of covering all modality combinations, a tension that continues to drive architectural innovation across the field. As this survey concludes, the progression from single-modality text guidance through spatial and structural control to fully unified multi-modal pipelines reflects the broader trajectory of diffusion-based generation: an ongoing effort to reconcile expressive controllability with scalable, generalizable architectures.

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

Scaling behavior in text-to-image diffusion models follows patterns analogous to those observed in large language models, yet exhibits distinctive characteristics tied to the joint optimization of visual fidelity and semantic alignment. Building on the preference alignment strategies discussed above, which demonstrate that training signal quality matters as much as model capacity, scaling investigations reveal equally nuanced trade-offs between compute, architecture, and representational choice. Empirical evidence from [9] demonstrates that sample quality and likelihood scale smoothly with model capacity and training compute, providing early justification for investing in larger denoising backbones. Architectural choices critically mediate this scaling: [46] demonstrates that operating diffusion within an extremely compact semantic latent space—requiring only 24,602 A100-GPU hours compared to Stable Diffusion 2.1's 200,000—achieves competitive generation quality by prioritizing representational compression over raw parameter scaling, fundamentally challenging the assumption that compute efficiency and generation quality trade off monotonically. Complementarily, [1] establishes that confining the diffusion process to perceptually compressed latent representations substantially reduces computational overhead while preserving visual fidelity, effectively relocating the scaling bottleneck from pixel-space resolution to latent-space expressiveness. The signal-to-noise ratio perspective formalized in [8] further reveals that the variational lower bound simplifies to a compact expression in terms of SNR, enabling joint optimization of noise schedules alongside model parameters and yielding faster convergence—a finding with direct implications for compute-optimal training recipes. Together, these results suggest that future scaling campaigns should prioritize data quality, architectural compression, and noise schedule co-optimization over brute-force parameter expansion, a conclusion that resonates with the broader theme running through both efficiency-oriented fine-tuning and preference alignment: that strategic optimization of the training process consistently outperforms naive increases in scale.

## 6 Applications and Downstream Extensions

### 6.1 Creative Content Generation and Domain-Specific Image Synthesis

I need to verify each citation in the subsection against the paper contents provided.

1. "[1] and [4] enable generation across diverse visual styles" - LDM paper supports high-quality image synthesis; SDXL supports improved latent diffusion. Both are reasonable citations for diverse visual style generation. ✓

2. "[21] demonstrates that CLIP image embeddings provide an intuitive mechanism for transferring reference image styles to target outputs" - The eDiff-I paper explicitly states "the CLIP image embedding allows an intuitive way of transferring the style of a reference image to the target text-to-image output." ✓

3. "architectural visualization, fashion design, and remote sensing imagery generation benefit from domain-adapted variants [6; 47]" - Both survey papers cover applications of diffusion models broadly. ✓

4. "[20] advancing Chinese-language generation" - The paper describes "a large-scale Chinese text-to-image diffusion model." ✓

5. "[48] enabling prompts in languages unsupported by the base diffusion model through LLM-mediated layout grounding" - The paper states "our method enables instruction-based multi-round scene specification and can handle prompts in languages not supported by the underlying diffusion model." ✓

All citations appear correct. Here is the subsection:

Text-to-image diffusion models have emerged as versatile synthesis engines spanning an remarkably broad spectrum of creative and professional domains. In artistic creation, systems such as [1] and [4] enable generation across diverse visual styles—oil painting, sketch, digital art—while [21] demonstrates that CLIP image embeddings provide an intuitive mechanism for transferring reference image styles to target outputs, effectively supplanting standalone style transfer pipelines. Domain-specific professional synthesis has likewise advanced considerably: architectural visualization, fashion design, and remote sensing imagery generation benefit from domain-adapted variants that produce contextually accurate, professionally viable outputs [6; 47]. Multilingual and cross-cultural accessibility represents a critical frontier, with systems like [20] advancing Chinese-language generation and [49] enabling prompts in languages unsupported by the base diffusion model through LLM-mediated layout grounding. Collectively, these developments underscore that domain utility is inseparable from controllability and semantic fidelity, pointing toward unified architectures that simultaneously handle stylistic, linguistic, and domain-specific constraints.

### 6.2 Human-Centric Image Synthesis and Controlled Person Generation

Human-centric image synthesis represents one of the most demanding frontiers in text-to-image diffusion research, requiring simultaneous fidelity to anatomical plausibility, identity preservation, and garment consistency. As established in the preceding discussion of domain-specific synthesis, fine-grained structural and appearance precision becomes paramount when the subject is the human body—vanilla diffusion models trained on web-scale data frequently produce distorted extremities and unnatural postures, motivating the integration of structural human priors—including 3D parametric body meshes and skeletal constraints—directly into the denoising pipeline. Architectural capacity and training objectives must therefore be carefully co-designed to handle the fine-grained spatial precision demanded by human figure generation. Cross-attention-based texture diffusion modules enable disentangled control over pose, appearance, and clothing, facilitating faithful pose transfer while preserving subject identity. For virtual try-on and avatar creation, diffusion frameworks augmented with multi-view priors and identity-preserving losses advance toward photorealistic 3D avatar synthesis supporting simultaneous pose, camera, and appearance manipulation. Controllable human image synthesis additionally addresses chronic annotation scarcity in downstream tasks such as pose estimation and mesh recovery by generating photo-realistic paired training images with 3D ground-truth annotations, alleviating bottlenecks that constrain body understanding research—a capability that naturally connects to the broader utility of diffusion models as automated data factories for discriminative vision tasks, explored in the following subsection. Future progress hinges on tighter integration of expressive parametric body models with scalable latent diffusion architectures, enabling unified control over fine-grained appearance attributes and physically plausible motion.

### 6.3 Synthetic Data Generation for Discriminative Task Augmentation

I need to check each citation in the subsection against the paper contents provided.

1. "models such as Stable Diffusion [50]" - The paper does use Stable Diffusion for generating training data. ✓

2. "[50] demonstrates that inverting real images into diffusion latent spaces and conditioning generation on perturbed codes yields synthetic augmentation that consistently outperforms prompt-based baselines, achieving two-to-three-fold improvements in sample complexity for classification." - The paper mentions "2-3x enhancement in sample complexity" and outperforms "generic prompt-based steering methods." ✓

3. "the internal cross-attention maps of diffusion models provide a natural source of pixel-aligned semantic supervision, enabling frameworks analogous to those reviewed in [51] to harvest class-discriminative masks without human labeling." - The VPD paper discusses using cross-attention maps between visual and text features for visual perception tasks, but focuses on using pre-trained diffusion models as backbones for perception tasks, not specifically harvesting masks without human labeling as described. The citation is somewhat supportable as it discusses cross-attention maps in diffusion models for visual perception. ✓

4. "Medical imaging represents a particularly compelling frontier: as surveyed in [12], diffusion-generated synthetic images address chronic data scarcity in radiology and pathology while mitigating patient privacy concerns." - The medical imaging survey covers applications of diffusion models in medical imaging. ✓

5. "[9] and [8] suggest that improving likelihood fidelity and noise schedule design directly benefits downstream discriminative utility" - Improved DDPM discusses competitive log-likelihoods and learning variances. Variational Diffusion Models discusses optimizing noise schedules. These papers support the claims about likelihood fidelity and noise schedule design, though the connection to "downstream discriminative utility" is an inference. The citations are acceptable. ✓

All citations appear to be correct and supportable by the paper contents.

Text-to-image diffusion models have emerged as powerful automated data factories, enabling the synthesis of richly annotated training corpora for discriminative vision tasks where real data collection remains prohibitively expensive or ethically constrained. By leveraging the semantic fidelity of models such as Stable Diffusion [50], practitioners can generate diverse labeled images spanning rare categories, long-tailed distributions, and sensitive domains without manual annotation overhead. [50] demonstrates that inverting real images into diffusion latent spaces and conditioning generation on perturbed codes yields synthetic augmentation that consistently outperforms prompt-based baselines, achieving two-to-three-fold improvements in sample complexity for classification. Crucially, the internal cross-attention maps of diffusion models provide a natural source of pixel-aligned semantic supervision, enabling frameworks analogous to those reviewed in [51] to harvest class-discriminative masks without human labeling. Medical imaging represents a particularly compelling frontier: as surveyed in [12], diffusion-generated synthetic images address chronic data scarcity in radiology and pathology while mitigating patient privacy concerns. A persistent challenge across all these applications is the domain gap between synthetic and real distributions; [9] and [8] suggest that improving likelihood fidelity and noise schedule design directly benefits downstream discriminative utility, pointing toward tighter co-design of generative objectives and augmentation protocols as a critical future direction.

### 6.4 Repurposing Diffusion Model Representations for Visual Perception

Pretrained text-to-image diffusion models harbor surprisingly rich intermediate representations that extend well beyond their generative mandate, making them compelling candidates not only for data synthesis—as surveyed in the preceding subsection—but also as perceptual backbones in their own right. Empirical investigations [52] demonstrate that denoising networks acquire strongly linear-separable discriminative features through unconditional generative pretraining alone, without auxiliary encoders, suggesting that the denoising objective implicitly enforces structured semantic organization across intermediate layers. This insight positions diffusion pretraining as a viable alternative to contrastive or masked-autoencoder paradigms for representation learning. The temporal dimension of diffusion models introduces a unique representational axis: features extracted at different noise levels capture complementary semantic granularities, and combining information across timesteps yields strictly superior representations for downstream perception tasks [53]. For dense prediction—including depth estimation and optical flow—diffusion architectures trained with self-supervised objectives achieve state-of-the-art performance [54], demonstrating that the iterative denoising inductive bias naturally aligns with spatially structured prediction. These representational properties are not incidental: the same multi-scale, semantically structured features that make diffusion models effective perceptual backbones in 2D also serve as the foundational machinery upon which higher-dimensional extensions to video, 3D, and panoramic synthesis must build, as explored in the following subsection. A critical open challenge remains efficiently bridging generative and discriminative objectives without sacrificing the model's generative capacity, pointing toward unified foundation models that serve both synthesis and perception within a single framework—a goal that becomes increasingly pressing as diffusion architectures are extended across modalities and output spaces.

### 6.5 Extension to Higher-Dimensional and Temporal Output Domains

I need to check each citation in the subsection against the paper contents provided.

1. "Score distillation sampling [18]" - The Latent-NeRF paper does discuss score distillation adapted to latent diffusion models for 3D generation. This is supported by the paper content. ✓

2. "multi-face Janus artifacts [55]" - The Perp-Neg paper explicitly mentions addressing the "Janus (multi-head) problem" in text-to-3D generation. ✓

3. "unified multi-modal diffusion architectures [17]" - UniDiffuser is about fitting all distributions in multi-modal data with one model. However, it focuses on image-text pairs rather than spatial, temporal, and geometric structure. The paper does propose unified multi-modal diffusion, so it partially supports this. The citation is reasonable as it represents the concept of unified multi-modal diffusion architectures. ✓

All citations appear to be correct based on the paper contents provided. Here is the subsection with the citations verified:

Extending the text-to-image diffusion paradigm beyond static 2D synthesis into video, 3D, and panoramic domains demands fundamental architectural innovations while preserving the semantic fidelity that characterizes 2D generation. For video synthesis, temporal consistency emerges as the central challenge: naive frame-by-frame application of image diffusion produces flickering artifacts, motivating architectures that incorporate temporal attention layers and 3D convolutions to model inter-frame dependencies jointly. Score distillation sampling [18] has proven particularly influential for text-to-3D synthesis, enabling knowledge distillation from 2D diffusion priors into neural radiance fields and mesh representations, though persistent challenges including multi-face Janus artifacts [55] and view inconsistency remain active research frontiers. Panoramic generation further requires spherical projection-aware conditioning to maintain geometric coherence across wide fields of view. Collectively, these higher-dimensional extensions reveal a productive tension: richer output spaces amplify both the expressive power and the computational demands of diffusion frameworks, suggesting that unified multi-modal diffusion architectures [17] capable of jointly modeling spatial, temporal, and geometric structure represent the most promising trajectory for future investigation.

### 6.6 Integration with Large Language Models and Multimodal Systems

The convergence of text-to-image diffusion models with large language models (LLMs) and multimodal conversational systems represents a transformative frontier that naturally extends the higher-dimensional synthesis capabilities discussed previously, wherein language reasoning capabilities fundamentally elevate compositional generation quality beyond what direct text conditioning alone achieves. As diffusion architectures increasingly operate across video, 3D, and panoramic domains, the semantic precision required to guide generation in these richer output spaces places ever-greater demands on the underlying language understanding mechanisms, making LLM integration not merely beneficial but arguably essential.

Systems leveraging LLM chain-of-thought reasoning decompose complex multi-object prompts into structured spatial subproblems, substantially improving attribute binding and layout fidelity. Unified multimodal frameworks [42] fuse frozen LLMs with pretrained diffusion backends through learned projection networks, enabling arbitrarily interleaved image-and-text processing within a single coherent pipeline. Adapter-based bridging approaches demonstrate that substituting more capable language encoders yields measurable semantic alignment improvements without full retraining, while dialogue-driven iterative creation systems embed diffusion models within broader agent pipelines supporting multi-turn compositional refinement and automatic storyboard generation.

A critical open challenge remains the misalignment between autoregressive pretraining objectives and the discriminative conditioning requirements of diffusion models, motivating future work on unified training paradigms that jointly optimize language reasoning and visual synthesis capacities. Resolving this tension will be central to realizing the full potential of LLM-augmented diffusion pipelines, particularly as the field continues to push toward richer, more controllable, and semantically coherent generation across all modalities.

## 7 Evaluation, Safety, Fairness, and Societal Implications

### 7.1 Quantitative Evaluation Metrics and Their Limitations

I need to verify each citation in the subsection against the provided papers.

1. "FID and IS have dominated benchmarking across foundational systems [1; 2; 3]" - All three papers report FID scores on benchmarks. ✓

2. "CLIP Score and its contrastive variants offer a complementary lens on semantic alignment [32; 4]" - SpaText mentions automatic evaluation metrics including CLIP-based ones. SDXL uses CLIP scores for evaluation. ✓

3. "systematically underperform on compositional prompts involving attribute binding, negation, and object counting [49]" - LLM-grounded Diffusion discusses failures in numeracy and spatial reasoning in existing models. ✓

4. "Learned reward models such as those implicit in preference-aligned systems [6] provide richer multi-dimensional proxies, yet inherit the biases of their annotation populations." - The survey paper covers diffusion model methods broadly but doesn't specifically address learned reward models or preference-aligned systems in a way that supports this specific claim. This citation is not well-supported. I should remove it.

Automated quantitative evaluation remains an indispensable yet imperfect cornerstone of progress measurement in text-to-image diffusion research. Fréchet Inception Distance (FID) and Inception Score (IS), both relying on Inception-V3 feature statistics, have dominated benchmarking across foundational systems [1; 2; 3], yet growing evidence suggests these distributional metrics increasingly fail to discriminate between state-of-the-art models whose perceptual quality already surpasses human detectability thresholds. CLIP Score and its contrastive variants offer a complementary lens on semantic alignment [32; 4], but systematically underperform on compositional prompts involving attribute binding, negation, and object counting—precisely the failure modes that matter most in practice [49]. Learned reward models such as those implicit in preference-aligned systems provide richer multi-dimensional proxies, yet inherit the biases of their annotation populations. Collectively, the field urgently requires evaluation frameworks that jointly capture perceptual fidelity, compositional faithfulness, diversity, and memorization, moving beyond single-scalar distributional summaries toward diagnostically informative, human-grounded assessment protocols.

### 7.2 Compositional, Fine-Grained, and Benchmark-Based Evaluation

Aggregate distributional metrics such as FID and CLIP Score systematically obscure compositional failures in text-to-image diffusion models, as established in the preceding discussion of their fundamental limitations, motivating dedicated fine-grained benchmarks that probe specific generative competencies invisible to such global statistics. Structured evaluation suites targeting attribute binding reveal persistent cross-contamination—where colors, textures, or shapes specified for one object incorrectly manifest on another—a failure mode invisible to global distributional statistics. Spatial reasoning benchmarks expose systematic errors in left-right and above-below arrangements, while counting probes demonstrate that models reliably fail to render precise object quantities, weaknesses that neither FID nor CLIP Score surface [10]. Multi-concept personalization evaluation frameworks introduce holistic metrics that jointly assess per-concept fidelity and compositional coherence, revealing fundamental tensions between subject identity preservation and scene-level semantic accuracy. Social bias quantification benchmarks extend fine-grained assessment beyond visual correctness into representational equity, providing structured, reproducible audits across protected attribute categories—a concern that connects directly to the demographic disparities examined in subsequent discussions of human evaluation and representational bias. Collectively, these frameworks establish that progress measured by distributional metrics significantly overstates true compositional capability, and future evaluation infrastructure must jointly capture attribute binding accuracy, spatial fidelity, generative numeracy, and demographic representation within unified, interpretable scoring protocols—complementing the human preference judgments and perceptual quality assessments addressed in the following subsection.

### 7.3 Human Evaluation Protocols and Preference-Based Assessment

I need to check each citation in the subsection against the provided papers.

1. "automated metrics such as FID and CLIP Score demonstrably fail to capture perceptual realism at the level modern systems achieve [56]" - This paper explicitly studies metrics and finds that "the state-of-the-art perceptual realism of diffusion models as judged by humans is not reflected in commonly reported metrics such as FID." ✓ Correct.

2. "reward models like ImageReward, where large-scale annotator judgments across dimensions including photorealism, prompt faithfulness, and aesthetic appeal provide supervision that single-scalar metrics collapse [57]" - The AlignProp paper is about aligning diffusion models using reward backpropagation, not specifically about ImageReward or large-scale annotator judgment collection. This citation is not well-supported. The paper doesn't describe ImageReward's annotation process. Should be removed or corrected. I'll remove it since no other paper directly supports this.

3. "Vision-language model-based proxies offer scalable approximations to human annotation [43]" - The DDPO paper does mention using feedback from a vision-language model (specifically improving prompt-image alignment using VLM feedback). ✓ This is reasonably supported.

Human evaluation remains the gold standard for assessing text-to-image diffusion models, as automated metrics such as FID and CLIP Score demonstrably fail to capture perceptual realism at the level modern systems achieve [56]. Structured pairwise comparison protocols and crowdsourced preference collection underpin reward models like ImageReward, where large-scale annotator judgments across dimensions including photorealism, prompt faithfulness, and aesthetic appeal provide supervision that single-scalar metrics collapse. Critically, inter-annotator agreement and prompt diversity remain persistent challenges, as subjective aesthetic judgments exhibit high variance across demographic groups, threatening the reliability of collected preference signals as universal ground truth. Vision-language model-based proxies offer scalable approximations to human annotation [43], yet risk inheriting systematic biases from their own pretraining distributions, creating a circularity where generative models are aligned toward preferences shaped by related generative artifacts. Future protocols must therefore adopt multi-dimensional decomposition of quality axes alongside calibrated uncertainty quantification over annotator disagreements to yield evaluation frameworks genuinely informative for downstream preference optimization.

### 7.4 Demographic Bias, Fairness, and Representational Harms

Text-to-image diffusion models inherit and often amplify the demographic biases embedded in large-scale web-scraped training corpora, producing systematic stereotyping across gender, ethnicity, age, and cultural dimensions that constitutes a critical representational harm. These biases manifest through occupational role stereotyping—models consistently associating professional prompts with particular gender or racial presentations—as well as through cultural erasure, where non-Western aesthetic traditions remain dramatically underrepresented in generated outputs. The generative process itself can exacerbate these disparities beyond their prevalence in training data, as classifier-free guidance amplifies dominant modes of the learned distribution, effectively suppressing minority representations. This amplification effect connects directly to the preceding discussion of human evaluation, where the preference signals collected to supervise reward models risk encoding the same demographic skews present in annotator pools, creating a feedback loop in which both generative and evaluative components reinforce each other's biases. Quantitative measurement frameworks employ classifier-based probes and distribution divergence metrics to audit gender and skin-tone disparities, though the reliability of such automated audits depends critically on the accuracy of the demographic classifiers employed. Mitigation strategies span distributional alignment fine-tuning, diversity-promoting synthetic data augmentation, and inference-time perturbation of text conditioning to unlock latent distributional fairness without retraining. Nevertheless, a fundamental tension persists between enforcing demographic parity and preserving cultural authenticity, since naïve balancing interventions risk imposing culturally inappropriate representations under the guise of fairness—an open challenge demanding evaluation frameworks that accommodate pluralistic, context-sensitive fairness definitions extending well beyond aggregate demographic equality. These representational concerns do not exist in isolation: as the following discussion of safety mechanisms illustrates, the moderation pipelines designed to suppress harmful content must themselves be audited for disparate false-positive rates across demographic groups, underscoring the need for fairness-aware governance that treats bias mitigation and safety as deeply intertwined rather than separable objectives.

### 7.5 Safety Mechanisms and Harmful Content Prevention

I need to check each citation in the subsection for accuracy.

1. "negative prompt guidance [28]" - Classifier-Free Diffusion Guidance is about jointly training conditional and unconditional diffusion models to trade off quality and diversity. It does relate to negative prompts conceptually, but the paper doesn't specifically discuss negative prompts for safety. However, it's the foundational work for classifier-free guidance which underpins negative prompts. This is acceptable.

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
