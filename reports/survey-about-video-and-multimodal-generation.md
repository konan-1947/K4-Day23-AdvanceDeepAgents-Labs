# Video and Multimodal Generation: Architectures, Temporal Dynamics, and Foundation Models

## TL;DR
- Video generation has transitioned from extending 2D U-Nets with temporal convolutions [1][2][3] to 3D spatio-temporal Diffusion Transformers (DiT) operating on compressed visual spacetime patches [4][5].
- Spatio-temporal latent autoencoders compress variable-duration video streams into continuous latent representations, enabling efficient generative modeling across compute budgets [2][5][6].
- Contemporary foundation models scale parameters to billions (e.g. Movie Gen [7], Wan [6]), integrating text-to-video, image-to-video, personalized video editing, and synchronized multimodal audio generation [7][8].
- Preserving physical causal consistency, long-horizon narrative coherence, and high-frequency anatomical fidelity across continuous motion remain significant open challenges [5][6][7].

## Background
Generative modeling for video extends image synthesis by introducing a temporal dimension that requires smooth transitions, coherent optical flow, and physical plausibility [1]. Early deep video synthesis methods struggled with blurriness and severe temporal flickering due to limited capacity and simplistic pixel-loss formulations [3]. The emergence of Denoising Diffusion Probabilistic Models (DDPM) revolutionized visual generation, initially through Video Diffusion Models (VDM) that inserted 1D temporal attention layers between pretrained 2D spatial convolution blocks [1]. To overcome the prohibitive memory demands of processing raw video frames at high frame rates, Video Latent Diffusion Models (VLDM) shifted the diffusion process into the latent space of pretrained spatial autoencoders, training temporal alignment layers on video clips while freezing image synthesis weights [2].

## Architectural Evolution: From 2D+1D U-Nets to Spatio-Temporal Diffusion Transformers
While U-Net backbones with alternating spatial-temporal attention dominated early architectures [1][2][3], they suffered from rigid inductive biases that limited compute scaling [4]. Diffusion Transformers (DiT) demonstrated that replacing convolutional U-Nets with isotropic transformer blocks yields predictable power-law scaling in visual generative quality [4]. Sora generalized DiT to video by representing video frames as continuous spatio-temporal patches ("spacetime latent patches") [5]. By treating visual tokens analogously to text tokens in language modeling, Diffusion Transformers handle arbitrary aspect ratios, resolutions, and video durations natively [5]. Recent open-weight models such as Wan-2.1 further optimize this design using unified 3D Variational Autoencoders and flow matching objectives, achieving photorealistic texture quality and coherent physical motion [6].

## Multimodal Conditioning and Controllable Generation
High-fidelity video synthesis requires conditioning beyond sparse text prompts [3][8]. Contemporary multimodal pipelines accommodate multimodal conditioning including reference images, dense edge trajectories, camera poses, and depth maps [7][8]. Movie Gen introduced a unified media foundation framework spanning high-definition text-to-video, localized instruction-based video editing, personalized subject preservation, and coordinated multi-channel audio synthesis [7]. By leveraging joint pretraining on video and synchronized sound, models capture cinematic temporal dynamics alongside natural acoustic alignment (such as ambient effects, footsteps, and environmental resonance) [7].

## Open-Weight Architectures and Efficient Sampling
The training compute required for foundation video models is immense, often involving tens of thousands of GPU hours [5][7]. Consequently, open-source efforts prioritize architectural efficiency [6]. Flow matching frameworks and rectified flow formulations straighten sampling trajectories, cutting diffusion inference steps from 50–100 down to 20–30 without sacrificing visual detail [6]. Furthermore, multi-stage cascading pipelines—separating base low-resolution video generation from temporal and spatial latent super-resolution—allow scalable inference on consumer hardware [2][6].

## Trends and open problems
Despite impressive visual fidelity, multimodal video generation faces foundational technical hurdles. First, temporal error compounding causes objects to deform, merge, or violate conservation laws over clips exceeding several seconds [5][6]. Second, physical common sense (intuitive physics) is learned purely from 2D pixel statistics rather than an embodied 3D physics engine, leading to common failure modes such as unnatural liquid flows, inverted collisions, and impossible biomechanical movements [5][7]. Finally, evaluation metrics remain unreliable: standard 2D metrics (like FVD and Fréchet Inception Distance) fail to capture temporal consistency, human anatomical accuracy, or prompt instruction adherence, demanding next-generation benchmark suites [6][7].

## References
[1] Video Diffusion Models. arxiv. https://arxiv.org/abs/2204.03458 (2022-04-07)
[2] Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models. arxiv. https://arxiv.org/abs/2304.08818 (2023-04-18)
[3] Make-A-Video: Text-to-Video Generation without Text-Video Data. arxiv. https://arxiv.org/abs/2209.14792 (2022-09-29)
[4] Scalable Diffusion Models with Transformers. arxiv. https://arxiv.org/abs/2212.09748 (2022-12-19)
[5] Video generation models as world simulators. web. https://openai.com/index/video-generation-models-as-world-simulators (2024-02-15)
[6] Wan: Open and Advanced Large-Scale Video Generative Models. hf-daily. https://huggingface.co/papers/2503.20314 (2025-03-01)
[7] Movie Gen: A Cast of Media Foundation Models. hf-search. https://huggingface.co/papers/2410.13720 (2024-10-17)
[8] Gen-2: The Next Step in Generative AI. web. https://runwayml.com/research/gen-2 (2023-03-20)
