# corecode

This directory holds the core components of EndoMEDE.

## `wave_dpt_head.py` — Wavelet DPT decoder

`WaveDPTHead` is a DPT-style decoder that does feature fusion and refinement in the wavelet domain.

- Four intermediate ViT layers are projected and resized into a multi-scale feature pyramid.
- `WaveFusionBlock` replaces the RefineNet fusion block of standard DPT. Each feature map is split by a Haar DWT into LL / HL / LH / HH sub-bands, and each band is handled differently:
  - LL (low frequency): isotropic 3×3 convolution.
  - HL / LH (vertical / horizontal edges): direction-aware 3×1 / 1×3 convolutions, guided by the LL band.
  - HH (diagonal detail): scaled by a learnable factor.
  
  The bands are then recombined with an IWT.
- `WaveOutputHead` runs one more wavelet refinement of the HL/LH bands before the final upsampling. A sigmoid then outputs relative depth, which is multiplied by `max_depth`.

## `lora.py` — PAS-LoRA (patch-adaptive-scale LoRA)

A parameter-efficient fine-tuning method for the DINOv3 ViT backbone. The pretrained backbone stays frozen.

- **MoE-LoRA**: low-rank adapters are added to the attention `qkv` and to the MLP `fc1` / `fc2`. Each adapter has several LoRA experts, mixed by an image-level softmax gate.
- **Patch-adaptive scale** (`StorageGuidedLoRAScaleGenerator`): predicts a separate LoRA scale for every patch token in every block. Different image regions, such as specular highlights, tissue folds, or dark lumen, therefore get different amounts of adaptation. Prefix tokens (cls and storage) keep a scale of 1.
- **Dynamic storage adapter** (`DynamicStorageAdapter`): adds an image-conditioned offset to the DINOv3 storage (register) tokens.

## `loss.py` — Training losses

The total loss is SiLog plus three structure-aware terms, each weighted 0.2:

- `SiLogLoss`: scale-invariant log loss on depth.
- `EdgeAwareGradientLoss`: L1 on log-depth gradients, with higher weight on image edges (Sobel).
- `EdgeAwareSmoothnessLoss`: encourages smooth depth in regions where the image is smooth. It does not need GT.
- `MultiScaleGradientLoss`: log-depth gradient matching at three scales (1, 1/2, 1/4).
- `TotalLoss`: combines the four terms above, with the same weights as in `train.py`.

## `c3vd_process.py` / `scared_process.py` — Dataset lists

These scripts build the train/val/test file lists. Each line is an `image_path depth_path` pair.

- **C3VD**: pairs `color/*_color.png` with `depth/*_depth.tiff` for the selected sequences.
- **SCARED**: pairs `left_undistorted/*.png` with `depthmap_undistorted/*.png` for each keyframe of the selected datasets.
