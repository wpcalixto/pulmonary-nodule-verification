# Methods summary

Ten nnU-Net v2.8.1 configurations (2D, 3D full-resolution, 3D low-resolution and cascade variants, standard and residual-encoder plans,
with and without 100%% foreground oversampling) were trained with five-fold cross-validation on the development part of BD1
(LIDC-IDRI). Predicted components were matched one-to-one to reference components with the Hungarian algorithm (IoU). Sixteen
multiscale morphological and anatomical responses were computed for each object and a post-detection verifier was trained on
out-of-fold predictions, frozen, and applied without re-tuning to the internal validation set (BD1), to RIDER test-retest pairs
(repeatability) and to LNDb (external validation). Stage-by-stage description: `docs/usage.md`; design: 718 development patients,
158 internal-validation patients, 31 test-retest pairs, 211 external scans.
