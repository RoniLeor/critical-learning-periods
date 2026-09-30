# Protocol and evidence boundaries

This is an independent reconstruction of the main accuracy experiment in
Achille, Rovere & Soatto, *Critical Learning Periods in Deep Networks* (ICLR 2019).
We did not locate verified author code. The implementation is not an exact
reproduction, and the repository does not reproduce every experiment in the paper.

## Published settings we follow

Full CIFAR-10, ten classes, 32×32 RGB inputs; the nine-block All-CNN listed in
Appendix A; SGD with batch 128, initial rate 0.05, multiplicative decay 0.97 per
epoch, and weight decay 0.001. Deficit images are bilinearly resized 32→8→32.
Every condition gets 160 clear epochs after blur removal.

Source: [paper, Section 2 and Appendix A](https://arxiv.org/html/1711.08856v3).

## Explicit reconstruction choices

These details are assumptions, not author-confirmed settings:

- No momentum or Nesterov; no learning-rate reset after blur removal.
- PyTorch default initialization and BatchNorm settings; convolution bias enabled.
- Same padding for 3×3 convolutions; Conv–BatchNorm–ReLU for every block,
  including the last 1×1 block. Average-pooled activations serve as logits.
- PIL bilinear resizing before augmentation. Pillow version is recorded.
- Zero padding of four pixels, random 32×32 crop, horizontal flip probability 0.5.
- Channel means `(0.4914, 0.4822, 0.4465)` and standard deviations
  `(0.2023, 0.1994, 0.2010)`.
- Blur durations 0, 40, 100; seeds 0, 1, 2. This samples three conditions rather
  than the paper's full blur-duration sweep.

Keeping decay running means the 100-blur condition starts clear training at a
lower learning rate. This experiment alone cannot separate that effect from
representation changes. Fixed-rate and rate-reset controls would be separate
experiments; we have not run them here.

## Split and checkpoint policy

Keep all 50,000 official training images and 10,000 official test images. Validate
archive checksums through torchvision; audit exact shared pixel hashes. The
recorded dataset had zero shared hashes. Do not select a checkpoint by test
accuracy. Report the last scheduled epoch.

`train` writes a resumable checkpoint after every epoch, using atomic replacement.
It contains model, optimizer, random-number states, predictions and full epoch
history. Completed runs are skipped when resumed. Source/configuration mismatch
is rejected. Use a new output directory after code or runtime changes. Checkpoints
are local trusted artifacts: load only files you created or trust.

The latest checkpoint replaces its predecessor. Historical epoch metrics survive,
but old weights do not; these files cannot reconstruct historical Fisher curves.

## Fisher measurement

`analyze` examines the three final seed-0 checkpoints. The same 300 balanced clear
training images (30/class) are used for every model. Five posterior-label sampling
seeds estimate the mean squared score-gradient norm, grouped into nine blocks.
Labels are sampled from the model distribution, not replaced with ground truth.
Conv and BatchNorm parameters contribute to each block's trace. Evaluation mode
freezes running statistics; parameters, buffers and existing gradients are preserved.

This is direct Monte Carlo model Fisher. The paper used a noise-based variational
approximation for All-CNN; this repository does not implement that approximation.
Error bars are standard deviations across five label-sampling repetitions,
not confidence intervals and not variability across training seeds or probe sets.

Trace magnitude depends on parameterization and model confidence. Higher trace
is prediction sensitivity to weight changes, not more knowledge, Shannon bits,
or proof of reduced plasticity. Layer shares hide changes in total magnitude.
An endpoint association cannot establish the causal mechanism or permanence.

## Included evidence

`results/seed0/` is a frozen snapshot: three completed seed-0 runs and their Fisher
measurements. Checkpoint hashes identify the original measured models; checkpoints
and datasets are not distributed. The remaining training seeds were still running
when this snapshot was prepared. The original execution configuration is retained
for provenance. Packaging changed paths and imports; the core architecture,
preprocessing and optimization are checked against the source implementation.
