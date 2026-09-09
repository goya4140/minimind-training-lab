# MiniMind-Lab Video-Omni Model Card

## Model

Video-and-question to text model. Eight uniformly sampled frames pass through frozen SigLIP2. A learned
spatial query pools every frame, a two-layer temporal Transformer models ordered frame features, and a
learned-query resampler produces 16 video tokens for the MiniMind LLM.

The formal configuration has 178,569,984 total parameters. Alignment trains 20,105,472 temporal/projector
parameters; video instruction tuning trains those modules plus the LLM boundary layers for 34,854,528
trainable parameters.

## Training data and policy

QIVD is pinned to revision `c5376ab0b9fd3643545a1503413aee64f26ba22a` and split by a stable seeded
hash into 2,400 train, 250 validation and 250 held-out test samples. QIVD is research-only, so videos are
not redistributed through this repository.

Only sampled visual frames and the written question enter the model; audio is ignored. SigLIP2 remains
frozen. The MiniMind language core and newly introduced temporal/projector modules follow the repository's
documented from-scratch/inheritance chain.

## Evaluation and limitations

Evaluation reports QIVD loss, exact match, containment, token F1, category breakdown and qualitative
answers. It also reverses frames and runs a separate 160-example controlled temporal benchmark covering
horizontal/vertical motion, size change and event order. A normal-over-reversed advantage is required as
evidence of temporal use.

Uniform eight-frame sampling can miss brief events. The model has no audio path, cannot localize exact
timestamps, and may rely on language priors instead of video. It is an educational research artifact, not a
general video safety, surveillance, medical, or autonomous-control system.
