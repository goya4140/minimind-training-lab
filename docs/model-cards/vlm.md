# MiniMind-Lab VLM Model Card

## Model

Image-and-text to text model built from the 63.9M MiniMind LLM, a frozen SigLIP2 Base P32 256 vision
encoder, and a 1.18M two-layer projector. Sixty-four image tokens replace `<|image_pad|>` embeddings
before the shared causal language model.

## Training

The model inherits the completed LLM SFT checkpoint. Stage one freezes both LLM and SigLIP2 and trains
the projector. Stage two keeps SigLIP2 frozen and trains the projector plus the first and last LLM layers.
Configurations are pinned under `configs/vlm/`.

## Evaluation and limitations

The fixed evaluation covers captioning, VQA, OCR-like prompts and visual hallucination cases. Final loss,
keyword recall, generation examples and language-regression evidence are collected in
`reports/final-results.md`.

The visual backbone is not trained from scratch. The small projector and bounded local training budget can
miss fine spatial detail, text in images, counting, unusual scenes and adversarial inputs. Generated claims
must be independently verified.
