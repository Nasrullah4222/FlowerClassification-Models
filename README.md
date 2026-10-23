# Tropical Flower Classification: CNNs, Vision Transformers, BYOL, and DINO

## Project Overview

This project explores tropical flower classification using four deep-learning notebooks. The experiments progress from supervised convolutional neural networks (CNNs), to supervised Vision Transformers, and then to self-supervised learning (SSL) with BYOL and DINO.

The dataset contains seven flower classes:

- Bougainvillea
- Crown of thorns
- Hibiscus
- Jungle geranium
- Madagascar periwinkle
- Marigold
- Rose

The notebooks investigate two related questions:

1. How accurately can pretrained models classify flower images when trained with labels?
2. How useful are visual representations learned without labels when evaluated using a lightweight classifier or nearest-neighbour search?

## Notebook Summaries

### 1. CNN Models --- Supervised Classification

**Notebook:** `cnn-cse-475-assignment-01-group-c.ipynb`

This notebook establishes a supervised baseline by fine-tuning pretrained CNN architectures on the seven-class flower dataset. It includes dataset exploration, class-distribution analysis, train/validation/test splitting, model training, and test-set evaluation.

**Models evaluated**

- MobileNetV3-Large
- EfficientNet-B3
- ResNeXt50-32x4d

**Reported results**

| Model | Test accuracy | Macro F1 | Trainable parameters |
| --- | ---: | ---: | ---: |
| EfficientNet-B3 | 100.00% | 100.00% | 10.7M |
| MobileNetV3-Large | 99.77% | 99.79% | 4.2M |
| ResNeXt50-32x4d | 99.32% | 99.33% | 23.0M |

**Main finding:** EfficientNet-B3 achieved the highest reported score in this notebook. MobileNetV3-Large was close behind with substantially fewer trainable parameters, making it an attractive lightweight alternative.

The notebook also examines performance under selected image corruptions. These robustness results should be interpreted separately from clean-test accuracy.

### 2. Transformer Models --- Supervised Classification

**Notebook:** `cse-475-transformer-models (1).ipynb`

This notebook compares pretrained transformer-based vision architectures using supervised fine-tuning. Beyond ordinary classification metrics, it includes confusion matrices, ROC analysis, corruption-based robustness testing, GradCAM++ visualisations, and confidence/calibration analysis.

**Models evaluated**

- ViT-B/16
- Swin-Tiny
- DeiT-Small

**Reported results**

| Model | Test accuracy | Macro F1 | AUC | Parameters |
| --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | 99.77% | 99.78% | 100.00% | 85.8M |
| DeiT-Small | 99.19% | 99.24% | 100.00% | 21.7M |
| Swin-Tiny | 97.92% | 97.99% | 99.90% | 27.5M |

**Main finding:** ViT-B/16 achieved the best transformer result, with DeiT-Small offering a smaller alternative and strong performance. ViT-B/16 also scored 99.77% accuracy on the reported clean test set.

The notebook's ViT-B/16 robustness tests show that performance depends on the type and severity of corruption. For example, accuracy was 94.12% under Gaussian noise at σ = 0.15 and 95.16% under Gaussian blur with kernel size 9. The reported expected calibration error (ECE) was 0.055, and one high-confidence incorrect prediction was detected.

### 3. BYOL --- Self-Supervised Representation Learning

**Notebook:** `byol-cse-475-assignment-02-group-c-ipynb.ipynb`

BYOL (Bootstrap Your Own Latent) learns image representations from two augmented views of the same image without using class labels in the self-supervised objective. It uses an online network and a slowly updated target network.

**Architecture and training**

- Backbone: EfficientNet-B3
- Online network: backbone + projector + predictor
- Target network: EMA-updated backbone + projector
- Training: 30 reported pre-training epochs
- Effective batch size: 128 (physical batch 32 with gradient accumulation of 4)
- Mixed-precision training and a learning-rate warm-up/cosine schedule

The notebook reports that the BYOL loss moved from -1.1537 to -1.7752 during pre-training.

**Downstream evaluation**

| Evaluation | Reported result |
| --- | ---: |
| Frozen-feature linear probe --- test accuracy | 98.85% |
| Linear probe --- macro F1 | 0.9870 |
| k-NN, k = 20 | 98.85% |

The k-NN evaluation also reports 98.61% at k = 1 and 98.85% at k = 5 and k = 10.

**Main finding:** BYOL produced highly useful representations for the reported downstream evaluation, despite not using class labels to train its self-supervised objective. The linear probe and k-NN results measure representation quality through different evaluation methods.

### 4. DINO --- Self-Distillation with No Labels

**Notebook:** `dino-cse-475-assignment-02-group-c (3).ipynb`

DINO learns visual representations using student and teacher networks. The teacher is updated using an exponential moving average (EMA) of the student. The notebook uses a ViT-B/16 backbone and multi-crop image augmentations.

**Architecture and training**

- Backbone: ViT-B/16
- Student and teacher networks with projection heads
- Reported pre-training: 60 epochs
- Batch size: 8
- Two global crops at 224 × 224 and local crops at 96 × 96
- Unlabelled images are used for self-supervised pre-training
- Attention maps are visualised to inspect the image regions the model attends to

The notebook records a final pre-training loss of 11.0891 and a minimum loss of 11.0606 at epoch 16. Its notes state that the DINO loss implementation was corrected to update the teacher-output center using raw teacher logits; the pre-training run needs to be rerun to obtain a valid corrected loss curve. Interpret the recorded loss with this caveat.

**Downstream evaluation**

| Evaluation | Reported result |
| --- | ---: |
| Frozen-feature linear probe --- test accuracy | 63.09% |
| Linear probe --- macro F1 | 0.6335 |
| k-NN, k = 1 | 55.25% |
| k-NN, k = 5 | 56.52% |
| k-NN, k = 10 | 55.59% |
| k-NN, k = 20 | 52.71% |

**Main finding:** In the recorded run, DINO's frozen representations were less effective for flower classification than the BYOL representations reported in the other notebook. The notebook also provides attention-map visualisations, which offer a qualitative view of the regions represented by the model.

## Results at a Glance

| Notebook / approach | Learning method | Backbone or model | Main reported result |
| --- | --- | --- | --- |
| CNN baseline | Supervised | EfficientNet-B3 | 100.00% test accuracy |
| Transformer comparison | Supervised | ViT-B/16 | 99.77% test accuracy |
| BYOL | Self-supervised, then linear probe | EfficientNet-B3 | 98.85% linear-probe test accuracy |
| DINO | Self-supervised, then linear probe | ViT-B/16 | 63.09% linear-probe test accuracy |

These scores are **not a strict head-to-head benchmark**. The notebooks use different evaluation protocols and test-set sizes. In particular, the CNN notebook reports 439 test examples, the transformer notebook reports 867, the BYOL notebook reports 433 for its downstream evaluation, and the DINO notebook reports 867. The supervised models are evaluated after labelled training, whereas BYOL and DINO are evaluated using frozen self-supervised features and a downstream classifier. Therefore, compare these numbers as results from each notebook's own experiment rather than as a definitive ranking under identical conditions.

## Overall Comparison

### Supervised CNNs vs. supervised transformers

Both approaches achieved very high accuracy in their respective experiments. EfficientNet-B3 reached 100.00% in the CNN notebook, while ViT-B/16 reached 99.77% in the transformer notebook. EfficientNet-B3 used far fewer trainable parameters than ViT-B/16 in the reported configurations. However, the test sets and experimental protocols differ, so these figures alone do not establish that one family is universally better.

### BYOL vs. DINO

Both methods learn representations without class labels in their self-supervised pre-training objective, but their recorded downstream results differ substantially. BYOL achieved 98.85% linear-probe test accuracy, while DINO achieved 63.09%. This is the outcome of the reported runs, not proof that BYOL is inherently superior to DINO. The DINO notebook documents a loss-centering correction and says the corrected pre-training run should be rerun, so its result should be treated as provisional.

### Accuracy is not the only consideration

- **EfficientNet-B3:** Best reported CNN accuracy in its notebook.
- **MobileNetV3-Large:** Strong CNN result with a comparatively small parameter count.
- **ViT-B/16:** Best transformer result in its notebook, with robustness and calibration analyses.
- **BYOL:** Strong recorded linear-probe and k-NN performance using self-supervised features.
- **DINO:** Lower recorded classification performance in this run, but includes attention-map analysis that can help inspect learned visual focus.

## Important Limitations

1. **Different splits and protocols:** Test-set sizes and evaluation setups vary across notebooks. Do not treat the table as a controlled comparison.
2. **Very high supervised scores:** Near-perfect accuracy should be validated with reproducible splits, duplicate/leakage checks, and ideally an independent test set before making deployment claims.
3. **Self-supervised results are downstream results:** Linear-probe accuracy and k-NN accuracy evaluate frozen representations; they are not directly equivalent to end-to-end supervised accuracy.
4. **DINO run caveat:** The notebook documents a loss-centering fix and says the corrected pre-training run needs to be rerun. Re-evaluate its downstream metrics after rerunning.
5. **Robustness and calibration:** Clean-test accuracy alone does not guarantee performance on images from different cameras, lighting conditions, backgrounds, or environments.

## Conclusion

Across the four notebooks, transfer learning with pretrained vision models produced strong results on the seven-class tropical flower dataset. The supervised CNN and transformer experiments achieved the highest classification scores in their respective notebooks. BYOL showed strong recorded self-supervised feature quality under both linear probing and k-NN evaluation. DINO's recorded downstream performance was lower, and its notebook identifies a training-loss correction that should be validated by rerunning the experiment.

The most useful next step is to standardise the train/validation/test split and evaluation pipeline, then rerun all four approaches under the same conditions. This would make the comparison more reliable and help separate differences caused by architecture from differences caused by data splits and training procedures.
