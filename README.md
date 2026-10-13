# Residual Networks

A from-scratch PyTorch implementation of the bottleneck ResNet-50/101/152 architectures from
"Deep Residual Learning for Image Recognition", with a config-driven training pipeline for CIFAR-100.

## What's here

- `src/model.py`
  - `block` - the bottleneck residual block: `1x1 -> 3x3 (stride) -> 1x1` convolutions with
    BatchNorm and ReLU, 4x channel expansion, and a projection shortcut (`1x1` conv + BN) when the
    stride or channel count changes.
  - `ResNet` - `7x7/2` conv stem + max-pool, four stages (64/128/256/512 bottleneck widths),
    global average pooling, and a linear classifier.
  - `ResNet50`, `ResNet101`, `ResNet152` - stage depths `[3,4,6,3]`, `[3,4,23,3]`, `[3,8,36,3]`.
- `src/train.py` - `Trainer`: per-epoch train/test loops with optional gradient-norm clipping and LR
  scheduler (`ReduceLROnPlateau` is stepped on test loss), accuracy via `src/helper.py`, TensorBoard logging.
- `main.py` - loads a YAML config, builds the CIFAR-100 loaders, model, Adam/SGD optimizer and
  StepLR/ReduceLROnPlateau/CosineAnnealingLR scheduler, then trains.
- `config/*.yaml` - experiment configs (model, data, optimizer, scheduler, epochs, logging).
- `src/datasets/cifar10.py`, `src/datasets/cifar100.py`, `src/transforms.py` - standalone data-loader
  helpers with augmentation; these are not used by `main.py`.

Notes on the setup:

- The network keeps the ImageNet stem, so a 32x32 CIFAR image is reduced to 8x8 after the stem and
  1x1 in the last stage. The paper's own CIFAR experiments used a smaller 3-stage ResNet instead.
- `main.py` applies only `ToTensor` + `Normalize` (no augmentation) and supports only `CIFAR100`.

## Project structure

```
.
├── main.py                          # entrypoint
├── config/
│   ├── resnet-cifar100.yaml         # ResNet50, Adam lr 1e-3
│   ├── resenet-101-cifar100.yaml    # ResNet101, Adam lr 1e-3
│   ├── resenet-152-cifar100.yaml    # ResNet152, Adam lr 1e-4 (used by main.py)
│   └── resnet-cifar10.yaml          # used only by src/datasets/cifar10.py
├── src/
│   ├── model.py
│   ├── train.py
│   ├── helper.py
│   ├── transforms.py
│   ├── datasets/{cifar10,cifar100}.py
│   └── utils/{config_loader,resnet_logging,schedulers}.py
├── runs/ResNet-Cifar100/            # TensorBoard event files from earlier runs
└── cifar100_experiment.log          # log from earlier short runs
```

## Getting started

```bash
git clone https://github.com/khushwant04/Residual-Networks.git
cd Residual-Networks
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py                       # run from the repo root
tensorboard --logdir runs            # loss/accuracy curves
```

CIFAR-100 is downloaded to `./data` on first run. Training logs go to the file named by
`logging.log_file` and to the console; TensorBoard events go to `logging.log_dir`.

To pick a different config, change the path in `main()` in `main.py`
(currently `config/resenet-152-cifar100.yaml`). Key config fields:

```yaml
model:     { name: ResNet50 | ResNet101 | ResNet152, num_classes: 100, input_channels: 3 }
optimizer: { name: Adam | SGD, lr: 0.001, weight_decay: 0.0001 }
scheduler: { name: StepLR | ReduceLROnPlateau | CosineAnnealingLR, step_size: 10, gamma: 0.1 }
training:  { num_epochs: 50, gradient_clip_value: 1.0, use_scheduler: true }
```

Other entrypoints:

```bash
python -m src.model                  # shape check: ResNet101 on a 4x3x224x224 batch
python -m src.datasets.cifar100      # load one augmented CIFAR-100 batch
```

## References

- He, Zhang, Ren, Sun, *Deep Residual Learning for Image Recognition*, CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Ioffe, Szegedy, *Batch Normalization*, 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
- Krizhevsky, *Learning Multiple Layers of Features from Tiny Images* (CIFAR-10/100), 2009. [Dataset page](https://www.cs.toronto.edu/~kriz/cifar.html)
