# Reproducibility environment

The reported experiments were run on Ubuntu 24.04 with an NVIDIA
A100-SXM4-80GB GPU, 116 GiB RAM, and 28 logical CPU cores. The run logger stores
the exact Git revision, source fingerprint, Python and PyTorch versions, CUDA
runtime, GPU name, resolved configuration, and parameter counts alongside every
training run.

The project does not depend on this exact machine. Full DINOv3 and DeepLabV3
training requires a CUDA-capable GPU, while tests and lightweight analysis can
run on CPU.

## Local environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,download]'
```

Model and dataset paths are relative to the repository and are excluded from
Git:

```text
data/
└── plantseg_v3/
models/
├── dinov3-vitb16-pretrain-lvd1689m/
├── dinov3-vitl16-pretrain-lvd1689m/
└── sam-vit-base/
```

PlantSeg v3 contains 7,916 train, 1,247 validation, and 2,295 test images. The
dataset audit command checks file presence and mask-label ranges:

```bash
python -m phytopathology.audit_data data/plantseg_v3
```

DINOv3 weights used in the experiments are available from ModelScope:

```bash
modelscope download --model facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local_dir models/dinov3-vitb16-pretrain-lvd1689m
```

The optional PlantWild v1 diagnostic uses only healthy images from its official
test split. Its manifest can be prepared with
`python -m phytopathology.prepare_plantwild_healthy`; this external evaluation
measures behaviour under domain shift and is not an in-domain specificity
estimate.

## Run artefacts

Every training run creates a separate timestamped directory under `runs/` with:

- the resolved YAML configuration;
- Git revision, dirty-worktree flag, and source snapshot hash;
- Python, PyTorch, CUDA, and GPU metadata;
- total and trainable parameter counts;
- JSONL metric history and the best checkpoint.

`runs/`, datasets, weights, cached features, and subset manifests are ignored by
Git. Concise protocols and derived metrics are retained in `docs/experiments/`
and `results/`.
