# cuDLA Python end-to-end sample — ResNet-50 on Jetson DLA

An end-to-end image classifier that runs the heavy ResNet-50
backbone on the Jetson **DLA** using **cuda-python + cuDLA python**, with the
lightweight stem/head on the GPU via **ONNX Runtime**. Pre/post-processing is
plain **NumPy**.

Given `dog.jpg`, it prints the predicted breed and writes an annotated image:

```
dog.jpg  ─►  golden retriever  ~98.1%
```

## The pipeline

```
image [1,3,224,224]
   │  NumPy preprocess (resize/crop/normalize)
   ▼
 stem   ──  ONNX Runtime (CUDA EP)      ──►  [1,64,56,56]      # GPU  (pre)
   ▼
 backbone ─ NumPy + cuda-python + cuDLA ──►  [1,2048,7,7]      # DLA  (real workload)
   ▼
 head   ──  ONNX Runtime (CUDA EP)      ──►  [1,1000] logits   # GPU  (post)
   │  NumPy softmax / top-k
   ▼
 "golden retriever 98.1%"  +  output/classified.jpg
```

The DLA stage is the interesting part, and `dla_runner.py` **is written to make
the cuDLA API easy to follow**: every `cudla.`* call is a labelled step in the
lifecycle (`create_device → module_load_from_memory → module_get_attributes → mem_register → Task → submit_task → …teardown`). The ordinary CUDA bookkeeping
(malloc/memcpy/stream) is delegated to `cuda_mem.py`, and the FP16 `CHW16`
surface reshaping to `packing.py`, so `dla_runner.py` reads as *only* the cuDLA
story.

## Library roles


| Stage                             | Library                           | Runs on |
| --------------------------------- | --------------------------------- | ------- |
| preprocess / postprocess          | NumPy (+ Pillow to load/annotate) | CPU     |
| stem (conv/bn/relu/maxpool)       | ONNX Runtime, CUDA EP             | GPU     |
| **backbone (ResNet stages 1–4)**  | **NumPy + cuda-python + cuDLA**   | **DLA** |
| head (globalavgpool/flatten/gemm) | ONNX Runtime, CUDA EP             | GPU     |




## Clone

```bash
git clone https://github.com/NVIDIA/Deep-Learning-Accelerator-SW.git
cd Deep-Learning-Accelerator-SW/samples/cuDLA/cuDLAPythonE2E
```

Model binaries are not included in this repository. Download the original
ResNet-50-v1-12 from the ONNX Model Zoo, prepare the subgraphs, and compile the
DLA loadable on the target Jetson as described below. Git LFS is not required.

## Setup

On Debian/Ubuntu-based Jetson systems, install venv support for the system
Python 3 interpreter before creating the virtual environment:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
```

If venv creation previously failed with `ensurepip is not available`, install
this package and rerun the venv command below. If `python3` points to a
non-default interpreter, install its matching `python3.X-venv` package instead.

Create a local virtual environment and install all dependencies from
`requirements.txt`. The file includes the Jetson package index needed for
`onnxruntime-gpu`:

```bash
python3 -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt
```



### Download and build the model on target

Install TensorRT 11.4.1 (including `trtexec`) and `curl` on the target Jetson. With
`trtexec` on `PATH`, run:

```bash
bash setup_models.sh
```

The script downloads [ResNet-50-v1-12 from the ONNX Model Zoo](https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx),
runs `model_prep/prepare_resnet50_dla.py`, and builds the standalone FP16 CHW16
DLA loadable with `trtexec` on DLA core 0. Preparation converts only the backbone
to FP16; the GPU stem/head stay FP32. The script uses TensorRT 11's format-only
I/O syntax. All artifacts go into the ignored `models/` directory.
The original model is reused if already present; the
subgraphs and loadable are rebuilt on each run. ONNX Model Zoo remains the
source of truth; no model binaries or prebuilt loadables are redistributed here.

For a custom TensorRT installation, set its library path and executable:

```bash
export TRT_DIR=/path/to/TensorRT
export LD_LIBRARY_PATH="${TRT_DIR}/lib:${LD_LIBRARY_PATH:-}"
TRTEXEC="${TRT_DIR}/bin/trtexec" bash setup_models.sh
```

`PYTHON` can override the default `./venv/bin/python`. Build the loadable on
the target with its installed TensorRT/DLA stack rather than copying one from
another system. The equivalent manual commands are under [Model](#model).

## Run it

From the `samples/cuDLA/cuDLAPythonE2E` directory:

```bash
./venv/bin/python run.py
```

Every normal invocation uses the hybrid pipeline: the stem runs on the GPU, the
backbone runs on the DLA, and the head runs on the GPU. If `--image` is omitted,
the bundled `assets/dog.jpg` is used.

Useful flags:

```bash
./venv/bin/python run.py --image assets/dog.jpg --topk 5
./venv/bin/python run.py --dla-core 1        # use the 2nd DLA core
./venv/bin/python run.py --validate          # hybrid run, then an additional full-GPU cross-check
./venv/bin/python run.py --verbose           # cuDLA handles, tensor descriptors, submission/copy progress
```

`--verbose` prints application-side cuDLA diagnostics, not tensor contents.
Submission is asynchronous; the host-copy completion message is printed only
after stream synchronization. Submission/copy messages appear for both the
warm-up and timed inference, and printing affects the reported timing.

`--validate` does not replace the hybrid pipeline. It runs the normal
GPU → DLA → GPU inference first, then runs the original full model on the GPU
and compares the predictions.

Expected: top-1 `golden retriever (207) ~98.1%`, and an annotated image at
`output/classified.jpg`.

On some Jetson systems, ONNX Runtime may print a device-discovery warning about
`/sys/class/drm/card1/device/vendor`. It is harmless when the hybrid result (and
the full-GPU result when using `--validate`) completes successfully.

### Sample images

Three bundled images under `assets/` (all standard ImageNet classes):

```bash
./venv/bin/python run.py --image assets/dog.jpg   --out output/dog.jpg    # golden retriever  ~98.1%
./venv/bin/python run.py --image assets/cat.jpg   --out output/cat.jpg    # tiger cat         ~53.0%
./venv/bin/python run.py --image assets/lion.jpg  --out output/lion.jpg   # lion              ~99.4%
```

Image sources and licenses are listed under [Credits](#credits) below.

## Files

```
cuDLAPythonE2E/
├── run.py          # CLI entry point (preprocess -> pipeline -> print + annotate)
├── pipeline.py     # GPU stem -> DLA backbone -> GPU head
├── dla_runner.py   # ★ the cuDLA API, front and centre (create_device ... submit_task)
├── cuda_mem.py     #   generic CUDA plumbing (malloc/memcpy/stream) kept out of the way
├── packing.py      #   NumPy NCHW <-> CHW16 surface reshaping (no math)
├── engine.py       # ONNX Runtime (CUDA EP) session builder for stem/head
├── imagenet.py     # NumPy/PIL preprocess, labels, softmax, top-k
├── runtime.py      # ensure_libs(): CUDA + cuDLA on LD_LIBRARY_PATH
├── setup_models.sh # download upstream model, prepare subgraphs, build on target
├── model_prep/
│   └── prepare_resnet50_dla.py   # regenerate stem + backbone + head from the full ONNX
├── models/         # created locally by setup_models.sh; not tracked
│   ├── resnet50_stem.onnx              # GPU pre
│   ├── resnet50_head.onnx              # GPU post
│   ├── resnet50_v1_prepared.loadable   # the DLA backbone (compiled loadable)
│   ├── resnet50_v1_prepared.onnx       # reference: the prepared backbone (DLA cut points)
│   └── resnet50-v1-12.onnx             # upstream full model (preparation + --validate)
├── assets/
│   ├── dog.jpg                         # sample: golden retriever
│   ├── cat.jpg                         # sample: tiger cat
│   ├── lion.jpg                        # sample: lion
│   └── synset.txt                      # ImageNet labels (ONNX Model Zoo)
└── output/                             # annotated result image lands here
```



## Requirements

Reference configuration: Jetson AGX Orin (2 DLA cores), JetPack 7.2-class image
(CUDA 13.4, L4T tegra kernel), Python 3.12. Setup uses the system `python3`;
Python 3.12 is not explicitly required by the sample. Other Python versions
need compatible wheels for the dependencies in `requirements.txt`, especially
`cuda-bindings` and `onnxruntime-gpu`.

Runtime Python dependencies (see **Setup** and `requirements.txt`):


| Package             | Role                            | License                            |
| ------------------- | ------------------------------- | ---------------------------------- |
| `numpy`             | preprocess / postprocess        | BSD-3-Clause (and bundled notices) |
| `pillow`            | image load / annotate           | MIT-CMU                            |
| `onnxruntime-gpu`   | GPU stem + head (CUDA EP)       | MIT                                |
| `cuda-bindings`     | cuDLA / CUDA Python bindings    | NVIDIA Software License            |
| `nvidia-cudnn-cu13` | `libcudnn.so.9` for ORT CUDA EP | NVIDIA Proprietary (cuDNN SDK)     |

`requirements.txt` also installs `onnx` and `onnx-graphsurgeon` for model
preparation.



## Notes

- The DLA is a **[low-power](https://github.com/NVIDIA/Deep-Learning-Accelerator-SW/tree/main#dla-performance-per-watt-power-efficiency)** accelerator, not a faster GPU: a single image is not
the speed win. The value is running the backbone at a small power budget while
the GPU stays free (higher throughput-per-watt when the GPU is also busy).
- Single-input/single-output FP16 `CHW16` loadable only (asserted at load).
- To classify a different DLA-compatible model, swap the three artifacts
(`resnet50_stem.onnx`, `resnet50_head.onnx`, `resnet50_v1_prepared.loadable`)
and update the pre/post-processing in `imagenet.py`.



## Tests

CPU-only tests cover backbone FP16 conversion and TensorRT 11 `trtexec`
arguments (the compiler is mocked):

```bash
./venv/bin/python -m unittest discover -s tests -v
```

The target build and `run.py --validate` are still needed to verify DLA inference.

## Credits

Sample images under `assets/` (from Wikimedia Commons):


| File       | Author      | License             | Source                                                                                                                   |
| ---------- | ----------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `cat.jpg`  | —           | Public domain (CC0) | [Tabby cat with blue eyes-3336579.jpg](https://commons.wikimedia.org/wiki/File:Tabby_cat_with_blue_eyes-3336579.jpg)     |
| `dog.jpg`  | —           | Public domain (CC0) | [Golden-retriever-dog-1362597631o6g.jpg](https://commons.wikimedia.org/wiki/File:Golden-retriever-dog-1362597631o6g.jpg) |
| `lion.jpg` | Kevin Pluck | CC BY 2.0           | [Lion waiting in Namibia.jpg](https://commons.wikimedia.org/wiki/File:Lion_waiting_in_Namibia.jpg)                       |


Each image was downscaled (long side ≤ 1024 px) for this repo; no other changes.

ImageNet class names come from the ONNX Model Zoo
[synset.txt](https://github.com/onnx/models/blob/main/validated/vision/classification/synset.txt)
(Apache-2.0), shipped unmodified as `assets/synset.txt`.

### Model

The locally generated ResNet-50 artifacts derive from **ResNet-50-v1-12** in the
[ONNX Model Zoo](https://github.com/onnx/models) (Apache-2.0):

```bash
mkdir -p models
curl --fail --location -o models/resnet50-v1-12.onnx \
  https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx
```

- `resnet50-v1-12.onnx` — the original full model (as downloaded above).
- `resnet50_stem.onnx` / `resnet50_head.onnx` — the GPU stem/head subgraphs, cut
out of the full model at the DLA boundaries.
- `resnet50_v1_prepared.onnx` — the DLA-compatible backbone (trimmed at the
DLA cut points and converted to FP16; BatchNormalization nodes are preserved).
- `resnet50_v1_prepared.loadable` — that backbone compiled to a DLA loadable for
this board (FP16, `CHW16`).

`model_prep/prepare_resnet50_dla.py` regenerates the stem, backbone, and head
subgraphs from the original model (needs `onnx onnx-graphsurgeon numpy`):

```bash
./venv/bin/python model_prep/prepare_resnet50_dla.py \
    --fp16 \
    --input models/resnet50-v1-12.onnx \
    --output models/resnet50_v1_prepared.onnx \
    --stem   models/resnet50_stem.onnx \
    --head   models/resnet50_head.onnx
```

The `.loadable` is produced by compiling `resnet50_v1_prepared.onnx` on the
target board (FP16 weights, `CHW16` I/O to match what `dla_runner.py` packs).
With TensorRT 11 installed, use `trtexec --buildDLAStandalone` from the sample directory
(adjust `TRT_DIR` to your install; the prepared ONNX already has a concrete
batch of 1, so do **not** pass `--shapes`):

```bash
export TRT_DIR=/path/to/TensorRT
export LD_LIBRARY_PATH=${TRT_DIR}/lib:/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda/targets/sbsa-linux/lib

${TRT_DIR}/bin/trtexec \
    --useDLACore=0 \
    --memPoolSize=dlaSRAM:1 \
    --inputIOFormats=chw16 \
    --outputIOFormats=chw16 \
    --onnx=models/resnet50_v1_prepared.onnx \
    --saveEngine=models/resnet50_v1_prepared.loadable \
    --buildDLAStandalone
```

TensorRT 11 takes precision from the model's tensor types, so it requires the
FP16 preparation step above; `--fp16` is no longer a `trtexec` option and the
I/O format options specify only the layout.

The resulting loadable's I/O descriptors are `resnetv17_pool0_fwd [1,64,56,56]`
→ `resnetv17_stage4_activation2 [1,2048,7,7]`, FP16 `CHW16` — exactly the cut
points the stem produces and the head consumes.

## License

Sample code is licensed under [Apache-2.0](LICENSE). The bundled images and
`assets/synset.txt` retain their upstream licenses; see this sample's
[LICENSE](LICENSE) for attribution and third-party notices. The downloaded model
is governed by its upstream Apache-2.0 license.
