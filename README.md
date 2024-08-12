# FLUX
introduced by Black Forest Labs: https://blackforestlabs.ai

adapted for MindSpore by @townwish



This repo contains minimal inference code to run text-to-image and image-to-image with Black Forest Labs' Flux latent rectified flow transformers.


## Local installation

```bash
cd $HOME && git clone https://github.com/black-forest-labs/flux
cd $HOME/flux
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e '.[all]'
```

### Models

We are offering three models:
- `FLUX.1 [pro]` the base model, available via API
- `FLUX.1 [dev]` guidance-distilled variant
- `FLUX.1 [schnell]` guidance and step-distilled variant

| Name   | HuggingFace repo   | License    | md5sum    |
|-------------|-------------|-------------|-------------|
| `FLUX.1 [schnell]` | https://huggingface.co/black-forest-labs/FLUX.1-schnell | [apache-2.0](model_licenses/LICENSE-FLUX1-schnell) | a9e1e277b9b16add186f38e3f5a34044 |
| `FLUX.1 [dev]` | https://huggingface.co/black-forest-labs/FLUX.1-dev| [FLUX.1-dev Non-Commercial License](model_licenses/LICENSE-FLUX1-dev) | a6bd8c16dfc23db6aee2f63a2eba78c0  |
| `FLUX.1 [pro]` | Only available in Black Forest Labs' API. |

The weights of the autoencoder are also released under [apache-2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) and can be found in either of the two HuggingFace repos above. They are the same for both models.


## Usage

The weights will be downloaded automatically from HuggingFace once you start one of the demos. To download `FLUX.1 [dev]`, you will need to be logged in, see [here](https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-login).
If you have downloaded the model weights manually, you can specify the downloaded paths via environment-variables:
```bash
export FLUX_SCHNELL=<path_to_flux_schnell_sft_file>
export FLUX_DEV=<path_to_flux_dev_sft_file>
export AE=<path_to_ae_sft_file>
```

For interactive sampling run
```bash
python -m flux --name <name> --loop
```
Or to generate a single sample run
```bash
python -m flux --name <name> \
  --height <height> --width <width> \
  --prompt "<prompt>"
```

Options:
- `--name`: Choose the model to use (options: "flux-schnell", "flux-dev")


## TODO: Diffusers integration

> [!WARNING]
> UNDO! UNDO! UNDO! TODO! TODO! TODO!

`FLUX.1 [schnell]` and `FLUX.1 [dev]` are integrated with the [🧨 mindone.diffusers](https://github.com/mindspore-lab/mindone) library. To use it with diffusers, install it:

```shell
pip install git+https://github.com/mindspore-lab/mindone.git
```

Then you can use `FluxPipeline` to run the model

```python
import mindspore
from mindone.diffusers import FluxPipeline

model_id = "black-forest-labs/FLUX.1-schnell" #you can also use `black-forest-labs/FLUX.1-dev`

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", mindspore_dtype=mindspore.bfloat16)

prompt = "A cat holding a sign that says hello world"
seed = 42
image = pipe(
    prompt,
    output_type="pil",
    num_inference_steps=4, #use a larger number if you are using [dev]
    generator=torch.Generator("cpu").manual_seed(seed)
)[0][0]
image.save("flux-schnell.png")
```

To learn more check out the [diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux) documentation
