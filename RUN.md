## `inference.py`

To run the the unedited Qwen3-VL model, first run
```
uv sync
```

A new virtual environment should now be created in the root of the project. Activate it

```
source .venv/bin/activate
```

With the venv activated, follow the instructions [https://github.com/mjun0812/flash-attention-prebuild-wheels](here) to find the correct `flash-attn` wheel to download

```
uv pip install <URL>
```

Then, download `vllm` with
```
uv pip install vllm --torch-backend=auto
```

The first time the script is run, HuggingFace will attempt to download and cache the model weights. Make sure it places them in the `.cache` directory

```
export HF_HOME=/workspace/Qwen3-VL/.cache/huggingface \
export HF_HUB_CACHE=/workspace/Qwen3-VL/.cache/huggingface/hub \
export XDG_CACHE_HOME=/workspace/Qwen3-VL/.cache
```



Your virtual environment should now have all the dependencies needed to run the script. Instead of manually activating the virtual environment each time, simply run the script with `uv`

```
uv run python inference.py
```
