#!/bin/bash
#SBATCH -J adaptvis.debug2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o logs/debug2-%j.out
#SBATCH -e logs/debug2-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs output outputs

CONDA_BASE="/oscar/rt/9.6/25/x86_64_v3/miniforge3-25.3.0-3-a6hhdjzejtacz63sugjqnvgosfqz63ul"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate adaptvis_py311

python - <<'PY'
import json
import torch
from dataset_zoo import get_dataset
from model_zoo import get_model

dataset_name = "Controlled_Images_A"
method = "base"
device = "cuda"

print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())

model, image_preprocess = get_model("llava1.5", device, method)
dataset = get_dataset(dataset_name, image_preprocess=image_preprocess, download=False)

item = dataset[0]
image = item["image_options"][0]

with open("prompts/Controlled_Images_A_with_answer_four_options.jsonl", "r", encoding="utf-8") as f:
    ex = json.loads(next(f))
prompt = ex["question"]
gold = ex["answer"][0]

# This is the only intentional difference from debug_one_sample.sbatch:
# no fixed max_length and no max_length padding.
single_input = model.processor(
    text=prompt,
    images=image,
    return_tensors="pt",
    truncation=True,
).to(device)

input_ids = single_input["input_ids"]
in_len = input_ids.shape[-1]
image_token_count = int((input_ids == 32001).sum().item())

print("Input length:", in_len)
print("Image token count (id=32001):", image_token_count)
print("Gold:", gold)

out = model.model.generate(
    **single_input,
    max_new_tokens=20,
    output_scores=True,
    return_dict_in_generate=True,
)

seq = out["sequences"][0]
new_tokens = seq[in_len:]

print("Output total length:", int(seq.shape[-1]))
print("New token length:", int(new_tokens.shape[-1]))
if new_tokens.shape[-1] > 0:
    print("First new token id:", int(new_tokens[0].item()))

decoded_new = model.processor.decode(new_tokens, skip_special_tokens=True)
decoded_full = model.processor.decode(seq, skip_special_tokens=True)

print("Decoded NEW:", repr(decoded_new))
print("Decoded FULL:", repr(decoded_full))
PY
