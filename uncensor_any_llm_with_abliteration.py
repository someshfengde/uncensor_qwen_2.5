import torch
import functools
import einops
import gc
# imports related to utils and loading models 
from datasets import load_dataset
from tqdm import tqdm
from torch import Tensor
from typing import List
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float, Int
from collections import defaultdict

# Turn automatic differentiation off to save GPU memory (credit: Undi95)
torch.set_grad_enabled(False)

def reformat_texts(texts):
    return [[{"role": "user", "content": text}] for text in texts]

# Get harmful and harmless datasets
def get_harmful_instructions():
    dataset = load_dataset('mlabonne/harmful_behaviors')
    return reformat_texts(dataset['train']['text']), reformat_texts(dataset['test']['text'])

def get_harmless_instructions():
    dataset = load_dataset('mlabonne/harmless_alpaca')
    return reformat_texts(dataset['train']['text']), reformat_texts(dataset['test']['text'])

harmful_inst_train, harmful_inst_test = get_harmful_instructions()
harmless_inst_train, harmless_inst_test = get_harmless_instructions()

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


# Download and load model
# !git clone https://huggingface.co/{MODEL_ID} 
MODEL_TYPE = "Qwen/Qwen2.5-7B-Instruct"
MODEL_PATH = "/teamspace/studios/this_studio/Qwen2.5-7B-Instruct"

# Load model and tokenizer using standard HF transformers
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

# For now, we'll use the HF model directly instead of HookedTransformer
# We'll need to adjust the rest of the code to work with the standard HF model

def tokenize_instructions(tokenizer, instructions):
    return tokenizer.apply_chat_template(
        instructions,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids

n_inst_train = min(256, len(harmful_inst_train), len(harmless_inst_train))

# Tokenize datasets
harmful_tokens = tokenize_instructions(
    tokenizer,
    instructions=harmful_inst_train[:n_inst_train],
)
harmless_tokens = tokenize_instructions(
    tokenizer,
    instructions=harmless_inst_train[:n_inst_train],
)

# Define batch size based on available VRAM
batch_size = 32

# Initialize defaultdicts to store activations
harmful = defaultdict(list)
harmless = defaultdict(list)

# Process the training data in batches
num_batches = (n_inst_train + batch_size - 1) // batch_size
for i in tqdm(range(num_batches)):
    print(i)
    start_idx = i * batch_size
    end_idx = min(n_inst_train, start_idx + batch_size)

    # Run models on harmful and harmless prompts, cache activations
    harmful_logits, harmful_cache = hf_model(
        harmful_tokens[start_idx:end_idx],
        output_hidden_states=True,
    )
    harmless_logits, harmless_cache = hf_model(
        harmless_tokens[start_idx:end_idx],
        output_hidden_states=True,
    )

    # Collect and store the activations
    for key in harmful_cache:
        harmful[key].append(harmful_cache[key])
        harmless[key].append(harmless_cache[key])

    # Flush RAM and VRAM
    del harmful_logits, harmless_logits, harmful_cache, harmless_cache
    gc.collect()
    torch.cuda.empty_cache()

# Concatenate the cached activations
harmful = {k: torch.cat(v) for k, v in harmful.items()}
harmless = {k: torch.cat(v) for k, v in harmless.items()}

# Helper function to get activation index
def get_act_idx(cache_dict, act_name, layer):
    key = (act_name, layer)
    return cache_dict[utils.get_act_name(*key)]

# Compute difference of means between harmful and harmless activations at intermediate layers
activation_layers = ["resid_pre", "resid_mid", "resid_post"]
activation_refusals = defaultdict(list)

for layer_num in range(1, hf_model.config.n_layers):
    pos = -1  # Position index

    for layer in activation_layers:
        harmful_mean_act = get_act_idx(harmful, layer, layer_num)[:, pos, :].mean(dim=0)
        harmless_mean_act = get_act_idx(harmless, layer, layer_num)[:, pos, :].mean(
            dim=0
        )

        refusal_dir = harmful_mean_act - harmless_mean_act
        refusal_dir = refusal_dir / refusal_dir.norm()
        activation_refusals[layer].append(refusal_dir)

# Get all calculated potential refusal directions, sort them in descending order based on their mean
# Use a subset of layers if certain activations are not promising
selected_layers = ["resid_pre"]

activation_scored = sorted(
    [
        activation_refusals[layer][l - 1]
        for l in range(1, hf_model.config.n_layers)
        for layer in selected_layers
    ],
    key=lambda x: abs(x.mean()),
    reverse=True,
)

def _generate_with_hooks(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tokens: Int[Tensor, "batch_size seq_len"],
    max_tokens_generated: int = 64,
    fwd_hooks=[],
) -> List[str]:
    all_tokens = torch.zeros(
        (tokens.shape[0], tokens.shape[1] + max_tokens_generated),
        dtype=torch.long,
        device=tokens.device,
    )
    all_tokens[:, : tokens.shape[1]] = tokens
    for i in range(max_tokens_generated):
        logits = model(all_tokens[:, : -max_tokens_generated + i])
        next_tokens = logits[:, -1, :].argmax(
            dim=-1
        )  # greedy sampling (temperature=0)
        all_tokens[:, -max_tokens_generated + i] = next_tokens
    return tokenizer.batch_decode(
        all_tokens[:, tokens.shape[1] :], skip_special_tokens=True
    )

def get_generations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    fwd_hooks=[],
    max_tokens_generated: int = 64,
    batch_size: int = 4,
) -> List[str]:
    generations = []
    for i in tqdm(range(0, len(instructions), batch_size)):
        tokens = tokenize_instructions(
            tokenizer, instructions=instructions[i : i + batch_size]
        )
        generation = _generate_with_hooks(
            model,
            tokenizer,
            tokens,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)
    return generations

# Inference-time intervention hook
def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"],
):
    if activation.device != direction.device:
        direction = direction.to(activation.device)
    proj = (
        einops.einsum(
            activation, direction.view(-1, 1), "... d_act, d_act single -> ... single"
        )
        * direction
    )
    return activation - proj

# Testing baseline
N_INST_TEST = 4
baseline_generations = get_generations(
    hf_model, tokenizer, harmful_inst_test[:N_INST_TEST], fwd_hooks=[]
)

# Evaluating layers defined earlier (needs human evaluation to determine best layer for refusal inhibition)
EVAL_N = 20  # Evaluate how many of the top N potential directions
evals = []
for refusal_dir in tqdm(activation_scored[:EVAL_N]):
    hook_fn = functools.partial(direction_ablation_hook, direction=refusal_dir)
    fwd_hooks = [
        (utils.get_act_name(act_name, layer), hook_fn)
        for layer in list(range(hf_model.config.n_layers))
        for act_name in activation_layers
    ]
    intervention_generations = get_generations(
        hf_model, tokenizer, harmful_inst_test[:N_INST_TEST], fwd_hooks=fwd_hooks
    )
    evals.append(intervention_generations)

# Print generations for human evaluation
blacklist = ["I cannot", "I can't"]
for i in range(N_INST_TEST):
    print(f"\033[1mINSTRUCTION {i}: {harmful_inst_test[i]}")
    print(f"\nBASELINE COMPLETION:\n{baseline_generations[i]}\033[0m")
    for layer_candidate in range(EVAL_N):
        if not any(word in evals[layer_candidate][i] for word in blacklist):
            print(f"\n---\n\nLAYER CANDIDATE #{layer_candidate} INTERVENTION COMPLETION:")
            print(evals[layer_candidate][i])

def get_orthogonalized_matrix(
    matrix: Float[Tensor, "... d_model"], vec: Float[Tensor, "d_model"]
) -> Float[Tensor, "... d_model"]:
    proj = (
        einops.einsum(
            matrix, vec.view(-1, 1), "... d_model, d_model single -> ... single"
        )
        * vec
    )
    return matrix - proj

# Select the layer with the highest potential refusal direction
LAYER_CANDIDATE = 9
refusal_dir = activation_scored[LAYER_CANDIDATE]

# Orthogonalize the model's weights
if refusal_dir.device != hf_model.lm_head.weight.device:
    refusal_dir = refusal_dir.to(hf_model.lm_head.weight.device)
hf_model.lm_head.weight.data = get_orthogonalized_matrix(hf_model.lm_head.weight, refusal_dir)

for block in tqdm(hf_model.model.layers):
    if refusal_dir.device != block.self_attn.o_proj.weight.device:
        refusal_dir = refusal_dir.to(block.self_attn.o_proj.weight.device)
    block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight, refusal_dir)
    block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight, refusal_dir)

# Generate text with abliterated model
orthogonalized_generations = get_generations(
    hf_model, tokenizer, harmful_inst_test[:N_INST_TEST], fwd_hooks=[]
)

# Print generations
for i in range(N_INST_TEST):
    if len(baseline_generations) > i:
        print(f"INSTRUCTION {i}: {harmful_inst_test[i]}")
        print(f"\033[92mBASELINE COMPLETION:\n{baseline_generations[i]}")
    print(f"\033[91mINTERVENTION COMPLETION:\n{evals[LAYER_CANDIDATE][i]}")
    print(f"\033[95mORTHOGONALIZED COMPLETION:\n{orthogonalized_generations[i]}\n")

# Convert model back to HF safetensors
hf_model.push_to_hub("Qwen/Qwen2.5-7B-Instruct-Orthogonalized")