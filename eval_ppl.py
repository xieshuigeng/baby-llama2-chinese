"""
Sample from the trained model with PyTorch
"""
import os
import json
from contextlib import nullcontext
import torch
from model import ModelArgs, Transformer
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import numpy as np
from datasets import load_dataset
import torch.nn.functional as F
import math

# def compute_bleu(labels, preds, weights=None):
#     from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
#     weights = weights or (0.25, 0.25, 0.25, 0.25)
#     return np.mean([sentence_bleu(references=[label],
#                                   hypothesis=pred,
#                                   smoothing_function=SmoothingFunction().method1,
#                                   weights=weights) for label, pred in zip(labels, preds)])
# -----------------------------------------------------------------------------
out_dir = 'out' # ignored if init_from is not 'resume'
start = "" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 100 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 30 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
dtype = "float32"
compile = False # use PyTorch 2.0 to compile the model to be faster
#exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
# max_seq_len = 512
# dim = 512
# n_layers = 8
# n_heads = 8

max_seq_len = 1024
dim = 1024
n_layers = 12
n_heads = 8
multiple_of = 32
dropout = 0.0 
model_args = dict(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        vocab_size=64793,#64793,
        multiple_of=multiple_of,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )  # s
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast()

# init from a model saved in a specific directory
ckpt_path = 'out/pretrain/epoch_0.pth'
state_dict = torch.load(ckpt_path, map_location=device)
gptconf = ModelArgs(**model_args)
model = Transformer(gptconf)
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(device)
if compile:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# load the tokenizer
tokenizer=ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')

## export HF_DATASETS_OFFLINE=1
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
# encodings = {k: v.to(device) for k, v in encodings.items()}
max_length = 1024
stride = 256
total_nll = 0.0
total_tokens = 0

with torch.no_grad():
    for text in dataset["text"]:
        if not text.strip():  # 跳过空行
            continue
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings.get("attention_mask", torch.ones_like(input_ids)).to(device)
        target_ids = input_ids.clone()
        target_ids[attention_mask == 0] = -100

        outputs = model(input_ids, targets=target_ids)
        n_pred_tokens = int((target_ids != -100).sum().item())

        shift_logits = outputs[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()

        # 忽略 padding 或指定 index
        loss_manual = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,  # HF 默认 ignore_index=-100
            reduction="mean"
        )

        neg_log_likelihood = loss_manual * n_pred_tokens
        # print(f"output.loss: {model.last_loss: .4f}")
        total_nll += neg_log_likelihood
        total_tokens += n_pred_tokens

        # print(f"Model loss:     {model.last_loss:.6f}")
        # print(f"Manual loss: {loss_manual:.6f}")

        # print(f"PPL: {ppl.item():.2f}")
        # print(f"output size: {outputs.size(1)}")
        # print(f"target_ids size: {target_ids.size(1)}")
        # print(f"target_ids sum: {n_pred_tokens}")

ppl = torch.exp(total_nll / (total_tokens))
print(f"PPL: {ppl.item():.2f}")