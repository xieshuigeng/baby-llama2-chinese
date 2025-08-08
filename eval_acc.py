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
dataset = load_dataset("ai2_arc", "ARC-Easy", split="test")

def label_to_index(label):
    label = label.strip()
    if label.isdigit():  # 数字字符串
        return int(label) - 1  # 假设数字是从1开始的
    elif len(label) == 1 and label.isalpha():
        return ord(label.upper()) - ord('A')
    else:
        raise ValueError(f"未知的答案标签格式: {label}")
    
correct = 0
total = 0

for ex in dataset:
    prompts = [ex["question"] + "\nAnswer: " + choice for choice in ex["choices"]["text"]]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model(inputs.input_ids, targets=inputs.input_ids)            
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = inputs.input_ids[..., 1:].contiguous()

        # mask 掉 -100 的位置
        loss_mask = (shift_labels != -100)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        token_losses = token_losses.view(shift_labels.size())

        # 只对有效 token 求和
        seq_losses = (token_losses * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)

    seq_losses = seq_losses.cpu().tolist()
    # print (f"Loss: {seq_losses}")
    pred = seq_losses.index(min(seq_losses))

    try:
        correct_idx = label_to_index(ex["answerKey"])
    except ValueError:
        print(f"跳过样本，无法识别答案标签: {ex['answerKey']}")
        continue

    if correct_idx < 0 or correct_idx >= len(ex["choices"]["text"]):
        print(f"跳过样本，答案索引越界: {correct_idx}")
        continue

    # print(f"pred: {pred} == correct_idx: {correct_idx}")
    total += 1
    if pred == correct_idx:
        correct += 1

acc = correct / total
print(f"ARC-Easy Accuracy: {acc*100:.2f}%")