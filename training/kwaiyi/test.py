import json
import os
import sys
from tokenization_llama_csharp_v2 import CustomLlamaTokenizer
import pickle
from transformers import AutoTokenizer

tk2 = './tokenizer.128k.data_ratio/'

# 注册你的自定义tokenizer
AutoTokenizer.register(CustomLlamaTokenizer, None, CustomLlamaTokenizer)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(tk2, trust_remote_code=True, add_bos_token=True, add_eos_token=True, use_fast=True)

# 测试序列化和反序列化
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('tokenizer.pkl', 'rb') as f:
    loaded_tokenizer = pickle.load(f)

# 使用加载的tokenizer
tokens = loaded_tokenizer.tokenize("Hello, world!")
print("Tokenized 'Hello, world!':", tokens)

print("Pad token ID:", tokenizer.pad_token_id)

ids = tokenizer("你好")
print("Tokenized '你好':", ids)

# 测试 save_pretrained
save_dir = "./saved_tokenizer"
tokenizer.save_pretrained(save_dir)
print(f"Tokenizer saved to {save_dir}")

# 从保存的目录加载tokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)

# 使用从保存的目录加载的tokenizer进行测试
test_text = "这是一个测试句子"
original_output = tokenizer(test_text)
loaded_output = loaded_tokenizer(test_text)

print("Original tokenizer output:", original_output)
print("Loaded tokenizer output:", loaded_output)

if original_output == loaded_output:
    print("save_pretrained test passed: Original and loaded tokenizer outputs match.")
else:
    print("save_pretrained test failed: Original and loaded tokenizer outputs do not match.")