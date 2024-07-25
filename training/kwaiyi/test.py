import json
import os
import sys

from tokenization_llama_csharp_v2 import CustomLlamaTokenizer  # 确保从正确的模块导入

                        
tk2 = './tokenizer.128k.data_ratio/'
# from tokenization_llama_csharp_v2 import LlamaTokenizer
# tokenizer = LlamaTokenizer.from_pretrained(tk2, trust_remote_code=True,  add_bos_token = True, add_eos_token=True)


# 注册你的自定义tokenizer
from transformers import AutoTokenizer
AutoTokenizer.register(CustomLlamaTokenizer, None, CustomLlamaTokenizer)
tokenizer = AutoTokenizer.from_pretrained(tk2, trust_remote_code=True,  add_bos_token = True, add_eos_token=True, use_fast=True)

import pickle

# 序列化
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# 反序列化
with open('tokenizer.pkl', 'rb') as f:
    loaded_tokenizer = pickle.load(f)

# 使用加载的tokenizer
tokens = loaded_tokenizer.tokenize("Hello, world!")

print(tokenizer.pad_token_id)
ids = tokenizer("你好")
print(ids)
# print(tokenizer.decode([tokenizer.bos_token_id,32, 6767, 49, 44, 10, 127963, 32, 6767, 50, 10, 10, 127962, 20636, 58, 1404, 2328, 135, 5439, 93, 10, 127962, 18096, 381, 58, 10, 127962, 25250, 58, 10, 127963, 32, 6767, 49, 44, 10, 127963, 32, 6767, 50, 10, 10, 127962, 20636, 58, 1404, 2328, 135, 5439, 93, 10, 127962, 18096, 381, 58, 10, 127962, 25250, 58, 10, 127963, 32, 6767, 49, 44, 10, 127963, 32, 6767, 50, 10, 10, 127962, 20636, 58, 1404, 2328, 135, 5439, 93, 10, 127962, 18096, 381, 58, 10, 127962, 25250, 58, 10, 127963, 32, 6767, 49, 44, 10, 127963, 32, 6767, 50, 10, 10, 127962, 20636, 58, 1404, 2328, 135, 5439, 93, 10, 127962, 18096, 381, 58, 10, 127962, 25250, 58, 10, 127963, 32, 6767, 49, 44, 10, 127963, 32, 6767, 50, 10, 10, 127962, 20636, 58, 1404, 2328, 135, 5439, 93, 10, 127962, 18096, 381, 58, 10, 127962, 25250, 58, 10, 127963, 32, 6767, 49, 44, 10, 127963, 32, 6767, 50, 10, 10, 127962, 20636, 58, 1404, 2328, 135, 5439, 93, 10, 127962, 18096, 381, 58, 10, 127962, 25250, 58, 10, 127963, 32, 6767, 49, 44, 10, 127963, 32, 6767, 50, 10, 10, 127962, 20636, 58, 1404, 2328, 135, 5439, 93, 10, 127962, 18096, 381, 58, 10, 127962, 25250, 58, 10, 127963, 32, 6767, 49, 44, 10, 127963, 32, 6767, 50, 10, 10, 127962, 20636, 58, 1404, 2328], skip_special_tokens = True))
print(tokenizer.decode(ids["input_ids"], skip_special_tokens = True))
# print(tokenizer.decode([127963], skip_special_tokens = True))
#print(tokenizer._added_tokens_decoder)
#print(tokenizer._added_tokens_encoder)

