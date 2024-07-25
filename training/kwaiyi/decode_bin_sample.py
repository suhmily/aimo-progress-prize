import json
from sys import argv
import numpy as np
                        
tk2 = '/nlp_group/liupeng15/toxiansheng/tokenizer.128k.data_ratio/'
from tokenization_llama_csharp_v2 import LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained(tk2, trust_remote_code=True,  add_bos_token = True, add_eos_token=True)


with open(argv[1], 'rb') as f:
    data = f.read(5000)
    # 将字节序列还原为numpy数组
    arr_restored = list(np.frombuffer(data, dtype=np.uint32))
    arr_restored = [int(one) for one in arr_restored]
    print(tokenizer.decode(arr_restored))

# print(tokenizer.pad_token_id)
# ids = tokenizer("你好")
# print(ids)
# # print(tokenizer.decode([tokenizer.bos_token_id,32, 6767, 49, 44, 10, 127963, 32, 6767, 50, 10, 10, 127962, 20636, 58, 1404, 2328, 135, 5439, 93, 10, 127962, 18096, 381, 58, 10, 127962, 25250, 58, 10, 127963, 32, 6767, 49, 44, 10, 127963, 32, 6767, 50, 10, 10, 127962, 20636, 58, 1404, 2328, 135, 5439, 93, 10, 127962, 18096, 381, 58, 10, 127962, 25250, 58, 10, 127963, 32, 6767, 49, 44, 10, 127963, 32, 6767, 50, 10, 10, 127962, 20636, 58, 1404, 2328, 135, 5439, 93, 10, 127962, 18096, 381, 58, 10, 127962, 25250, 58, 10, 127963, 32, 6767, 49, 44, 10, 127963, 32, 6767, 50, 10, 10, 127962, 20636, 58, 1404, 2328, 135, 5439, 93, 10, 127962, 18096, 381, 58, 10, 127962, 25250, 58, 10, 127963, 32, 6767, 49, 44, 10, 127963, 32, 6767, 50, 10, 10, 127962, 20636, 58, 1404, 2328, 135, 5439, 93, 10, 127962, 18096, 381, 58, 10, 127962, 25250, 58, 10, 127963, 32, 6767, 49, 44, 10, 127963, 32, 6767, 50, 10, 10, 127962, 20636, 58, 1404, 2328, 135, 5439, 93, 10, 127962, 18096, 381, 58, 10, 127962, 25250, 58, 10, 127963, 32, 6767, 49, 44, 10, 127963, 32, 6767, 50, 10, 10, 127962, 20636, 58, 1404, 2328, 135, 5439, 93, 10, 127962, 18096, 381, 58, 10, 127962, 25250, 58, 10, 127963, 32, 6767, 49, 44, 10, 127963, 32, 6767, 50, 10, 10, 127962, 20636, 58, 1404, 2328], skip_special_tokens = True))
# print(tokenizer.decode(ids["input_ids"], skip_special_tokens = True))
# # print(tokenizer.decode([127963], skip_special_tokens = True))
# #print(tokenizer._added_tokens_decoder)
# #print(tokenizer._added_tokens_encoder)
