import transformers

AUTO_TOKENIZER_CLASS = transformers.LlamaTokenizer
tokenizer = AUTO_TOKENIZER_CLASS.from_pretrained('/nlp_group/liupeng15/toxiansheng/tokenizer.128k.data_ratio', trust_remote_code=True, add_bos_token = True, add_eos_token = True)
print(tokenizer.bos_token_id)
tokenized = tokenizer("<s>",
    add_bos_token = True
)
print(tokenized)
