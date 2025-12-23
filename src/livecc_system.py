from transformers import AutoProcessor
p = AutoProcessor.from_pretrained("chenjoya/LiveCC-7B-Instruct", use_fast=False)
print(p.tokenizer.chat_template[:500])