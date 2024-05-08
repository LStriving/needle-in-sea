from transformers import AutoModelForCausalLM
import torch
# Below is slow and hard to control in a cluster
# Unless you insist, **we recommend you download the model to local first**
model = AutoModelForCausalLM.from_pretrained("yaofu/llama-2-7b-80k", 
                                             attn_implementation="flash_attention_2",
                                             cache_dir='.',
                                             torch_dtype=torch.bfloat16
                                             ) 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",cache_dir='.')