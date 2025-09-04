from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import time
start_time = time.time()

model_name_or_path = "TheBloke/Mistral-7B-v0.1-AWQ"

# Load model
# model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
#                                           trust_remote_code=False, safetensors=True)
model = AutoAWQForCausalLM.from_quantized(
    model_name_or_path,
    fuse_layers=True,
    trust_remote_code=False,
    safetensors=True,
    device_map="auto",          # biar otomatis taruh ke GPU/CPU
    dtype="auto"          # biar dtype sesuai model
)
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = "What is the 2nd amendment rights in United States about? Summarize it to 2 sentences."
prompt_template=f'''{prompt}

'''
print("\n\n*** Generate:")
elapsed = time.time() - start_time 
print(f"Model loaded in {elapsed:.2f} seconds")
tokens = tokenizer(
    prompt_template,
    return_tensors='pt'
).input_ids.cuda()

# Generate output
generation_output = model.generate(
    tokens,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_new_tokens=120,
    repetition_penalty=1.5
)
print("Output: ", tokenizer.decode(generation_output[0]), "Model Thime: ", time.time() - start_time)

"""
# Inference should be possible with transformers pipeline as well in future
# But currently this is not yet supported by AutoAWQ (correct as of September 25th 2023)
from transformers import pipeline

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

print(pipe(prompt_template)[0]['generated_text'])
"""
