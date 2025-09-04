from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

messages = [
    {"role": "user", "content": "Menurutmu apa itu warna?"},
    {"role": "assistant", "content": "Warna merupakan salah satu produk dari cahaya yang dipantulkan oleh benda. Warna dapat memengaruhi suasana hati dan persepsi kita terhadap lingkungan sekitar."},
    {"role": "user", "content": "Lalu mengapa warna bisa memengaruhi suasana hati?"},
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])