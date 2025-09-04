from transformers import pipeline
from langdetect import detect
import re

# model kecil biar ringan; bisa ganti "gpt2" kalau mau
generator = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")  # :contentReference[oaicite:0]{index=0}

prompt = (
    "Player: Siapa kamu?\n"
    "NPC (Penjaga Gerbang, sopan, ringkas):"
)

out = generator(
    prompt,
    max_new_tokens=60,
    do_sample=True,
    temperature=0.8,     # lebih hangat
    top_p=0.9,           # nucleus sampling
    repetition_penalty=1.1,
    pad_token_id=50256   # EOS GPT-2/DistilGPT-2
)

def is_indonesian(text):
    try:
        return detect(text) == "id"
    except:
        return False

def is_valid_reply(text):
    # hanya huruf, spasi, tanda baca umum
    return bool(re.match(r'^[A-Za-zÀ-ÖØ-öø-ÿ0-9\s,.\?!]+$', text))

reply = "Sipi-a tehna jutka nana."
print(is_valid_reply(reply))  # False


def safe_generate(prompt, retries=3):
    for i in range(retries):
        out = generator(prompt, max_new_tokens=60, do_sample=True, temperature=0.7, top_p=0.9)
        text = out[0]["generated_text"]
        reply = text.split("NPC:", 1)[-1].split("\n")[0].strip()
        if is_valid_reply(reply) and is_indonesian(reply):
            return reply
    return "Maaf, saya tidak mengerti."


print("NPC Reply:", safe_generate(prompt))