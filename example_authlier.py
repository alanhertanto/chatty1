import torch
import json
import random
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline, 
    AutoModel
)
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# =========================
# MODEL SETUP
# =========================
# Ganti model_name kalau pakai fine-tune checkpoint
model_name = "EleutherAI/gpt-neo-1.3B"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = 0 if torch.cuda.is_available() else -1

gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tok,
    pad_token_id=tok.eos_token_id,
    device=device,
    repetition_penalty=1.5,
)

# =========================
# RETRIEVER SETUP (FAISS)
# =========================
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # kecil, cepat
knowledge_texts = []
with open("npc_guard.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        knowledge_texts.append(json.loads(line)["text"])

# Buat embeddings
kb_embeddings = embedder.encode(knowledge_texts, convert_to_numpy=True)

# Index FAISS
dim = kb_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(kb_embeddings)

def retrieve_fact(player_input: str, top_k: int = 1):
    query_emb = embedder.encode([player_input], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k)
    if len(I[0]) > 0 and D[0][0] < 1.0:  # threshold bisa di-tweak
        return knowledge_texts[I[0][0]]
    return None

# =========================
# GENERATOR
# =========================
def generate_reply(prompt):
    out = gen(prompt,
              max_new_tokens=80,
              do_sample=True,
              temperature=0.7,
              top_p=0.9,
              repetition_penalty=1.5)
    reply_full = out[0]["generated_text"]
    reply = reply_full.split("NPC:", 1)[-1].split("\n")[0].strip()
    return reply

# =========================
# CHAT LOOP
# =========================
history = []
fallbacks = [
    "I guard the village gate and ensure everyone’s safety.",
    "It is my duty to watch over this entrance.",
    "My job is to protect the village from harm."
]

print("NPC Guard Chat. Type 'quit' to exit.\n")

while True:
    player = input("Player: ")
    if player.lower() in ["quit", "exit"]:
        break

    fact = retrieve_fact(player)
    context = "\n".join(history[-4:])  # short-term memory

    if fact:
        prompt = (
            f"NPC is the village guard. Answer politely and consistently.\n"
            f"Relevant fact: {fact}\n"
            f"{context}\nPlayer: {player}\nNPC:"
        )
    else:
        prompt = (
            "NPC is the village guard. Answer politely, concisely, and stay in character.\n"
            f"{context}\nPlayer: {player}\nNPC:"
        )

    answer = generate_reply(prompt)

    # Eval loop sederhana: hindari pengulangan
    last_answer = history[-1].replace("NPC: ", "") if history else ""
    if answer.lower() == last_answer.lower():
        answer = random.choice(fallbacks)

    print(f"NPC: {answer}\n")

    history.append(f"Player: {player}")
    history.append(f"NPC: {answer}")
