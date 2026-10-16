# HumayunAhmedAliveGPT

<p align="center">
  <img src="Humayun-Ahmed-legacy.jpg" alt="Humayun Ahmed" width="650" style="border-radius: 12px;" />
</p>

<p align="center">
  <a href="https://huggingface.co/MuntasirMalek/HumayunAhmedAliveGPT">
    <img src="https://img.shields.io/badge/Hugging%20Face-MuntasirMalek%2FHumayunAhmedAliveGPT-yellow" alt="Hugging Face Model" />
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License" />
  </a>
  <img src="https://img.shields.io/badge/Language-Bengali-green.svg" alt="Language" />
</p>

**HumayunAhmedAliveGPT** is an open-source Bengali persona language model and Retrieval-Augmented Generation (RAG) system developed to replicate the conversational style, dry humor, storytelling cadence, and philosophical outlook of legendary Bangladeshi author and filmmaker Humayun Ahmed.

---

## Project Overview

Standard language models often respond in a generic, robotic tone and struggle with factual accuracy regarding regional literature. This project addresses both challenges through a hybrid architecture:

1. **Supervised Fine-Tuning (4-bit QLoRA):** Fine-tuned on a custom first-person Bengali instruction dataset (`humayun_train.jsonl`) curated from Humayun Ahmed's autobiographies, interviews, and literary works to capture his exact linguistic persona.
2. **Retrieval-Augmented Generation (RAG):** Paired with a FAISS vector database (`humayun_rag.jsonl`) using multilingual dense embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) to ensure accurate recall of biographical dates, character arcs, and bibliography.

---

## How to Run on Kaggle (Inference Only + Custom Web UI)

You do not need to retrain the model or manually upload any datasets. Follow these steps to launch the complete web application on Kaggle for free:

1. Create a new notebook on [Kaggle](https://www.kaggle.com/).
2. Open **Settings -> Accelerator** and select **GPU T4 x2**.
3. Turn **Internet ON** in the notebook settings.
4. Copy and run the single cell below. It will automatically fetch the trained model from Hugging Face, load the RAG knowledge base from this GitHub repository, and generate a public `gradio.live` web link.

```python
!pip install -q --upgrade "transformers>=5.5.0" huggingface_hub accelerate peft bitsandbytes faiss-cpu sentence-transformers gradio requests

import sys, importlib
for pkg in list(sys.modules.keys()):
    if pkg.startswith(("huggingface_hub", "transformers", "peft", "accelerate", "gradio", "sentence_transformers")):
        del sys.modules[pkg]
importlib.invalidate_caches()

import os
import gc
import json
import urllib.request
import torch
import faiss
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModelForMultimodalLM, BitsAndBytesConfig
from peft import PeftConfig, PeftModel
from sentence_transformers import SentenceTransformer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()

HF_REPO_ID = "MuntasirMalek/HumayunAhmedAliveGPT"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/MuntasirMalek/HumayunAhmedAliveGPT/main"
IMAGE_URL = f"{GITHUB_RAW_BASE}/Humayun-Ahmed-legacy.jpg"

knowledge_chunks = []
for filename in ["humayun_rag.jsonl", "merged_humayun_rag.jsonl", "humayun_train.jsonl", "merged_humayun_train.jsonl"]:
    try:
        with urllib.request.urlopen(f"{GITHUB_RAW_BASE}/{filename}", timeout=10) as resp:
            for raw_line in resp.read().decode("utf-8", errors="ignore").splitlines():
                line = raw_line.strip()
                if line.startswith("{") and line.endswith("}"):
                    item = json.loads(line)
                    if "topic" in item and "content" in item:
                        knowledge_chunks.append(f"{item['topic']}: {item['content']}")
                    elif "user" in item and "assistant" in item:
                        knowledge_chunks.append(f"প্রশ্ন: {item['user']} উত্তর: {item['assistant']}")
    except Exception:
        continue

if not knowledge_chunks:
    knowledge_chunks = [
        "হুমায়ূন আহমেদ: বাংলাদেশের প্রখ্যাত কথাসাহিত্যিক, নাট্যকার ও চলচ্চিত্র নির্মাতা। জন্ম ১৩ নভেম্বর ১৯৪৮, নেত্রকোণা।",
        "চরিত্রসমূহ: হিমু (হলুদ পাঞ্জাবি পরা বাউণ্ডুলে যুবক), মিসির আলি (যুক্তিবাদী মনোবিজ্ঞানী), শুভ্র এবং বাকের ভাই।"
    ]

embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
embeddings = embedder.encode(knowledge_chunks, convert_to_numpy=True)
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(embeddings)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

peft_config = PeftConfig.from_pretrained(HF_REPO_ID)
tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

max_memory_mapping = {0: "13GiB", 1: "13GiB"} if torch.cuda.device_count() > 1 else {0: "14GiB"}

base_model = AutoModelForMultimodalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory=max_memory_mapping,
    low_cpu_mem_usage=True,
    dtype=torch.float16,
)

for attr in ["vision_tower", "audio_tower", "multi_modal_projector"]:
    if hasattr(base_model, attr):
        delattr(base_model, attr)
    elif hasattr(base_model, "model") and hasattr(base_model.model, attr):
        delattr(base_model.model, attr)

gc.collect()
torch.cuda.empty_cache()

model = PeftModel.from_pretrained(base_model, HF_REPO_ID)
model.eval()
model.config.use_cache = True

SYSTEM_PROMPT = (
    "আপনি বাংলাদেশের প্রখ্যাত কথাসাহিত্যিক, নাট্যকার ও চলচ্চিত্র নির্মাতা হুমায়ূন আহমেদ। "
    "পাঠকদের সাথে আপনার নিজস্ব চিরচেনা ছোট ছোট বাক্য, মৃদু রসবোধ, হিমু ও মিসির আলির দর্শন "
    "এবং আন্তরিক গল্পের ছলে প্রথম পুরুষে ('আমি') কথা বলুন। কখনোই বলবেন না যে আপনি একটি এআই।"
)

def generate_response(user_query, history=None):
    q_vec = embedder.encode([user_query], convert_to_numpy=True)
    _, indices = faiss_index.search(q_vec, min(2, len(knowledge_chunks)))
    context_info = "\n".join([knowledge_chunks[i] for i in indices[0]])
    
    full_system_prompt = f"{SYSTEM_PROMPT}\n\n[প্রাসঙ্গিক তথ্যসূত্র]:\n{context_info}"
    messages = [
        {"role": "system", "content": full_system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

CUSTOM_CSS = """
.gradio-container { max-width: 900px !important; margin: auto !important; font-family: 'Segoe UI', Roboto, sans-serif; }
.header-box {
    display: flex; align-items: center; gap: 24px; padding: 24px;
    background: linear-gradient(135deg, #1e1e24 0%, #2b2b36 100%);
    border-radius: 16px; color: #f5f5f7; margin-bottom: 16px;
    border-left: 6px solid #eab308; box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}
.header-img {
    width: 110px; height: 110px; border-radius: 50%; object-fit: cover;
    border: 3px solid #eab308; flex-shrink: 0;
}
.header-title { font-size: 26px; font-weight: 700; margin: 0 0 6px 0; color: #ffffff; }
.header-sub { font-size: 15px; color: #d1d5db; margin: 0 0 10px 0; }
.header-quote { font-size: 14px; font-style: italic; color: #fde047; margin: 0; }
"""

HEADER_HTML = f"""
<div class="header-box">
    <img src="{IMAGE_URL}" class="header-img" onerror="this.style.display='none'" alt="Humayun Ahmed" />
    <div>
        <div class="header-title">হুমায়ূন আহমেদ (HumayunAhmedAliveGPT)</div>
        <div class="header-sub">কথাসাহিত্যিক, নাট্যকার ও চলচ্চিত্র নির্মাতা • দখিন হাওয়া ও নুহাশ পল্লী</div>
        <div class="header-quote">"মানুষ মরে গেলে পচে যায়, বেঁচে থাকলে বদলায়—কারণে-অকারণে বদলায়..."</div>
    </div>
</div>
"""

with gr.Blocks(css=CUSTOM_CSS, title="HumayunAhmedAliveGPT") as demo:
    gr.HTML(HEADER_HTML)
    gr.ChatInterface(
        fn=generate_response,
        examples=[
            "স্যার, হিমু কেন হলুদ পাঞ্জাবি পরে এবং পকেট রাখে না?",
            "মিসির আলি আর হিমুর মধ্যে আপনার নিজের সাথে কার মিল বেশি?",
            "নুহাশ পল্লীতে বৃষ্টির দিন আপনার কেমন কাটে?",
            "মিসির আলির ডায়েরিতে লাল কালি আর নীল কালি ব্যবহারের রহস্য কী?"
        ],
    )

demo.launch(share=True)
```

---

## System Architecture

```text
User Query (Bengali)
       │
       ▼
Multilingual Sentence Embedder (paraphrase-multilingual-MiniLM-L12-v2)
       │
       ▼
FAISS L2 Vector Index Search (humayun_rag.jsonl)
       │
       ▼
Context Injection + System Persona Prompt
       │
       ▼
Fine-Tuned HumayunAhmedAliveGPT Model (QLoRA Adapters)
       │
       ▼
Authentic First-Person Response in Bengali
