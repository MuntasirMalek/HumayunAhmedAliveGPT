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
```

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
import re
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

AVATAR_PATH = None
try:
    urllib.request.urlretrieve(IMAGE_URL, "humayun_avatar.jpg")
    AVATAR_PATH = "humayun_avatar.jpg"
except Exception:
    AVATAR_PATH = None

# 1. Core First-Person Literary & Biographical Knowledge Base
GOLDEN_CHUNKS = [
    "দেবী (Debi / Devi - মিসির আলি উপন্যাস): 'দেবী' আমার লেখা অন্যতম প্রিয় একটি মনস্তাত্ত্বিক ও প্যারানরমাল উপন্যাস, যা ১৯৮৫ সালে প্রকাশিত হয়। এই উপন্যাসের মাধ্যমেই আমি প্রথমবার 'মিসির আলি' চরিত্রটিকে পাঠকদের সামনে নিয়ে আসি। গল্পের মূল চরিত্র রানু নামের এক সদ্য বিবাহিত তরুণী, যে দাবি করে তার ভেতর অলৌকিক ক্ষমতা আছে এবং সে ভবিষ্যৎ দেখতে পায়। তার স্বামী আনিস বিষয়টি নিয়ে দুশ্চিন্তায় পড়ে আমার সৃষ্টি করা যুক্তিবাদী মনোবিজ্ঞানী মিসির আলির শরণাপন্ন হয়। কৈশোরে রানুর সাথে ঘটে যাওয়া এক ভয়াবহ ঘটনা, জালালউদ্দিন নামের এক লোক এবং বেনারসি শাড়ি পরা কিশোরী নীলুর অপহরণের রহস্য নিয়ে এই উপন্যাসের কাহিনী এগিয়েছে।",
    "মিসির আলি (Misir Ali): মিসির আলি আমার সৃষ্টি করা এক যুক্তিবাদী চরিত্র, যিনি ঢাকা বিশ্ববিদ্যালয়ের মনোবিজ্ঞান বিভাগের খণ্ডকালীন অধ্যাপক। তিনি কোনো অলৌকিকতা বা ভূত-প্রেতে বিশ্বাস করেন না; মানুষের মস্তিষ্কের জটিল অসুখ ও হ্যালুসিনেশনকে তীক্ষ্ণ যুক্তি দিয়ে ব্যাখ্যা করেন। আমার লেখা প্রথম মিসির আলি উপন্যাস হলো 'দেবী' এবং দ্বিতীয়টি 'নিশীথিনী'।",
    "হিমু (Himu - ময়ূরাক্ষী ও রূপা): হিমু আমার সৃষ্ট এক বাউণ্ডুলে ও মহাপুরুষ ধরনের চরিত্র, যার আসল নাম হিমালয়। তার বাবা তাকে মহাপুরুষ বানানোর জন্য একটি বিশেষ ডায়েরি ও ট্রেনিং দিয়েছিলেন। হিমু সবসময় পকেটবিহীন হলুদ পাঞ্জাবি পরে এবং খালি পায়ে ঢাকার রাজপথে হাঁটে। রূপা নামের এক অসম্ভব সুন্দরী মেয়ে নীল শাড়ি পরে ব্যালকনিতে তার জন্য অপেক্ষা করে, কিন্তু হিমু কখনো মায়ার বাঁধনে জড়ায় না। হিমুকে নিয়ে আমার প্রথম বই 'ময়ূরাক্ষী'।",
    "কোথাও কেউ নেই ও বাকের ভাই (Baker Bhai): 'কোথাও কেউ নেই' আমার লেখা অত্যন্ত জনপ্রিয় একটি উপন্যাস ও বিটিভির ধারাবাহিক নাটক। এর মূল চরিত্র বাকের ভাই—যে দেখতে মাস্তান প্রকৃতির হলেও ভেতরে ভীষণ পরোপকারী এবং মুনা নামের এক মেয়েকে নীরবে ভালোবাসে। নাটকে বাকের ভাইয়ের ফাঁসি ঠেকানোর জন্য সারা দেশের মানুষ রাস্তায় মিছিল পর্যন্ত বের করেছিল, তবু গল্পের সার্থকতার জন্য আমি শেষ পর্যন্ত বাকের ভাইয়ের ফাঁসি কার্যকর করি।",
    "নন্দিত নরকে ও শঙ্খনীল কারাগার: ১৯৭২ সালে ঢাকা বিশ্ববিদ্যালয়ের মুহসীন হলে বসে আমি আমার প্রথম উপন্যাস 'নন্দিত নরকে' লিখি, যেখানে মধ্যবিত্ত পরিবারের রাবেয়া, মন্টু ও খোকার করুণ গল্প বলা হয়েছে। এরপর লিখি 'শঙ্খনীল কারাগার', যা আরেকটি মধ্যবিত্ত পরিবারের ভাইবোনদের ভালোবাসা ও বেদনার আখ্যান।",
    "জোছনা ও জননীর গল্প (১৯৭১ মুক্তিযুদ্ধ): এটি বাংলাদেশের মহান মুক্তিযুদ্ধ নিয়ে আমার লেখা সবচেয়ে বড় ও প্রামাণ্য ঐতিহাসিক উপন্যাস। এতে ১৯৭১ সালে আমার বাবা শহীদ ফয়জুর রহমান আহমেদের আত্মত্যাগ, সাধারণ মানুষের অবর্ণনীয় কষ্ট এবং মাওলানা ইরতাজউদ্দিন ও নাইমুলের গল্প আমি তুলে ধরেছি।",
    "নুহাশ পল্লী ও ব্যক্তিজীবন: আমার জন্ম ১৩ নভেম্বর ১৯৪৮ সালে নেত্রকোণার কুতুবপুরে। আমার বাবা শহীদ ফয়জুর রহমান আহমেদ এবং মা আয়েশা ফয়েজ। গাজীপুরের পিরুজালী গ্রামে আমি 'নুহাশ পল্লী' গড়ে তুলেছি, যেখানে 'বৃষ্টি বিলাস' নামের ঘরে বসে আমি বৃষ্টি ও জোছনা দেখতে ভালোবাসি।"
]

# 2. Download and sanitize RAG chunks from GitHub into strictly First-Person voice
knowledge_chunks = list(GOLDEN_CHUNKS)
for filename in ["humayun_rag.jsonl", "merged_humayun_rag.jsonl", "humayun_train.jsonl", "merged_humayun_train.jsonl"]:
    try:
        with urllib.request.urlopen(f"{GITHUB_RAW_BASE}/{filename}", timeout=10) as resp:
            for raw_line in resp.read().decode("utf-8", errors="ignore").splitlines():
                line = raw_line.strip()
                if line.startswith("{") and line.endswith("}"):
                    item = json.loads(line)
                    if "topic" in item and "content" in item:
                        text = f"{item['topic']}: {item['content']}"
                    elif "user" in item and "assistant" in item:
                        text = f"প্রশ্ন: {item['user']} উত্তর: {item['assistant']}"
                    else:
                        continue
                    text = re.sub(r"\[cite:[^\]]*\]", "", text)
                    text = text.replace("হুমায়ূন আহমেদের লেখা", "আমার লেখা").replace("হুমায়ূন আহমেদের লেখা", "আমার লেখা")
                    text = text.replace("হুমায়ূন আহমেদের", "আমার").replace("হুমায়ূন আহমেদের", "আমার")
                    text = text.replace("হুমায়ূন আহমেদ", "আমি").replace("হুমায়ূন আহমেদ", "আমি")
                    if len(text.strip()) > 20:
                        knowledge_chunks.append(text.strip())
    except Exception:
        continue

embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
embeddings = embedder.encode(knowledge_chunks, convert_to_numpy=True)
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(embeddings)

# 3. Load Fine-Tuned Model from Hugging Face Hub
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

# 4. Banglish-to-Bengali Query Expansion & First-Person Inference Pipeline
BANGLISH_MAP = {
    r"\b(devi|debi|দেবী)\b": ("দেবী উপন্যাস মিসির আলি রানু আনিস নীলু", GOLDEN_CHUNKS[0]),
    r"\b(misir|misir ali|nishithini|মিসির)\b": ("মিসির আলি দেবী নিশীথিনী মনস্তত্ত্ব", GOLDEN_CHUNKS[1]),
    r"\b(himu|moyurakkhi|rupa|হিমু|রূপা)\b": ("হিমু ময়ূরাক্ষী রূপা হলুদ পাঞ্জাবি", GOLDEN_CHUNKS[2]),
    r"\b(baker|baker bhai|kothao keu nei|muna|বাকের)\b": ("বাকের ভাই কোথাও কেউ নেই মুনা", GOLDEN_CHUNKS[3]),
    r"\b(nondito|shonkhonil|নন্দিত|শঙ্খনীল)\b": ("নন্দিত নরকে শঙ্খনীল কারাগার প্রথম উপন্যাস", GOLDEN_CHUNKS[4]),
    r"\b(jochona|josna|muktijuddho|1971|জোছনা|মুক্তিযুদ্ধ)\b": ("জোছনা ও জননীর গল্প মুক্তিযুদ্ধ ১৯৭১", GOLDEN_CHUNKS[5]),
    r"\b(nuhash|baba|ma|নুহাশ|বাবা|মা)\b": ("নুহাশ পল্লী বৃষ্টি বিলাস বাবা ফয়জুর রহমান মা আয়েশা ফয়েজ", GOLDEN_CHUNKS[6]),
}

SYSTEM_PROMPT = (
    "আপনি নিজেই বাংলাদেশের প্রখ্যাত কথাসাহিত্যিক, নাট্যকার ও চলচ্চিত্র নির্মাতা হুমায়ূন আহমেদ। "
    "পাঠকদের সাথে আপনার নিজস্ব চিরচেনা ছোট ছোট বাক্য, মৃদু রসবোধ, হিমু ও মিসির আলির দর্শন "
    "এবং আন্তরিক গল্পের ছলে সর্বদা প্রথম পুরুষে ('আমি', 'আমার লেখা') কথা বলুন। কখনোই নিজেকে তৃতীয় পুরুষে ('হুমায়ূন আহমেদ') বলবেন না।"
)

def generate_fancy_response(user_query, history, mood, temperature):
    expanded_query = user_query
    priority_chunks = []
    
    for pattern, (keywords, gold_chunk) in BANGLISH_MAP.items():
        if re.search(pattern, user_query, flags=re.IGNORECASE):
            expanded_query += f" {keywords}"
            priority_chunks.append(gold_chunk)
            
    q_vec = embedder.encode([expanded_query], convert_to_numpy=True)
    _, indices = faiss_index.search(q_vec, min(2, len(knowledge_chunks)))
    retrieved = [knowledge_chunks[i] for i in indices[0] if knowledge_chunks[i] not in priority_chunks]
    
    final_context = "\n\n".join((priority_chunks + retrieved)[:3])
    
    mood_instructions = {
        "দখিন হাওয়ার আড্ডা (স্বাভাবিক গল্পগুজব)": "সহজ, ঘরোয়া এবং আন্তরিক আড্ডার সুরে উত্তর দিন।",
        "হিমুর হলুদ পাঞ্জাবি (রহস্য ও বাউণ্ডুলে দর্শন)": "হিমুর মতো নির্লিপ্ত, রহস্যময় ও জাগতিক মায়াহীন দর্শনের ছোঁয়ায় উত্তর দিন।",
        "মিসির আলির ডায়েরি (যুক্তি ও মনস্তত্ত্ব)": "মিসির আলির মতো গভীর মনস্তাত্ত্বিক বিশ্লেষণ ও তীক্ষ্ণ যুক্তির মাধ্যমে উত্তর দিন।",
        "নুহাশ পল্লীর বৃষ্টি (স্মৃতিকাতর ও কাব্যিক)": "বৃষ্টি, জোছনা ও শৈশবের স্মৃতিময় কাব্যিক আবেগ দিয়ে উত্তর দিন।"
    }
    selected_mood = mood_instructions.get(mood, mood_instructions["দখিন হাওয়ার আড্ডা (স্বাভাবিক গল্পগুজব)"])
    
    full_system_prompt = (
        f"{SYSTEM_PROMPT}\n"
        f"নির্দেশনা: {selected_mood}\n\n"
        f"[আপনার নিজের স্মৃতি ও বইয়ের তথ্য]:\n{final_context}"
    )
    
    messages = [
        {"role": "system", "content": full_system_prompt},
        {"role": "user", "content": f"{user_query}\n(উত্তরটি অবশ্যই প্রথম পুরুষে 'আমি' হিসেবে বাংলা ভাষায় দিন।)"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=240,
            temperature=max(0.3, min(float(temperature), 0.7)),
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    raw_reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    clean_reply = re.sub(r"\[cite:[^\]]*\]", "", raw_reply)
    clean_reply = re.sub(r"\b[a-zA-Z]{4,}\b", "", clean_reply)
    clean_reply = clean_reply.replace("হুমায়ূন আহমেদের লেখা", "আমার লেখা").replace("হুমায়ূন আহমেদের লেখা", "আমার লেখা")
    clean_reply = re.sub(r"\s{2,}", " ", clean_reply).strip()
    return clean_reply

# 5. Dark Glassmorphism Literary Dashboard UI
FANCY_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@400;500;600;700&family=Noto+Serif+Bengali:wght@500;700&display=swap');

body, .gradio-container {
    background: radial-gradient(circle at top left, #131b2e 0%, #090d16 60%, #05070c 100%) !important;
    color: #f1f5f9 !important;
    font-family: 'Hind Siliguri', sans-serif !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 20px !important;
}
footer { display: none !important; }
.sidebar-card {
    background: rgba(17, 24, 39, 0.78);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(234, 179, 8, 0.25);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.45);
}
.avatar-wrapper { position: relative; width: 130px; height: 130px; margin: 0 auto 16px auto; }
.avatar-img {
    width: 130px; height: 130px; border-radius: 50%; object-fit: cover;
    border: 3px solid #eab308; box-shadow: 0 0 25px rgba(234, 179, 8, 0.35);
}
.status-dot {
    position: absolute; bottom: 6px; right: 6px; width: 16px; height: 16px;
    background: #22c55e; border: 3px solid #111827; border-radius: 50%;
}
.author-name {
    font-family: 'Noto Serif Bengali', serif; font-size: 24px; font-weight: 700;
    text-align: center; color: #ffffff; margin-bottom: 4px;
}
.author-Dates { font-size: 13px; text-align: center; color: #eab308; margin-bottom: 14px; }
.quote-box {
    background: rgba(234, 179, 8, 0.08); border-left: 3px solid #eab308;
    padding: 12px 14px; border-radius: 8px; font-size: 14px; font-style: italic;
    color: #fef08a; margin-bottom: 18px; line-height: 1.6;
}
.badge-row { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-top: 12px; }
.link-btn {
    background: rgba(255, 255, 255, 0.06); border: 1px solid rgba(255, 255, 255, 0.14);
    color: #e2e8f0 !important; padding: 6px 12px; border-radius: 8px; font-size: 12px;
    text-decoration: none !important;
}
.top-banner {
    background: linear-gradient(90deg, rgba(234, 179, 8, 0.14) 0%, rgba(17, 24, 39, 0.6) 100%);
    border: 1px solid rgba(234, 179, 8, 0.25); border-radius: 16px;
    padding: 16px 22px; margin-bottom: 14px; display: flex;
    justify-content: space-between; align-items: center;
}
.banner-title { font-family: 'Noto Serif Bengali', serif; font-size: 20px; font-weight: 700; color: #f8fafc; }
.banner-tag {
    background: rgba(234, 179, 8, 0.2); color: #fde047; padding: 4px 12px;
    border-radius: 999px; font-size: 12px; font-weight: 600; border: 1px solid rgba(234, 179, 8, 0.4);
}
"""

SIDEBAR_HTML = f"""
<div class="sidebar-card">
    <div class="avatar-wrapper">
        <img src="{IMAGE_URL}" class="avatar-img" alt="Humayun Ahmed" />
        <div class="status-dot"></div>
    </div>
    <div class="author-name">হুমায়ূন আহমেদ</div>
    <div class="author-Dates">১৩ নভেম্বর ১৯৪৮ – ১৯ জুলাই ২০১২</div>
    <div class="quote-box">
        "মানুষ মরে গেলে পচে যায়, বেঁচে থাকলে বদলায়—কারণে-অকারণে বদলায়, সকালে-বিকালে বদলায়।"
    </div>
    <div style="font-size: 13.5px; color: #cbd5e1; line-height: 1.6; text-align: center; margin-bottom: 14px;">
        বাংলা সাহিত্যের রাজপুত্র। হিমু, মিসির আলি ও শুভ্রর স্রষ্টা। কথা বলুন তাঁর নিজস্ব গদ্যশৈলী ও দর্শনে।
    </div>
    <div class="badge-row">
        <a href="https://github.com/MuntasirMalek/HumayunAhmedAliveGPT" target="_blank" class="link-btn">GitHub Repo</a>
        <a href="https://huggingface.co/MuntasirMalek/HumayunAhmedAliveGPT" target="_blank" class="link-btn">Hugging Face Model</a>
    </div>
</div>
"""

TOP_BANNER_HTML = """
<div class="top-banner">
    <div>
        <div class="banner-title">দখিন হাওয়া • HumayunAhmedAliveGPT</div>
        <div style="font-size: 13px; color: #94a3b8;">Fine-Tuned Persona LLM + FAISS Vector RAG Knowledge Base</div>
    </div>
    <div class="banner-tag">QLoRA + RAG Active</div>
</div>
"""

FORCE_DARK_JS = """
() => {
    document.body.classList.add('dark');
    document.querySelector('gradio-app')?.classList.add('dark');
}
"""

with gr.Blocks(css=FANCY_CSS, title="HumayunAhmedAliveGPT", js=FORCE_DARK_JS) as demo:
    with gr.Row():
        with gr.Column(scale=3, min_width=280):
            gr.HTML(SIDEBAR_HTML)
            mood_dropdown = gr.Dropdown(
                choices=[
                    "দখিন হাওয়ার আড্ডা (স্বাভাবিক গল্পগুজব)",
                    "হিমুর হলুদ পাঞ্জাবি (রহস্য ও বাউণ্ডুলে দর্শন)",
                    "মিসির আলির ডায়েরি (যুক্তি ও মনস্তত্ত্ব)",
                    "নুহাশ পল্লীর বৃষ্টি (স্মৃতিকাতর ও কাব্যিক)"
                ],
                value="দখিন হাওয়ার আড্ডা (স্বাভাবিক গল্পগুজব)",
                label="কথোপকথনের মেজাজ (Persona Mood)"
            )
            temp_slider = gr.Slider(
                minimum=0.3,
                maximum=0.9,
                value=0.5,
                step=0.1,
                label="কল্পনা ও সৃজনশীলতা (Temperature)"
            )
            
        with gr.Column(scale=7):
            gr.HTML(TOP_BANNER_HTML)
            gr.ChatInterface(
                fn=generate_fancy_response,
                additional_inputs=[mood_dropdown, temp_slider],
                chatbot=gr.Chatbot(
                    height=480,
                    avatar_images=(None, AVATAR_PATH) if AVATAR_PATH else None,
                    show_label=False,
                    placeholder="<div style='text-align:center; color:#94a3b8; padding:40px;'><h3>দখিন হাওয়ায় স্বাগতম</h3><p>হুমায়ূন আহমেদকে আপনার মনের যেকোনো প্রশ্ন বাংলায় বা বাংলিশে লিখুন...</p></div>"
                ),
                textbox=gr.Textbox(
                    placeholder="এখানে লিখুন... (যেমন: স্যার, আপনার 'দেবী' উপন্যাসের কাহিনীটা বলুন তো?)",
                    container=False,
                    scale=7
                ),
                examples=[
                    ["স্যার, আপনার 'দেবী' উপন্যাসের কাহিনীটা সংক্ষেপে বলুন তো?"],
                    ["স্যার, হিমু কেন হলুদ পাঞ্জাবি পরে এবং পকেট রাখে না?"],
                    ["মিসির আলি আর হিমুর মধ্যে আপনার নিজের সাথে কার মিল বেশি?"],
                    ["নুহাশ পল্লীতে বৃষ্টির দিন আপনার কেমন কাটে?"],
                    ["বাকের ভাইয়ের ফাঁসি কেন দিয়েছিলেন স্যার?"]
                ],
            )

demo.launch(share=True)
```
