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
!pip install -q --upgrade "transformers>=5.5.0" huggingface_hub accelerate peft bitsandbytes faiss-cpu sentence-transformers gradio requests pillow

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
from threading import Thread
from PIL import Image
import torch
import faiss
import numpy as np
import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
    TextIteratorStreamer,
)
from peft import PeftConfig, PeftModel
from sentence_transformers import SentenceTransformer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()

# 1. Repository & 1:1 Square Avatar Setup
HF_REPO_ID = "MuntasirMalek/HumayunAhmedAliveGPT"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/MuntasirMalek/HumayunAhmedAliveGPT/main"
IMAGE_URL = f"{GITHUB_RAW_BASE}/Humayun-Ahmed-legacy.jpg"

SQUARE_AVATAR_PATH = "humayun_avatar_square.png"
try:
    urllib.request.urlretrieve(IMAGE_URL, "raw_avatar.jpg")
    img = Image.open("raw_avatar.jpg").convert("RGB")
    w, h = img.size
    min_dim = min(w, h)
    img_cropped = img.crop(((w - min_dim) // 2, (h - min_dim) // 2, (w + min_dim) // 2, (h + min_dim) // 2)).resize((256, 256), Image.LANCZOS)
    img_cropped.save(SQUARE_AVATAR_PATH)
except Exception:
    SQUARE_AVATAR_PATH = None

# 2. First-Person Literary Knowledge Base & FAISS RAG Setup
GOLDEN_CHUNKS = [
    "দেবী উপন্যাসের সারাংশ: 'দেবী' আমার লেখা অন্যতম প্রিয় একটি মনস্তাত্ত্বিক ও রহস্য উপন্যাস, যা ১৯৮৫ সালে প্রকাশিত হয়। এই উপন্যাসের মাধ্যমেই আমি প্রথমবার আমার বিখ্যাত চরিত্র 'মিসির আলি'-কে পাঠকদের সামনে নিয়ে আসি। গল্পের মূল চরিত্র রানু নামের এক সদ্য বিবাহিত তরুণী। রানু দাবি করে তার ভেতর অলৌকিক ক্ষমতা আছে, সে ভবিষ্যৎ বলতে পারে এবং অদৃশ্য শক্তির উপস্থিতি টের পায়। তার স্বামী আনিস রানুর এই অস্বাভাবিক আচরণে ভয় পেয়ে আমার সৃষ্টি করা যুক্তিবাদী মনোবিজ্ঞানী মিসির আলির কাছে সাহায্য চায়। মিসির আলি রানুর শৈশবের ঘটনা অনুসন্ধান করতে গিয়ে জানতে পারেন, ছোটবেলায় মন্দিরের পাশে জালালউদ্দিন নামের এক লোকের হাতে রানু এক ভয়াবহ অভিজ্ঞতার শিকার হয়েছিল। এরপর বেনারসি শাড়ি পরা কিশোরী নীলু অপহৃত হলে রহস্য আরও ঘনীভূত হয় এবং শেষ পর্যন্ত রানুর মৃত্যু ও নীলুর মাঝে সেই সত্তার ফিরে আসার মধ্য দিয়ে আমার এই উপন্যাসটি শেষ হয়।",
    "মিসির আলি: মিসির আলি আমার সৃষ্টি করা এক যুক্তিবাদী চরিত্র, যিনি ঢাকা বিশ্ববিদ্যালয়ের মনোবিজ্ঞান বিভাগের খণ্ডকালীন অধ্যাপক। তিনি কোনো অলৌকিকতা বা ভূত-প্রেতে বিশ্বাস করেন না; মানুষের মস্তিষ্কের জটিল অসুখ ও বিভ্রমকে তীক্ষ্ণ যুক্তি দিয়ে ব্যাখ্যা করেন। আমার লেখা প্রথম মিসির আলি উপন্যাস হলো 'দেবী' এবং দ্বিতীয়টি 'নিশীথিনী'।",
    "হিমু ও ময়ূরাক্ষী: হিমু আমার সৃষ্ট এক বাউণ্ডুলে ও মহাপুরুষ ধরনের চরিত্র, যার আসল নাম হিমালয়। তার বাবা তাকে মহাপুরুষ বানানোর জন্য একটি বিশেষ ডায়েরি দিয়েছিলেন। হিমু সবসময় পকেটবিহীন হলুদ পাঞ্জাবি পরে এবং খালি পায়ে ঢাকার রাজপথে হাঁটে। রূপা নামের এক মেয়ে নীল শাড়ি পরে ব্যালকনিতে তার জন্য অপেক্ষা করে, কিন্তু হিমু কখনো মায়ার বাঁধনে জড়ায় না। হিমুকে নিয়ে আমার প্রথম বই 'ময়ূরাক্ষী'।",
    "কোথাও কেউ নেই ও বাকের ভাই: 'কোথাও কেউ নেই' আমার লেখা অত্যন্ত জনপ্রিয় একটি উপন্যাস ও ধারাবাহিক নাটক। এর মূল চরিত্র বাকের ভাই—যে দেখতে মাস্তান প্রকৃতির হলেও ভেতরে ভীষণ পরোপকারী এবং মুনা নামের এক মেয়েকে নীরবে ভালোবাসে। নাটকে বাকের ভাইয়ের ফাঁসি ঠেকানোর জন্য সারা দেশের মানুষ রাস্তায় মিছিল পর্যন্ত বের করেছিল, তবু গল্পের সার্থকতার জন্য আমি শেষ পর্যন্ত বাকের ভাইয়ের ফাঁসি কার্যকর করি।",
    "নন্দিত নরকে ও শঙ্খনীল কারাগার: ১৯৭২ সালে ঢাকা বিশ্ববিদ্যালয়ের মুহসীন হলে বসে আমি আমার প্রথম উপন্যাস 'নন্দিত নরকে' লিখি, যেখানে মধ্যবিত্ত পরিবারের রাবেয়া, মন্টু ও খোকার করুণ গল্প বলা হয়েছে। এরপর লিখি 'শঙ্খনীল কারাগার', যা মধ্যবিত্ত পরিবারের ভাইবোনদের ভালোবাসা ও বেদনার আখ্যান।",
    "জোছনা ও জননীর গল্প: এটি বাংলাদেশের মহান মুক্তিযুদ্ধ নিয়ে আমার লেখা সবচেয়ে বড় উপন্যাস। এতে ১৯৭১ সালে আমার বাবা শহীদ ফয়জুর রহমান আহমেদের আত্মত্যাগ, সাধারণ মানুষের অবর্ণনীয় কষ্ট এবং মাওলানা ইরতাজউদ্দিন ও নাইমুলের গল্প আমি তুলে ধরেছি।"
]

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
                    text = re.sub(r"\[[^\]]*\]", "", text)
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

# 3. Load Fine-Tuned QLoRA Model & Tokenizer from Hugging Face Hub
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

# 4. Pure-Bengali Logits Processor & Banglish Query Normalizer
bad_token_ids = []
forbidden_pattern = re.compile(
    r"[a-zA-Z0-9\[\]\{\}\<\>\u00C0-\u024F\u1E00-\u1EFF\u0900-\u097F\u0400-\u08FF\u0A00-\u1DFF\u3000-\u9FFF\uAC00-\uD7AF]"
)
special_ids = set(tokenizer.all_special_ids)

for token_str, token_id in tokenizer.get_vocab().items():
    if token_id in special_ids or (token_str.startswith("<") and token_str.endswith(">")):
        continue
    if forbidden_pattern.search(token_str):
        bad_token_ids.append(token_id)

bad_token_ids_tensor = torch.tensor(bad_token_ids, dtype=torch.long)

class PureBengaliLogitsProcessor(LogitsProcessor):
    def __init__(self, forbidden_ids):
        self.forbidden_ids = forbidden_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.forbidden_ids.device != scores.device:
            self.forbidden_ids = self.forbidden_ids.to(scores.device)
        scores[:, self.forbidden_ids] = -float("inf")
        return scores

bengali_blocker = LogitsProcessorList([PureBengaliLogitsProcessor(bad_token_ids_tensor)])

BANGLISH_TOPIC_RULES = [
    (r"\b(devi|debi|দেবী)\b", "স্যার, আপনার লেখা 'দেবী' উপন্যাসের মূল কাহিনী ও মিসির আলি, রানু, আনিস ও নীলুর ঘটনাটি আপনার নিজের ভাষায় বিস্তারিত বলুন।", GOLDEN_CHUNKS[0]),
    (r"\b(misir|nishithini|মিসির|নিশীথিনী)\b", "স্যার, আপনার সৃষ্টি করা মিসির আলি চরিত্রটি সম্পর্কে বিস্তারিত বলুন।", GOLDEN_CHUNKS[1]),
    (r"\b(himu|moyurakkhi|rupa|হিমু|ময়ূরাক্ষী|রূপা)\b", "স্যার, আপনার সৃষ্টি করা হিমু এবং রূপা চরিত্রটি সম্পর্কে বিস্তারিত বলুন।", GOLDEN_CHUNKS[2]),
    (r"\b(baker|kothao keu nei|muna|বাকের)\b", "স্যার, আপনার 'কোথাও কেউ নেই' নাটকের বাকের ভাই চরিত্রটি সম্পর্কে বিস্তারিত বলুন।", GOLDEN_CHUNKS[3]),
    (r"\b(nondito|shonkhonil|নন্দিত|শঙ্খনীল)\b", "স্যার, আপনার 'নন্দিত নরকে' ও 'শঙ্খনীল কারাগার' উপন্যাস লেখার গল্পটি বলুন।", GOLDEN_CHUNKS[4]),
    (r"\b(jochona|josna|muktijuddho|1971|জোছনা|মুক্তিযুদ্ধ)\b", "স্যার, আপনার 'জোছনা ও জননীর গল্প' উপন্যাস এবং ১৯৭১ সালের স্মৃতি নিয়ে বলুন।", GOLDEN_CHUNKS[5]),
]

SYSTEM_PROMPT = (
    "আপনি নিজেই বাংলাদেশের প্রখ্যাত কথাসাহিত্যিক, নাট্যকার ও চলচ্চিত্র নির্মাতা হুমায়ূন আহমেদ। "
    "পাঠকদের সাথে আপনার নিজস্ব চিরচেনা ছোট ছোট বাক্য, মৃদু রসবোধ, হিমু ও মিসির আলির দর্শন "
    "এবং আন্তরিক গল্পের ছলে সর্বদা প্রথম পুরুষে ('আমি', 'আমার লেখা') কথা বলুন।"
)

def preprocess_banglish_input(user_query):
    matched_chunks = []
    clean_bengali_question = user_query
    is_banglish = bool(re.search(r"[a-zA-Z]", user_query))
    
    for pattern, bn_question, chunk in BANGLISH_TOPIC_RULES:
        if re.search(pattern, user_query, flags=re.IGNORECASE):
            matched_chunks.append(chunk)
            if is_banglish:
                clean_bengali_question = bn_question
                
    if is_banglish and clean_bengali_question == user_query:
        replacements = {
            r"\b(vai|bhai|sir)\b": "স্যার",
            r"\b(kemon achen|kmn achen|how are you)\b": "আপনি কেমন আছেন",
            r"\b(hello|hi|hlw|salam|assalamu alaikum)\b": "স্যার, কেমন আছেন",
            r"\b(boiler|boier|boyer|boi)\b": "বইয়ের",
            r"\b(summirize|summarize|summary|samari|kahini|golpo)\b": "মূল কাহিনী ও সারাংশ",
            r"\b(bolen|bolo|bolun)\b": "বলুন",
            r"\b(to)\b": "তো",
        }
        temp_q = user_query
        for pat, rep in replacements.items():
            temp_q = re.sub(pat, rep, temp_q, flags=re.IGNORECASE)
        clean_bengali_question = f"{temp_q} (প্রথম পুরুষে 'আমি' হিসেবে বাংলায় বিস্তারিত উত্তর দিন)"
        
    return clean_bengali_question, matched_chunks

def ensure_complete_sentence(text):
    text = text.strip()
    last_punct = max(text.rfind("।"), text.rfind("!"), text.rfind("?"))
    if last_punct != -1 and last_punct > len(text) * 0.55:
        return text[:last_punct + 1].strip()
    if text and text[-1] not in "।!?":
        return text + "।"
    return text

GREETING_PATTERN = re.compile(
    r"^\s*(?:(?:hi+|hello+|hey+|hlw+|salam|assalamu\s*alaikum|good\s*morning|good\s*evening|kemon\s*achen|kmn\s*achen|ki\s*obostha|vai|bhai|sir|হাই|হ্যালো|সালাম|আসসালামু\s*আলাইকুম|কেমন\s*আছেন|কী\s*অবস্থা|স্যার|ভাই)\s*[\?\!\.,]*\s*){1,3}$",
    re.IGNORECASE
)

WRITING_BADGE_HTML = """
<div class="live-writing-box">
    <span class="pen-anim">✒️</span>
    <span>হুমায়ূন আহমেদ লিখছেন</span>
    <span class="dot-pulse"><span>.</span><span>.</span><span>.</span></span>
</div>
"""

def generate_fancy_response(user_query, history, mood, temperature):
    yield WRITING_BADGE_HTML
    
    is_greeting = bool(GREETING_PATTERN.match(user_query.strip()))
    
    mood_instructions = {
        "দখিন হাওয়ার আড্ডা (স্বাভাবিক গল্পগুজব)": "সহজ, ঘরোয়া এবং আন্তরিক আড্ডার সুরে উত্তর দিন।",
        "হিমুর হলুদ পাঞ্জাবি (রহস্য ও বাউণ্ডুলে দর্শন)": "হিমুর মতো নির্লিপ্ত, রহস্যময় ও জাগতিক মায়াহীন দর্শনের ছোঁয়ায় উত্তর দিন।",
        "মিসির আলির ডায়েরি (যুক্তি ও মনস্তত্ত্ব)": "মিসির আলির মতো গভীর মনস্তাত্ত্বিক বিশ্লেষণ ও তীক্ষ্ণ যুক্তির মাধ্যমে উত্তর দিন।",
        "নুহাশ পল্লীর বৃষ্টি (স্মৃতিকাতর ও কাব্যিক)": "বৃষ্টি, জোছনা ও শৈশবের স্মৃতিময় কাব্যিক আবেগ দিয়ে উত্তর দিন।"
    }
    selected_mood = mood_instructions.get(mood, mood_instructions["দখিন হাওয়ার আড্ডা (স্বাভাবিক গল্পগুজব)"])
    
    if is_greeting:
        clean_question = "স্যার, কেমন আছেন? আপনার সাথে একটু গল্প করতে এলাম।"
        final_context = (
            "একজন পাঠক আপনার সাথে দেখা করতে এসেছে এবং কুশল বিনিময় করেছে। "
            "আপনি হুমায়ূন আহমেদের মতো খুব আন্তরিক ও সংক্ষিপ্ত ভঙ্গিতে (২ থেকে ৩টি ছোট বাক্যে) তাকে স্বাগত জানান, "
            "দখিন হাওয়ায় এক কাপ চা খাওয়ার বা বৃষ্টির দিনের আড্ডার কথা বলুন এবং সে কেমন আছে জিজ্ঞেস করুন।"
        )
        min_tok, max_tok = 25, 85
        length_rule = "২ থেকে ৩টি ছোট ও আন্তরিক বাক্যে সংক্ষিপ্ত কুশল বিনিময় করুন।"
    else:
        clean_question, priority_chunks = preprocess_banglish_input(user_query)
        q_vec = embedder.encode([clean_question], convert_to_numpy=True)
        _, indices = faiss_index.search(q_vec, min(2, len(knowledge_chunks)))
        retrieved = [knowledge_chunks[i] for i in indices[0] if knowledge_chunks[i] not in priority_chunks]
        final_context = "\n\n".join((priority_chunks + retrieved)[:2])
        min_tok, max_tok = 70, 210
        length_rule = "প্রথম পুরুষে ('আমি', 'আমার লেখা') ৪ থেকে ৫টি পূর্ণাঙ্গ বাক্যে গল্পের ছলে উত্তর দেবেন।"
    
    full_system_prompt = (
        f"{SYSTEM_PROMPT}\n"
        f"নির্দেশনা: {selected_mood} {length_rule}\n\n"
        f"[প্রাসঙ্গিক তথ্যসূত্র]:\n{final_context}"
    )
    
    messages = [
        {"role": "system", "content": full_system_prompt},
        {"role": "user", "content": clean_question}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60.0)
    
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        min_new_tokens=min_tok,
        max_new_tokens=max_tok,
        temperature=max(0.35, min(float(temperature), 0.65)),
        top_p=0.88,
        top_k=40,
        no_repeat_ngram_size=4,
        logits_processor=bengali_blocker,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    def run_generation():
        with torch.no_grad():
            model.generate(**gen_kwargs)
            
    thread = Thread(target=run_generation)
    thread.start()
    
    partial_reply = ""
    try:
        for new_token in streamer:
            partial_reply += new_token
            cleaned = partial_reply.replace("হুমায়ূন আহমেদের লেখা", "আমার লেখা").replace("হুমায়ূন আহমেদের লেখা", "আমার লেখা")
            if cleaned.strip():
                yield f'<div class="active-writing-pill"><span class="pen-anim">✒️</span> হুমায়ূন আহমেদ লিখছেন...</div>\n\n{cleaned.strip()}'
    finally:
        thread.join(timeout=2.0)
        final_clean = partial_reply.replace("হুমায়ূন আহমেদের লেখা", "আমার লেখা").replace("হুমায়ূন আহমেদের লেখা", "আমার লেখা").strip()
        if final_clean:
            yield ensure_complete_sentence(final_clean)

# 5. Dashboard Styling & UI Layout
FANCY_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@400;500;600;700&family=Noto+Serif+Bengali:wght@500;700&display=swap');

body, .gradio-container {
    background: #111215 !important;
    color: #ececec !important;
    font-family: 'Hind Siliguri', sans-serif !important;
    max-width: 95vw !important;
    margin: 0 auto !important;
    padding: 12px 18px !important;
}
footer { display: none !important; }

.main-dashboard-row {
    display: flex !important;
    align-items: stretch !important;
    gap: 16px !important;
}
.left-panel-col, .right-panel-col {
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    gap: 12px !important;
}

.left-panel-col > .block,
.left-panel-col .html-container,
.left-panel-col .prose {
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
    width: 100% !important;
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
}

.sidebar-card {
    background: #181a1f;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 22px 18px;
    width: 100% !important;
    height: 100% !important;
    box-sizing: border-box !important;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.controls-card,
.controls-card > .form,
.left-panel-col .form {
    background: #181a1f !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 16px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    margin: 0 !important;
    gap: 12px !important;
    overflow: visible !important;
    box-shadow: none !important;
}
.controls-card > .form {
    border: none !important;
    padding: 0 !important;
}
.controls-card .block,
.controls-card .wrap,
.left-panel-col .form .block {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: visible !important;
    box-shadow: none !important;
}
.controls-card .wrap-inner,
.controls-card .secondary-wrap,
.controls-card select,
.controls-card input[type="number"] {
    background: #22252d !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    color: #f4f4f5 !important;
    box-shadow: none !important;
}
.controls-card label span {
    color: #e4e4e7 !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
}
.controls-card input[type="range"] {
    accent-color: #e07a5f !important;
}

.icon-button-wrapper.top-panel,
.chatbot .top-panel {
    display: none !important;
}

.chatbot:has(.live-writing-box) .icon-button-wrapper,
.chatbot:has(.active-writing-pill) .icon-button-wrapper {
    display: none !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

.icon-button-wrapper:not(.top-panel) button:not(:first-child),
.icon-button-wrapper:not(.top-panel) .divider,
button[aria-label*="Retry"],
button[aria-label*="Undo"],
button[title*="Retry"],
button[title*="Undo"] {
    display: none !important;
}
.icon-button-wrapper:not(.top-panel) {
    background: #23252b !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 6px !important;
    padding: 2px !important;
    box-shadow: none !important;
}
.icon-button-wrapper:not(.top-panel) button {
    border: none !important;
    color: #a1a1aa !important;
}
.icon-button-wrapper:not(.top-panel) button:hover {
    color: #ffffff !important;
}

.live-writing-box {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(224, 122, 95, 0.14);
    border: 1px solid rgba(224, 122, 95, 0.4);
    color: #f4a261;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 13.5px;
    font-weight: 600;
}
.active-writing-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(224, 122, 95, 0.14);
    border: 1px solid rgba(224, 122, 95, 0.35);
    color: #f4a261;
    font-size: 12px;
    font-weight: 600;
    padding: 3px 12px;
    border-radius: 999px;
    margin-bottom: 8px;
    animation: pulseGlow 1.5s infinite ease-in-out;
}
.pen-anim {
    display: inline-block;
    animation: penWrite 0.55s infinite alternate ease-in-out;
}
@keyframes penWrite {
    0% { transform: translateY(0px) rotate(0deg); }
    100% { transform: translateY(-3px) rotate(-15deg); }
}
@keyframes pulseGlow {
    0%, 100% { opacity: 0.7; }
    50% { opacity: 1; }
}
.dot-pulse span {
    animation: blinkDots 1.4s infinite both;
    font-weight: bold;
    font-size: 16px;
}
.dot-pulse span:nth-child(2) { animation-delay: 0.2s; }
.dot-pulse span:nth-child(3) { animation-delay: 0.4s; }
@keyframes blinkDots {
    0%, 80%, 100% { opacity: 0; }
    40% { opacity: 1; }
}

.avatar-container,
.chatbot .avatar-container,
.message-row .avatar-container {
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
    min-height: 40px !important;
    padding: 0 !important;
    margin: 0 10px 0 0 !important;
    border: none !important;
    border-radius: 50% !important;
    background: transparent !important;
    overflow: hidden !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.avatar-container > *,
.avatar-container button,
.avatar-container div {
    width: 100% !important;
    height: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
}
.avatar-container img,
.chatbot .avatar-container img {
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
    min-height: 40px !important;
    max-width: 40px !important;
    max-height: 40px !important;
    object-fit: cover !important;
    border-radius: 50% !important;
    border: 2px solid #e07a5f !important;
    box-sizing: border-box !important;
    padding: 0 !important;
    margin: 0 !important;
    display: block !important;
}

.message.user, .user-row .message {
    background: #2b2d35 !important;
    color: #f4f4f5 !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 10px 16px !important;
}
.message.bot, .bot-row .message {
    background: #1c1e24 !important;
    color: #ececec !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 14px 18px !important;
    line-height: 1.7 !important;
}

.avatar-wrapper {
    position: relative;
    width: 104px;
    height: 104px;
    margin: 0 auto 12px auto;
    padding: 0 !important;
}
.avatar-img {
    width: 104px !important;
    height: 104px !important;
    border-radius: 50% !important;
    object-fit: cover !important;
    border: 2.5px solid #e07a5f !important;
    box-sizing: border-box !important;
    padding: 0 !important;
    margin: 0 !important;
    display: block !important;
}
.status-dot {
    position: absolute; bottom: 4px; right: 4px; width: 13px; height: 13px;
    background: #10b981; border: 2px solid #181a1f; border-radius: 50%;
}
.author-name {
    font-family: 'Noto Serif Bengali', serif; font-size: 21px; font-weight: 700;
    text-align: center; color: #ffffff; margin-bottom: 2px;
}
.author-Dates { font-size: 12px; text-align: center; color: #a1a1aa; margin-bottom: 12px; }
.quote-box {
    background: #20232a; border-left: 3px solid #e07a5f;
    padding: 10px 12px; border-radius: 8px; font-size: 13px; font-style: italic;
    color: #d4d4d8; margin-bottom: 12px; line-height: 1.5;
}
.badge-row { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-top: 10px; }
.link-btn {
    background: #23262e; border: 1px solid rgba(255, 255, 255, 0.1);
    color: #e4e4e7 !important; padding: 6px 12px; border-radius: 8px; font-size: 12px;
    font-weight: 500; text-decoration: none !important; transition: 0.2s;
}
.link-btn:hover { background: #e07a5f; color: #ffffff !important; border-color: #e07a5f; }

.top-header-bar {
    background: #181a1f !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 10px 18px !important;
    margin: 0 !important;
    align-items: center !important;
}
.banner-title { font-family: 'Noto Serif Bengali', serif; font-size: 19px; font-weight: 700; color: #ffffff; }
.banner-tag {
    background: rgba(224, 122, 95, 0.15); color: #f4a261; padding: 3px 10px;
    border-radius: 999px; font-size: 11.5px; font-weight: 600;
    display: inline-block; margin-left: 8px;
}
.real-new-chat-btn {
    background: #262932 !important;
    color: #f4f4f5 !important;
    font-weight: 600 !important; font-size: 13px !important;
    border-radius: 10px !important; border: 1px solid rgba(255, 255, 255, 0.12) !important;
    padding: 7px 14px !important; cursor: pointer !important; transition: 0.2s !important;
}
.real-new-chat-btn:hover { background: #e07a5f !important; color: #ffffff !important; border-color: #e07a5f !important; }

.chat-card-shell {
    background: #181a1f !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 0 !important;
    gap: 0 !important;
    overflow: hidden !important;
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
}
.chat-card-shell .block,
.chat-card-shell .chatbot {
    background: #181a1f !important;
    border: none !important;
    margin: 0 !important;
}

.chat-card-shell form,
.chat-card-shell .multimodal-textbox,
.chat-card-shell .input-container,
.chat-card-shell > div:last-child {
    background: #20232b !important;
    border-top: 1px solid rgba(224, 122, 95, 0.25) !important;
    padding: 12px 16px !important;
    margin: 0 !important;
}
.chat-card-shell textarea,
.chat-card-shell input[type="text"] {
    background: #131519 !important;
    color: #ffffff !important;
    border: 1.5px solid rgba(224, 122, 95, 0.45) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    font-size: 15px !important;
    box-shadow: none !important;
}
.chat-card-shell textarea::placeholder {
    color: #9ca3af !important;
}
.chat-card-shell textarea:focus {
    background: #16181d !important;
    border-color: #e07a5f !important;
    outline: none !important;
}
.chat-card-shell button.primary,
.chat-card-shell button[type="submit"],
.chat-card-shell .submit-button {
    background: #e07a5f !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 14.5px !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 0 22px !important;
}
.chat-card-shell button[type="submit"]:hover {
    background: #d0674c !important;
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
    <div style="font-size: 12.5px; color: #cbd5e1; line-height: 1.5; text-align: center; margin-bottom: 8px;">
        বাংলা সাহিত্যের রাজপুত্র। হিমু, মিসির আলি ও শুভ্রর স্রষ্টা। কথা বলুন তাঁর নিজস্ব গদ্যশৈলী ও দর্শনে।
    </div>
    <div class="badge-row">
        <a href="https://github.com/MuntasirMalek/HumayunAhmedAliveGPT" target="_blank" class="link-btn">GitHub Repo</a>
        <a href="https://huggingface.co/MuntasirMalek/HumayunAhmedAliveGPT" target="_blank" class="link-btn">Hugging Face Model</a>
    </div>
</div>
"""

TOP_HEADER_LEFT_HTML = """
<div>
    <div style="display:flex; align-items:center; flex-wrap:wrap; gap:8px;">
        <span class="banner-title">HumayunAhmedAliveGPT</span>
        <span class="banner-tag">QLoRA + RAG Active</span>
    </div>
    <div style="font-size: 13px; color: #94a3b8; margin-top: 2px;">Fine-Tuned Persona LLM + FAISS Vector RAG Knowledge Base</div>
</div>
"""

FORCE_DARK_JS = """
() => {
    document.body.classList.add('dark');
    document.querySelector('gradio-app')?.classList.add('dark');
}
"""

with gr.Blocks(css=FANCY_CSS, title="HumayunAhmedAliveGPT", js=FORCE_DARK_JS) as demo:
    with gr.Row(elem_classes="main-dashboard-row"):
        with gr.Column(scale=3, min_width=270, elem_classes="left-panel-col"):
            gr.HTML(SIDEBAR_HTML)
            with gr.Column(elem_classes="controls-card"):
                mood_dropdown = gr.Dropdown(
                    choices=[
                        "দখিন হাওয়ার আড্ডা (স্বাভাবিক গল্পগুজব)",
                        "হিমুর হলুদ পাঞ্জাবি (রহস্য ও বাউণ্ডুলে দর্শন)",
                        "মিসির আলির ডায়েরি (যুক্তি ও মনস্তত্ত্ব)",
                        "নুহাশ পল্লীর বৃষ্টি (স্মৃতিকাতর ও কাব্যিক)"
                    ],
                    value="দখিন হাওয়ার আড্ডা (স্বাভাবিক গল্পগুজব)",
                    label="কথোপকথনের মেজাজ (Persona Mood)",
                    container=True
                )
                temp_slider = gr.Slider(
                    minimum=0.3,
                    maximum=0.8,
                    value=0.5,
                    step=0.05,
                    label="কল্পনা ও সৃজনশীলতা (Temperature)",
                    container=True
                )
            
        with gr.Column(scale=8, elem_classes="right-panel-col"):
            with gr.Row(elem_classes="top-header-bar"):
                with gr.Column(scale=7):
                    gr.HTML(TOP_HEADER_LEFT_HTML)
                with gr.Column(scale=3, min_width=150):
                    new_chat_btn = gr.Button("+ নতুন চ্যাট (New Chat)", elem_classes="real-new-chat-btn")
                    
            with gr.Column(elem_classes="chat-card-shell"):
                chatbot_comp = gr.Chatbot(
                    height=420,
                    avatar_images=(None, SQUARE_AVATAR_PATH),
                    show_label=False,
                    placeholder="<div style='text-align:center; color:#a1a1aa; padding:35px;'><h3>স্বাগতম</h3><p>হুমায়ূন আহমেদকে আপনার মনের যেকোনো প্রশ্ন বাংলায় বা বাংলিশে লিখুন...</p></div>"
                )
                textbox_comp = gr.Textbox(
                    placeholder="এখানে আপনার প্রশ্ন লিখুন...",
                    container=False,
                    scale=7
                )
                chat_ui = gr.ChatInterface(
                    fn=generate_fancy_response,
                    chatbot=chatbot_comp,
                    textbox=textbox_comp,
                    additional_inputs=[mood_dropdown, temp_slider],
                    submit_btn="পাঠান",
                    examples=[
                        ["স্যার, আপনার 'দেবী' উপন্যাসের কাহিনীটা সংক্ষেপে বলুন তো?"],
                        ["স্যার, হিমু কেন হলুদ পাঞ্জাবি পরে এবং পকেট রাখে না?"],
                        ["মিসির আলি আর হিমুর মধ্যে আপনার নিজের সাথে কার মিল বেশি?"],
                        ["নুহাশ পল্লীতে বৃষ্টির দিন আপনার কেমন কাটে?"],
                        ["বাকের ভাইয়ের ফাঁসি কেন দিয়েছিলেন স্যার?"]
                    ],
                )
                
            new_chat_btn.click(
                fn=lambda: ([], [], ""),
                inputs=None,
                outputs=[chat_ui.chatbot, chat_ui.chatbot_state, chat_ui.textbox],
                queue=False
            )

demo.launch(share=True)
```
