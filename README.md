# dialogue-summarization-flan-t5
Summarizing multi-turn dialogues using FLAN-T5 and prompt engineering (zero-shot, one-shot, few-shot)


# 🧠 Dialogue Summarizer with FLAN-T5

This project uses the `google/flan-t5-base` language model to summarize multi-turn human dialogues from the **DailyDialog** dataset using **prompt engineering**. It explores three powerful inference techniques:

- 🔹 Zero-Shot
- 🔹 One-Shot
- 🔹 Few-Shot

All running **free** using open-source tools — no API keys needed!

---

## 🧩 What This Project Does

> 💬 Turns a conversation like this...
>
> Speaker 1: I’m feeling a little sick today.
Speaker 2: Oh no, you should get some rest.
Speaker 1: Yeah, I might take a nap.
Speaker 2: Good idea. Hope you feel better soon.
>


> ...into a short summary like this:


---

## 📦 Tech Stack

| Tool        | Purpose                             |
|-------------|-------------------------------------|
| 🧠 FLAN-T5    | Language model for summarization     |
| 🤗 Hugging Face Datasets | To load the `daily_dialog` dataset |
| 🐍 Python     | All code written in Python          |
| 💻 Google Colab | Free platform to run everything    |

---

## 🚀 How to Use It

### 1️⃣ Open in Google Colab

> ✅ You can use this even if you have no Python installed!

- Download the notebook or copy-paste code into [Google Colab](https://colab.research.google.com/)
- Install dependencies:
  ```python
  !pip install transformers datasets sentencepiece
  
2️⃣ Load the Dialogue Dataset

from datasets import load_dataset
dataset = load_dataset("daily_dialog", trust_remote_code=True)



3️⃣ Pick a Dialogue

def format_dialogue(dialogue_lines):
    return "\n".join([f"Speaker {i % 2 + 1}: {line}" for i, line in enumerate(dialogue_lines)])

sample_dialogue = format_dialogue(dataset["test"][0]["dialog"])


4️⃣ Choose a Prompting Style
zero_shot_prompt(dialogue)

one_shot_prompt(dialogue)

few_shot_prompt(dialogue)


5️⃣ Load the Model & Generate Summary

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)




def generate_summary(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    output = model.generate(**inputs, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)



📚 Dataset Info
Dataset: DailyDialog on Hugging Face

Type: Multi-turn dialogue dataset

Use Case: Everyday conversations in English

📌 What I Learned
How to use the flan-t5 model for real-world summarization

Prompt engineering techniques (zero-shot, one-shot, few-shot)

Using Hugging Face datasets and models with no API

How model prompts affect the quality of generative output



🧠 Future Improvements
Add ROUGE evaluation for automatic scoring

Try other models like bart-large-cnn or mistral

Build a web app version using Gradio or Streamlit
