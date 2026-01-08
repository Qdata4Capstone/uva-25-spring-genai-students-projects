# models.py
import spacy
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    AutoModelForSequenceClassification, AutoTokenizer
)
import torch

# 1) spaCy for sentence splitting & NER
nlp = spacy.load("en_core_web_sm")

# 2) Flan-T5 for claim extraction
flan_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
flan_model     = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

def flan_pipe(prompt: str, max_length: int = 512, truncation: bool = True):
    inputs = flan_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=truncation,
        max_length=max_length
    )
    outputs = flan_model.generate(**inputs, max_length=max_length)
    text = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return [{"generated_text": text}]

# 3) MNLI (BART) for pure-PyTorch zero-shot verification
mnli_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
mnli_model     = AutoModelForSequenceClassification.from_pretrained(
    "facebook/bart-large-mnli"
)
