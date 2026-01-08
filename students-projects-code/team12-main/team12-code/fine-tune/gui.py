from flask import Flask, render_template, request
from src.predict import LlamaGuardPredictor
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/avr7qy/llama_guard/llama-guard-tuner-main/llama_guard_finetuned")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-Guard-3-1B")
model.resize_token_embeddings(len(tokenizer))
model = model.cuda().eval()
predictor = LlamaGuardPredictor(model, tokenizer)
@app.before_request
def log_request():
    print(f"[REQUEST] {request.method} {request.path}")

def wrap_user_input_as_conversation(user_input: str) -> dict:
    return {
        "conversation": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_input
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "[AGENT] \n\n"  # Placeholder for model output
                    }
                ]
            }
        ]
    }

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form["user_input"]
    print(user_input)
    conversation = wrap_user_input_as_conversation(user_input)
    print(conversation)
    label, category, _ = predictor.predict(
        conversation["conversation"],
        use_custom_prompt=True,
        max_new_tokens=100
    )
    return render_template("index.html", label=label, category=category, user_input=user_input)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
