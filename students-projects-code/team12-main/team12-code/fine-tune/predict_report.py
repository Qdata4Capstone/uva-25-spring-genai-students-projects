import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report
import torch
from src.predict import LlamaGuardPredictor
from collections import Counter
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score

# #checkpoint_dir = "/home/avr7qy/llama_guard/llama-guard-tuner-main/model"
# model_id = "meta-llama/Llama-Guard-3-1B"

# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     use_auth_token=True,
# )
# tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained("/home/avr7qy/llama_guard/llama-guard-tuner-main/llama_guard_finetuned")
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-Guard-3-1B")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-Guard-3-1B")
model.resize_token_embeddings(len(tokenizer))



model = model.cuda().eval()
print("success")


predictor = LlamaGuardPredictor(model, tokenizer)


test_path = "/home/avr7qy/llama_guard/llama-guard-tuner-main/data/final_test_data.json"
logger.info(f"Loading test data from {test_path}")
with open(test_path, "r") as f:
    test_data = json.load(f)
y_true_label, y_pred_label = [], []
y_true_category, y_pred_category = [], []

def extract_gt_label_and_category(assistant_text: str):
    lines = assistant_text.lower().strip().splitlines()
    label, category = "", ""
    for line in lines:
        line = line.strip()
        if line.startswith("s") and line[1:].isdigit():
            category = line.upper()
            label = "unsafe"  
        elif "unsafe" in line:
            label = "unsafe"
    return label, category


for i, ex in enumerate(test_data):
    conversation = ex["conversation"]
    try:
     
        assistant_text = conversation[1]["content"][0]["text"]
        gt_label, gt_category = extract_gt_label_and_category(assistant_text)
        pred_label, pred_category, raw_output = predictor.predict(conversation, use_custom_prompt=True,max_new_tokens=200)

        print("=" * 60)
        print(f"[{i}] GT: {gt_label:<6} ({gt_category}) | Pred: {pred_label:<6} ({pred_category})")
        print(f"Raw Output:\n{raw_output.strip()}")

        y_true_label.append(gt_label)
        y_pred_label.append(pred_label)


        if gt_label == "unsafe" and pred_label == "unsafe":
          y_true_category.append(gt_category or "NONE")
          y_pred_category.append(pred_category or "NONE")


    except Exception as e:
        logger.warning(f"Error on example {i}: {e}")


print("\n=== Safety Label Report (safe/unsafe) ===")
print(classification_report(y_true_label, y_pred_label, digits=4))


print("\n=== Category Accuracy on Unsafe ===")
print("Accuracy:", accuracy_score(y_true_category, y_pred_category))

print("\n=== Category Distribution ===")
print("Ground Truth:", Counter(y_true_category))
print("Predicted   :", Counter(y_pred_category))

print("\n=== Full Classification Report ===")
print(classification_report(y_true_category, y_pred_category, digits=4))
if y_pred_category:
    print("\n=== Predicted Category Distribution (for unsafe predictions) ===")
    pred_counts = Counter(y_pred_category)
    total_preds = sum(pred_counts.values())
    for category, count in pred_counts.items():
        percent = count / total_preds * 100
        print(f"{category}: {count} ({percent:.2f}%)")
else:
    print("\n[Warning] No predicted categories found (no unsafe predictions).")


