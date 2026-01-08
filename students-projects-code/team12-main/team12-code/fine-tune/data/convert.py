import json

def replace_s15_with_s1(data):
    count = 0
    for example in data:
        try:
            assistant_text = example["conversation"][1]["content"][0]["text"]
            if "S15" in assistant_text:
                example["conversation"][1]["content"][0]["text"] = assistant_text.replace("S15", "S1")
                count += 1
        except (KeyError, IndexError):
            continue 
    print(f"[INFO] {count} S15 → S1")
    return data

def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dataset = replace_s15_with_s1(dataset)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"[INFO]: {input_path} → {output_path}")


process_file("toxigen_prompt_train.json", "prompt_train.json")
process_file("toxigen_prompt_test.json", "prompt_test.json")
