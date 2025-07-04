#zero shot
import json
import torch
import gc
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- CONFIG ----------------
MODEL_ID = "google/gemma-3-1b-it"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/kaggle/input/create2/TeleQnA_training.txt"
MAX_QUESTIONS = 500
# ----------------------------------------

# ---------------- LOAD MODELS ----------------
def load_model_and_tokenizer(model_id=MODEL_ID, device=DEVICE):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval().to(device)
    return tokenizer, model

# ---------------- LOAD DATA ----------------
def load_teleqna_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# ---------------- EXTRACT OPTION NUMBER ----------------
def extract_option_number(text, num_options):
    matches = re.findall(r"\b[1-5]\b", text)
    for match in matches:
        i = int(match)
        if 1 <= i <= num_options:
            return i
    return -1

# ---------------- PREDICT OPTION ----------------
def predict_option_fast(question, options, tokenizer, model, device):
    prompt = f"""You are a telecom expert.

Question: {question}
Options:
""" + "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]) + """

Choose the best option number only:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)
    result = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()

    return extract_option_number(result, len(options))

# ---------------- GENERATE SHORT EXPLANATION ----------------
def generate_explanation_short(option_text, tokenizer, model, device):
    prompt = f"""You are a telecom expert. In 2–3 lines, explain why this answer is correct:

Answer: {option_text}"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()

# ---------------- MAIN LOOP ----------------
gc.collect()
torch.cuda.empty_cache()

tokenizer, model = load_model_and_tokenizer()
parsed_data = load_teleqna_data(DATA_PATH)

correct = 0
total = 0
skipped = 0

print("Starting zero-shot evaluation...\n")

for idx, key in enumerate(parsed_data.keys()):
    if total >= MAX_QUESTIONS:
        break

    entry = parsed_data[key]
    question = entry.get("question", "").strip()
    options = [entry.get(f"option {i}") for i in range(1, 6) if entry.get(f"option {i}")]
    answer_field = entry.get("answer", "")

    if not question or len(options) < 2:
        skipped += 1
        continue

    if "option" in answer_field:
        try:
            correct_option = int(answer_field.split("option ")[1].split(":")[0])
        except:
            skipped += 1
            continue
    else:
        skipped += 1
        continue

    print(f"\nQ{total+1}: {question}")

    predicted_option = predict_option_fast(question, options, tokenizer, model, DEVICE)

    if predicted_option == -1:
        print("Could not predict a valid option.")
        skipped += 1
        continue

    selected_option_text = options[predicted_option - 1]
    print(f"Predicted Option: {predicted_option}")

    explanation = generate_explanation_short(selected_option_text, tokenizer, model, DEVICE)
    print(f"Explanation: {explanation}")

    if predicted_option == correct_option:
        correct += 1

    total += 1

# ---------------- SUMMARY ----------------
print("\n\n========== SUMMARY ==========")
print(f"Questions Evaluated: {total}")
print(f"Skipped: {skipped}")
print(f"Correct Predictions: {correct}")
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"Baseline Accuracy: {correct}/{total} = {accuracy:.2f}%")
