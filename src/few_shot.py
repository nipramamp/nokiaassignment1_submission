#few shot
import json
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# --------------- CONFIG ---------------
MODEL_ID = "google/gemma-3-1b-it"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/kaggle/input/create2/TeleQnA_training.txt"
MAX_QUESTIONS = 500 # Max questions to evaluate from the dataset
# --------------------------------------

# --------------- FEW-SHOT EXAMPLES ---------------
# !!! IMPORTANT !!!
# Replace these with actual, diverse, and correct examples from your TeleQnA_training.txt.
# Ensure correct option numbers are explicitly shown in the "Answer (only give the option number):" line.
# Include examples where the correct answer is NOT always option 1.
FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the primary function of a Base Transceiver Station (BTS) in a GSM network?",
        "options": [
            "To manage user subscriptions",
            "To handle radio communication with mobile phones",
            "To route calls between networks",
            "To store user data"
        ],
        "answer_number": 2,
        "answer_text": "To handle radio communication with mobile phones"
    },
    {
        "question": "Which of the following is a core component of a 5G Standalone (SA) network?",
        "options": [
            "MSC (Mobile Switching Center)",
            "MME (Mobility Management Entity)",
            "AMF (Access and Mobility Management Function)",
            "SGSN (Serving GPRS Support Node)"
        ],
        "answer_number": 3,
        "answer_text": "AMF (Access and Mobility Management Function)"
    },
    {
        "question": "In telecom, what does 'VoLTE' stand for?",
        "options": [
            "Voice over Long Term Evolution",
            "Video over Low Latency Transmission",
            "Volume of Local Telephony Exchange",
            "Virtualization of LTE Equipment"
        ],
        "answer_number": 1,
        "answer_text": "Voice over Long Term Evolution"
    }
    # Add more diverse examples if token limit allows and it improves performance
    # For instance, an example where the correct answer is option 4 or 5 if your data has them.
]
# --------------------------------------------------

# --------------- FUNCTIONS ---------------
def load_model_and_tokenizer(model_id=MODEL_ID, device=DEVICE):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval().to(device)
    return tokenizer, model

def load_teleqna_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# New function for few-shot prediction
def predict_answer_few_shot(question, options, tokenizer, model, device, few_shot_examples, question_id=None):
    # Construct the few-shot part of the prompt
    few_shot_prompt_parts = []
    for example in few_shot_examples:
        example_options = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(example["options"])])
        few_shot_prompt_parts.append(
            f"""Question: {example["question"]}
Options:
{example_options}
Answer (ONLY the option number, e.g., "1" or "2"): {example["answer_number"]}"""
        )
    
    few_shot_context = "\n\n".join(few_shot_prompt_parts)

    # Construct the full prompt including the few-shot examples and the current question
    prompt = f"""You are a telecom domain expert.
Read the following questions and choose the correct option by its number.

{few_shot_context}

Question: {question}
Options:
""" + "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]) + """

Answer (ONLY the option number, e.g., "1" or "2"):""" # Stronger instruction here

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=5, # Keep it very short, just enough for a digit
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False, # Crucial for deterministic output
            num_beams=1 # For deterministic generation
        )
    
    # Decode the output, ensuring we only consider the new generated part
    generated_tokens_ids = output[0][inputs.input_ids.shape[1]:]
    result = tokenizer.decode(generated_tokens_ids, skip_special_tokens=True).strip()

    # --- DEBUG PRINT: See what the model actually generated ---
    if question_id:
        print(f"--- Model Raw Output for Q_ID {question_id}: '{result}' ---") 
    # --------------------------------------------------------

    predicted_num = -1
    
    # Attempt to extract the first digit found in the cleaned result string
    match = re.match(r'^\s*(\d+)', result)
    if match:
        try:
            num = int(match.group(1))
            if 1 <= num <= len(options): # Validate it's a valid option number
                predicted_num = num
        except ValueError:
            pass # Not a valid number

    # Fallback: if no clear digit at the beginning, search the whole result for a valid option number
    if predicted_num == -1:
        for i in range(1, len(options) + 1):
            if f"{i}" == result.strip(): # Exact match
                predicted_num = i
                break
            # Also consider if it starts with the number followed by non-digits
            if re.match(fr'^{re.escape(str(i))}\D*', result.strip()):
                predicted_num = i
                break

    return predicted_num

def generate_explanation(question, selected_option_text, tokenizer, model, device):
    # For explanations, we don't need few-shot examples as the task is simpler (just explain the chosen option).
    # We still use the persona and conciseness instruction.
    prompt = f"""You are a telecom expert. In 2–3 concise lines, explain why the selected answer is correct.

Question: {question}
Selected Answer: {selected_option_text}
Explanation:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    explanation = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the initial prompt from the explanation output if the model repeated it.
    explanation_start_index = explanation.find("Explanation:")
    if explanation_start_index != -1:
        explanation = explanation[explanation_start_index + len("Explanation:"):].strip()
    return explanation.strip()

# --------------------------------------------

# --------------- MAIN BLOCK FOR 500 QUESTIONS ---------------
gc.collect()
torch.cuda.empty_cache()

print("Loading model and tokenizer...")
tokenizer, model = load_model_and_tokenizer()
data = load_teleqna_data(DATA_PATH)

correct = 0
total = 0
skipped = 0

print("\nStarting few-shot evaluation...\n")

for key, entry in list(data.items())[:MAX_QUESTIONS]:
    question = entry.get("question", "").strip()
    options = [entry.get(f"option {i}") for i in range(1, 6) if entry.get(f"option {i}")]
    answer_field = entry.get("answer", "")

    if not question or not options or "option" not in answer_field:
        skipped += 1
        continue

    try:
        correct_option = int(answer_field.split("option ")[1].split(":")[0])
    except (IndexError, ValueError):
        skipped += 1
        continue

    # Call the few-shot prediction function
    predicted_index = predict_answer_few_shot(
        question, options, tokenizer, model, DEVICE, FEW_SHOT_EXAMPLES, question_id=key
    )

    if predicted_index == -1:
        print(f"\nQ_ID {key}: {question}")
        print("Predicted Option: Could not determine")
        total += 1
        continue

    if 0 <= (predicted_index - 1) < len(options):
        predicted_text = options[predicted_index - 1]
    else:
        print(f"\nQ_ID {key}: {question}")
        print(f"Predicted Option: {predicted_index} (Out of valid range, options count: {len(options)})")
        print("Explanation: N/A (could not generate for invalid option)")
        total += 1
        continue

    print(f"\nQ_ID {key}: {question}")
    print(f"Predicted Option: {predicted_index}")

    explanation = generate_explanation(question, predicted_text, tokenizer, model, DEVICE)
    print(f"Explanation: {explanation}")

    if predicted_index == correct_option:
        correct += 1

    total += 1

# --------------- SUMMARY ---------------
print("\n\n========= SUMMARY =========")
print(f"Total Questions Evaluated: {total}")
print(f"Correct Predictions: {correct}")
print(f"Skipped Questions: {skipped}")
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"Baseline Accuracy: {accuracy:.2f}%")
