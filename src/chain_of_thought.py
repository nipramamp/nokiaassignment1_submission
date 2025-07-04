#chain of thought
import json
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import re

# --- IMPORTANT: Hugging Face Login for Gated Models ---
# If you are running this in Google Colab, Kaggle, or a similar environment,
# you MUST log in to Hugging Face to access Gemma models.
# 1. Go to https://huggingface.co/settings/tokens and generate a new token (with 'read' access).
# 2. Uncomment the following lines and run them at the start of your notebook/script.
from huggingface_hub import login
# login() # Uncomment this line and run to log in. You will be prompted to paste your token.
# --------------------------------------------------------

# --------------- CONFIGURATION SETTINGS ---------------
MODEL_ID = "google/gemma-3-1b-it" # The specific Gemma model version
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available, else CPU
DATA_PATH = "/kaggle/input/create2/TeleQnA_training.txt" # Path to your Q&A dataset
MAX_QUESTIONS = 500 # Maximum number of questions to evaluate from the dataset
# ------------------------------------------------------

# --------------- CHAIN OF THOUGHT FEW-SHOT EXAMPLES ---------------
# These examples are CRUCIAL for the Chain of Thought (CoT) process.
# They teach the model the desired format: Question -> Options -> Reasoning -> Answer.
# Ensure these examples are:
# 1. Highly relevant to your telecom domain.
# 2. Diverse in terms of question types and correct answers (not always option 1).
# 3. Have clear, concise, and accurate reasoning steps that logically lead to the answer.
FEW_SHOT_EXAMPLES_COT = [
    {
        "question": "What is the primary function of a Base Transceiver Station (BTS) in a GSM network?",
        "options": [
            "To manage user subscriptions",
            "To handle radio communication with mobile phones",
            "To route calls between networks",
            "To store user data"
        ],
        "reasoning": "A BTS is the radio equipment in a cell tower. Its role is to manage the wireless connection between the network and individual mobile phones, transmitting and receiving radio signals.",
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
        "reasoning": "5G SA introduces new network functions. MSC, MME, and SGSN are 2G/3G/4G components. The AMF is a key 5G Core function for connection and mobility management.",
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
        "reasoning": "VoLTE is a standard for making voice calls using the 4G LTE data network, providing better call quality and integrated data usage.",
        "answer_number": 1,
        "answer_text": "Voice over Long Term Evolution"
    },
    {
        "question": "What is the role of a Home Subscriber Server (HSS) in a 4G LTE network?",
        "options": [
            "Manages data traffic routing",
            "Stores subscriber profiles and authentication information",
            "Handles call session control",
            "Provides IP address allocation"
        ],
        "reasoning": "The HSS acts as a central database for subscriber information. It stores details like user authentication credentials, service subscriptions, and routing information for mobile users.",
        "answer_number": 2,
        "answer_text": "Stores subscriber profiles and authentication information"
    }
]
# ---------------------------------------------------------------------

# Custom stopping criteria class
# This class tells the model to stop generating tokens when it encounters specific sequences,
# preventing it from writing beyond the desired output for a single question.
class StopOnTokens(StoppingCriteria):
    def _init_(self, stop_token_ids, device):
        super()._init_()
        # Convert stop token ID lists to PyTorch tensors and move them to the device (CPU/GPU)
        self.stop_token_ids = [torch.tensor(ids).to(device) for ids in stop_token_ids]
        self.device = device

    def _call_(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if the currently generated sequence ends with any of the defined stop sequences
        for stop_ids in self.stop_token_ids:
            if input_ids.shape[1] >= stop_ids.shape[0] and torch.equal(input_ids[0][-stop_ids.shape[0]:], stop_ids):
                return True
        return False

# --------------- CORE FUNCTIONS ---------------
def load_model_and_tokenizer(model_id=MODEL_ID, device=DEVICE):
    """Loads the pre-trained model and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval().to(device) # Set model to evaluation mode and move to device
    return tokenizer, model

def load_teleqna_data(filepath):
    """Loads the TeleQnA dataset from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def predict_answer_cot(question, options, tokenizer, model, device, few_shot_examples_cot, question_id=None):
    """
    Predicts an answer using the Chain of Thought prompting approach.
    Generates reasoning and then extracts the final answer.
    """
    # 1. Construct the few-shot part of the prompt
    few_shot_prompt_parts = []
    for example in few_shot_examples_cot:
        example_options = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(example["options"])])
        few_shot_prompt_parts.append(
            f"""Question: {example["question"]}
Options:
{example_options}
Reasoning: {example["reasoning"]}
Answer (ONLY the option number): {example["answer_number"]}"""
        )
    
    few_shot_context = "\n\n".join(few_shot_prompt_parts)

    # 2. Construct the full prompt for the current question
    prompt = f"""You are a telecom domain expert.
Read the following questions and choose the correct option by its number.

{few_shot_context}

Question: {question}
Options:
""" + "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]) + """

Reasoning:""" # Model is expected to start generating reasoning here

    # Tokenize the prompt and move to the device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    # 3. Define stop sequences for controlled generation
    # We want the model to stop after it has generated its reasoning and answer,
    # before it tries to start a new 'Question:' block (as seen in few-shot).
    stop_sequence_strs = [
        "\n\nQuestion:", # Standard separator for new questions
        "\nQuestion:",   # Single newline variation
    ]
    
    # Convert string stop sequences to token IDs
    stop_token_ids_list = [
        tokenizer.encode(s, add_special_tokens=False) for s in stop_sequence_strs
    ]
    
    # Create the stopping criteria list
    stopping_criteria_list = StoppingCriteriaList([StopOnTokens(stop_token_ids=stop_token_ids_list, device=device)])

    # 4. Generate the model's output
    with torch.no_grad(): # Disable gradient calculation for faster inference
        output = model.generate(
            **inputs, 
            max_new_tokens=500, # Max tokens to generate for reasoning + answer. Tuned for verbosity.
            pad_token_id=tokenizer.eos_token_id, # Pad token for batching (important if batching)
            do_sample=False, # Set to False for deterministic output (no randomness)
            num_beams=1, # Number of beams for beam search (1 means greedy search)
            stopping_criteria=stopping_criteria_list # Apply the custom stopping criteria
        )
    
    # Decode only the newly generated tokens (excluding the prompt itself)
    generated_tokens_ids = output[0][inputs.input_ids.shape[1]:]
    full_model_output = tokenizer.decode(generated_tokens_ids, skip_special_tokens=True).strip()

    # 5. Aggressive Post-Processing for Cleaner Output and Robust Parsing
    # This step ensures that any extraneous text (like premature starts of new questions
    # or repeated prompt patterns) are removed before displaying or parsing.
    
    processed_output_for_parsing = full_model_output
    
    # Patterns to identify and cut off unwanted generated content
    cut_off_patterns = [
        r'\n\nQuestion:',             # Catches the start of a new question block
        r'\n\nReasoning:',            # Catches if the model starts new reasoning after its answer
        r'Answer \(ONLY the option number\):$' # Catches if it repeats the answer prompt without a number
    ]

    for pattern in cut_off_patterns:
        match = re.search(pattern, processed_output_for_parsing)
        if match:
            processed_output_for_parsing = processed_output_for_parsing[:match.start()].strip()
            break # Cut off at the first pattern found and stop checking

    # --- DEBUG PRINT: Shows the model's output after cleaning/processing ---
    # This block is COMMENTED OUT by default to keep the main output clean.
    # Uncomment if you need to inspect the raw processed output for debugging.
    # if question_id:
    #     print(f"--- Model Raw Output (Processed) for Q_ID {question_id}:\n'{processed_output_for_parsing}'\n---")
    # ----------------------------------------------------------------------

    # 6. Parse the Reasoning and Predicted Answer from the processed output
    reasoning_text = ""
    predicted_num = -1 # Default to -1 if no valid answer is found

    # Regex to find the "Answer (ONLY the option number): X" pattern and capture the digit.
    # This is the primary parsing logic.
    answer_pattern = r'Answer \(ONLY the option number\):\s*(\d+)'
    answer_match = re.search(answer_pattern, processed_output_for_parsing)

    if answer_match:
        # If a valid answer number is found
        reasoning_text = processed_output_for_parsing[:answer_match.start()].strip()
        try:
            num = int(answer_match.group(1)) # Extract the digit
            if 1 <= num <= len(options): # Validate against available options
                predicted_num = num
        except ValueError:
            pass # Parsing failed (e.g., non-digit where digit expected, highly unlikely with regex)
    else:
        # If the explicit answer pattern (with a number) is NOT found
        # This means the model failed to output a parsable answer.
        reasoning_text = processed_output_for_parsing # Treat entire output as reasoning
        # predicted_num remains -1, signaling "Could not determine"

    return predicted_num, reasoning_text # Return parsed answer number and generated reasoning

# (This function is kept for completeness but not directly used in the main evaluation loop
# as CoT approach implicitly provides reasoning.)
def generate_explanation(question, selected_option_text, tokenizer, model, device):
    """Generates a concise explanation for a given question and selected option."""
    prompt = f"""You are a telecom expert. In 2–3 concise lines, explain why the selected answer is correct.

Question: {question}
Selected Answer: {selected_option_text}
Explanation:"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    explanation = tokenizer.decode(output[0], skip_special_tokens=True)
    explanation_start_index = explanation.find("Explanation:")
    if explanation_start_index != -1:
        explanation = explanation[explanation_start_index + len("Explanation:"):].strip()
    return explanation.strip()

# --------------------------------------------

# --------------- MAIN EVALUATION BLOCK ---------------
# Clean up GPU memory before starting
gc.collect()
torch.cuda.empty_cache()

print("Loading model and tokenizer...")
tokenizer, model = load_model_and_tokenizer()
data = load_teleqna_data(DATA_PATH)

correct = 0
total = 0
skipped = 0 # To count entries that couldn't be processed (e.g., missing data)

print("\nStarting Chain of Thought evaluation...\n")

# Iterate through a subset of the questions (defined by MAX_QUESTIONS)
for key, entry in list(data.items())[:MAX_QUESTIONS]:
    question = entry.get("question", "").strip()
    # Collect options (assuming they are "option 1", "option 2", etc.)
    options = [entry.get(f"option {i}") for i in range(1, 6) if entry.get(f"option {i}")]
    answer_field = entry.get("answer", "") # The ground truth answer from the dataset

    # Skip questions with missing data
    if not question or not options or "option" not in answer_field:
        skipped += 1
        continue

    # Extract the correct option number from the answer field (e.g., "option 2: To...")
    try:
        correct_option = int(answer_field.split("option ")[1].split(":")[0])
    except (IndexError, ValueError):
        skipped += 1
        continue

    # Call the CoT prediction function
    predicted_index, generated_reasoning = predict_answer_cot(
        question, options, tokenizer, model, DEVICE, FEW_SHOT_EXAMPLES_COT, question_id=key
    )

    # Handle cases where the prediction could not be determined by parsing
    if predicted_index == -1:
        print(f"\nQ_ID {key}: {question}")
        print("Predicted Option: Could not determine")
        print(f"Reasoning (if any): {generated_reasoning if generated_reasoning else 'N/A'}")
        total += 1
        continue

    # Determine the predicted option text (for display)
    if 0 <= (predicted_index - 1) < len(options):
        predicted_text = options[predicted_index - 1]
    else:
        # Handle cases where parsed index is out of valid bounds (should be rare with validation)
        # If the predicted_index is invalid, we can't get text, so mark it.
        predicted_text = "[Invalid Option Index]"
        print(f"\nQ_ID {key}: {question}")
        print(f"Predicted Option: {predicted_index} ({predicted_text}, options count: {len(options)})")
        print(f"Reasoning (if any): {generated_reasoning if generated_reasoning else 'N/A'}")
        total += 1
        continue

    # Print the question, predicted option (number AND text), and generated reasoning
    print(f"\nQ_ID {key}: {question}")
    print(f"Predicted Option: {predicted_index} ({predicted_text})") # MODIFIED LINE
    print(f"Reasoning: {generated_reasoning}")

    # Check if the prediction is correct and update counters
    if predicted_index == correct_option:
        correct += 1

    total += 1

# --------------- EVALUATION SUMMARY ---------------
print("\n\n========= SUMMARY =========")
print(f"Total Questions Evaluated: {total}")
print(f"Correct Predictions: {correct}")
print(f"Skipped Questions: {skipped}")
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"Baseline Accuracy (CoT): {accuracy:.2f}%")
