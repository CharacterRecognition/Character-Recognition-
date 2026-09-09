import os
import torch
import re
from transformers import BertTokenizer, BertForMaskedLM
from spellchecker import SpellChecker

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
spell = SpellChecker()
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForMaskedLM.from_pretrained(MODEL_NAME)
model.eval()


def is_valid_word(word):
    return re.fullmatch(r"[a-zA-Z]+", word) is not None

# Stage 1: Spell Correction (ONLY alphabetic words)
def spell_correction(text):
    words = text.split()
    corrected = []

    for word in words:
        if is_valid_word(word):
            c = spell.correction(word)
            corrected.append(c if c else word)
        else:
            corrected.append(word)

    return " ".join(corrected)

# Stage 2: Contextual Correction (BERT, SAFE)
def contextual_correction(text):
    words = text.split()
    corrected_words = words.copy()

    for i, word in enumerate(words):
        if not is_valid_word(word):
            continue  # skip numbers, symbols, domains

        masked_words = words.copy()
        masked_words[i] = tokenizer.mask_token
        masked_text = " ".join(masked_words)

        inputs = tokenizer(masked_text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        mask_index = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_index) == 0:
            continue

        logits = outputs.logits[0, mask_index[0]]
        predicted_id = torch.argmax(logits).item()
        predicted_word = tokenizer.decode([predicted_id]).strip()

        # Replace only if prediction is alphabetic
        if is_valid_word(predicted_word):
            corrected_words[i] = predicted_word

    return " ".join(corrected_words)

def normalize_text(text):
    if isinstance(text, (list, tuple)):
        return " ".join([str(t) for t in text if str(t).strip() != ""])
    else:
        return str(text)

def contextualSeq_AI(text_array):
    corrected_outputs = []
    for text in text_array if isinstance(text_array, (list, tuple)) else [text_array]:
        # Convert list entries to sentence
        if isinstance(text, (list, tuple)):
            text = " ".join(text)
        else:
            text = str(text)

        # Stage 1: Spell correction
        spell_text = spell_correction(text)
        # Stage 2: Contextual correction
        final_text = contextual_correction(spell_text)
        corrected_outputs.append(final_text)
    return corrected_outputs[0] if isinstance(text_array, str) else corrected_outputs


