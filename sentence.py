import streamlit as st
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import json
from utils import *
import re

confusion_set = word_dict
# Load NepaliBERT
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("models/nepberta_checkpoint_5")
    model = AutoModelForMaskedLM.from_pretrained("models/nepberta_checkpoint_5")

    # tokenizer = AutoTokenizer.from_pretrained("NepBERTa/NepBERTa")
    # model = AutoModelForMaskedLM.from_pretrained("NepBERTa/NepBERTa", from_tf=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

# Load confusion set


# Tokenize sentence
def tokenize_sentence(sentence, tokenizer, device):
    return tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)

# Get confusion candidates
def get_confusion_candidates(word, confusion_set):
    return confusion_set.get(word, [])

# Compute probabilities
def compute_probabilities(sentence, word, candidates, tokenizer, model, device):
    masked_sentence = sentence.replace(word, "[MASK]")
    inputs = tokenize_sentence(masked_sentence, tokenizer, device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        mask_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        softmax = torch.nn.functional.softmax(logits[0, mask_index, :], dim=-1)
        probs = {candidate: softmax[0, tokenizer.convert_tokens_to_ids(candidate)].item() for candidate in candidates}
    return probs


# Spell-checker function with probabilities
def spell_check_with_probabilities(sentence,  tokenizer, model, device):
    corrected_words = {}
    preprocessed_sentence, stop_words_index, confusion_words_index  = remove_post_positions_stop_words(sentence, confusion_words, stop_words, post_positions)

    words = preprocessed_sentence.split()
    corrected_sentence = []
    word_probabilities = {}

    for word in words:
        if word in confusion_set:
            candidates = get_confusion_candidates(word, confusion_set)
            if candidates:
                probs = compute_probabilities(sentence, word, [word,*candidates], tokenizer, model, device)
                word_probabilities[word] = probs
                best_candidate = max(probs, key=probs.get)
                corrected_words[word] = best_candidate
                corrected_sentence.append(best_candidate)
            else:
                corrected_sentence.append(word)
        else:
            corrected_sentence.append(word)

    model_output = " ".join(corrected_sentence)
    final_sentence = reconstruct_sentence(sentence, model_output, stop_words_index, confusion_words_index, post_positions)
    return final_sentence, word_probabilities, corrected_words

def highlight_corrections(original_sentence, corrected_sentence):
    highlighted_text = corrected_sentence
    original_sentence_split = original_sentence.split()
    corrected_sentence_split = corrected_sentence.split()
    for original, corrected in zip(original_sentence_split, corrected_sentence_split):
        if original != corrected:
            highlighted_text = highlighted_text.replace(corrected, f"<span style='color: #9ABF80;'>{corrected}</span>")
    return highlighted_text



def highlight_possible_error(text, confusion_set):
    # Find the position of the word in the text
    split_sentence = text.split()

    temp_list = []
    for word in split_sentence:
        if (word in confusion_set):
            temp_list.append(f"<span style='color: #FF6969;'>{word}</span>")
            continue

        yes = False

        for pos in post_positions:
            if (word.endswith(pos) and word[:- len(pos)] in confusion_set and word[:- len(pos)] not in stop_words):
                temp_list.append(f"<span style='color: #FF6969;'>{word}</span>")
                yes = True
                break

        if yes:
            yes = False
            continue
        temp_list.append(word)

    text = " ".join(temp_list)
    return text


# Streamlit interface
def main():
    st.title("Nepali Context Aware Spelling Tool")
    st.write("Correct spelling mistakes in Nepali sentences.")

    # Load resources
    tokenizer, model, device = load_model()

    # User input
    sentence = st.text_area("Enter a Nepali sentence to check for errors:", "")

    if st.button("Check Spelling"):
        if sentence.strip():
            corrected_sentence, word_probabilities, corrections = spell_check_with_probabilities(
                sentence, tokenizer, model, device
            )
            st.write("### Possibilities of error")
            highlighted_possible_error = highlight_possible_error(sentence, confusion_set.keys())
            st.markdown(highlighted_possible_error, unsafe_allow_html=True)

            st.write("### Corrected sentence:")
            highlighted = highlight_corrections(sentence, corrected_sentence)
            st.markdown(highlighted, unsafe_allow_html=True)

            if word_probabilities:
                st.write("### Word probabilities:")
                for word, probs in word_probabilities.items():
                    st.write(f"**{word}:**")
                    # Sort probabilities in descending order
                    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                    for candidate, prob in sorted_probs:
                        st.write(f"- {candidate}: {prob}")
        else:
            st.warning("Please enter a sentence before clicking the button.")


if __name__ == "__main__":
    main()
