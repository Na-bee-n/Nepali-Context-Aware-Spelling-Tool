import pandas as pd
import numpy as np


confusion_set_df = pd.read_csv('./datasets/final_confusion_set.csv')
# Sample data as described

# Initialize an empty dictionary
word_dict = {}

# Populate the dictionary
for _, row in confusion_set_df.iloc[:,:2].iterrows():
    word = row['Word']
    confusion_words = [word.strip() for word in row['Confusions'].split(',')]  # Handle multiple confusion words
    word_dict[word] = confusion_words

    # Add reverse mappings for all confusion words
    for confusion_word in confusion_words:
        if confusion_word not in word_dict:
            word_dict[confusion_word] = []
        if word not in word_dict[confusion_word]:
            word_dict[confusion_word].append(word)

confusion_set = word_dict
# Print the resulting dictionary

confusion_words = []
for word, value in word_dict.items():
    if word is not np.nan and word.count(" ") == 0:
        confusion_words.append(word.strip())
        
post_positions =  [
    "ले", "लागि", "निम्ति", "लाई", "देखि", "बाट", "बाटै", "प्रति", "द्वारा", "को", "का", "की", "मा", 
    "मै", "कै", "हरु", "संग", "संगै", "लगायत", "माथि", "अनुसार", "रहे", "बिना", "तुल्य", "झैँ", 
    "समेत", "चाहिँ", "तर्फ", "तिर", "जस्तो", "जस्ता", "जस्तै", "बीच", "सँग", "सम्म", "वाला", "पट्टि", 
    "बारे", "नै", "भित्र", "माथि", "मुनि", "पछि", "पछाडि", "अगाडि", "अघि", "अनुरूप", "जत्रो", "वाद", 
    "वटा", "मध्ये", "मार्फत", "साथ", "बमोजिम", "खेरि", "निर", "वारि", "पारि"
]

stop_words = [
    "अनि", "अब", "अरू", "आदि", "आफू", "उ", "उन", "उनी", "ऊ",
    "कसरी", "कस्तो", "कि", "किन", "किनकि", "किनभने",
    "के", "केहि", "केही", "को", "चाहीं", "छ", "जता", "जताततै",
    "जब", "जबकि", "जस्ता", "जस्तै", "जस्तो", "जहाँ", ''
    "जुन", "जुनै", "जे", "जो", "जोपनि", "जोपनी",'जस',
    "झैं", "त", "तत्काल", "तथा", "तपाईं", "तब",
    "तर", "तल", "तापनि", "तिनी", "तिनै", "तिमी",
    "ती", "त्यस","त्यसै", "त्यसकारण",  "त्यसो", "त्यस्तै", "त्यस्तो", "त्यहाँ", "त्यहीँ",
    "त्यो", "थिए", "थिएँ", "थियो",
    "देखि", "द्वारा", "न", "नि", "नै", "नौ",
    "पछि", "पछी", "पनि",'पनी',
    "बरु", "बाट",
    "मा", "मेरो", "मै", "यति", "यदि", "यद्यपि", "यसरी", "यसओ", "यस्तै", "यस्तो", "यहाँ",
    "यही", "या", "यी", "यो", "र",'यहि',
    "रे", "लाई", "लाख", "लागि", "ले",
    "वा", "वाट",
    "सँग", "सँगै",
    "सय", "सहित",
    "सहितै",
    "सो",
    "हामी", "हाम्रा", "हाम्रो", "हुँ",
    "म", "तँ", "तिमी",
    "ऊ", "त्यो", "उ", "ती", "उनी", "उहाँ",
    "तिम्रो", "उस",
    "कुन", "कहाँ", "कसै", "सबै", "आफ्नै",
    "हजुर", "वहाँ",
    "हो", "अहो",
    "च", "है", "ल", "लौं",
    "ला", "अथवा", "नत्र",
    "हाइत", "छि", "वाह", "अरे", 
    "कुनै"
]
# stop_words += post_positions
stop_words = list(set(stop_words))






################################## 
 # Reconstruction


def remove_post_positions_stop_words(sentence, confusion_set_words, stop_words, post_positions):
    words = sentence.split()  # Split sentence into words
    temp_filtered_words = []

    # Step 1: Remove postpositions
    for word in words:
        for pos in post_positions:
            if word.endswith(pos):  # Check if the word ends with the postposition
                word = word[: -len(pos)]  # Remove the postposition
                break  # Stop checking once the postposition is removed
        temp_filtered_words.append(word)

    # Step 2: Replace original words with filtered words if they are in confusion_set_words or stop_words
    confusion_words_index = []
    stop_words_index = []
    for index, removed_pp_word in enumerate(temp_filtered_words):
        if removed_pp_word == '':
            continue
        elif removed_pp_word in stop_words:
            words[index] = ''
            stop_words_index.append(index)
        elif removed_pp_word in confusion_set_words:
            words[index] = removed_pp_word
            confusion_words_index.append(index)

    words = filter(lambda x: x != '', words)
    return " ".join(words), stop_words_index, confusion_words_index


def reconstruct_sentence(input_sentence, model_output, stop_words_index,confusion_words_index, post_positions):
    print(input_sentence)
    print(model_output)
    input_words = input_sentence.split()
    model_output_words = model_output.split()

    for index in stop_words_index:
        model_output_words.insert(index, input_words[index])
    

    for index in confusion_words_index:
        for pos in post_positions:
            if input_words[index].endswith(pos):
                model_output_words[index]+= pos
                
    return " ".join(model_output_words)
        