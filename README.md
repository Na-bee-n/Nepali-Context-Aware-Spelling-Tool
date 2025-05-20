This is a spell checker that corrects spelling error of similar 
sounding Nepali words.

ABSTRACT
 This project introduces a novel approach to context-based spelling correction for
 the Nepali language, addressing the unique challenges posed by similar sounding
 confusing Nepali words. In Part A, we employed the Word2Vec model, as limited
 work had been done for this task using this approach. However, Word2Vec’s static
 embeddings and lack of positional encoding resulted in poor performance. To address
 this, we collected around 16,000 similar-sounding words (e.g., homophones) from
 Nepali dictionaries using transliteration and manual search, organizing them into a
 confusion set. Additionally, we expanded our dataset to around 16 million sentences
 for fine-tuning. In Part B, we transitioned to a transformer-based encoder model, BERT,
 achieving a baseline accuracy of 78%. We performed multiple experiments where we
 found that the post-position in the Nepali like को, मा, हरु, etc. significantly contributes
 to NepBERTa’s learning tendency. Building on this, we fine-tuned NepBERTa and
 observed impressive loss curves. To evaluate its performance, we randomly sampled
 a test set of 956 sentences, with 878 sentences containing frequently repeated confusion
 words. For these, the test accuracy was 81.2%, while for less frequently occurring
 confusion words, the accuracy was 60.27%. This highlights NepBERTa’s effectiveness
 in context-aware spelling correction, particularly for commonly repeated confusion
 words in the training set.
