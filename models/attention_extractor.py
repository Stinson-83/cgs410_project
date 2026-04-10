"""
models/attention_extractor.py — Transformer Attention Weight Extraction

Extracts attention weights from all layers and heads of a pretrained
transformer (BERT / GPT-2). Handles wordpiece-to-word alignment so that
the output attention matrices are over *words*, not subword tokens.

Key design decisions:
    - [CLS] and [SEP] tokens are excluded from the aligned matrices
    - Subword→word alignment: average attention *from* subwords (rows),
      sum attention *to* subwords (columns), then re-normalize rows
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Optional, Dict, Any


class AttentionExtractor:
    """
    Wraps a HuggingFace transformer model for attention extraction.

    Usage:
        extractor = AttentionExtractor("bert-base-uncased", device="cpu")
        result = extractor.extract("The dog barked.")
        # result["attention"].shape == (n_layers, n_heads, n_words, n_words)
    """

    def __init__(self, model_name: str = "bert-base-uncased", device: str = "cpu"):
        """
        Args:
            model_name: HuggingFace model identifier.
            device: "cpu" or "cuda".
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, output_attentions=True
        ).to(device)
        self.model.eval()

    def _get_word_to_subword_mapping(
        self, sentence: str, words: List[str]
    ) -> List[List[int]]:
        """
        Compute word→subword index mapping.

        For each word in the sentence, find which subword token indices
        (in the model's tokenization) correspond to it. We skip special
        tokens ([CLS], [SEP]) by offsetting indices.

        Args:
            sentence: Original sentence string.
            words: List of word strings (e.g., from spaCy tokenization).

        Returns:
            List of lists: word_to_subword[i] = [subword_idx_1, subword_idx_2, ...]
            These indices are into the *full* tokenized sequence (including [CLS]/[SEP]).
        """
        encoded = self.tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=True)
        offsets = encoded["offset_mapping"]  # (start, end) for each subword token

        # Build character→word mapping
        char_to_word = [None] * len(sentence)
        pos = 0
        for word_idx, word in enumerate(words):
            # Find the word in the sentence starting from current position
            start = sentence.lower().find(word.lower(), pos)
            if start == -1:
                # Fallback: just use current position
                start = pos
            for c in range(start, min(start + len(word), len(sentence))):
                char_to_word[c] = word_idx
            pos = start + len(word)

        # Map subword tokens to words
        word_to_subword: List[List[int]] = [[] for _ in range(len(words))]

        for subword_idx, (char_start, char_end) in enumerate(offsets):
            if char_start == 0 and char_end == 0:
                # Special token ([CLS], [SEP], [PAD])
                continue
            # Find which word this subword belongs to
            mid_char = (char_start + char_end) // 2
            if mid_char < len(char_to_word) and char_to_word[mid_char] is not None:
                word_idx = char_to_word[mid_char]
                word_to_subword[word_idx].append(subword_idx)

        # Ensure every word has at least one mapping (fallback for edge cases)
        for i in range(len(words)):
            if not word_to_subword[i]:
                # Try to find the nearest mapped subword
                word_to_subword[i] = [i + 1]  # offset by 1 for [CLS]

        return word_to_subword

    def _align_attention_to_words(
        self,
        attention: np.ndarray,
        word_to_subword: List[List[int]],
        n_subwords: int,
    ) -> np.ndarray:
        """
        Align subword-level attention matrices to word-level.

        Strategy:
            - Rows (attention FROM): average across subwords of the same word
            - Columns (attention TO): sum across subwords of the same word
            - Re-normalize each row to sum to 1

        Args:
            attention: shape (n_layers, n_heads, n_subwords, n_subwords)
            word_to_subword: mapping from word indices to subword indices
            n_subwords: total number of subword tokens (including specials)

        Returns:
            Aligned attention of shape (n_layers, n_heads, n_words, n_words)
        """
        n_layers, n_heads = attention.shape[0], attention.shape[1]
        n_words = len(word_to_subword)

        aligned = np.zeros((n_layers, n_heads, n_words, n_words), dtype=np.float32)

        for w_i in range(n_words):
            for w_j in range(n_words):
                subwords_i = word_to_subword[w_i]
                subwords_j = word_to_subword[w_j]

                # Average over source subwords, sum over target subwords
                for si in subwords_i:
                    for sj in subwords_j:
                        if si < n_subwords and sj < n_subwords:
                            aligned[:, :, w_i, w_j] += attention[:, :, si, sj]

                # Average over source subwords
                if len(subwords_i) > 0:
                    aligned[:, :, w_i, w_j] /= len(subwords_i)

        # Re-normalize rows to sum to 1
        row_sums = aligned.sum(axis=-1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid division by zero
        aligned = aligned / row_sums

        return aligned

    @torch.no_grad()
    def extract(self, sentence: str, words: List[str]) -> Dict[str, Any]:
        """
        Extract word-aligned attention matrices from the transformer.

        Args:
            sentence: Raw input sentence.
            words: List of word tokens (e.g., from spaCy tokenization).

        Returns:
            Dictionary with:
                - "attention": np.ndarray of shape (n_layers, n_heads, n_words, n_words)
                - "subword_tokens": list of subword token strings
                - "word_to_subword": the alignment mapping
        """
        # Tokenize
        inputs = self.tokenizer(
            sentence, return_tensors="pt", add_special_tokens=True
        ).to(self.device)

        # Forward pass
        outputs = self.model(**inputs)
        attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)

        # Stack all layers → (n_layers, n_heads, seq_len, seq_len)
        attention_tensor = torch.stack(attentions, dim=0).squeeze(1)  # remove batch dim
        attention_np = attention_tensor.cpu().numpy()

        n_subwords = attention_np.shape[-1]

        # Get subword tokens for debugging
        subword_tokens = self.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0].cpu().tolist()
        )

        # Compute word→subword mapping
        word_to_subword = self._get_word_to_subword_mapping(sentence, words)

        # Align attention to word level
        aligned_attention = self._align_attention_to_words(
            attention_np, word_to_subword, n_subwords
        )

        return {
            "attention": aligned_attention,
            "subword_tokens": subword_tokens,
            "word_to_subword": word_to_subword,
        }


# ─── CLI test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    extractor = AttentionExtractor("bert-base-uncased", device="cpu")

    sentence = "The dog that chased the cat barked."
    words = ["The", "dog", "that", "chased", "the", "cat", "barked", "."]

    result = extractor.extract(sentence, words)
    print(f"Attention shape: {result['attention'].shape}")
    print(f"Subword tokens: {result['subword_tokens']}")
    print(f"Word-to-subword mapping: {result['word_to_subword']}")
