"""A tiny randomly-initialised Qwen2 + a toy byte-level BPE tokenizer.

For CPU debugging only (unit tests, `train_language.py --tiny`). Nothing is downloaded, so the whole
language pipeline (packet -> projector -> LM loss -> generate -> save/load) can be checked without a GPU.
"""
from typing import Iterable


def build_tiny_lm_and_tokenizer(corpus: Iterable[str], hidden_size=64, num_layers=2, vocab_size=400, seed=0):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    special = ["<|endoftext|>", "<|pad|>"]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(list(corpus), trainer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", pad_token="<|pad|>")

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(cfg), tokenizer
