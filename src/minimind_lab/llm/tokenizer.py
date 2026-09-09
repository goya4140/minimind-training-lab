class ByteTokenizer:
    """Dependency-free tokenizer used only for pipeline smoke tests.

    The formal 64M run uses the pinned MiniMind BPE tokenizer (vocab 6400).
    """

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 259

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [byte + 3 for byte in text.encode("utf-8")]
        return [self.bos_token_id, *ids, self.eos_token_id] if add_special_tokens else ids

    def decode(self, ids: list[int]) -> str:
        data = bytes(token - 3 for token in ids if 3 <= token < self.vocab_size)
        return data.decode("utf-8", errors="replace")

