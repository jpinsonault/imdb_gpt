class CharacterTokenizer:
    SPECIAL_PAD = '\u200C'
    SPECIAL_START = '\u200D'
    SPECIAL_END = '\u200E'
    SPECIAL_SEP = '\u200F'
    
    def __init__(self, special_tokens=None):
        if special_tokens is None:
            special_tokens = [self.SPECIAL_PAD, self.SPECIAL_START, self.SPECIAL_END, self.SPECIAL_SEP]
        self.special_tokens = special_tokens
        self.char_to_index = {}
        self.index_to_char = {}
        self.alphabet = set()
        self.trained = False

    def train(self, corpus):
        for text in corpus:
            if text:
                self.alphabet.update(text)
        # Remove any special tokens from the alphabet to avoid duplicates.
        remaining_chars = sorted(self.alphabet - set(self.special_tokens))
        full_vocab = self.special_tokens + remaining_chars
        self.char_to_index = {ch: idx for idx, ch in enumerate(full_vocab)}
        self.index_to_char = {idx: ch for ch, idx in self.char_to_index.items()}
        self.trained = True

    def encode(self, text):
        if not self.trained:
            raise RuntimeError("Tokenizer has not been trained. Call train(corpus) first.")
        # Use first special token as fallback (typically UNK) if a character is missing.
        fallback = self.char_to_index[self.special_tokens[0]]
        tokens = [self.char_to_index.get(ch, fallback) for ch in text]
        tokens = [self.char_to_index[self.special_tokens[1]]] + tokens + [self.char_to_index[self.special_tokens[2]]]
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        if not self.trained:
            raise RuntimeError("Tokenizer has not been trained. Call train(corpus) first.")
        tokens = [self.index_to_char.get(idx, '') for idx in token_ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]
        return ''.join(tokens)

    def token_to_id(self, token: str):
        return self.char_to_index.get(token, self.char_to_index[self.special_tokens[0]])

    def id_to_token(self, token_id: int):
        return self.index_to_char.get(token_id, "")

    def get_vocab_size(self):
        return len(self.char_to_index)

