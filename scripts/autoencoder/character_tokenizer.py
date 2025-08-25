class CharacterTokenizer:
    SPECIAL_UNK = "<unk>"
    SPECIAL_PAD = "<pad>"
    SPECIAL_START = "<s>"
    SPECIAL_END = "</s>"
    SPECIAL_SEP = "<sep>"

    def __init__(self, special_tokens=None):
        if special_tokens is None:
            special_tokens = [
                self.SPECIAL_UNK,
                self.SPECIAL_PAD,
                self.SPECIAL_START,
                self.SPECIAL_END,
                self.SPECIAL_SEP,
            ]
        self.special_tokens = special_tokens
        self.char_to_index = {}
        self.index_to_char = {}
        self.alphabet = set()
        self.trained = False

    def train(self, corpus):
        for text in corpus:
            if text:
                self.alphabet.update(text)
        remaining_chars = sorted(self.alphabet - set(self.special_tokens))
        full_vocab = list(self.special_tokens) + remaining_chars
        self.char_to_index = {ch: idx for idx, ch in enumerate(full_vocab)}
        self.index_to_char = {idx: ch for ch, idx in self.char_to_index.items()}
        self.trained = True

    def encode(self, text):
        if not self.trained:
            raise RuntimeError("Tokenizer has not been trained. Call train(corpus) first.")
        unk_id = self.char_to_index[self.special_tokens[0]]
        start_id = self.char_to_index[self.special_tokens[2]]
        end_id = self.char_to_index[self.special_tokens[3]]
        body = [self.char_to_index.get(ch, unk_id) for ch in text]
        return [start_id] + body + [end_id]

    def decode(self, token_ids, skip_special_tokens=True):
        if not self.trained:
            raise RuntimeError("Tokenizer has not been trained. Call train(corpus) first.")
        toks = [self.index_to_char.get(int(idx), "") for idx in token_ids]
        if skip_special_tokens:
            toks = [t for t in toks if t not in self.special_tokens]
        return "".join(toks)

    def token_to_id(self, token):
        return self.char_to_index.get(token, self.char_to_index[self.special_tokens[0]])

    def id_to_token(self, token_id):
        return self.index_to_char.get(int(token_id), "?")

    def get_vocab_size(self):
        return len(self.char_to_index)
