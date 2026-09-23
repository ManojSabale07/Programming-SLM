from pathlib import Path

from tokenizers import Tokenizer


class ProgrammingTokenizer:
    def __init__(self, tokenizer_path=None):
        if tokenizer_path is None:
            tokenizer_path = (
                Path(__file__).resolve().parent
                / "programming_bpe_tokenizer_full.json"
            )

        tokenizer_path = Path(tokenizer_path)

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {tokenizer_path}"
            )

        self.tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @property
    def pad_id(self):
        return self.tokenizer.token_to_id("<PAD>")

    @property
    def unk_id(self):
        return self.tokenizer.token_to_id("<UNK>")

    @property
    def bos_id(self):
        return self.tokenizer.token_to_id("<BOS>")

    @property
    def eos_id(self):
        return self.tokenizer.token_to_id("<EOS>")

    def encode(self, text, add_bos=True, add_eos=False):
        token_ids = self.tokenizer.encode(text).ids

        if add_bos and self.bos_id is not None:
            token_ids = [self.bos_id] + token_ids

        if add_eos and self.eos_id is not None:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, token_ids):
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )

    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id):
        return self.tokenizer.id_to_token(token_id)
