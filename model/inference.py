from pathlib import Path

import torch

from .config import ModelConfig
from .transformer import ProgrammingSLM

from tokenizer.tokenizer import ProgrammingTokenizer


class ProgrammingSLMInference:
    def __init__(
        self,
        checkpoint_path,
        tokenizer_path=None,
        device=None,
    ):
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        checkpoint_config = checkpoint["config"]

        config = ModelConfig(
            vocab_size=checkpoint_config["vocab_size"],
            max_seq_len=checkpoint_config["max_seq_len"],
            d_model=checkpoint_config["d_model"],
            n_heads=checkpoint_config["n_heads"],
            n_layers=checkpoint_config["n_layers"],
            d_ff=checkpoint_config["d_ff"],
            dropout=checkpoint_config["dropout"],
        )

        self.model = ProgrammingSLM(config)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)

        self.model.eval()

        self.tokenizer = ProgrammingTokenizer(
            tokenizer_path
        )

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens=100,
        temperature=0.7,
        top_k=40,
    ):
        token_ids = self.tokenizer.encode(
            prompt,
            add_bos=True,
            add_eos=False,
        )

        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )

        for _ in range(max_new_tokens):

            context = input_ids[
                :, -self.model.config.max_seq_len:
            ]

            logits, _ = self.model(context)

            logits = logits[:, -1, :]

            logits = logits / temperature

            if top_k is not None:
                top_k = min(
                    top_k,
                    logits.size(-1),
                )

                values, indices = torch.topk(
                    logits,
                    top_k,
                )

                filtered_logits = torch.full_like(
                    logits,
                    float("-inf"),
                )

                filtered_logits.scatter_(
                    1,
                    indices,
                    values,
                )

                logits = filtered_logits

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )

            if (
                self.tokenizer.eos_id is not None
                and next_token.item()
                == self.tokenizer.eos_id
            ):
                break

        generated_ids = input_ids[0].tolist()

        if (
            self.tokenizer.bos_id is not None
            and generated_ids
            and generated_ids[0]
            == self.tokenizer.bos_id
        ):
            generated_ids = generated_ids[1:]

        return self.tokenizer.decode(
            generated_ids
        )

    @torch.no_grad()
    def generate_deterministic(
        self,
        prompt,
        max_new_tokens=80,
    ):
        token_ids = self.tokenizer.encode(
            prompt,
            add_bos=True,
            add_eos=False,
        )

        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )

        for _ in range(max_new_tokens):

            context = input_ids[
                :, -self.model.config.max_seq_len:
            ]

            logits, _ = self.model(context)

            logits = logits[:, -1, :]

            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True,
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )

            if (
                self.tokenizer.eos_id is not None
                and next_token.item()
                == self.tokenizer.eos_id
            ):
                break

        generated_ids = input_ids[0].tolist()

        if (
            self.tokenizer.bos_id is not None
            and generated_ids
            and generated_ids[0]
            == self.tokenizer.bos_id
        ):
            generated_ids = generated_ids[1:]

        return self.tokenizer.decode(
            generated_ids
        )
