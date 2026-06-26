import argparse
import json
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from mteb import MTEB
from mteb.models.text_formatting_utils import corpus_to_texts
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def query_prompt(query: str) -> str:
    instruction = "Given a query, retrieve relevant passages that answer the query."
    return f"<instruct>{instruction}\n<query>{query}"


class EffiRRetriever:
    def __init__(
        self,
        model_name_or_path: str,
        lora_dir: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        cache_dir: str | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            attn_implementation="eager",
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )

        if lora_dir:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, lora_dir)
            self.model = self.model.merge_and_unload()

            input_emb_path = Path(lora_dir) / "embedding" / "input_emb.pth"
            lm_head_path = Path(lora_dir) / "embedding" / "lm_head.pth"
            if input_emb_path.is_file() and lm_head_path.is_file():
                self.model.set_input_embeddings(torch.load(input_emb_path))
                self.model.lm_head = torch.load(lm_head_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode(self, sentences, batch_size=None, **kwargs):
        return self.encode_queries(sentences, batch_size=batch_size, **kwargs)

    def encode_queries(self, queries: Union[List[str], str], batch_size=None, **kwargs):
        sentences = [queries] if isinstance(queries, str) else list(queries)
        prompted = [query_prompt(query) for query in sentences]
        return self._encode(prompted, batch_size=batch_size)

    def encode_corpus(self, corpus, batch_size=None, **kwargs):
        sentences = corpus_to_texts(corpus, sep=" ")
        return self._encode(sentences, batch_size=batch_size)

    @torch.no_grad()
    def _encode(self, sentences: List[str], batch_size=None) -> np.ndarray:
        batch_size = batch_size or self.batch_size
        all_embeddings = []

        for start in tqdm(range(0, len(sentences), batch_size), desc="Encoding"):
            batch = sentences[start : start + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(
                **inputs,
                return_dict=True,
                output_hidden_states=True,
            )
            embeddings = last_token_pool(outputs.hidden_states[-1], inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.float().cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--lora-dir", default=None)
    parser.add_argument("--task", default="NFCorpus")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default="mteb_results")
    return parser.parse_args()


def main():
    args = parse_args()
    model = EffiRRetriever(
        model_name_or_path=args.model_name_or_path,
        lora_dir=args.lora_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        cache_dir=args.cache_dir,
    )

    output_dir = Path(args.output_dir) / args.task
    evaluation = MTEB(tasks=[args.task], task_langs=["en"])
    results = evaluation.run(
        model,
        output_folder=str(output_dir),
        batch_size=args.batch_size,
        eval_splits=["test"],
        overwrite_results=True,
    )

    score = results[0].scores["test"][0]["main_score"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w") as f:
        json.dump({"task": args.task, "main_score": score}, f, indent=2)
    print(f"{args.task} main_score={score:.4f}")


if __name__ == "__main__":
    main()
