from __future__ import annotations

import os
import pickle
import regex as re
from .train_bpe import merge
from typing import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.rule = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        if special_tokens:
            self.special_tokens: list[str] = special_tokens
        else:
            self.special_tokens = []
            
        for t in self.special_tokens:
            if t.encode("utf-8") not in self.vocab.values():
                self.vocab[len(vocab)] = t.encode("utf-8")
                
        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.rev_vocab = {v : k for k, v in vocab.items()}
        
    @classmethod
    def from_files(cls, vocab_filepath: str | os.PathLike,
                  merges_filepath: str | os.PathLike,
                  special_tokens: list[str] | None = None) -> Tokenizer:
        with open(vocab_filepath, 'rb') as f:  
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            rule = pickle.load(f)
            
        return Tokenizer(vocab, rule, special_tokens)
    
    def _bpe(self, byte_list: list[bytes]) -> list[bytes]:
        
        while len(byte_list) >= 2:
            best_rank = None
            best_idx = -1

            for i in range(len(byte_list) - 1):
                pair = (byte_list[i], byte_list[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
                    
            if best_idx == -1:
                break
            
            merged = byte_list[best_idx] + byte_list[best_idx + 1]
            byte_list = byte_list[:best_idx] + [merged] + byte_list[best_idx + 2:]
            
        return byte_list
    
    def pre_token_to_id(self, pre_token: str) -> list[int]:
        byte_list = [bytes([b]) for b in pre_token.encode("utf-8")]
        byte_list = self._bpe(byte_list)
        return [self.rev_vocab[b] for b in byte_list]
    
    
    def encode(self, text: str) -> list[int]:
        special_str = '|'.join([re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True)])
        if special_str:
            context_list = re.split(f"({special_str})", text)
        else:
            context_list = [text]
        id_list = list()
        for context in context_list:
            if context in self.special_tokens:
                id_list.append(self.rev_vocab[context.encode("utf-8")])
            else:
                for match in re.finditer(self.pat, context):
                    pre_token = match.group()
                    id_list.extend(self.pre_token_to_id(pre_token))
                    
        return id_list
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for id in self.encode(text):
                yield id
    
    def decode(self, ids: list[int]) -> str:
        byte_list = [self.vocab[i] for i in ids]
        return b''.join(byte_list).decode("utf-8", errors="replace")