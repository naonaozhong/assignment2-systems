import os
import regex as re
import pickle

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import BinaryIO
from collections import Counter

from .pretokenization_example import find_chunk_boundaries

## Parameters
NUM_PROCESSES = 4
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

## Split the context
def pre_token_count(text:str, special_tokens:list[str])\
    ->dict[str, int]:
    special_tokens_prep = [re.escape(s) for s in sorted(special_tokens, key=len, reverse=True)]
    if special_tokens_prep:
        context_list = re.split("|".join(special_tokens_prep), text)
    else:
        context_list = [text]
    pre_token_counter = Counter()
    for context in context_list:
        for token in re.finditer(PAT, context):
            pre_token_counter.update([token.group()])
    return pre_token_counter

## merge a pre-token list
def merge(lst:list[bytes], item:tuple[bytes, bytes]) -> list[bytes]:
    prev, next = item
    output = list()
    k = 0
    while k < len(lst) - 1:
        if lst[k] == prev and lst[k+1] == next:
            output.append(prev + next)
            k += 2
        else:
            output.append(lst[k])
            k += 1
    if k == len(lst) - 1:
        output.append(lst[k])
        
    return output

def counter_to_token(counter:dict[str, int], vocab_size:int)\
    -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    pre_token_index_map = list(counter.keys())
    corpus:list[list[bytes]]\
            = list(list(bytes([ch]) for ch in key.encode("utf-8")) for key in counter.keys())
    vocab = {id : bytes([id]) for id in range(256)}
    rule = list()
    pair_counts_num:dict[tuple[bytes, bytes], int] = Counter()
    pair_counts_index:dict[tuple[bytes, bytes], list[int]] = dict()
    ## initialize pair_counts diction
    for i in range(len(pre_token_index_map)):
        token = corpus[i]
        multiplier = counter[pre_token_index_map[i]]
        
        for prev, next in zip(token[:-1],token[1:]):
            pair_counts_num.update({(prev, next): multiplier})
            if not pair_counts_index.get((prev, next)):
                pair_counts_index[(prev, next)] = [i]
            else:
                pair_counts_index[(prev, next)].append(i)
    ## one bpe merge            
    while len(vocab) < vocab_size:
        most_counts = pair_counts_num.most_common(1)[0][1]
        prev, next = max([k for k, v in pair_counts_num.items() if v == most_counts]) 
        vocab[len(vocab)]= (prev + next)
        rule.append((prev, next))
        for i in pair_counts_index[(prev, next)]:
            multiplier = counter[pre_token_index_map[i]]
            for p, n in zip(corpus[i][:-1],corpus[i][1:]):
                pair_counts_num.subtract({(p, n): multiplier})
            corpus[i] = merge(corpus[i], (prev, next))
            for p, n in zip(corpus[i][:-1],corpus[i][1:]):
                pair_counts_num.update({(p, n): multiplier})
                if not pair_counts_index.get((p, n)):
                    pair_counts_index[(p, n)] = [i]
                else:
                    pair_counts_index[(p, n)].append(i)
        pair_counts_index[(prev, next)] = []
    return vocab, rule

def train_bpe(input_path:str | os.PathLike, 
              vocab_size:int, 
              special_tokens:list[str],
              num_process = 4)\
    -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        
    if num_process <= 1:
        pre_token_counter = _count_serial(input_path, special_tokens)
    else:
        pre_token_counter = _count_parallel(input_path, special_tokens, num_process)
            
    ids_map, rule = counter_to_token(pre_token_counter, vocab_size=vocab_size-len(special_tokens))
    
    m = len(ids_map)
    for i in range(len(special_tokens)):
        ids_map[i + m] = special_tokens[i].encode("utf-8")
        
    return ids_map, rule

def _process_chunk(args):
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return pre_token_count(chunk, special_tokens)

def _count_serial(input_path, special_tokens):
    counter = Counter()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 1, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            counter += pre_token_count(chunk, special_tokens)
    return counter


def _count_parallel(input_path, special_tokens, num_process):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_process, b"<|endoftext|>")
    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    counter = Counter()
    with ProcessPoolExecutor(max_workers=num_process) as ex:
        for partial in ex.map(_process_chunk, tasks):
            counter.update(partial)
    return counter

if __name__ == "__main__":
    ## parameters
    Valid_Text = Path("./data/TinyStoriesV2-GPT4-valid.txt")
    Train_Text = Path("./data/TinyStoriesV2-GPT4-train.txt")
    
    weight_path = Path(__file__).resolve().parent.parent / Path("weights")
    vocab_path = weight_path / Path("valid_vocab.pkl")
    rule_path = weight_path / Path("valid_rule.pkl")
      
    vocab, rule = train_bpe(Valid_Text, 10_000, ['<|endoftext|>'])
        
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(rule_path, "wb") as f:
        pickle.dump(rule, f)
    
    