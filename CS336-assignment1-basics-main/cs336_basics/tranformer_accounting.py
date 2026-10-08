
def tranformer_accounting(vocab_size: int, context_length: int, num_layers: int,
                          d_model: int, num_heads: int, d_ff: int):
    emb = vocab_size * d_model
    layer = 2 * d_model + 4 * d_model ** 2 + 3 * d_ff * d_model
    final_rmsnorm = d_model
    final_linear = d_model * vocab_size
    
    return emb + num_layers * layer + final_linear + final_rmsnorm

if __name__ == "__main__":
    count = tranformer_accounting(50_257, 1024, 48, 1600, 25, 4288)
    print(count)