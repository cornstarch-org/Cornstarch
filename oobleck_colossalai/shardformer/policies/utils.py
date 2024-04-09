from torch import nn


def resize_token_embeddings(new_num_tokens: int, embedding: nn.Embedding):
    # In-place resize of the token embeddings
    embedding.num_embeddings = new_num_tokens

    if embedding.weight is not None:
        embedding.weight.data = nn.functional.pad(
            embedding.weight.data,
            (0, 0, 0, new_num_tokens - embedding.weight.size(0)),
            "constant",
            0,
        )
