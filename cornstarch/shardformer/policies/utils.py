from torch import nn


def resize_embeddings(new_num_tokens: int, embedding: nn.Embedding):
    if embedding is None:
        return

    # In-place resize of the token embeddings
    embedding.num_embeddings = new_num_tokens

    if embedding.weight is not None:
        embedding.weight.data = nn.functional.pad(
            embedding.weight.data,
            (0, 0, 0, new_num_tokens - embedding.weight.size(0)),
            "constant",
            0,
        )


def resize_lm_head(new_num_tokens: int, lm_head: nn.Linear):
    if lm_head is None:
        return

    # In-place resize of the lm head
    lm_head.out_features = new_num_tokens

    if lm_head.weight is not None:
        lm_head.weight.data = nn.functional.pad(
            lm_head.weight.data,
            (0, 0, 0, new_num_tokens - lm_head.weight.size(0)),
            "constant",
            0,
        )
