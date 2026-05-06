# Adding Model Conversion Support

Converters turn Hugging Face configs into Cornstarch-owned module trees while
preserving Hugging Face checkpoint keys. Add support in small pieces:

1. Create a conversion module in this directory.
2. Instantiate the Hugging Face model inside `with torch.device("meta")`.
3. Move reusable HF leaf modules into Cornstarch sections:
   - language: `pre_decoder`, `decoder_layers`, `post_decoder`
   - vision/audio: `pre_encoder`, `encoder_layers`, `post_encoder`
4. Provide `hf_to_cornstarch_prefixes` for every state-dict prefix that moved.
   Cornstarch imports and exports Hugging Face-shaped keys, so this mapping is
   the checkpoint compatibility contract.
5. Implement a `TransformerForwardSpec` when the family needs custom native
   forward behavior such as masks, rotary embeddings, pooling, MoE routing, or
   loss/logit post-processing.
6. Register the converter in `cornstarch.models.hf_conversion.from_hf_config`
   and export it from `cornstarch.models.conversions`.
7. Add focused integration coverage in `tests/model/test_model_conversion.py`
   and materialization coverage in `tests/model/test_model_materialization.py`.

Keep converters lazy: construction should leave parameters and buffers on
`meta`. Do not store a root Hugging Face model on the Cornstarch object. Reuse
Hugging Face submodules as leaves when useful, but keep the visible repeated
layer stack as a Cornstarch `nn.ModuleList` so later materialization/offload can
operate layer by layer.
