# Code copied from https://github.com/hpcaitech/ColossalAI/blob/v0.3.5/colossalai/shardformer/policies/auto_policy.py

import importlib
from typing import Type

from colossalai.shardformer.policies.auto_policy import PolicyLocation
from colossalai.shardformer.policies.base_policy import Policy

__all__ = ["get_policy_type", "get_autopolicy", "import_policy"]


_POLICY_LIST = {
    # Multimodal
    "cornstarch.models.multimodal_language_model.modeling_multimodal_language_model.MultimodalProjector": PolicyLocation(
        file_name="multimodal", class_name="MultimodalProjectorPolicy"
    ),
    "cornstarch.models.multimodal_language_model.modeling_multimodal_language_model.ModalEncoderModule": PolicyLocation(
        file_name="multimodal", class_name="ModalModulePolicy"
    ),
    "cornstarch.models.multimodal_language_model.modeling_multimodal_language_model.ModalDecoderModule": PolicyLocation(
        file_name="multimodal", class_name="ModalModulePolicy"
    ),
    # BERT
    "transformers.models.bert.modeling_bert.BertModel": PolicyLocation(
        file_name="bert", class_name="BertModelPolicy"
    ),
    "transformers.models.bert.modeling_bert.BertForPreTraining": PolicyLocation(
        file_name="bert", class_name="BertForPreTrainingPolicy"
    ),
    "transformers.models.bert.modeling_bert.BertLMHeadModel": PolicyLocation(
        file_name="bert", class_name="BertLMHeadModelPolicy"
    ),
    "transformers.models.bert.modeling_bert.BertForMaskedLM": PolicyLocation(
        file_name="bert", class_name="BertForMaskedLMPolicy"
    ),
    "transformers.models.bert.modeling_bert.BertForSequenceClassification": PolicyLocation(
        file_name="bert", class_name="BertForSequenceClassificationPolicy"
    ),
    "transformers.models.bert.modeling_bert.BertForTokenClassification": PolicyLocation(
        file_name="bert", class_name="BertForTokenClassificationPolicy"
    ),
    "transformers.models.bert.modeling_bert.BertForNextSentencePrediction": PolicyLocation(
        file_name="bert", class_name="BertForNextSentencePredictionPolicy"
    ),
    "transformers.models.bert.modeling_bert.BertForMultipleChoice": PolicyLocation(
        file_name="bert", class_name="BertForMultipleChoicePolicy"
    ),
    "transformers.models.bert.modeling_bert.BertForQuestionAnswering": PolicyLocation(
        file_name="bert", class_name="BertForQuestionAnsweringPolicy"
    ),
    # LLaMA
    "transformers.models.llama.modeling_llama.LlamaModel": PolicyLocation(
        file_name="llama", class_name="LlamaModelPolicy"
    ),
    "transformers.models.llama.modeling_llama.LlamaForCausalLM": PolicyLocation(
        file_name="llama", class_name="LlamaForCausalLMPolicy"
    ),
    "transformers.models.llama.modeling_llama.LlamaForSequenceClassification": PolicyLocation(
        file_name="llama", class_name="LlamaForSequenceClassificationPolicy"
    ),
    # LLaMA4
    "transformers.models.llama4.modeling_llama4.Llama4TextModel": PolicyLocation(
        file_name="llama4", class_name="Llama4TextModelPolicy"
    ),
    "transformers.models.llama4.modeling_llama4.Llama4ForCausalLM": PolicyLocation(
        file_name="llama4", class_name="Llama4ForCausalLMPolicy"
    ),
    # GPT2
    "transformers.models.gpt2.modeling_gpt2.GPT2Model": PolicyLocation(
        file_name="gpt2", class_name="GPT2ModelPolicy"
    ),
    "transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel": PolicyLocation(
        file_name="gpt2", class_name="GPT2LMHeadModelPolicy"
    ),
    "transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModel": PolicyLocation(
        file_name="gpt2", class_name="GPT2DoubleHeadsModelPolicy"
    ),
    "transformers.models.gpt2.modeling_gpt2.GPT2ForQuestionAnswering": PolicyLocation(
        file_name="gpt2", class_name="GPT2ForQuestionAnsweringPolicy"
    ),
    "transformers.models.gpt2.modeling_gpt2.GPT2ForTokenClassification": PolicyLocation(
        file_name="gpt2", class_name="GPT2ForTokenClassificationPolicy"
    ),
    "transformers.models.gpt2.modeling_gpt2.GPT2ForSequenceClassification": PolicyLocation(
        file_name="gpt2", class_name="GPT2ForSequenceClassificationPolicy"
    ),
    # ViT
    "transformers.models.vit.modeling_vit.ViTModel": PolicyLocation(
        file_name="vit", class_name="ViTModelPolicy"
    ),
    # OPT
    "transformers.models.opt.modeling_opt.OPTModel": PolicyLocation(
        file_name="opt", class_name="OPTModelPolicy"
    ),
    "transformers.models.opt.modeling_opt.OPTForCausalLM": PolicyLocation(
        file_name="opt", class_name="OPTForCausalLMPolicy"
    ),
    "transformers.models.opt.modeling_opt.OPTForSequenceClassification": PolicyLocation(
        file_name="opt", class_name="OPTForSequenceClassificationPolicy"
    ),
    "transformers.models.opt.modeling_opt.OPTForQuestionAnswering": PolicyLocation(
        file_name="opt", class_name="OPTForQuestionAnsweringPolicy"
    ),
    # CLIP
    "transformers.models.clip.modeling_clip.CLIPVisionModel": PolicyLocation(
        file_name="clip", class_name="CLIPVisionModelPolicy"
    ),
    # EvaCLIP
    "cornstarch.models.evaclip.modeling_evaclip.EvaCLIPVisionModel": PolicyLocation(
        file_name="evaclip", class_name="EvaCLIPVisionModelPolicy"
    ),
    # InternViT
    "cornstarch.models.intern_vit.modeling_intern_vit.InternVisionModel": PolicyLocation(
        file_name="intern_vit", class_name="InternVisionModelPolicy"
    ),
    # Siglip
    "transformers.models.siglip.modeling_siglip.SiglipVisionModel": PolicyLocation(
        file_name="siglip", class_name="SiglipVisionModelPolicy"
    ),
    # Mistral
    "transformers.models.mistral.modeling_mistral.MistralModel": PolicyLocation(
        file_name="mistral", class_name="MistralModelPolicy"
    ),
    "transformers.models.mistral.modeling_mistral.MistralForCausalLM": PolicyLocation(
        file_name="mistral", class_name="MistralForCausalLMPolicy"
    ),
    # Mixtral
    "transformers.models.mixtral.modeling_mixtral.MixtralModel": PolicyLocation(
        file_name="mixtral", class_name="MixtralModelPolicy"
    ),
    "transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM": PolicyLocation(
        file_name="mixtral", class_name="MixtralForCausalLMPolicy"
    ),
    # Pixtral
    "transformers.models.pixtral.modeling_pixtral.PixtralVisionModel": PolicyLocation(
        file_name="pixtral", class_name="PixtralVisionModelPolicy"
    ),
    # DINOv2
    "transformers.models.dinov2.modeling_dinov2.Dinov2Model": PolicyLocation(
        file_name="dinov2", class_name="Dinov2ModelPolicy"
    ),
    "transformers.models.dinov2.modeling_dinov2.Dinov2ForImageClassification": PolicyLocation(
        file_name="dinov2", class_name="Dinov2ForImageClassificationPolicy"
    ),
    "transformers.models.dinov2.modeling_dinov2.Dinov2Backbone": PolicyLocation(
        file_name="dinov2", class_name="Dinov2BackbonePolicy"
    ),
    # Gemma
    "transformers.models.gemma.modeling_gemma.GemmaModel": PolicyLocation(
        file_name="gemma", class_name="GemmaModelPolicy"
    ),
    "transformers.models.gemma.modeling_gemma.GemmaForCausalLM": PolicyLocation(
        file_name="gemma", class_name="GemmaForCausalLMPolicy"
    ),
    # Gemma2
    "transformers.models.gemma2.modeling_gemma2.Gemma2Model": PolicyLocation(
        file_name="gemma2", class_name="Gemma2ModelPolicy"
    ),
    "transformers.models.gemma2.modeling_gemma2.Gemma2ForCausalLM": PolicyLocation(
        file_name="gemma2", class_name="Gemma2ForCausalLMPolicy"
    ),
    # Gemma3
    "transformers.models.gemma3.modeling_gemma3.Gemma3TextModel": PolicyLocation(
        file_name="gemma3", class_name="Gemma3TextModelPolicy"
    ),
    "transformers.models.gemma3.modeling_gemma3.Gemma3ForCausalLM": PolicyLocation(
        file_name="gemma3", class_name="Gemma3ForCausalLMPolicy"
    ),
    # Phi3
    "transformers.models.phi3.modeling_phi3.Phi3Model": PolicyLocation(
        file_name="phi3", class_name="Phi3ModelPolicy"
    ),
    "transformers.models.phi3.modeling_phi3.Phi3ForCausalLM": PolicyLocation(
        file_name="phi3", class_name="Phi3ForCausalLMPolicy"
    ),
    # Phi4
    "transformers.models.phi4_multimodal.modeling_phi4_multimodal.Phi4MultimodalAudioModel": PolicyLocation(
        file_name="phi4_multimodal", class_name="Phi4MultimodalAudioModelPolicy"
    ),
    # Qwen2
    "transformers.models.qwen2.modeling_qwen2.Qwen2Model": PolicyLocation(
        file_name="qwen2", class_name="Qwen2ModelPolicy"
    ),
    "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM": PolicyLocation(
        file_name="qwen2", class_name="Qwen2ForCausalLMPolicy"
    ),
    # Qwen2Audio
    "transformers.models.qwen2_audio.modeling_qwen2_audio.Qwen2AudioEncoder": PolicyLocation(
        file_name="qwen2_audio", class_name="Qwen2AudioEncoderPolicy"
    ),
    # Qwen2Vision
    "transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VisionTransformerPretrainedModel": PolicyLocation(
        file_name="qwen2_vision", class_name="Qwen2VisionTransformerPolicy"
    ),
    "transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration": PolicyLocation(
        file_name="qwen2", class_name="Qwen2ForCausalLMPolicy"
    ),
    # Qwen3
    "transformers.models.qwen3.modeling_qwen3.Qwen3Model": PolicyLocation(
        file_name="qwen3", class_name="Qwen3ModelPolicy"
    ),
    "transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM": PolicyLocation(
        file_name="qwen3", class_name="Qwen3ForCausalLMPolicy"
    ),
    # Qwen3Moe
    "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeModel": PolicyLocation(
        file_name="qwen3_moe", class_name="Qwen3MoeModelPolicy"
    ),
    "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeForCausalLM": PolicyLocation(
        file_name="qwen3_moe", class_name="Qwen3MoeForCausalLMPolicy"
    ),
    # Whisper
    "transformers.models.whisper.modeling_whisper.WhisperEncoder": PolicyLocation(
        file_name="whisper", class_name="WhisperEncoderPolicy"
    ),
}


def import_policy(policy_location: PolicyLocation) -> Policy:
    """
    Dynamically import a Policy class based on the policy location.
    """
    module_name = f"cornstarch.shardformer.policies.{policy_location.file_name}"
    module = importlib.import_module(module_name)
    return getattr(module, policy_location.class_name)


def get_policy_type(model_name: str) -> Type[Policy]:
    policy_location = _POLICY_LIST.get(model_name, None)
    if policy_location is None:
        raise NotImplementedError(
            f"Auto policy for {model_name} is not implemented\n. Supported models are {list(_POLICY_LIST.keys())}"
        )

    return import_policy(policy_location)


def get_autopolicy(model_name: str) -> Policy:
    r"""
    Return the auto policy for the model

    Args:
        pipeline_template (:class:`PipelineTemplate`): The pipeline template to get the corresponding policy

    Return:
        :class:`Policy`: The auto policy for the model
    """
    return get_policy_type(model_name)()
