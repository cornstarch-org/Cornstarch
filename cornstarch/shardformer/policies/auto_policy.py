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
    "cornstarch.models.multimodal_language_model.modeling_multimodal_language_model.ModalModule": PolicyLocation(
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
    "transformers.models.vit.modeling_vit.ViTForImageClassification": PolicyLocation(
        file_name="vit", class_name="ViTForImageClassificationPolicy"
    ),
    "transformers.models.vit.modeling_vit.ViTForMaskedImageModeling": PolicyLocation(
        file_name="vit", class_name="ViTForMaskedImageModelingPolicy"
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
