# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/14

from .configuration_mixtral_hybrid import (
    MixtralHybridConfig
)
from .modeling_mixtral_hybrid import (
    MixtralHybridModel,
    MixtralHybridConfig,
    MixtralHybridForCausalLM,
    MixtralHybridDecoderLayer
)

MixtralHybridConfig.register_for_auto_class("AutoConfig")
MixtralHybridModel.register_for_auto_class("AutoModel")
MixtralHybridForCausalLM.register_for_auto_class("AutoModelForCausalLM")
