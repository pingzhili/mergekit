# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/14
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig

from .configuration_mixtral_hybrid import (
    MixtralHybridConfig
)
from .modeling_mixtral_hybrid import (
    MixtralHybridModel,
    MixtralHybridConfig,
    MixtralHybridForCausalLM,
    MixtralHybridDecoderLayer
)

AutoConfig.register("mixtral_hybrid", MixtralHybridConfig)
AutoModel.register(MixtralHybridConfig, MixtralHybridModel)
AutoModelForCausalLM.register(MixtralHybridConfig, MixtralHybridForCausalLM)
