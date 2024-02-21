# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/14

from .configuration_mixtral_hybrid import (
    MixtralBlockLevelConfig
)
from .modeling_mixtral_hybrid import (
    MixtralBlockLevelModel,
    MixtralBlockLevelConfig,
    MixtralBlockLevelForCausalLM,
    MixtralBlockLevelDecoderLayer
)

MixtralBlockLevelConfig.register_for_auto_class("AutoConfig")
MixtralBlockLevelModel.register_for_auto_class("AutoModel")
MixtralBlockLevelForCausalLM.register_for_auto_class("AutoModelForCausalLM")
