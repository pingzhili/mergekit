# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/14

from .configuration_mixtral_block_level import (
    MixtralLmLevelConfig
)
from .modeling_mixtral_block_level import (
    MixtralLmLevelModel,
    MixtralLmLevelForCausalLM,
    MixtralLmLevelSparseDecoderLayer
)

MixtralLmLevelConfig.register_for_auto_class("AutoConfig")
MixtralLmLevelModel.register_for_auto_class("AutoModel")
MixtralLmLevelForCausalLM.register_for_auto_class("AutoModelForCausalLM")
