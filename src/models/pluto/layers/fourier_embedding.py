# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import List, Optional

import torch
import torch.nn as nn


class FourierEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_freq_bands: int) -> None:
        super(FourierEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 创建[input_dim:3, num_freq_bands:64]的可学习频率参数
        # 注意：与传统固定频率序列（base_freq * 2^i * 2*pi）不同，
        # 这里使用可学习的 Embedding，允许模型自适应学习最优频率模式
        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(input_dim)
            ]
        )
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        continuous_inputs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # [4, 215, 3, 64] = [4, 215, 3, 1] * [3, 64]
        # 使用可学习的频率权重（而非传统的固定 2^i 指数序列）
        x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi

        # [4， 215， 3， 64+64+1]
        x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
        continuous_embs: List[Optional[torch.Tensor]] = [None] * self.input_dim
        for i in range(self.input_dim):
            # [B, T, hidden_dim] = [B, T, 2 * num_freq_bands + 1] * [num_freq_bands + 1, hidden_dim]
            continuous_embs[i] = self.mlps[i](x[..., i, :])
        # [4, 215, 128]
        x = torch.stack(continuous_embs).sum(dim=0)
        # import pdb; pdb.set_trace()
        return self.to_out(x)
