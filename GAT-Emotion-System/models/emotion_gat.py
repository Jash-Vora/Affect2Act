# Copyright 2025 Jash Vora
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class EmotionGAT(nn.Module):
    """
    Fixed GAT-based emotion network addressing key representation learning issues
    """
    def __init__(self, node_features=12, hidden_dim=64, num_emotions=6, num_heads=4):
        super(EmotionGAT, self).__init__()
        self.num_emotions = num_emotions
        self.hidden_dim = hidden_dim
        
        # Multi-head GAT layers with proper dimensions
        self.gat1 = GATConv(
            node_features, hidden_dim // num_heads, 
            heads=num_heads, dropout=0.2, concat=True
        )
        self.gat2 = GATConv(
            hidden_dim, hidden_dim // num_heads, 
            heads=num_heads, dropout=0.2, concat=True
        )
        
        # Final GAT layer that outputs emotion-specific representations
        self.gat3 = GATConv(
            hidden_dim, hidden_dim // 2, 
            heads=2, dropout=0.2, concat=False
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        
        # FIX 1: Proper emotion-specific transformations
        # Instead of reducing to 1D, maintain rich representations
        self.emotion_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.LayerNorm(hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 4, hidden_dim // 4)  # Keep rich representation
            ) for _ in range(num_emotions)
        ])
        
        # FIX 2: Proper emotion aggregation that preserves information
        self.emotion_aggregator = nn.Sequential(
            nn.Linear(num_emotions * (hidden_dim // 4), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_emotions)  # Final emotion representation
        )
        
        # FIX 3: Learnable emotion interaction weights
        self.emotion_interaction = nn.Parameter(torch.randn(num_emotions, num_emotions) * 0.1)
        
        # Temperature parameter for sharpness control
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)  # Start with higher temperature
        
    def forward(self, x, edge_index, batch=None):
        # Apply GAT layers with proper residual connections
        x1 = self.gat1(x, edge_index)
        x1 = self.ln1(x1)
        x1 = F.elu(x1)
        
        x2 = self.gat2(x1, edge_index)
        x2 = self.ln2(x2)
        x2 = F.elu(x2)
        x2 = x2 + x1  # Residual connection
        
        x3 = self.gat3(x2, edge_index)
        x3 = self.ln3(x3)
        x3 = F.elu(x3)
        
        # FIX 4: Proper batching and emotion processing
        if batch is None:
            # Single graph case - reshape to process each emotion node
            batch_size = 1
            emotion_features = x3.view(self.num_emotions, -1)
        else:
            # Multiple graphs case
            batch_size = torch.max(batch) + 1
            emotion_features = []
            for b in range(batch_size):
                mask = batch == b
                batch_emotions = x3[mask]
                if batch_emotions.size(0) == self.num_emotions:
                    emotion_features.append(batch_emotions)
                else:
                    # Handle incomplete batches
                    padded = torch.zeros(self.num_emotions, x3.size(1), device=x3.device)
                    padded[:batch_emotions.size(0)] = batch_emotions
                    emotion_features.append(padded)
            emotion_features = torch.stack(emotion_features, dim=0)
        
        # Process each emotion with its specific transformer
        transformed_emotions = []
        for i, transformer in enumerate(self.emotion_transformers):
            if batch is None:
                emotion_repr = transformer(emotion_features[i:i+1])
            else:
                emotion_repr = transformer(emotion_features[:, i, :])
            transformed_emotions.append(emotion_repr)
        
        # FIX 5: Proper emotion interaction modeling
        if batch is None:
            emotion_stack = torch.cat(transformed_emotions, dim=1)
        else:
            emotion_stack = torch.stack(transformed_emotions, dim=1)
            emotion_stack = emotion_stack.view(batch_size, -1)
        
        # Apply emotion interactions
        raw_emotions = self.emotion_aggregator(emotion_stack)
        
        # Apply learnable emotion interactions
        interaction_weights = torch.softmax(self.emotion_interaction, dim=1)
        interacted_emotions = torch.matmul(raw_emotions.unsqueeze(1), interaction_weights).squeeze(1)
        
        # Final emotion representation with temperature scaling
        final_emotions = (raw_emotions + interacted_emotions) / 2.0
        scaled_emotions = final_emotions / torch.clamp(self.temperature, min=0.1)
        
        return scaled_emotions