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


class DecisionMakingNN(nn.Module):
    """
    Enhanced decision network with better emotion integration
    """
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=3, num_emotions=6):
        super(DecisionMakingNN, self).__init__()
        self.num_emotions = num_emotions
        
        # Environment processing
        self.env_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # FIX 8: Better emotion processing
        self.emotion_encoder = nn.Sequential(
            nn.Linear(num_emotions, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # FIX 9: Improved attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.1
        )
        
        # Emotion-environment fusion
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # FIX 10: Specialized decision pathways
        self.decision_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, emotion_repr):
        # Process inputs
        env_features = self.env_encoder(x)
        emotion_features = self.emotion_encoder(emotion_repr)
        
        # Apply attention between environment and emotions
        env_query = env_features.unsqueeze(1)
        emotion_kv = emotion_features.unsqueeze(1)
        
        attended_env, attention_weights = self.attention(
            env_query, emotion_kv, emotion_kv
        )
        attended_env = attended_env.squeeze(1)
        
        # Fuse features
        combined = torch.cat([attended_env, emotion_features], dim=1)
        fused_features = self.fusion_layers(combined)
        
        # Generate decision
        decision_logits = self.decision_head(fused_features)
        return F.softmax(decision_logits, dim=1), attention_weights