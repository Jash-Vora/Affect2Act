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
import numpy as np
import random


class SyntheticDataGenerator:
    """
    Enhanced data generator with more diverse and realistic emotion patterns
    """
    def __init__(self, num_emotions=6):
        self.num_emotions = num_emotions
        self.emotions = ['joy', 'fear', 'anger', 'sadness', 'surprise', 'trust']
        self.base_seed = None
        
        # FIX 6: More diverse and realistic emotion patterns
        self.scenario_templates = {
            'emergency': {
                'environment': [0.9, 0.1, 0.8, 0.2, 0.9, 0.1],
                'emotion_patterns': [
                    [0.1, 0.8, 0.6, 0.3, 0.7, 0.2],  # High fear, some anger
                    [0.0, 0.9, 0.4, 0.5, 0.8, 0.1],  # Very high fear, surprise
                    [0.2, 0.7, 0.8, 0.2, 0.6, 0.3],  # Fear + anger mix
                ],
                'preferred_decision': 0,
                'variance': 0.15
            },
            'social': {
                'environment': [0.2, 0.9, 0.2, 0.8, 0.3, 0.8],
                'emotion_patterns': [
                    [0.9, 0.1, 0.05, 0.05, 0.4, 0.8],  # High joy, trust
                    [0.7, 0.2, 0.1, 0.1, 0.6, 0.9],   # Joy + surprise + trust
                    [0.8, 0.0, 0.0, 0.2, 0.3, 0.9],   # Joy + trust, some sadness
                ],
                'preferred_decision': 1,
                'variance': 0.12
            },
            'analytical': {
                'environment': [0.3, 0.4, 0.4, 0.7, 0.5, 0.7],
                'emotion_patterns': [
                    [0.4, 0.2, 0.1, 0.1, 0.2, 0.9],   # High trust, low arousal
                    [0.5, 0.3, 0.2, 0.2, 0.4, 0.8],   # Balanced but trust-dominant
                    [0.3, 0.1, 0.0, 0.3, 0.1, 0.9],   # Trust + some contemplation
                ],
                'preferred_decision': 2,
                'variance': 0.1
            },
            'conflicted': {
                'environment': [0.6, 0.5, 0.8, 0.3, 0.7, 0.4],
                'emotion_patterns': [
                    [0.2, 0.6, 0.7, 0.5, 0.4, 0.2],   # Mixed negative
                    [0.1, 0.5, 0.8, 0.6, 0.3, 0.1],   # Anger + sadness dominant
                    [0.3, 0.7, 0.6, 0.4, 0.5, 0.3],   # Fear + anger + some joy
                ],
                'preferred_decision': 1,
                'variance': 0.2
            },
            'creative': {
                'environment': [0.4, 0.6, 0.3, 0.8, 0.6, 0.6],
                'emotion_patterns': [
                    [0.8, 0.2, 0.1, 0.1, 0.9, 0.6],   # Joy + high surprise
                    [0.7, 0.3, 0.2, 0.2, 0.8, 0.7],   # Joy + surprise + trust
                    [0.6, 0.1, 0.0, 0.3, 0.9, 0.5],   # Surprise dominant
                ],
                'preferred_decision': 2,
                'variance': 0.15
            },
            'supportive': {
                'environment': [0.2, 0.8, 0.4, 0.7, 0.3, 0.8],
                'emotion_patterns': [
                    [0.6, 0.2, 0.1, 0.4, 0.2, 0.9],   # Trust + empathetic sadness
                    [0.7, 0.3, 0.2, 0.5, 0.3, 0.8],   # Joy + sadness + trust
                    [0.4, 0.1, 0.0, 0.6, 0.1, 0.9],   # High trust, empathetic
                ],
                'preferred_decision': 1,
                'variance': 0.12
            }
        }
    
    def set_seed(self, seed):
        """Set seed for reproducible data generation"""
        self.base_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_sample(self, scenario_type=None):
        if scenario_type is None:
            scenario_type = random.choice(list(self.scenario_templates.keys()))
        
        template = self.scenario_templates[scenario_type]
        
        # Generate environment with controlled noise
        environment = torch.tensor(template['environment'], dtype=torch.float32)
        env_noise = torch.randn(6) * template['variance'] * 0.3
        environment = torch.clamp(environment + env_noise, 0.0, 1.0)
        
        # FIX 7: Select from multiple emotion patterns for diversity
        emotion_pattern = random.choice(template['emotion_patterns'])
        base_emotions = torch.tensor(emotion_pattern, dtype=torch.float32)
        
        # Add controlled noise
        emotion_noise = torch.randn(self.num_emotions) * template['variance'] * 0.5
        emotions = torch.clamp(base_emotions + emotion_noise, 0.02, 1.0)
        
        # Ensure some emotional activation
        if torch.max(emotions) < 0.25:
            emotions = emotions + 0.15
            emotions = torch.clamp(emotions, 0.0, 1.0)
        
        # Target decision with some controlled randomness
        if random.random() < 0.85:
            target_decision = template['preferred_decision']
        else:
            target_decision = random.randint(0, 2)
        
        return {
            'environment': environment,
            'emotions': emotions,
            'target_decision': target_decision,
            'scenario_type': scenario_type
        }
    
    def generate_dataset(self, num_samples, scenario_distribution=None):
        if scenario_distribution is None:
            scenario_distribution = {
                'emergency': 0.18,
                'social': 0.18,
                'analytical': 0.18,
                'conflicted': 0.18,
                'creative': 0.14,
                'supportive': 0.14
            }
        
        dataset = []
        for _ in range(num_samples):
            rand_val = random.random()
            cumulative = 0
            scenario_type = 'emergency'
            
            for scenario, prob in scenario_distribution.items():
                cumulative += prob
                if rand_val <= cumulative:
                    scenario_type = scenario
                    break
            
            sample = self.generate_sample(scenario_type)
            dataset.append(sample)
        
        return dataset
    
    def generate_dataset_with_split(self, total_samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """Generate dataset with proper train/val/test splits to prevent leakage"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Set seed for reproducible splits
        self.set_seed(seed)
        
        # Calculate split sizes
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        test_size = total_samples - train_size - val_size
        
        # Generate all data with different seeds to ensure no overlap
        datasets = {}
        
        # Train set with base seed
        self.set_seed(seed)
        datasets['train'] = self.generate_dataset(train_size)
        
        # Validation set with different seed
        self.set_seed(seed + 1000)
        datasets['val'] = self.generate_dataset(val_size)
        
        # Test set with different seed
        self.set_seed(seed + 2000)
        datasets['test'] = self.generate_dataset(test_size)
        
        return datasets