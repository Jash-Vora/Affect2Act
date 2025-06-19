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
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import random
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict

from models.emotion_gat import EmotionGAT
from models.decision_nn import DecisionMakingNN
from data.data_generator import SyntheticDataGenerator


class EmotionModulationSystem:
    """
    Fixed emotion modulation system addressing key learning issues
    """
    def __init__(self):
        self.emotion_gat = EmotionGAT(node_features=12, num_heads=4)
        self.decision_nn = DecisionMakingNN(input_dim=6)
        
        self.data_generator = SyntheticDataGenerator()
        
        self.emotions = ['joy', 'fear', 'anger', 'sadness', 'surprise', 'trust']
        self.training_history = {
            'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [],
            'emotion_entropy': [], 'emotion_balance': [], 'emotion_variance': []
        }
    
    def create_emotion_graph(self, environment_features, emotion_intensity):
        """Create enhanced emotion graph with better connectivity"""
        num_nodes = len(self.emotions)
        
        # Enhanced node features
        node_features = torch.zeros(num_nodes, 12)
        for i, intensity in enumerate(emotion_intensity):
            node_features[i, :6] = environment_features
            node_features[i, 6] = intensity
            node_features[i, 7] = i / (num_nodes - 1)
            
            # Emotion-specific profiles
            emotion_profiles = {
                0: [0.8, 0.2, 0.9, 0.3],    # joy: high valence, low arousal
                1: [0.1, 0.9, 0.2, 0.8],    # fear: low valence, high arousal
                2: [0.1, 0.8, 0.6, 0.7],    # anger: low valence, high arousal
                3: [0.2, 0.3, 0.1, 0.4],    # sadness: low valence, low arousal
                4: [0.5, 0.7, 0.5, 0.9],    # surprise: neutral valence, high arousal
                5: [0.7, 0.3, 0.8, 0.6]     # trust: high valence, moderate arousal
            }
            node_features[i, 8:] = torch.tensor(emotion_profiles[i])
        
        # FIX 11: Enhanced edge connectivity for better emotion interactions
        emotion_relationships = [
            # Primary relationships
            (0, 5), (5, 0),  # joy-trust
            (0, 4), (4, 0),  # joy-surprise
            (1, 2), (2, 1),  # fear-anger
            (1, 3), (3, 1),  # fear-sadness
            (2, 3), (3, 2),  # anger-sadness
            
            # Secondary relationships
            (0, 1), (1, 0),  # joy can counter fear
            (0, 2), (2, 0),  # joy can counter anger
            (0, 3), (3, 0),  # joy can counter sadness
            (5, 1), (1, 5),  # trust-fear interaction
            (5, 2), (2, 5),  # trust-anger interaction
            (4, 1), (1, 4),  # surprise-fear
            (4, 2), (2, 4),  # surprise-anger
            (4, 3), (3, 4),  # surprise-sadness
            
            # Self-loops for stability
            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)
        ]
        
        edge_index = torch.tensor(emotion_relationships, dtype=torch.long).t().contiguous()
        return Data(x=node_features, edge_index=edge_index)
    
    def advanced_emotion_regularization(self, emotion_repr):
        """Enhanced regularization - FIXED to not use validation data statistics"""
        # Normalize emotions for analysis
        emotion_probs = F.softmax(emotion_repr, dim=1)
        
        # 1. Entropy regularization (prevent collapse to single emotion)
        entropy = -torch.sum(emotion_probs * torch.log(emotion_probs + 1e-8), dim=1)
        # FIX: Use theoretical target, not validation-derived target
        target_entropy = np.log(self.emotion_gat.num_emotions) * 0.7  # 70% of max entropy
        entropy_loss = F.mse_loss(entropy, torch.full_like(entropy, target_entropy))
        
        # 2. Balance regularization - use theoretical uniform distribution
        mean_probs = torch.mean(emotion_probs, dim=0)
        uniform_target = torch.full_like(mean_probs, 1.0/self.emotion_gat.num_emotions)
        balance_loss = F.kl_div(
            torch.log(mean_probs + 1e-8), uniform_target, reduction='batchmean'
        )
        
        # 3. Variance regularization with fixed theoretical minimum
        if emotion_probs.size(0) > 1:
            emotion_variance = torch.var(emotion_probs, dim=0)
            # FIX: Use theoretical minimum, not data-derived
            min_variance = 0.01  # Fixed theoretical minimum
            variance_loss = F.relu(min_variance - emotion_variance).mean()
        else:
            variance_loss = torch.tensor(0.0, device=emotion_repr.device)
        
        # 4. Correlation regularization with fixed penalty
        if emotion_probs.size(0) > 1:
            correlation_matrix = torch.corrcoef(emotion_probs.T)
            off_diagonal = correlation_matrix - torch.eye(
                correlation_matrix.size(0), device=correlation_matrix.device
            )
            correlation_loss = torch.mean(torch.abs(off_diagonal))
        else:
            correlation_loss = torch.tensor(0.0, device=emotion_repr.device)
        
        total_reg_loss = (
            entropy_loss + 
            0.3 * balance_loss + 
            0.2 * variance_loss + 
            0.1 * correlation_loss
        )
        
        return total_reg_loss, {
            'entropy': entropy.mean().item(),
            'balance': 1.0 - torch.max(mean_probs).item(),
            'variance': emotion_variance.mean().item() if emotion_probs.size(0) > 1 else 0.0
        }
    
    def train_model(self, num_epochs=80, batch_size=32, total_samples=2000, lr=0.001, seed=42):
        print("=== Training Fixed GAT-Based Emotion Modulation System ===")
        print("Key fixes implemented:")
        print("✓ Proper train/val/test splits with no data leakage")
        print("✓ Separate seeds for each dataset split")
        print("✓ No validation statistics used in training")
        print("✓ Fixed theoretical targets for regularization\n")
        
        # FIX: Generate properly split datasets
        datasets = self.data_generator.generate_dataset_with_split(
            total_samples=total_samples, 
            train_ratio=0.7, 
            val_ratio=0.15, 
            test_ratio=0.15,
            seed=seed
        )
        
        train_data = datasets['train']
        val_data = datasets['val']
        # Store test data separately - don't use during training!
        self.test_data = datasets['test']
        
        print(f"Dataset sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(self.test_data)}")
        
        # Better optimizer configuration with weight decay
        optimizer = torch.optim.AdamW([
            {'params': self.emotion_gat.parameters(), 'lr': lr * 0.5, 'weight_decay': 1e-4},
            {'params': self.decision_nn.parameters(), 'lr': lr, 'weight_decay': 1e-5}
        ])
        
        # FIX: Use deterministic scheduler (no validation-based adjustments during training)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=lr * 0.01
        )
        
        best_val_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            self.emotion_gat.train()
            self.decision_nn.train()
            
            # FIX: Set epoch-specific seed for reproducible shuffling
            random.seed(seed + epoch)
            random.shuffle(train_data)
            
            epoch_metrics = {
                'loss': 0.0, 'accuracy': 0.0, 'emotion_entropy': 0.0,
                'emotion_balance': 0.0, 'emotion_variance': 0.0
            }
            
            num_batches = 0
            
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i+batch_size]
                
                optimizer.zero_grad()
                
                batch_emotion_reprs = []
                batch_losses = []
                batch_predictions = []
                batch_targets = []
                
                for sample in batch_data:
                    emotion_graph = self.create_emotion_graph(
                        sample['environment'], sample['emotions']
                    )
                    
                    # Forward pass through GAT
                    emotion_repr = self.emotion_gat(
                        emotion_graph.x, emotion_graph.edge_index
                    )
                    batch_emotion_reprs.append(emotion_repr)
                    
                    # Forward pass through decision network
                    decision_input = sample['environment'].unsqueeze(0)
                    decision_output, _ = self.decision_nn(decision_input, emotion_repr)
                    
                    # Compute losses
                    target = torch.tensor([sample['target_decision']], dtype=torch.long)
                    classification_loss = F.cross_entropy(decision_output, target)
                    
                    batch_losses.append(classification_loss)
                    batch_predictions.append(torch.argmax(decision_output, dim=1).item())
                    batch_targets.append(sample['target_decision'])
                
                # Compute regularization on batch of emotion representations
                if batch_emotion_reprs:
                    combined_emotions = torch.cat(batch_emotion_reprs, dim=0)
                    reg_loss, reg_metrics = self.advanced_emotion_regularization(combined_emotions)
                    
                    # Total loss
                    classification_loss = torch.stack(batch_losses).mean()
                    total_loss = classification_loss + 0.1 * reg_loss
                    
                    # Backward pass
                    total_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        list(self.emotion_gat.parameters()) + 
                        list(self.decision_nn.parameters()), 
                        max_norm=1.0
                    )
                    
                    optimizer.step()
                    
                    # Update metrics
                    epoch_metrics['loss'] += total_loss.item()
                    epoch_metrics['accuracy'] += sum([p == t for p, t in zip(batch_predictions, batch_targets)]) / len(batch_targets)
                    epoch_metrics['emotion_entropy'] += reg_metrics['entropy']
                    epoch_metrics['emotion_balance'] += reg_metrics['balance']
                    epoch_metrics['emotion_variance'] += reg_metrics['variance']
                    
                    num_batches += 1
            
            # Average metrics
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
            
            # FIX: Validation evaluation (properly separated)
            val_loss, val_accuracy, _, _ = self.evaluate_model(val_data)
            
            # Update scheduler (deterministic, not validation-dependent)
            scheduler.step()
            
            # Store history
            self.training_history['loss'].append(epoch_metrics['loss'])
            self.training_history['accuracy'].append(epoch_metrics['accuracy'])
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            self.training_history['emotion_entropy'].append(epoch_metrics['emotion_entropy'])
            self.training_history['emotion_balance'].append(epoch_metrics['emotion_balance'])
            self.training_history['emotion_variance'].append(epoch_metrics['emotion_variance'])
            
            # Early stopping based on validation (this is acceptable)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_best_model()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Progress reporting
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs}")
                print(f"  Train - Loss: {epoch_metrics['loss']:.4f}, Acc: {epoch_metrics['accuracy']:.4f}")
                print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}")
                print(f"  Emotion Metrics:")
                print(f"    Entropy: {epoch_metrics['emotion_entropy']:.4f} (target: {np.log(6)*0.7:.4f})")
                print(f"    Balance: {epoch_metrics['emotion_balance']:.4f}")
                print(f"    Variance: {epoch_metrics['emotion_variance']:.4f}")
                print(f"  Best Val Acc: {best_val_accuracy:.4f}")
                print(f"  Temperature: {self.emotion_gat.temperature.item():.4f}")
                print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if patience_counter >= 15:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_accuracy:.4f}")
        print("Test data reserved for final evaluation only.")
        return self.training_history
    
    def evaluate_model(self, data):
        """Comprehensive model evaluation"""
        self.emotion_gat.eval()
        self.decision_nn.eval()
        
        total_loss = 0.0
        predictions = []
        targets = []
        all_emotion_reprs = []
        
        with torch.no_grad():
            for sample in data:
                emotion_graph = self.create_emotion_graph(
                    sample['environment'], sample['emotions']
                )
                
                emotion_repr = self.emotion_gat(
                    emotion_graph.x, emotion_graph.edge_index
                )
                
                decision_input = sample['environment'].unsqueeze(0)
                decision_output, _ = self.decision_nn(decision_input, emotion_repr)
                
                target_decision = torch.tensor([sample['target_decision']], dtype=torch.long)
                loss = F.cross_entropy(decision_output, target_decision)
                total_loss += loss.item()
                
                pred = torch.argmax(decision_output, dim=1).item()
                predictions.append(pred)
                targets.append(sample['target_decision'])
                all_emotion_reprs.append(emotion_repr)
        
        avg_loss = total_loss / len(data)
        accuracy = accuracy_score(targets, predictions)
        
        # Compute emotion representation metrics
        if all_emotion_reprs:
            combined_emotions = torch.cat(all_emotion_reprs, dim=0)
            emotion_probs = F.softmax(combined_emotions, dim=1)
            
            # Entropy analysis
            entropy = -torch.sum(emotion_probs * torch.log(emotion_probs + 1e-8), dim=1)
            avg_entropy = entropy.mean().item()
            
            # Balance analysis
            mean_probs = torch.mean(emotion_probs, dim=0)
            balance_score = 1.0 - torch.max(mean_probs).item()
        else:
            avg_entropy = 0.0
            balance_score = 0.0
        
        return avg_loss, accuracy, avg_entropy, balance_score

    def save_best_model(self):
        """Save the best model state"""
        torch.save({
            'emotion_gat_state_dict': self.emotion_gat.state_dict(),
            'decision_nn_state_dict': self.decision_nn.state_dict(),
            'training_history': self.training_history
        }, 'best_emotion_model.pth')

    def load_model(self, path):
        """Load a saved model"""
        checkpoint = torch.load(path)
        self.emotion_gat.load_state_dict(checkpoint['emotion_gat_state_dict'])
        self.decision_nn.load_state_dict(checkpoint['decision_nn_state_dict'])
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']

    def predict_decision(self, environment_features, emotion_intensity):
        """Make a prediction for new input"""
        self.emotion_gat.eval()
        self.decision_nn.eval()
        
        with torch.no_grad():
            # Create emotion graph
            emotion_graph = self.create_emotion_graph(environment_features, emotion_intensity)
            
            # Get emotion representation
            emotion_repr = self.emotion_gat(emotion_graph.x, emotion_graph.edge_index)
            
            # Make decision
            decision_input = environment_features.unsqueeze(0)
            decision_output, attention_weights = self.decision_nn(decision_input, emotion_repr)
            
            # Get prediction and confidence
            probabilities = decision_output.squeeze(0)
            predicted_decision = torch.argmax(probabilities).item()
            confidence = torch.max(probabilities).item()
            
            # Get emotion analysis
            emotion_probs = F.softmax(emotion_repr, dim=1).squeeze(0)
            
            return {
                'predicted_decision': predicted_decision,
                'confidence': confidence,
                'decision_probabilities': probabilities.numpy(),
                'emotion_activations': emotion_probs.numpy(),
                'dominant_emotions': [self.emotions[i] for i in torch.topk(emotion_probs, 3).indices.numpy()],
                'attention_weights': attention_weights
            }

    def analyze_emotion_patterns(self, data):
        """Analyze emotion patterns in the data"""
        self.emotion_gat.eval()
        
        emotion_patterns = defaultdict(list)
        scenario_emotions = defaultdict(list)
        
        with torch.no_grad():
            for sample in data:
                emotion_graph = self.create_emotion_graph(
                    sample['environment'], sample['emotions']
                )
                
                emotion_repr = self.emotion_gat(emotion_graph.x, emotion_graph.edge_index)
                emotion_probs = F.softmax(emotion_repr, dim=1).squeeze(0)
                
                # Store patterns by decision
                decision = sample['target_decision']
                emotion_patterns[decision].append(emotion_probs.numpy())
                
                # Store patterns by scenario type if available
                if 'scenario_type' in sample:
                    scenario_emotions[sample['scenario_type']].append(emotion_probs.numpy())
        
        # Compute average patterns
        decision_patterns = {}
        for decision, patterns in emotion_patterns.items():
            if patterns:
                decision_patterns[decision] = np.mean(patterns, axis=0)
        
        scenario_patterns = {}
        for scenario, patterns in scenario_emotions.items():
            if patterns:
                scenario_patterns[scenario] = np.mean(patterns, axis=0)
        
        return decision_patterns, scenario_patterns

    def plot_training_history(self):
        """Plot comprehensive training history"""
        history = self.training_history
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training History - Fixed GAT Emotion System', fontsize=16)
        
        # Loss plot
        axes[0,0].plot(history['loss'], label='Train Loss', alpha=0.7)
        axes[0,0].plot(history['val_loss'], label='Val Loss', alpha=0.7)
        axes[0,0].set_title('Loss Over Time')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0,1].plot(history['accuracy'], label='Train Acc', alpha=0.7)
        axes[0,1].plot(history['val_accuracy'], label='Val Acc', alpha=0.7)
        axes[0,1].set_title('Accuracy Over Time')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Emotion entropy
        axes[0,2].plot(history['emotion_entropy'], label='Entropy', alpha=0.7)
        axes[0,2].axhline(y=np.log(6)*0.7, color='r', linestyle='--', label='Target')
        axes[0,2].set_title('Emotion Entropy')
        axes[0,2].set_xlabel('Epoch')
        axes[0,2].set_ylabel('Entropy')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # Emotion balance
        axes[1,0].plot(history['emotion_balance'], alpha=0.7)
        axes[1,0].set_title('Emotion Balance Score')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Balance (1 = perfect)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Emotion variance
        axes[1,1].plot(history['emotion_variance'], alpha=0.7)
        axes[1,1].set_title('Emotion Variance')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Variance')
        axes[1,1].grid(True, alpha=0.3)
        
        # Combined metric
        if len(history['accuracy']) > 0 and len(history['emotion_entropy']) > 0:
            combined_score = np.array(history['val_accuracy']) * np.array(history['emotion_balance'])
            axes[1,2].plot(combined_score, alpha=0.7)
            axes[1,2].set_title('Combined Performance Score')
            axes[1,2].set_xlabel('Epoch')
            axes[1,2].set_ylabel('Accuracy × Balance')
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_emotion_analysis(self, test_data):
        """Plot comprehensive emotion analysis"""
        decision_patterns, scenario_patterns = self.analyze_emotion_patterns(test_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Emotion Pattern Analysis', fontsize=16)
        
        # Decision-based emotion patterns
        if decision_patterns:
            decision_matrix = np.array([decision_patterns[i] for i in sorted(decision_patterns.keys())])
            im1 = axes[0,0].imshow(decision_matrix, cmap='viridis', aspect='auto')
            axes[0,0].set_title('Emotion Patterns by Decision')
            axes[0,0].set_xlabel('Emotions')
            axes[0,0].set_ylabel('Decision Type')
            axes[0,0].set_xticks(range(len(self.emotions)))
            axes[0,0].set_xticklabels(self.emotions, rotation=45)
            axes[0,0].set_yticks(range(len(decision_patterns)))
            axes[0,0].set_yticklabels([f'Decision {i}' for i in sorted(decision_patterns.keys())])
            plt.colorbar(im1, ax=axes[0,0])
        
        # Scenario-based emotion patterns
        if scenario_patterns:
            scenario_matrix = np.array(list(scenario_patterns.values()))
            im2 = axes[0,1].imshow(scenario_matrix, cmap='plasma', aspect='auto')
            axes[0,1].set_title('Emotion Patterns by Scenario')
            axes[0,1].set_xlabel('Emotions')
            axes[0,1].set_ylabel('Scenario Type')
            axes[0,1].set_xticks(range(len(self.emotions)))
            axes[0,1].set_xticklabels(self.emotions, rotation=45)
            axes[0,1].set_yticks(range(len(scenario_patterns)))
            axes[0,1].set_yticklabels(list(scenario_patterns.keys()), rotation=0)
            plt.colorbar(im2, ax=axes[0,1])
        
        # Sample predictions visualization
        sample_results = []
        for i, sample in enumerate(test_data[:20]):  # First 20 samples
            result = self.predict_decision(sample['environment'], sample['emotions'])
            sample_results.append({
                'predicted': result['predicted_decision'],
                'actual': sample['target_decision'],
                'confidence': result['confidence'],
                'emotions': result['emotion_activations']
            })
        
        # Prediction accuracy scatter
        predicted = [r['predicted'] for r in sample_results]
        actual = [r['actual'] for r in sample_results]
        confidence = [r['confidence'] for r in sample_results]
        
        scatter = axes[1,0].scatter(actual, predicted, c=confidence, cmap='coolwarm', alpha=0.7)
        axes[1,0].set_title('Prediction vs Actual (colored by confidence)')
        axes[1,0].set_xlabel('Actual Decision')
        axes[1,0].set_ylabel('Predicted Decision')
        axes[1,0].set_xticks([0, 1, 2])
        axes[1,0].set_yticks([0, 1, 2])
        plt.colorbar(scatter, ax=axes[1,0])
        
        # Emotion activation distribution
        all_emotions = np.array([r['emotions'] for r in sample_results])
        axes[1,1].boxplot(all_emotions, labels=self.emotions)
        axes[1,1].set_title('Emotion Activation Distribution')
        axes[1,1].set_xlabel('Emotions')
        axes[1,1].set_ylabel('Activation Level')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

    def generate_emotion_report(self, test_data):
        """Generate comprehensive emotion analysis report"""
        print("=== FIXED GAT EMOTION SYSTEM - ANALYSIS REPORT ===\n")
        
        # Evaluate model
        test_loss, test_accuracy, avg_entropy, balance_score = self.evaluate_model(test_data)
        
        print(f"Model Performance:")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Emotion Entropy: {avg_entropy:.4f} (target: {np.log(6)*0.7:.4f})")
        print(f"  Emotion Balance: {balance_score:.4f}")
        
        # Analyze patterns
        decision_patterns, scenario_patterns = self.analyze_emotion_patterns(test_data)
        
        print(f"\nDecision-Emotion Patterns:")
        for decision, pattern in decision_patterns.items():
            dominant_emotions = np.argsort(pattern)[-3:][::-1]
            print(f"  Decision {decision}: {', '.join([f'{self.emotions[i]} ({pattern[i]:.3f})' for i in dominant_emotions])}")
        
        if scenario_patterns:
            print(f"\nScenario-Emotion Patterns:")
            for scenario, pattern in scenario_patterns.items():
                dominant_emotions = np.argsort(pattern)[-2:][::-1]
                print(f"  {scenario}: {', '.join([f'{self.emotions[i]} ({pattern[i]:.3f})' for i in dominant_emotions])}")
        
        # Sample predictions
        print(f"\nSample Predictions:")
        for i, sample in enumerate(test_data[:5]):
            result = self.predict_decision(sample['environment'], sample['emotions'])
            print(f"  Sample {i+1}:")
            print(f"    Predicted: {result['predicted_decision']} (confidence: {result['confidence']:.3f})")
            print(f"    Actual: {sample['target_decision']}")
            print(f"    Top emotions: {', '.join(result['dominant_emotions'])}")
        
        return {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'emotion_entropy': avg_entropy,
            'emotion_balance': balance_score,
            'decision_patterns': decision_patterns,
            'scenario_patterns': scenario_patterns
        }
    def final_test_evaluation(self):
        """Perform final test evaluation ONLY after training is complete"""
        if not hasattr(self, 'test_data'):
            raise ValueError("No test data available. Run train_model first.")
        
        print("=== FINAL TEST EVALUATION (Data Never Seen During Training) ===")
        
        # Load best model
        try:
            self.load_model('best_emotion_model.pth')
            print("Loaded best model from training.")
        except:
            print("Using current model (best model file not found).")
        
        # Evaluate on test data
        test_loss, test_accuracy, avg_entropy, balance_score = self.evaluate_model(self.test_data)
        
        print(f"\nFinal Test Results:")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Emotion Entropy: {avg_entropy:.4f}")
        print(f"  Emotion Balance: {balance_score:.4f}")
        
        # Generate comprehensive report
        report = self.generate_emotion_report(self.test_data)
        
        return report