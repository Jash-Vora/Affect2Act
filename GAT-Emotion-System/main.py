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


#!/usr/bin/env python3
"""
Main runner for the GAT-Based Emotion Modulation System
Imports and executes the complete emotion analysis experiment
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

# Import your custom classes (assuming they're in separate files)
try:
    from models.emotion_gat import EmotionGAT
    from data.data_generator import SyntheticDataGenerator
    from models.decision_nn import DecisionMakingNN
    from emotion_system import EmotionModulationSystem
    print("✓ Successfully imported all custom modules")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure all your class files are in the same directory as main.py")
    print("Expected files: emotion_gat.py, synthetic_data_generator.py, decision_making_nn.py, emotion_modulation_system.py")
    exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

def setup_environment():
    """Setup the training environment and check dependencies"""
    print("=== Environment Setup ===")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check PyTorch GPU availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("✓ Using CPU for computation")
    
    # Set matplotlib backend for plotting
    plt.style.use('default')
    
    return device

def run_quick_test():
    """Run a quick functionality test"""
    print("\n=== Quick Functionality Test ===")
    
    try:
        # Test data generation
        generator = SyntheticDataGenerator()
        test_sample = generator.generate_sample('emergency')
        print(f"✓ Data generation test passed")
        print(f"  Sample scenario: {test_sample.get('scenario_type', 'unknown')}")
        print(f"  Environment shape: {test_sample['environment'].shape}")
        print(f"  Emotions shape: {test_sample['emotions'].shape}")
        
        # Test model initialization
        emotion_gat = EmotionGAT()
        decision_nn = DecisionMakingNN()
        print(f"✓ Model initialization test passed")
        print(f"  EmotionGAT parameters: {sum(p.numel() for p in emotion_gat.parameters()):,}")
        print(f"  DecisionNN parameters: {sum(p.numel() for p in decision_nn.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def run_full_experiment(config):
    """Run the complete emotion modulation experiment"""
    print("\n=== Starting Full Experiment ===")
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    try:
        # Initialize system
        print("\nInitializing Emotion Modulation System...")
        system = EmotionModulationSystem()
        
        # Train model
        print("\nStarting training...")
        training_history = system.train_model(
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            total_samples=config['total_samples'],
            lr=config['learning_rate'],
            seed=config['seed']
        )
        
        # Final evaluation
        print("\nPerforming final test evaluation...")
        final_report = system.final_test_evaluation()
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        system.plot_training_history()
        system.plot_emotion_analysis(system.test_data)
        
        # Save results
        save_results(system, final_report, config)
        
        return system, final_report, training_history
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def save_results(system, final_report, config):
    """Save experiment results"""
    print("\n=== Saving Results ===")
    
    try:
        # Create results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = results_dir / "final_emotion_model.pth"
        torch.save({
            'emotion_gat_state_dict': system.emotion_gat.state_dict(),
            'decision_nn_state_dict': system.decision_nn.state_dict(),
            'training_history': system.training_history,
            'config': config
        }, model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Save report
        report_path = results_dir / "experiment_report.txt"
        with open(report_path, 'w') as f:
            f.write("=== GAT Emotion Modulation System - Experiment Report ===\n\n")
            f.write(f"Configuration:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nFinal Results:\n")
            f.write(f"  Test Accuracy: {final_report['test_accuracy']:.4f}\n")
            f.write(f"  Test Loss: {final_report['test_loss']:.4f}\n")
            f.write(f"  Emotion Entropy: {final_report['emotion_entropy']:.4f}\n")
            f.write(f"  Emotion Balance: {final_report['emotion_balance']:.4f}\n")
        print(f"✓ Report saved to {report_path}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def run_interactive_demo(system):
    """Run interactive demo with the trained system"""
    print("\n=== Interactive Demo ===")
    print("Test the trained emotion system with custom inputs!")
    
    emotions = ['joy', 'fear', 'anger', 'sadness', 'surprise', 'trust']
    
    while True:
        try:
            print(f"\nEnter environment features (6 values between 0-1, space-separated):")
            print("Or type 'quit' to exit, 'random' for random input")
            
            user_input = input("> ").strip().lower()
            
            if user_input == 'quit':
                break
            elif user_input == 'random':
                env_features = torch.rand(6)
                emotion_intensities = torch.rand(6)
                print(f"Generated random input:")
                print(f"  Environment: {env_features.numpy()}")
                print(f"  Emotions: {emotion_intensities.numpy()}")
            else:
                try:
                    env_values = [float(x) for x in user_input.split()]
                    if len(env_values) != 6:
                        print("Please enter exactly 6 values")
                        continue
                    env_features = torch.tensor(env_values, dtype=torch.float32)
                    
                    print("Enter emotion intensities (6 values between 0-1, space-separated):")
                    emotion_input = input("> ").strip()
                    emotion_values = [float(x) for x in emotion_input.split()]
                    if len(emotion_values) != 6:
                        print("Please enter exactly 6 emotion values")
                        continue
                    emotion_intensities = torch.tensor(emotion_values, dtype=torch.float32)
                    
                except ValueError:
                    print("Invalid input. Please enter numeric values.")
                    continue
            
            # Make prediction
            result = system.predict_decision(env_features, emotion_intensities)
            
            print(f"\n--- Prediction Results ---")
            print(f"Predicted Decision: {result['predicted_decision']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Decision Probabilities: {result['decision_probabilities']}")
            print(f"Top 3 Active Emotions: {', '.join(result['dominant_emotions'])}")
            
            print(f"\nEmotion Activations:")
            for i, emotion in enumerate(emotions):
                activation = result['emotion_activations'][i]
                bar = '█' * int(activation * 20)
                print(f"  {emotion:8s}: {activation:.3f} {bar}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Demo ended.")

def main():
    """Main execution function"""
    print("🧠 GAT-Based Emotion Modulation System")
    print("=====================================")
    
    # Setup environment
    device = setup_environment()
    
    # Run quick test
    if not run_quick_test():
        print("❌ Quick test failed. Please check your imports and dependencies.")
        return
    
    # Configuration
    config = {
        'num_epochs': 60,
        'batch_size': 32,
        'total_samples': 2000,
        'learning_rate': 0.001,
        'seed': 42
    }
    
    # Ask user for experiment type
    print("\n=== Experiment Options ===")
    print("1. Full experiment (recommended)")
    print("2. Quick experiment (fewer epochs)")
    print("3. Custom configuration")
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == '2':
        config.update({
            'num_epochs': 20,
            'total_samples': 1000
        })
        print("Using quick experiment settings")
    elif choice == '3':
        print("Enter custom configuration:")
        try:
            config['num_epochs'] = int(input(f"Number of epochs ({config['num_epochs']}): ") or config['num_epochs'])
            config['batch_size'] = int(input(f"Batch size ({config['batch_size']}): ") or config['batch_size'])
            config['total_samples'] = int(input(f"Total samples ({config['total_samples']}): ") or config['total_samples'])
            config['learning_rate'] = float(input(f"Learning rate ({config['learning_rate']}): ") or config['learning_rate'])
        except ValueError:
            print("Invalid input, using default configuration")
    
    # Run experiment
    system, final_report, training_history = run_full_experiment(config)
    
    if system is None:
        print("❌ Experiment failed")
        return
    
    # Print final summary
    print(f"\n🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"📊 Final Test Accuracy: {final_report['test_accuracy']:.4f}")
    print(f"🧠 Emotion Balance Score: {final_report['emotion_balance']:.4f}")
    
    # Offer interactive demo
    demo_choice = input("\nWould you like to run the interactive demo? (y/n): ").strip().lower()
    if demo_choice in ['y', 'yes']:
        run_interactive_demo(system)
    
    print("\n✨ All done! Check the 'results' folder for saved outputs.")

if __name__ == "__main__":
    main()