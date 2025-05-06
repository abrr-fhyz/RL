import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from NNModel import NeuralNetwork
from NPModel import NPNeuralNetwork

class PlasticityTester:
    def __init__(self, input_size=20, num_classes=5):
        self.input_size = input_size
        self.num_classes = num_classes
        self.task_types = ['linear', 'nonlinear', 'sparse', 'imbalanced', 'noisy']
        self.scaler = StandardScaler()
        
    def copy_model(self, model):
        import copy
        new_model = type(model)(model.layers)
        new_model.weights = [w.copy() for w in model.weights]
        new_model.biases = [b.copy() for b in model.biases]
        return new_model

    def generate_task(self, task_type, num_samples=1000, difficulty=1.0):
        """Generate different types of classification tasks with safe parameters"""

        n_informative = max(self.num_classes, 5) 
        n_redundant = min(5, self.input_size - n_informative)
        
        if task_type == 'linear':
            params = {
                'n_informative': n_informative,
                'n_redundant': 0,
                'n_clusters_per_class': 1,
                'class_sep': 2.0 * difficulty,
                'flip_y': 0.01
            }
        elif task_type == 'nonlinear':
            params = {
                'n_informative': n_informative,
                'n_redundant': n_redundant,
                'n_clusters_per_class': min(2, 2**n_informative // self.num_classes),
                'class_sep': 1.5 * difficulty,
                'flip_y': 0.1 * (1/difficulty)
            }
        elif task_type == 'sparse':
            params = {
                'n_informative': max(self.num_classes, 2),
                'n_redundant': 0,
                'n_clusters_per_class': 1,
                'class_sep': 1.8 * difficulty,
                'flip_y': 0.05
            }
        elif task_type == 'imbalanced':
            weights = np.linspace(1, 0.2, self.num_classes)
            params = {
                'n_informative': n_informative,
                'n_redundant': 0,
                'n_clusters_per_class': 1,
                'class_sep': 1.6 * difficulty,
                'weights': weights/np.sum(weights)
            }
        elif task_type == 'noisy':
            params = {
                'n_informative': max(3, int(n_informative/2)),
                'n_redundant': int(self.input_size * 0.7),
                'n_clusters_per_class': 1,
                'class_sep': 1.7 * difficulty,
                'flip_y': 0.3 * (1/difficulty)
            }
        
        # Generate the data with safe parameters
        X, y = make_classification(
            n_samples=num_samples,
            n_features=self.input_size,
            n_classes=self.num_classes,
            random_state=42,
            **params
        )
        
        X = self.scaler.fit_transform(X)
        y_onehot = np.eye(self.num_classes)[y]
        return X, y_onehot
    
    def evaluate_model(self, model, X, y):
        """Evaluate model performance"""
        y_pred = model.predict(X)
        loss = model.cross_entropy_loss(y, y_pred)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        curvature = self.estimate_loss_curvature(model, X, y)
        return loss, accuracy, curvature
    
    def train_model(self, model, X, y, epochs=300, batch_size=32, lr=0.001):
        """Train model with appropriate method based on type"""
        if isinstance(model, NPNeuralNetwork):
            model.train(X, y, epochs=epochs, batch_size=batch_size)
        else:
            model.train(X, y, epochs=epochs, l_r=lr, batch_size=batch_size)
    
    def run_plasticity_test(self, model_class, model_name, num_base_tasks=3, num_tests_per_task=2):
        """Comprehensive plasticity testing protocol"""
        results = []
        
        layers = [self.input_size, 64, 64, self.num_classes]
        if model_class == NPNeuralNetwork:
            model = model_class(layers, initial_lr=0.001)
        else:
            model = model_class(layers)
        
        print(f"\n=== {model_name} - Sequential Task Learning ===")
        for task_num in range(num_base_tasks):
            task_type = self.task_types[task_num % len(self.task_types)]
            X_train, y_train = self.generate_task(task_type, difficulty=1.0 + task_num*0.2)
            
            print(f"\nTraining on {task_type} task {task_num+1}/{num_base_tasks}")
            self.train_model(model, X_train, y_train, epochs=300, batch_size=64)
            
            train_loss, train_acc, train_curvature = self.evaluate_model(model, X_train, y_train)  # Now unpacking 3 values
            print(f"Final training - Loss: {train_loss:.4f}, Accuracy: {train_acc*100:.2f}%, Curvature: {train_curvature:.4f}")
            
            for test_num in range(num_tests_per_task):
                test_type = self.task_types[(task_num + test_num + 1) % len(self.task_types)]
                try:
                    X_test, y_test = self.generate_task(test_type, difficulty=1.0 + task_num*0.2)
                    
                    init_loss, init_acc, init_curvature = self.evaluate_model(model, X_test, y_test)
                    
                    self.train_model(model, X_test, y_test, epochs=300, batch_size=64)
                    
                    final_loss, final_acc, final_curvature = self.evaluate_model(model, X_test, y_test)
                    improvement = init_loss - final_loss
                    
                    print(f"Test {test_num+1} on {test_type}: "
                          f"Loss {init_loss:.4f}→{final_loss:.4f} "
                          f"Acc {init_acc*100:.2f}%→{final_acc*100:.2f}% "
                          f"Curvature {init_curvature:.4f}→{final_curvature:.4f} "
                          f"Improvement: {improvement:.4f}")
                    
                    results.append({
                        'phase': 'sequential',
                        'task_num': task_num,
                        'test_num': test_num,
                        'task_type': task_type,
                        'test_type': test_type,
                        'init_loss': init_loss,
                        'init_acc': init_acc,
                        'init_curvature': init_curvature,  
                        'final_loss': final_loss,
                        'final_acc': final_acc,
                        'final_curvature': final_curvature,  
                        'improvement': improvement,
                        'curvature_change': final_curvature - init_curvature
                    })
                except ValueError as e:
                    print(f"Skipping {test_type} test due to parameter constraints: {str(e)}")
                    continue
        
        print(f"\n=== {model_name} - Catastrophic Forgetting Test ===")
        original_task = self.task_types[0]
        try:
            X_orig, y_orig = self.generate_task(original_task)
            
            orig_loss, orig_acc, orig_curvature = self.evaluate_model(model, X_orig, y_orig)
            print(f"Performance on original {original_task} task: "
                  f"Loss {orig_loss:.4f}, Accuracy {orig_acc*100:.2f}%, Curvature {orig_curvature:.4f}")
            
            results.append({
                'phase': 'forgetting',
                'task_type': original_task,
                'loss': orig_loss,
                'accuracy': orig_acc,
                'curvature': orig_curvature
            })
        except ValueError as e:
            print(f"Skipping forgetting test due to parameter constraints: {str(e)}")
        
        print(f"\n=== {model_name} - Transfer Learning Test ===")
        hard_task = 'nonlinear'
        try:
            X_hard, y_hard = self.generate_task(hard_task, difficulty=2.0)
            
            init_loss, init_acc, init_curvature = self.evaluate_model(model, X_hard, y_hard)
            print(f"Initial performance on hard {hard_task} task: "
                  f"Loss {init_loss:.4f}, Accuracy {init_acc*100:.2f}%, Curvature {init_curvature:.4f}")
            
            self.train_model(model, X_hard, y_hard, epochs=300, batch_size=64)
            
            final_loss, final_acc, final_curvature = self.evaluate_model(model, X_hard, y_hard)
            improvement = init_loss - final_loss
            print(f"Final performance after fine-tuning: "
                  f"Loss {final_loss:.4f}, Accuracy {final_acc*100:.2f}%, "
                  f"Curvature {final_curvature:.4f}, Improvement: {improvement:.4f}")
            
            results.append({
                'phase': 'transfer',
                'task_type': hard_task,
                'init_loss': init_loss,
                'init_acc': init_acc,
                'init_curvature': init_curvature,  
                'final_loss': final_loss,
                'final_acc': final_acc,
                'final_curvature': final_curvature,
                'improvement': improvement
            })
        except ValueError as e:
            print(f"Skipping transfer test due to parameter constraints: {str(e)}")
        
        return results

    def estimate_loss_curvature(self, model, X, y, epsilon=1e-3, samples=50):
        """Robust curvature estimation that works for both NN types"""
        original_loss = model.cross_entropy_loss(y, model.predict(X))
        total_curvature = 0.0
        
        # Get current parameter values
        original_weights = [w.copy() for w in model.weights]
        original_biases = [b.copy() for b in model.biases]
        
        for _ in range(samples):
            # Create meaningful perturbation directions
            direction = []
            for w in model.weights:
                # Create direction that considers weight magnitudes
                d = np.random.randn(*w.shape) * (0.1 + np.abs(w))
                direction.append(d)
            
            # Normalize the direction vector
            norm = np.sqrt(sum(np.sum(d**2) for d in direction))
            if norm < 1e-8:
                continue  # Skip degenerate directions
                
            direction = [d/norm for d in direction]
            
            # Evaluate perturbed models
            def evaluate_perturbed(sign):
                # Apply perturbation
                for i in range(len(model.weights)):
                    model.weights[i] = original_weights[i] + sign * epsilon * direction[i]
                y_pred = model.predict(X)
                return model.cross_entropy_loss(y, y_pred)
            
            # Calculate finite differences
            loss_plus = evaluate_perturbed(+1)
            loss_minus = evaluate_perturbed(-1)
            curvature = (loss_plus + loss_minus - 2*original_loss) / (epsilon**2)
            total_curvature += curvature
            
            # Debug print (optional)
            print(f"Sample {_}: L0={original_loss:.4f}, L+={loss_plus:.4f}, L-={loss_minus:.4f}, Curv={curvature:.4f}")
        
        # Restore original parameters
        for i in range(len(model.weights)):
            model.weights[i] = original_weights[i]
        for i in range(len(model.biases)):
            model.biases[i] = original_biases[i]
        
        return total_curvature / samples

    def visualize_results(self, nn_results, np_results):
        """Split visualization into two clear plots"""
        
        # Plot 1: Original Metrics
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Sequential Improvement
        plt.subplot(2, 2, 1)
        seq_nn = [r for r in nn_results if r['phase'] == 'sequential']
        seq_np = [r for r in np_results if r['phase'] == 'sequential']
        
        if seq_nn and seq_np:
            plt.plot([r['task_num'] for r in seq_nn], 
                    [r['improvement'] for r in seq_nn], 
                    'b-o', label='Standard NN')
            plt.plot([r['task_num'] for r in seq_np], 
                    [r['improvement'] for r in seq_np], 
                    'r-o', label='Neuroplastic NN')
            plt.title('Plasticity Improvement Over Tasks')
            plt.xlabel('Task Number')
            plt.ylabel('Loss Improvement')
            plt.legend()
            plt.grid(True)
        
        # Subplot 2: Forgetting Test
        plt.subplot(2, 2, 2)
        forget_nn = [r for r in nn_results if r['phase'] == 'forgetting']
        forget_np = [r for r in np_results if r['phase'] == 'forgetting']
        
        if forget_nn and forget_np:
            plt.bar(['Standard NN', 'Neuroplastic NN'], 
                   [forget_nn[0]['accuracy'], forget_np[0]['accuracy']],
                   color=['blue', 'red'])
            plt.title('Catastrophic Forgetting Test')
            plt.ylabel('Accuracy')
            plt.grid(True)
        
        # Subplot 3: Transfer Learning
        plt.subplot(2, 2, 3)
        transfer_nn = [r for r in nn_results if r['phase'] == 'transfer']
        transfer_np = [r for r in np_results if r['phase'] == 'transfer']
        
        if transfer_nn and transfer_np:
            plt.bar(['Standard NN Init', 'Neuroplastic NN Init'], 
                   [transfer_nn[0]['init_acc'], transfer_np[0]['init_acc']],
                   color=['lightblue', 'pink'], label='Initial')
            plt.bar(['Standard NN Final', 'Neuroplastic NN Final'], 
                   [transfer_nn[0]['final_acc'], transfer_np[0]['final_acc']],
                   color=['blue', 'red'], label='After Fine-tuning')
            plt.title('Transfer Learning Performance')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
        
        # Subplot 4: Task-Type Improvement
        plt.subplot(2, 2, 4)
        if seq_nn and seq_np:
            task_types = sorted(list(set(r['test_type'] for r in seq_nn)))
            nn_means = [np.mean([r['improvement'] for r in seq_nn if r['test_type'] == t]) for t in task_types]
            np_means = [np.mean([r['improvement'] for r in seq_np if r['test_type'] == t]) for t in task_types]
            
            x = np.arange(len(task_types))
            width = 0.35
            plt.bar(x - width/2, nn_means, width, label='Standard NN')
            plt.bar(x + width/2, np_means, width, label='Neuroplastic NN')
            plt.title('Improvement by Task Type')
            plt.xticks(x, task_types, rotation=45)
            plt.ylabel('Loss Improvement')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Plot 2: Curvature Analysis (New Separate Figure)
        plt.figure(figsize=(15, 6))

        # Subplot 1: Curvature Over Tasks
        plt.subplot(1, 2, 1)
        if seq_nn and seq_np:
            all_curvatures = [r.get('init_curvature', 0) for r in seq_nn + seq_np] + \
                            [r.get('final_curvature', 0) for r in seq_nn + seq_np]
            y_min = min(all_curvatures) * 0.9 if min(all_curvatures) > 0 else min(all_curvatures) * 1.1
            y_max = max(all_curvatures) * 1.1
            
            plt.ylim(y_min, y_max)
            
            nn_init, = plt.plot([r['task_num'] for r in seq_nn], 
                               [r.get('init_curvature', 0) for r in seq_nn],
                               'b-o', label='Standard NN (Init)', markersize=8)
            nn_final, = plt.plot([r['task_num'] for r in seq_nn], 
                                [r.get('final_curvature', 0) for r in seq_nn],
                                'b--x', label='Standard NN (Final)', markersize=8)
            
            np_init, = plt.plot([r['task_num'] for r in seq_np], 
                               [r.get('init_curvature', 0) for r in seq_np],
                               'r-o', label='Neuroplastic NN (Init)', markersize=8)
            np_final, = plt.plot([r['task_num'] for r in seq_np], 
                                [r.get('final_curvature', 0) for r in seq_np],
                                'r--x', label='Neuroplastic NN (Final)', markersize=8)
            
            plt.title('Loss Curvature Evolution', fontsize=12)
            plt.xlabel('Task Number', fontsize=10)
            plt.ylabel('Curvature Estimate', fontsize=10)
            plt.legend(handles=[nn_init, nn_final, np_init, np_final], fontsize=9)
            plt.grid(True, alpha=0.3)

        # Subplot 2: Curvature vs Improvement
        plt.subplot(1, 2, 2)
        if seq_nn and seq_np:
            x_min = min([r.get('init_curvature', 0) for r in seq_nn + seq_np]) * 0.9
            x_max = max([r.get('init_curvature', 0) for r in seq_nn + seq_np]) * 1.1
            y_min = min([r['improvement'] for r in seq_nn + seq_np]) * 0.9
            y_max = max([r['improvement'] for r in seq_nn + seq_np]) * 1.1
            
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            
            z_nn = np.polyfit([r.get('init_curvature', 0) for r in seq_nn],
                             [r['improvement'] for r in seq_nn], 1)
            z_np = np.polyfit([r.get('init_curvature', 0) for r in seq_np],
                             [r['improvement'] for r in seq_np], 1)
            
            x_vals = np.array([x_min, x_max])
            plt.plot(x_vals, z_nn[0]*x_vals + z_nn[1], 'b-', alpha=0.3)
            plt.plot(x_vals, z_np[0]*x_vals + z_np[1], 'r-', alpha=0.3)
            
            plt.scatter(
                [r.get('init_curvature', 0) for r in seq_nn],
                [r['improvement'] for r in seq_nn],
                c='blue', label='Standard NN', s=80, alpha=0.7
            )
            plt.scatter(
                [r.get('init_curvature', 0) for r in seq_np],
                [r['improvement'] for r in seq_np],
                c='red', label='Neuroplastic NN', s=80, alpha=0.7
            )
            
            plt.title('Initial Curvature vs Plasticity Improvement', fontsize=12)
            plt.xlabel('Initial Curvature', fontsize=10)
            plt.ylabel('Improvement', fontsize=10)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    tester = PlasticityTester(input_size=20, num_classes=5)
    
    print("=== Testing Standard Neural Network ===")
    nn_results = tester.run_plasticity_test(
        NeuralNetwork, 
        "Standard Neural Network",
        num_base_tasks=5,
        num_tests_per_task=2
    )
    
    print("\n=== Testing Neuroplastic Neural Network ===")
    np_results = tester.run_plasticity_test(
        NPNeuralNetwork, 
        "Neuroplastic Neural Network",
        num_base_tasks=5,
        num_tests_per_task=2
    )
    
    print("\n=== Generating Comparative Visualizations ===")
    tester.visualize_results(nn_results, np_results)