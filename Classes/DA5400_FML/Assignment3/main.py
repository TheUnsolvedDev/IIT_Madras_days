import numpy as np
import matplotlib.pyplot as plt
import tqdm

from dataset import *
from algorithms import *

class Solution1:
    def __init__(self) -> None:
        self.data_train = Dataset1().get_data()
        print(self.data_train.shape)
    
    def question1(self):
        components = 9
        self.train_data = self.data_train.reshape(-1,784)/255.0
        transformed_data, principal_components, eigenvalues, mean = principal_component_analysis(self.train_data, 9)
        fig,axes = plt.subplots(3,3,figsize=(10,10))
        plt.tight_layout(pad=5)
        for i in range(9):
            ax = axes[i//3][i%3]
            ax.imshow(principal_components[:, i].reshape(28, 28), cmap='gray')
            ax.set_title(f'PC {i+1}')
            ax.axis('off')
        plt.suptitle("Principal Components")
        plt.savefig("images/principal_components.png")
        plt.show()
        
        print('Variance explained:')
        total_variance = np.sum(eigenvalues)
        for i, eigenvalue in enumerate(eigenvalues):
            print(f'PC {i+1}: {eigenvalue / total_variance * 100:.2f}%')
            if i == 9:
                break
            
        
    def question2(self):
        dimensions = [5, 10, 50, 100]
        sample_indices = [0, 1, 2, 3, 4] 
        self.train_data = self.data_train.reshape(-1,784)/255.0
        for d in dimensions:
            transformed_data, principal_components, eigenvalues, mean = principal_component_analysis(self.train_data, d)
            reconstructed_data = reconstruct_data(transformed_data, principal_components, mean)
            
            original = self.train_data[sample_indices]
            reconstructed = reconstructed_data[sample_indices]
            indices = sample_indices
            num_components = d
            fig, axes = plt.subplots(len(indices), 2, figsize=(len(indices), len(indices) * 2))
            plt.tight_layout(pad=5)
            for i, idx in enumerate(indices):
                axes[i, 0].imshow(original[idx].reshape(28, 28), cmap='gray')
                axes[i, 0].set_title("Original")
                axes[i, 0].axis('off')

                axes[i, 1].imshow(reconstructed[idx].reshape(28, 28), cmap='gray')
                axes[i, 1].set_title(f"Reconstructed with {num_components} PCs")
                axes[i, 1].axis('off')
            plt.savefig(f"images/reconstruction_{d}.png")
            plt.show()
            
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = np.cumsum(eigenvalues) / total_variance
        for i, ratio in enumerate(explained_variance_ratio):
            if ratio >= 0.95:  # Find the dimension explaining at least 95% of the variance
                print(f"At least 95% of the variance is explained by {i+1} components.")
                break
    

class Solution2:
    def __init__(self) -> None:
        self.data = Dataset2().data_numpy
        self.llyod = LlyodsAlgorithm()
        
    def question1(self):
        for seed in range(5):
            fig,ax = plt.subplots(1,2,figsize=(14,5))
            # plt.tight_layout()
            error_history = []
            initial_centroids = self.llyod.initialize_centroids(self.data, 2, seed=seed)
            temp_centroids = initial_centroids.copy()
            final_centroids = None
            
            for _ in tqdm.tqdm(range(100)):
                old_centroids = temp_centroids
                labels = self.llyod.closest_centroid(self.data, old_centroids)
                error_history.append(self.llyod.compute_error(self.data, labels, temp_centroids))
                temp_centroids = self.llyod.move_centroids(self.data, labels, 2)
                if np.all(np.abs(old_centroids - temp_centroids) < 1e-4):
                    final_centroids = temp_centroids
                    break
            error = self.llyod.compute_error(self.data, labels, final_centroids)
            print(f'seed {seed} error: {error}')
            
            ax[0].scatter(range(len(error_history)), error_history)
            ax[0].plot(error_history)
            ax[0].set_xlabel('iteration')
            ax[0].set_ylabel('error')
            ax[0].set_title('error history for seed ' + str(seed))
            
            ax[1].scatter(self.data[:, 0], self.data[:, 1], c=labels,alpha=0.1)
            ax[1].scatter(initial_centroids[:, 0], initial_centroids[:, 1], c=range(len(initial_centroids)), marker='*',alpha=0.5)
            for i in range(len(initial_centroids)):
                ax[1].text(initial_centroids[i, 0], initial_centroids[i, 1], 'initial')
            ax[1].scatter(final_centroids[:, 0], final_centroids[:, 1], c=range(len(final_centroids)), marker='o')
            for i in range(len(final_centroids)):
                ax[1].text(final_centroids[i, 0], final_centroids[i, 1], 'final')
            ax[1].legend(['data','initial centroids', 'final centroids'])
            ax[1].set_title('initial and final centroids for seed ' + str(seed))
            ax[1].set_xlabel('feature 1')
            ax[1].set_ylabel('feature 2')
            plt.savefig(f'images/seed_{seed}.png')
            plt.show()
        
            
    def question2(self):
        initial_centroids = self.llyod.initialize_centroids(self.data, 1, seed=42)
        for k in [2,3,4,5]:
            fig,ax = plt.subplots(1,2,figsize=(14,5))
            initial_centroids = np.concatenate((initial_centroids, self.llyod.initialize_centroids(self.data, 1, seed=k)), axis=0)
            temp_centroids = initial_centroids.copy()
            for _ in tqdm.tqdm(range(100)):
                old_centroids = temp_centroids
                labels = self.llyod.closest_centroid(self.data, old_centroids)
                temp_centroids = self.llyod.move_centroids(self.data, labels, len(temp_centroids))
                if np.all(np.abs(old_centroids - temp_centroids) < 1e-4):
                    final_centroids = temp_centroids
                    break
            error = self.llyod.compute_error(self.data, labels, final_centroids)
            ax[0].scatter(self.data[:, 0], self.data[:, 1], c=labels,alpha=0.1)
            ax[0].scatter(initial_centroids[:, 0], initial_centroids[:, 1], c=range(len(initial_centroids)), marker='*',alpha=0.5)
            for i in range(len(initial_centroids)):
                ax[0].text(initial_centroids[i, 0], initial_centroids[i, 1], 'initial')
            ax[0].scatter(final_centroids[:, 0], final_centroids[:, 1], c=range(len(final_centroids)), marker='o')
            for i in range(len(final_centroids)):
                ax[0].text(final_centroids[i, 0], final_centroids[i, 1], 'final')
            ax[0].legend(['data','initial centroids', 'final centroids'])
            ax[0].set_xlabel('feature 1')
            ax[0].set_ylabel('feature 2')
            ax[0].set_title('initial and final centroids for k=' + str(k))
            
            x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
            y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            
            distances = np.sqrt(((grid_points[:, np.newaxis] - final_centroids)**2).sum(axis=2))
            closest_centroids = np.argmin(distances, axis=1)
            voronoi_regions = closest_centroids.reshape(xx.shape)
            
            # plt.figure(figsize=(8,6))
            ax[1].imshow(voronoi_regions, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='Pastel1', alpha=0.5)
            ax[1].scatter(self.data[:, 0], self.data[:, 1], c=labels, cmap='Pastel1', marker='o', s=30, edgecolor='k', alpha=0.6)
            ax[1].scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', marker='x', s=100, label="Centroids")
            for i in range(len(final_centroids)):
                ax[1].text(final_centroids[i, 0], final_centroids[i, 1], 'final')
            ax[1].set_xlabel('feature 1')
            ax[1].set_ylabel('feature 2')
            ax[1].set_title(f'Voronoi regions for k={k}')
            ax[1].legend()
            plt.savefig(f'images/k_voronoi_{k}.png')
            plt.show()
            
            
if __name__ == '__main__':
    solution = Solution1()
    solution.question1()
    solution.question2()
    
    solution = Solution2()
    solution.question1()
    solution.question2()