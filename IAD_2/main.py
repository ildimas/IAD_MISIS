import matplotlib.pyplot as plt
import numpy as np

def generate_terrain_model():
    # Define the size of the grid
    grid_size = 100
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    x, y = np.meshgrid(x, y)

    # Create a synthetic terrain using a mathematical function
    z = np.sin(np.sqrt(x**2 + y**2))

    # Plot the contour map
    plt.figure(figsize=(8, 6))
    contour = plt.contour(x, y, z, levels=15, cmap='terrain')
    plt.clabel(contour, inline=True, fontsize=8)
    plt.title('Digital Terrain Model')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.colorbar(contour, label='Elevation')
    plt.show()

if __name__ == "__main__":
    generate_terrain_model()
