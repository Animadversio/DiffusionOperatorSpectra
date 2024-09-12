import numpy as np
import torch as th
from scipy.sparse import diags, kron, eye, diags_array
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

def create_diff_operator(N_x, N_y, L_x, L_y):
    # Grid spacing in x and y directions
    dx = L_x / (N_x - 1)
    dy = L_y / (N_y - 1)
    # Central difference matrix in 1D for x and y
    D_x_1d = (1 / (2 * dx)) * diags([-1, 0, 1], [-1, 0, 1], shape=(N_x, N_x))
    D_y_1d = (1 / (2 * dy)) * diags([-1, 0, 1], [-1, 0, 1], shape=(N_y, N_y))
    DD_x_1d = (1 / dx**2) * diags([1, -2, 1], [-1, 0, 1], shape=(N_x, N_x))
    DD_y_1d = (1 / dy**2) * diags([1, -2, 1], [-1, 0, 1], shape=(N_y, N_y))
    I_x = eye(N_x)
    I_y = eye(N_y)
    # d/dx and d/dy operators for 2D
    D_x = kron(I_y, D_x_1d)
    D_y = kron(D_y_1d, I_x)
    # Laplacian operator
    Laplacian = kron(I_y, DD_x_1d) + kron(DD_y_1d, I_x)
    return D_x, D_y, Laplacian


import numpy as np
from scipy.sparse import diags, kron, eye

def apply_dirichlet_boundary(D_1d, DD_1d):
    """
    Apply Dirichlet boundary conditions (zero value at boundary).
    Modify the first and last rows of the derivative matrices.
    """
    D_1d = D_1d.tolil()
    DD_1d = DD_1d.tolil()

    # Set boundary rows to 0 for first and second derivative matrices
    D_1d[0, :] = 0
    D_1d[-1, :] = 0
    DD_1d[0, :] = 0
    DD_1d[-1, :] = 0

    return D_1d.tocsr(), DD_1d.tocsr()


def apply_neumann_boundary(D_1d, DD_1d, dx):
    """
    Apply Neumann boundary conditions (zero derivative at boundary).
    Modify the first and last rows of the derivative matrices.
    """
    D_1d = D_1d.tolil()
    DD_1d = DD_1d.tolil()

    # Neumann boundary: forward/backward difference for first derivative
    D_1d[0, 0], D_1d[0, 1] = -1 / dx, 1 / dx
    D_1d[-1, -1], D_1d[-1, -2] = 1 / dx, -1 / dx

    # Adjust second derivative (Laplacian) at boundaries
    DD_1d[0, 0], DD_1d[0, 1] = -2 / dx**2, 2 / dx**2
    DD_1d[-1, -1], DD_1d[-1, -2] = -2 / dx**2, 2 / dx**2

    return D_1d.tocsr(), DD_1d.tocsr()


def create_1d_operators(N, L, boundary_condition):
    """
    Create 1D difference operators for first and second derivatives,
    applying the specified boundary condition (Dirichlet or Neumann).
    """
    dx = L / (N - 1)

    # Central difference operators (interior points)
    D_1d = (1 / (2 * dx)) * diags([-1, 0, 1], [-1, 0, 1], shape=(N, N))
    DD_1d = (1 / dx**2) * diags([1, -2, 1], [-1, 0, 1], shape=(N, N))

    # Apply boundary conditions
    if boundary_condition == "Dirichlet":
        D_1d, DD_1d = apply_dirichlet_boundary(D_1d, DD_1d)
    elif boundary_condition == "Neumann":
        D_1d, DD_1d = apply_neumann_boundary(D_1d, DD_1d, dx)

    return D_1d, DD_1d


def create_diff_operator_boundary(N_x, N_y, L_x, L_y, boundary_condition="Neumann"):
    """
    Create 2D difference operators (first derivative and Laplacian)
    with the specified boundary condition (Dirichlet or Neumann).
    """
    assert boundary_condition in ["Dirichlet", "Neumann"], \
        "boundary_condition must be 'Dirichlet' or 'Neumann'"
    # Create 1D operators for x and y directions
    D_x_1d, DD_x_1d = create_1d_operators(N_x, L_x, boundary_condition)
    D_y_1d, DD_y_1d = create_1d_operators(N_y, L_y, boundary_condition)

    # Identity matrices for Kronecker product
    I_x = eye(N_x)
    I_y = eye(N_y)

    # Construct 2D operators via Kronecker product
    D_x = kron(I_y, D_x_1d)
    D_y = kron(D_y_1d, I_x)
    Laplacian = kron(I_y, DD_x_1d) + kron(DD_y_1d, I_x)

    return D_x, D_y, Laplacian


def visualize_eigenmodes(eigenvalues, eigenvectors, shape, num2plot=20):
    plt.figure(figsize=(6, 4))
    plt.plot(eigenvalues.real, 'o-', label='real', alpha=0.7)
    plt.plot(eigenvalues.imag, 'o-', label='imag', alpha=0.7)
    plt.xlabel('rank')
    plt.ylabel('value')
    plt.title('Eigenvalues of the operator')
    plt.show()
    
    if isinstance(num2plot, int):
        num_eigenmodes = min(num2plot, eigenvectors.shape[1])
        eigenids = range(num_eigenmodes)
    elif isinstance(num2plot, tuple):
        eigenids = range(*num2plot)
        num_eigenmodes = len(eigenids)
    elif isinstance(num2plot, list):
        eigenids = num2plot
        num_eigenmodes = len(eigenids)
    nrow = int(np.ceil(num_eigenmodes / 5))
    fig, axes = plt.subplots(nrow, 5, figsize=(13, nrow* 2.4))
    axes = axes.flatten()
    for i, eigi in enumerate(eigenids):
        reshaped_eigenvec = eigenvectors[:, eigi].real.reshape(shape)
        ax = axes[i]
        im = ax.imshow(reshaped_eigenvec, cmap='coolwarm')
        ax.set_title(f'Eig {eigi+1} {eigenvalues[eigi].real:.2f}+{eigenvalues[eigi].imag:.2f}i')
        ax.axis('off')
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
    

def compute_FPoperator_eigenmodes_from_score_func(gmm_score_func, num_eigenvalues=25,
                        grid_size=101, xlim=(-5, 5), ylim=(-5, 5), showfig=True,
                        boundary_condition="Neumann"):
    """
    Modularized function to compute eigenmodes from Gaussian Mixture Model (GMM).
    
    Parameters:
    - gmm_score_func: Function handle to compute the score of the GMM
    - mus: Tensor of Gaussian means
    - covs: Tensor of Gaussian covariances
    - sigma: Standard deviation of the Gaussians
    - grid_size: Size of the grid for visualization
    - num_eigenvalues: Number of eigenvalues to compute
    
    Returns:
    - eigenvalues: The computed eigenvalues
    - eigenfunctions: The computed eigenfunctions
    """
    # Create grid points
    xspan = np.linspace(*xlim, grid_size)
    yspan = np.linspace(*ylim, grid_size)
    XX, YY = np.meshgrid(xspan, yspan)
    dx = xspan[1] - xspan[0]
    dy = yspan[1] - yspan[0]
    assert dx == dy
    # Reshape points into the format required by the GMM
    pnts = th.tensor(np.stack([XX, YY], axis=-1).reshape(-1, 2)).float()
    # Compute score vectors and probability density function
    score_vecs = gmm_score_func(pnts)
    score_tsrs = score_vecs.reshape(*XX.shape, -1)
    # probs = gmm.pdf(pnts).reshape(*XX.shape)
    # Create differential operators
    if boundary_condition is not None:  
        D_x, D_y, Laplacian = create_diff_operator_boundary(len(xspan), len(yspan), xspan[-1] - xspan[0], yspan[-1] - yspan[0], boundary_condition)
    else:
        D_x, D_y, Laplacian = create_diff_operator(len(xspan), len(yspan), xspan[-1] - xspan[0], yspan[-1] - yspan[0])
    # Define the Hamiltonian operator
    Hopts = Laplacian \
            - (D_x @ diags_array(score_vecs[:, 0].numpy()) + D_y @ diags_array(score_vecs[:, 1].numpy()))
    # Check if Hopts is symmetric
    is_symmetric = (Hopts - Hopts.transpose()).nnz == 0
    print(Hopts.shape, "Hopts is symmetric?", is_symmetric)
    # Access the values stored in the sparse CSR matrix
    values = Hopts.data
    # Use NumPy to check for NaN and inf values
    nancounts = np.isnan(values).sum()
    infcounts = np.isinf(values).sum()
    if nancounts > 0 or infcounts > 0:
        print("nan counts in Hopts:", nancounts)
        print("inf counts in Hopts:", infcounts)
        
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigs(Hopts, k=num_eigenvalues, which='SM')
    # Reshape eigenvectors to 2D grid
    eigenfunctions = [np.real(eigenvectors[:, i]).reshape(XX.shape) for i in range(num_eigenvalues)]
    # Output first eigenvalue and eigenfunction
    print("First eigenvalue:", eigenvalues[0])
    print("First eigenfunction shape:", eigenfunctions[0].shape)
    if showfig:
        # Visualize eigenmodes
        visualize_eigenmodes(eigenvalues, eigenvectors, XX.shape, num2plot=num_eigenvalues)
    return eigenvalues, eigenfunctions


if __name__ == "__main__":
    import sys
    sys.path.append("/n/home12/binxuwang/Github/DiffusionOperatorSpectra")
    import torch as th 
    import numpy as np
    import matplotlib.pyplot as plt
    from core.gaussian_mixture_lib import GaussianMixture, GaussianMixture_torch
    sigma = 0.2
    mus = th.tensor([np.cos(np.pi * np.arange(6) / 3), np.sin(np.pi * np.arange(6) / 3)]).T.float()
    # covs = th.repeat_interleave(th.eye(2).unsqueeze(0), 3, dim=0) * sigma ** 2
    covs = th.repeat_interleave(th.eye(2).unsqueeze(0), 6, dim=0) * sigma ** 2
    gmm = GaussianMixture_torch(mus, covs, th.ones(mus.shape[0]) / mus.shape[0])
    eigenvalues, eigenfunctions = compute_FPoperator_eigenmodes_from_score_func(
            gmm.score, grid_size=101, num_eigenvalues=25)
    
    # # visualize the density, score function vector field
    # xspan = np.linspace(-5, 5, 101)
    # yspan = np.linspace(-5, 5, 101)
    # XX, YY = np.meshgrid(xspan, yspan)
    # dx = xspan[1] - xspan[0]
    # dy = yspan[1] - yspan[0]
    # assert dx == dy
    # pnts = th.tensor(np.stack([XX, YY], axis=-1).reshape(-1, 2)).float()
    # score_vecs = gmm.score(pnts)
    # score_tsrs = score_vecs.reshape(*XX.shape, -1)
    # probs = gmm.pdf(pnts).reshape(*XX.shape)

    # D_x, D_y, Laplacian = create_diff_operator(len(xspan), len(yspan), xspan[-1] - xspan[0], yspan[-1] - yspan[0])
    # Hopts =  Laplacian \
    #     - (D_x @ diags_array(score_vecs[:, 0].numpy()) + D_y @ diags_array(score_vecs[:, 1].numpy()))
    # # is Hopts symmetric? no, it's not! 
    # print(Hopts.shape, "Hopts is symmetric?", (Hopts - Hopts.transpose()).nnz == 0)
    # num_eigenvalues = 25
    # eigenvalues, eigenvectors = eigs(Hopts, k=num_eigenvalues, which='SM')
    # # Reshape eigenvectors to 2D grid
    # eigenfunctions = [np.real(eigenvectors[:, i]).reshape(XX.shape) for i in range(num_eigenvalues)]
    # # Output first eigenvalue and eigenfunction
    # print("First eigenvalue:", eigenvalues[0])
    # print("First eigenfunction shape:", eigenfunctions[0].shape)
    # visualize_eigenmodes(eigenvalues, eigenvectors, XX.shape, num2plot=25)
    
    
    
    