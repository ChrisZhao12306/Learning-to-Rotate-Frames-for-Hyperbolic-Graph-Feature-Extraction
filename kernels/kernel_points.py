import sys
sys.path.append('/data1/ziyan/RFN2')# Please change accordingly!

from distributions.Lorentz_wrapped_normal import LorentzWrappedNormal
from geoopt import ManifoldParameter
from optim import RiemannianAdam, RiemannianSGD
from os import makedirs
import torch
import torch.nn.functional as F
from os.path import join, exists
import math

def get_frame_kernel_points_hyperboloid(scale_j, dimension, manifold, rotation_matrix=None):
    """
    Generate the representation (tangent vector) of the framework kernel point in the tangent space using a learnable rotation matrix
    
    Args:
    scale_j: Scale parameter j
    dimension: Dimension D
    manifold: Hyperbolic manifold
    rotation_matrix: Learnable rotation matrix with shape (D, D)

Returns:
    The tangent vector corresponding to the framework kernel point, with shape (2*D, D+1) for the Lorentz model
    """
    # K = 2D
    num_kernel = 2 * dimension
    sinh_val = torch.sinh(torch.tensor(2.0 ** scale_j, dtype=torch.float32))

    # Part 1: k = 1, ..., D
    eye_d = torch.eye(dimension, device=rotation_matrix.device if rotation_matrix is not None else None)
    if rotation_matrix is not None:
        
        eye_d = torch.mm(rotation_matrix, eye_d)
    
    tangent_vecs_part1 = sinh_val * eye_d

    # Part 2: k = D+1, ..., 2D
    
    e_sum_matrix = torch.zeros((dimension, dimension), device=rotation_matrix.device if rotation_matrix is not None else None)
    indices = torch.arange(dimension, device=rotation_matrix.device if rotation_matrix is not None else None)
    e_sum_matrix[indices, indices] = 1.0 / math.sqrt(2)
    e_sum_matrix[indices, (indices + 1) % dimension] = 1.0 / math.sqrt(2)
    
   
    if rotation_matrix is not None:
        
        e_sum_matrix = torch.mm(rotation_matrix, e_sum_matrix)
    
    tangent_vecs_part2 = sinh_val * e_sum_matrix
    
    
    spatial_vectors = torch.cat([tangent_vecs_part1, tangent_vecs_part2], dim=0)
    
    # Add 0 along the time dimension to form the complete tangent vector
    # The tangent vector is at the origin, with the time component being 0
    time_coord = torch.zeros(num_kernel, 1, device=rotation_matrix.device if rotation_matrix is not None else None)
    tangent_vectors = torch.cat([time_coord, spatial_vectors], dim=1)
    
    return tangent_vectors

class LearnableRotationMatrix(torch.nn.Module):
    def __init__(self, dimension, temperature=1.0):
        super(LearnableRotationMatrix, self).__init__()
        init_matrix = torch.eye(dimension) + 0.01 * torch.randn(dimension, dimension)
        q, r = torch.linalg.qr(init_matrix)
        self.rotation_matrix = torch.nn.Parameter(q)
        # temperature
        self.temperature = torch.nn.Parameter(torch.tensor(temperature))
        self.min_temp = 0.1  
        
    def forward(self):
        current_temp = F.softplus(self.temperature) + self.min_temp
        
        # Use QR decomposition to ensure the output is always an orthogonal matrix.
        q, r = torch.linalg.qr(self.rotation_matrix)
        det = torch.det(q)
        q[:, -1] = q[:, -1] * det.sign()
        eye = torch.eye(q.shape[0], device=q.device)
        soft_rotation = eye + (q - eye) * current_temp
        
        q_soft, r = torch.linalg.qr(soft_rotation)
        det = torch.det(q_soft)
        q_soft[:, -1] = q_soft[:, -1] * det.sign()
        
        return q_soft

def get_multi_scale_frame_kernel_points_hyperboloid(dimension, manifold, temperature=1.0, use_learnable_rotation=True):
    """
    Generate framework kernel points using multiple scale parameters and combine them, with an option to include a learnable rotation matrix.
    
    Args:
    dimension: Dimension D
    manifold: Hyperbolic manifold
    temperature: Temperature parameter controlling the softness of the rotation matrix
    use_learnable_rotation: Whether to use a learnable rotation matrix

    Returns:
    combined_kernel_points: Combined framework kernel points
    rotation: Learnable rotation matrix module (if enabled) or None
    """
    scale_values = [0.5, 1.0, 1.5, 2.0]
    
    accumulated_tangents = torch.zeros(2 * dimension, dimension + 1)
    
    # Determine whether to use a learnable rotation matrix based on the parameters.
    if use_learnable_rotation:
        
        rotation = LearnableRotationMatrix(dimension, temperature)
        rotation_matrix = rotation()
        print(f"[INFO] Using learnable rotation matrix")
        print(f"[INFO] Current temperature: {F.softplus(rotation.temperature.data) + rotation.min_temp:.4f}")
    else:
        rotation = None
        rotation_matrix = torch.eye(dimension)
        print(f"[INFO] Using fixed identity rotation matrix (no learnable rotation)")
    
    decay_rate = 0.5  
    indices = torch.arange(len(scale_values), dtype=torch.float32)
    weights = torch.exp(-decay_rate * indices)
    weights = weights / weights.sum()
    
    print(f"[INFO] Using scales: {scale_values}")
    print(f"[INFO] Using weights: {weights.tolist()}")
    
    
    for scale_j, weight in zip(scale_values, weights):
        kernel_tangents = get_frame_kernel_points_hyperboloid(scale_j, dimension, manifold, rotation_matrix)
        
        norm = torch.norm(kernel_tangents, p=2, dim=1, keepdim=True)
        kernel_tangents = kernel_tangents / (norm + 1e-8)
        
        accumulated_tangents.add_(weight * kernel_tangents)
    
    combined_kernel_points = manifold.expmap0(accumulated_tangents)
    
    return combined_kernel_points, rotation

def get_sampled_frame_kernel_points_hyperboloid(dimension, manifold, sample_ratio=0.5, temperature=1.0, use_learnable_rotation=True):
    """
    Memory-efficient version: Reduce the number of framework kernel points through sampling.
    
    Args:
    dimension: Dimension D
    manifold: Hyperbolic manifold
    sample_ratio: Sampling ratio (0, 1], default 0.5 indicates using half of the kernel points
    temperature: Temperature parameter controlling the softness of the rotation matrix
    use_learnable_rotation: Whether to use a learnable rotation matrix

    Returns:
    sampled_kernel_points: Sampled framework kernel points
    rotation: Learnable rotation matrix module (if enabled) or None
    """
    
    all_kernel_points, rotation = get_multi_scale_frame_kernel_points_hyperboloid(dimension, manifold, temperature, use_learnable_rotation)
    
    total_kernels = all_kernel_points.shape[0]  # 2*dimension
    num_sampled = max(dimension, int(total_kernels * sample_ratio))
    
    if num_sampled <= dimension:
       
        indices = torch.arange(num_sampled)
    else:
        
        first_part = torch.arange(dimension)
        
        remaining_needed = num_sampled - dimension
        second_part_indices = torch.randperm(dimension)[:remaining_needed] + dimension
        indices = torch.cat([first_part, second_part_indices])
    
   
    sampled_kernel_points = all_kernel_points[indices]
    
    print(f"[INFO] Sampled {num_sampled} kernels from {total_kernels} total kernels (ratio: {sample_ratio})")
    
    return sampled_kernel_points, rotation

def compute_hyperbolic_distance(x, z, manifold):
    """
    Compute the hyperbolic distance d_H(x, z)
    
    Args:
    x: Input point with shape (batch_size, dim+1)
    z: Kernel point with shape (num_kernels, dim+1)
    manifold: Hyperbolic manifold

    Returns:
    Distance matrix with shape (batch_size, num_kernels)
    """
    
    x_expanded = x.unsqueeze(1)  # (batch_size, 1, dim+1)
    z_expanded = z.unsqueeze(0)  # (1, num_kernels, dim+1)
    
   
    distances = manifold.dist(x_expanded, z_expanded)
    
    return distances

def frechet_mean_hyperboloid(points, weights, manifold, max_iter=100, tol=1e-6):
    """
    Compute the Fréchet mean in hyperbolic space
    
    Args:
    points: Point set with shape (batch_size, num_points, dim+1)
    weights: Weights with shape (batch_size, num_points)
    manifold: Hyperbolic manifold
    max_iter: Maximum number of iterations
    tol: Convergence tolerance

    Returns:
    Fréchet mean with shape (batch_size, dim+1)
    """
    weights = weights / weights.sum(dim=1, keepdim=True)
    
    
    batch_size = points.shape[0]
    mean_points = []
    
    for i in range(batch_size):
        
        weighted_mean = manifold.mid_point(points[i], weights[i])
        mean_points.append(weighted_mean)
    
    
    mean = torch.stack(mean_points, dim=0)
    
    return mean

def load_kernels(manifold, radius, num_kpoints, dimension, random=False, use_frame=False, temperature=1.0, sample_ratio=None, use_learnable_rotation=True):
    """
    Load or generate kernel points
    
    Args:
    manifold: Hyperbolic manifold
    radius: Radius (used for scaling)
    num_kpoints: Number of kernel points
    dimension: Space dimension (excluding time coordinate)
    random: Whether to use random kernel points
    use_frame: Whether to use the framework kernel points method (will use multiple scales)
    temperature: Temperature parameter controlling the softness of the rotation matrix
    sample_ratio: Sampling ratio for framework kernel points, None means no sampling
    use_learnable_rotation: Whether to use a learnable rotation matrix
    """
    
    print(f"[DEBUG] load_kernels called with: use_frame={use_frame}, num_kpoints={num_kpoints}, dimension={dimension}, sample_ratio={sample_ratio}, use_learnable_rotation={use_learnable_rotation}")
    
    if use_frame:
        if sample_ratio is not None and sample_ratio < 1.0:
            print(f"[DEBUG] Using sampled frame kernel points with ratio {sample_ratio}")
            kernel_points, rotation = get_sampled_frame_kernel_points_hyperboloid(dimension, manifold, sample_ratio, temperature, use_learnable_rotation)
        else:
            print(f"[DEBUG] Using multi-scale frame kernel points (scale_j=1,2,3,4)")
            kernel_points, rotation = get_multi_scale_frame_kernel_points_hyperboloid(dimension, manifold, temperature, use_learnable_rotation)
        
        kernel_tangents = manifold.logmap0(kernel_points)
        
        dis = manifold.dist0(kernel_points).max()
        if dis > 0:
            kernel_tangents *= radius / dis
        
        return kernel_tangents, rotation
    
    if random:
        kernel_points = manifold.random_normal((num_kpoints, dimension + 1))
        kernel_tangents = manifold.logmap0(kernel_points)
        dis = manifold.dist0(kernel_points).max()
        kernel_tangents *= radius / dis

        return kernel_tangents

    # Kernel directory
    kernel_dir = 'kernels/dispositions'
    if not exists(kernel_dir):
        makedirs(kernel_dir)

    # Kernel_file
    kernel_file = join(kernel_dir, 'hyperboloid_k_{:03d}_{:d}D.pt'.format(num_kpoints, dimension))

    # Check if already done
    if not exists(kernel_file):
        kernel_points = get_origin_kernel_points_lorentz(num_kernel = num_kpoints, 
                                                         dim = dimension + 1, 
                                                         manifold = manifold, 
                                                         max_iter = 1000, 
                                                         verbose = False)
        kernel_tangents = manifold.logmap0(kernel_points[1:])
        kernel_tangents = torch.concat([torch.zeros(1, dimension + 1), kernel_tangents])

        torch.save(kernel_tangents, kernel_file)

    else:
        kernel_tangents = torch.load(kernel_file)
        
        if kernel_tangents.shape[1] == dimension:
            kernel_tangents = torch.cat([torch.zeros(kernel_tangents.shape[0], 1), kernel_tangents], dim=1)
        kernel_points = manifold.expmap0(kernel_tangents)

    # Scale kernels
    dis = manifold.dist0(kernel_points).max()
    kernel_tangents *= radius / dis

    return kernel_tangents
