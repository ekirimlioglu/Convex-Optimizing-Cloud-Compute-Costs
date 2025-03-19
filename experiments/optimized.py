import numpy as np
import cvxpy as cp

# ----- OPTIMIZATION APPROACHES -----

def optimize_mixed_problem(c, alpha, p, beta1, beta2, beta3, gamma, E, K, d, mu, g, n, m,
                          initial_x=None, max_iterations=100, tolerance=1e-4):
    """
    Solve the mixed optimization problem using CVXPY with successive approximation

    Parameters:
    ----------
    c : array-like
        Linear term coefficient vector
    alpha, p, beta1, beta2, beta3, gamma : float
        Scalar parameters
    E, K : array-like
        Constraint matrices
    d, mu, g : array-like
        Constraint vectors
    n : int
        Dimension of variable x
    m : int
        Number of constraints
    initial_x : array-like, optional
        Initial guess for x
    max_iterations : int, optional
        Maximum number of iterations
    tolerance : float, optional
        Convergence tolerance

    Returns:
    -------
    x_opt : numpy.ndarray
        Optimal solution
    f_opt : float
        Optimal objective value
    """
    # Initialize
    if initial_x is None:
        x_current = np.ones(n)  # Initial guess
    else:
        x_current = initial_x

    # Convert inputs to numpy arrays if they aren't already
    c = np.array(c)
    E = np.array(E)
    K = np.array(K)
    d = np.array(d)
    mu = np.array(mu)
    g = np.array(g)

    # Begin iterations
    for iter_num in range(max_iterations):
        # Create variable
        x = cp.Variable(n, nonneg=True)

        # Linearize the concave part around current point
        # For Term 2: -alpha * 1^T e^(-beta1 * Ex)
        Ex_current = E @ x_current
        term2_value = -alpha * np.sum(np.exp(-beta1 * Ex_current))

        # Gradient of Term 2 at x_current
        grad_term2 = alpha * beta1 * E.T @ np.exp(-beta1 * Ex_current)

        # Linearized approximation of Term 2
        term2_approx = term2_value + grad_term2 @ (x - x_current)

        # Convex parts directly
        # Term 1: c^T x + alpha * p (linear)
        term1 = c @ x + alpha * p

        # Term 3: -gamma * 1^T log(1 + beta2 * Ex) (convex)
        Ex_current = E @ x_current
        term3_value = -gamma * np.sum(np.log(1 + beta2 * Ex_current))

        # Gradient of Term 3 at x_current
        grad_term3 = -gamma * beta2 * E.T @ (1 / (1 + beta2 * Ex_current))

        # Linearized approximation of Term 3
        term3_approx = term3_value + grad_term3 @ (x - x_current)

        # Term 4: beta3 * sum(max(0, d_r - (Kx)_r)^2) (convex)
        term4_components = []
        for r in range(m):
            term4_components.append(cp.maximum(0, d[r] - K[r] @ x)**2)
        term4 = beta3 * cp.sum(cp.hstack(term4_components))

        # Objective: minimize Term 1 - Term 2 + Term 3 + Term 4
        # For Term 2, we use the linearized approximation
        objective = cp.Minimize(term1 - term2_approx - term3_approx + term4)

        # Constraints
        constraints = [
            K @ x >= d - mu,
            K @ x <= d + g
        ]

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)

        # Check solution status
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: Problem status is {problem.status}")
            # Try another solver if the first one fails
            problem.solve(solver=cp.SCS)
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Failed to solve the problem. Status: {problem.status}")
                return x_current, None

        # Extract solution
        x_new = x.value

        # Check convergence
        rel_change = np.linalg.norm(x_new - x_current) / (np.linalg.norm(x_current) + 1e-10)
        print(f"Iteration {iter_num}: relative change = {rel_change}")

        if rel_change < tolerance:
            print(f"Converged after {iter_num+1} iterations.")
            break

        # Update current point
        x_current = x_new

    # Calculate actual objective value at the solution
    x_opt = x_current
    Ex_opt = E @ x_opt
    Kx_opt = K @ x_opt

    term1_val = c @ x_opt + alpha * p
    term2_val = -alpha * np.sum(np.exp(-beta1 * Ex_opt))
    term3_val = -gamma * np.sum(np.log(1 + beta2 * Ex_opt))

    term4_val = 0
    for r in range(m):
        term4_val += beta3 * max(0, d[r] - Kx_opt[r])**2

    f_opt = term1_val + term2_val - term3_val + term4_val

    return x_opt, f_opt

# ----- OPTIMIZATION MODEL SETUP -----

def setup_optimization_problem(cloud_data, demand_vector, uncertainty_radius, acceptable_waste, 
                               alpha=1.0, beta1=0.1, beta2=0.01, beta3=10.0, gamma=0.5):
    """
    Set up the convex optimization problem as described in the paper
    
    Parameters:
    cloud_data: DataFrame containing cloud instance specifications
    demand_vector: array of resource demands (vCPU, memory, storage)
    uncertainty_radius: array of resource demand uncertainties
    acceptable_waste: array of acceptable resource waste
    alpha, beta1, beta2, beta3, gamma: model parameters
    
    Returns:
    problem: cvxpy problem object
    x: cvxpy variable representing allocation
    """
    # Extract relevant data
    n = len(cloud_data)  # Number of instance types
    m = len(demand_vector)  # Number of resource types
    p = len(cloud_data['provider'].unique())  # Number of providers
    
    # Create resource composition matrix K
    K = np.zeros((m, n))
    for i, instance in cloud_data.iterrows():
        K[0, i] = instance['vcpu']      # vCPU
        K[1, i] = instance['memory']    # Memory (GB)
        K[2, i] = instance['storage']   # Storage (GB)
    
    # Create cost vector c
    c = cloud_data['cost'].values
    
    # Create provider selector matrix E
    E = np.zeros((p, n))
    providers = cloud_data['provider'].unique()
    for i, instance in cloud_data.iterrows():
        provider_idx = np.where(providers == instance['provider'])[0][0]
        E[provider_idx, i] = 1
    
    # Define the variable
    x = cp.Variable(n, nonneg=True)
    
    # Define the objective function components
    base_cost = c @ x
    
    # Provider consolidation penalty (approximating indicator function)
    provider_usage = E @ x
    consolidation_penalty = alpha * p - alpha * cp.sum(cp.exp(-beta1 * provider_usage))
    
    # Volume discount term
    volume_discount = -gamma * cp.sum(cp.log(1 + beta2 * provider_usage))
    
    # Resource shortage penalty
    Kx = K @ x
    shortage_penalty = beta3 * cp.sum_squares(cp.maximum(0, demand_vector - Kx))
    
    # Full objective function
    objective = base_cost + consolidation_penalty + volume_discount + shortage_penalty
    
    # Constraints
    constraints = [
        Kx >= demand_vector - uncertainty_radius,  # Resource sufficiency with uncertainty
        Kx <= demand_vector + acceptable_waste     # Limit on resource waste
    ]
    
    # Create and return the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, K, E, c, providers
