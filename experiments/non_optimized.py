import numpy as np
import cvxpy as cp
import pandas as pd

def kubernetes_autoscaler_allocation(cloud_data, demand_vector, K, E, providers, 
                                    existing_allocation=None, node_pools=None):
    """
    Simulates a realistic Kubernetes cluster autoscaler behavior
    
    In Kubernetes, autoscalers can only scale existing node pools up or down.
    They cannot arbitrarily mix and match different instance types for optimal resource allocation.
    
    Parameters:
    cloud_data: DataFrame containing cost information
    demand_vector: Vector of resource demands
    K: Resource matrix
    E: Provider matrix 
    providers: List of providers
    existing_allocation: Current allocation of instances (if None, assumes starting from scratch)
    node_pools: List of allowed node types in format [(provider_idx, instance_idx),...] 
                If None, all instance types are considered as separate node pools
    
    Returns:
    allocation: Vector of the number of each instance type to allocate
    """
    m, n = K.shape
    costs = cloud_data['cost'].values
    
    # If no node pools defined, create one pool per instance type
    if node_pools is None:
        node_pools = []
        for p_idx in range(len(providers)):
            for i_idx in range(n):
                if E[p_idx, i_idx] == 1:
                    node_pools.append((p_idx, i_idx))
    
    # If no existing allocation, start with zero
    if existing_allocation is None:
        existing_allocation = np.zeros(n, dtype=int)
    
    # Create CVXPY problem with node pool constraints
    # Fix: Create integer variable first, then add nonnegativity constraint separately
    x = cp.Variable(n, integer=True)
    
    # Objective: minimize cost
    objective = cp.Minimize(costs @ x)
    
    # Basic constraint: meet or exceed demand
    constraints = [K @ x >= demand_vector]
    
    # Add nonnegativity constraint
    constraints.append(x >= 0)
    
    # Kubernetes constraint: can only use predefined node pools
    for i in range(n):
        # Check if this instance type is in any node pool
        is_in_pool = False
        for p_idx, i_idx in node_pools:
            if i_idx == i:
                is_in_pool = True
                break
        
        # If not in any pool, constrain to 0 (or existing allocation)
        if not is_in_pool:
            constraints.append(x[i] == existing_allocation[i])
    
    # Kubernetes constraint: cannot reduce existing allocation (only scale up)
    # This simulates how Kubernetes CA typically works - it scales up when needed
    # but only scales down when specific conditions are met (which we don't model here)
    constraints.append(x >= existing_allocation)
    
    # Solve the problem using a more basic approach
    problem = cp.Problem(objective, constraints)
    
    try:
        # Try to solve the problem with SCIPY's default configuration
        # This should work with most standard installations
        problem.solve(solver=cp.SCIPY, scipy_options={'method': 'highs'})
        
        # If the solver fails, try a relaxed approach with integer rounding
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            # Create a relaxed problem without integer constraints
            x_relaxed = cp.Variable(n)
            relaxed_constraints = [K @ x_relaxed >= demand_vector, x_relaxed >= 0]
            
            for i in range(n):
                is_in_pool = False
                for p_idx, i_idx in node_pools:
                    if i_idx == i:
                        is_in_pool = True
                        break
                
                if not is_in_pool:
                    relaxed_constraints.append(x_relaxed[i] == existing_allocation[i])
            
            relaxed_constraints.append(x_relaxed >= existing_allocation)
            
            relaxed_prob = cp.Problem(cp.Minimize(costs @ x_relaxed), relaxed_constraints)
            relaxed_prob.solve(solver=cp.SCIPY)
            
            # Round the solution to get integer values
            x_value = np.ceil(x_relaxed.value).astype(int)
            
            # Make sure the rounded solution still satisfies constraints
            resources = K @ x_value
            if np.all(resources >= demand_vector):
                return x_value
            else:
                # If rounding violates constraints, incrementally add instances
                while not np.all(resources >= demand_vector):
                    # Find the resource with the largest deficit
                    deficit = demand_vector - resources
                    deficit[deficit < 0] = 0  # Only consider resources that are short
                    
                    if np.sum(deficit) == 0:
                        break
                    
                    # Find most efficient instance to add for the deficient resource
                    resource_idx = np.argmax(deficit)
                    
                    # Calculate efficiency (resource per cost) for valid instances
                    efficiencies = np.zeros(n)
                    for i in range(n):
                        is_in_pool = False
                        for p_idx, i_idx in node_pools:
                            if i_idx == i:
                                is_in_pool = True
                                break
                        
                        if is_in_pool and costs[i] > 0:
                            efficiencies[i] = K[resource_idx, i] / costs[i]
                    
                    # Choose most efficient instance
                    instance_idx = np.argmax(efficiencies)
                    x_value[instance_idx] += 1
                    resources = K @ x_value
                
                return x_value
    except Exception as e:
        raise Exception(f"K8s autoscaler optimization failed: {str(e)}")
    
    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        return np.round(x.value).astype(int)
    else:
        raise Exception(f"K8s autoscaler optimization failed. Status: {problem.status}")

