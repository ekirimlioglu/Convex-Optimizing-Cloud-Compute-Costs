import numpy as np
import cvxpy as cp
import pandas as pd

def greedy_allocation(cloud_data, demand_vector, K):
    """
    Implement a greedy allocation strategy for comparison
    Selects instances based on most cost-effective resource per dollar
    """
    m, n = K.shape  # m resources, n instance types
    
    # Calculate the resource-to-cost efficiency for each instance
    efficiency = np.zeros(n)
    costs = cloud_data['cost'].values
    
    # Weight efficiency by demand to prioritize scarce resources
    demand_weights = demand_vector / np.sum(demand_vector)
    
    for i in range(n):
        # Weighted sum of normalized resources
        resources_per_dollar = 0
        for j in range(m):
            if costs[i] > 0:
                resources_per_dollar += (K[j, i] / costs[i]) * demand_weights[j]
        efficiency[i] = resources_per_dollar
    
    # Sort instances by efficiency
    sorted_indices = np.argsort(-efficiency)
    
    # Perform allocation
    allocation = np.zeros(n, dtype=int)
    remaining_demand = demand_vector.copy()
    
    while np.any(remaining_demand > 0):
        # Find best instance for remaining demand
        best_instance = -1
        best_contribution = 0
        
        for idx in sorted_indices:
            # Skip if this instance doesn't contribute to remaining demand
            instance_contribution = np.minimum(K[:, idx], remaining_demand)
            contribution_metric = np.sum(instance_contribution)
            
            if contribution_metric > best_contribution:
                best_contribution = contribution_metric
                best_instance = idx
        
        if best_instance == -1 or best_contribution == 0:
            break  # No instance can further reduce the demand
        
        # Allocate the best instance
        allocation[best_instance] += 1
        remaining_demand = np.maximum(0, remaining_demand - K[:, best_instance])
    
    return allocation

def single_provider_allocation(cloud_data, demand_vector, K, E, provider_idx):
    """
    Implement an allocation strategy that uses only a single provider
    Uses CVXPY for robust optimization
    
    Parameters:
    cloud_data: DataFrame containing cost information
    demand_vector: Vector of resource demands
    K: Resource matrix where K[i,j] = amount of resource i provided by instance j
    E: Provider matrix where E[i,j] = 1 if instance j belongs to provider i
    provider_idx: Index of the provider to use
    
    Returns:
    allocation: Vector of the number of each instance type to allocate
    """
    m, n = K.shape  # m resources, n instance types
    
    # Filter to only use instances from the specified provider
    provider_mask = E[provider_idx, :] == 1
    K_provider = K[:, provider_mask]
    costs_provider = cloud_data.loc[provider_mask, 'cost'].values
    
    # Check if this provider has any instances
    if K_provider.size == 0:
        return np.zeros(n)
    
    # Create the optimization problem using CVXPY
    num_instances = K_provider.shape[1]
    x = cp.Variable(num_instances, integer=True, nonneg=True)
    
    # Objective: minimize cost
    objective = cp.Minimize(costs_provider @ x)
    
    # Constraint: meet or exceed demand for each resource
    constraints = [K_provider @ x >= demand_vector]
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.GLPK_MI)  # Use GLPK_MI for integer programming
        
        # Check if a solution was found
        if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
            x_provider = np.round(x.value).astype(int)
            
            # Convert back to full allocation vector
            x_full = np.zeros(n)
            x_full[provider_mask] = x_provider
            
            return x_full
        else:
            # No solution with this provider
            return np.zeros(n)
    except Exception as e:
        print(f"Optimization error for provider {provider_idx}: {e}")
        # If solver fails, return zeros
        return np.zeros(n)

def lowest_cost_provider_allocation(cloud_data, demand_vector, K, E, providers):
    """Find the single provider that can meet demand at lowest cost"""
    best_allocation = None
    best_cost = float('inf')
    
    for i in range(len(providers)):
        allocation = single_provider_allocation(cloud_data, demand_vector, K, E, i)
        
        # Check if this allocation meets the demand
        resources = K @ allocation
        if np.all(resources >= demand_vector):
            cost = np.sum(allocation * cloud_data['cost'].values)
            if cost < best_cost:
                best_cost = cost
                best_allocation = allocation
    
    if best_allocation is None:
        # Fall back to greedy if no single provider can meet demand
        return greedy_allocation(cloud_data, demand_vector, K)
    
    return best_allocation

def multi_provider_allocation_cvxpy(cloud_data, demand_vector, K, E, providers):
    """
    Find the optimal allocation across all providers using CVXPY
    This overcomes the limitation of only using a single provider
    
    Parameters:
    Same as other functions
    
    Returns:
    allocation: Vector of the number of each instance type to allocate
    """
    m, n = K.shape  # m resources, n instance types
    costs = cloud_data['cost'].values
    
    # Create the optimization problem
    x = cp.Variable(n, integer=True, nonneg=True)
    
    # Objective: minimize cost
    objective = cp.Minimize(costs @ x)
    
    # Constraint: meet or exceed demand for each resource
    constraints = [K @ x >= demand_vector]
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.GLPK_MI)  # Use GLPK_MI for integer programming
        
        # Check if a solution was found
        if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
            return np.round(x.value).astype(int)
        else:
            # If no optimal solution, fall back to greedy
            print(f"CVXPY could not find optimal solution. Status: {problem.status}")
            return greedy_allocation(cloud_data, demand_vector, K)
    except Exception as e:
        print(f"Multi-provider optimization error: {e}")
        # If solver fails, fall back to the heuristic approach
        return lowest_cost_provider_allocation(cloud_data, demand_vector, K, E, providers)

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
    x = cp.Variable(n, integer=True, nonneg=True)
    
    # Objective: minimize cost
    objective = cp.Minimize(costs @ x)
    
    # Basic constraint: meet or exceed demand
    constraints = [K @ x >= demand_vector]
    
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
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.GLPK_MI)
        
        if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
            return np.round(x.value).astype(int)
        else:
            print(f"K8s autoscaler optimization failed. Status: {problem.status}")
            # If the problem is infeasible with node pool constraints,
            # fall back to the baseline multi-provider allocation
            return multi_provider_allocation_cvxpy(cloud_data, demand_vector, K, E, providers)
    except Exception as e:
        print(f"K8s autoscaler optimization error: {e}")
        return multi_provider_allocation_cvxpy(cloud_data, demand_vector, K, E, providers)

def kubernetes_resource_based_autoscaler(cloud_data, demand_vector, K, E, providers,
                                        existing_allocation=None, resource_idx=0):
    """
    Simulates a Kubernetes autoscaler that scales based on a single resource metric
    
    Most Kubernetes autoscalers focus on CPU or memory (not multiple resources simultaneously).
    This function simulates how a traditional autoscaler might work by focusing on a single
    resource dimension (e.g., CPU) and scales node pools accordingly.
    
    Parameters:
    cloud_data: DataFrame containing cost information
    demand_vector: Vector of resource demands
    K: Resource matrix
    E: Provider matrix
    providers: List of providers
    existing_allocation: Current allocation of instances
    resource_idx: Index of the primary resource to scale on (e.g., 0 for CPU, 1 for memory)
    
    Returns:
    allocation: Vector of the number of each instance type to allocate
    """
    m, n = K.shape
    costs = cloud_data['cost'].values
    
    # If no existing allocation, start with zero
    if existing_allocation is None:
        existing_allocation = np.zeros(n, dtype=int)
    
    # Current resources provided by existing allocation
    current_resources = K @ existing_allocation
    
    # Check if we need to scale up
    if np.all(current_resources >= demand_vector):
        return existing_allocation  # No need to scale up
    
    # Focus on the primary resource (usually CPU or memory in Kubernetes)
    primary_resource_deficit = max(0, demand_vector[resource_idx] - current_resources[resource_idx])
    
    if primary_resource_deficit == 0:
        # If the primary resource is satisfied but others aren't, this simulates
        # the limitation of traditional autoscalers focusing on a single metric
        print(f"Warning: Primary resource {resource_idx} is satisfied, but other resources are not.")
        print(f"This reflects a limitation of single-resource autoscalers.")
        return existing_allocation
    
    # Create optimization problem focused on the primary resource
    x = cp.Variable(n, integer=True, nonneg=True)
    
    # Objective: minimize cost while meeting primary resource demand
    objective = cp.Minimize(costs @ x)
    
    # Main constraint: satisfy primary resource demand
    constraints = [
        K[resource_idx, :] @ x >= demand_vector[resource_idx],
        x >= existing_allocation  # Cannot reduce existing allocation
    ]
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.GLPK_MI)
        
        if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
            allocation = np.round(x.value).astype(int)
            
            # Check if the solution satisfies all resource demands
            all_resources = K @ allocation
            if not np.all(all_resources >= demand_vector):
                print("Warning: The single-resource autoscaler allocation does not satisfy all resource demands.")
                print(f"Demand: {demand_vector}")
                print(f"Provided: {all_resources}")
                print("This reflects a real limitation of traditional autoscalers.")
                
            return allocation
        else:
            print(f"Single-resource autoscaler optimization failed. Status: {problem.status}")
            return kubernetes_autoscaler_allocation(cloud_data, demand_vector, K, E, providers, existing_allocation)
    except Exception as e:
        print(f"Single-resource autoscaler error: {e}")
        return kubernetes_autoscaler_allocation(cloud_data, demand_vector, K, E, providers, existing_allocation)
