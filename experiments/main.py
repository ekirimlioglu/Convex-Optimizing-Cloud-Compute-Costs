import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import requests
import json
from scipy.optimize import linprog
import os

# Set random seed for reproducibility
np.random.seed(42)

# ----- OPTIMIZATION APPROACHES -----

def optimize_mixed_problem(c, alpha, p, beta1, beta2, beta3, gamma, E, K, d, mu, g, n, m,
                          initial_x=None, max_iterations=100, tolerance=1e-4):
    """Improved successive approximation method"""
    # Initialize
    if initial_x is None:
        # Better initialization: start with a basic feasible solution
        # Solve a simpler problem to get a good starting point
        x_current = np.zeros(n)
        for i in range(m):
            for j in range(n):
                if K[i,j] > 0 and d[i] > 0:
                    # Use instances that provide required resources
                    amount_needed = d[i] / K[i,j]
                    x_current[j] = max(x_current[j], amount_needed)
    else:
        x_current = initial_x

    # Convert inputs to numpy arrays
    c = np.array(c)
    E = np.array(E)
    K = np.array(K)
    d = np.array(d)
    mu = np.array(mu)
    g = np.array(g)
    
    # Track best solution
    best_x = x_current
    best_obj = float('inf')
    
    # Parameters for adaptive damping
    damping = 0.5  # Start with 50% damping
    no_improvement_count = 0
    
    print("Starting iterative optimization with improved convergence...")
    
    # Begin iterations
    for iter_num in range(max_iterations):
        # Create variable
        x = cp.Variable(n, nonneg=True)
        
        # Linearize all non-convex terms
        Ex_current = E @ x_current
        
        # Term 2: Provider consolidation penalty
        term2_value = -alpha * np.sum(np.exp(-beta1 * Ex_current))
        grad_term2 = alpha * beta1 * E.T @ np.exp(-beta1 * Ex_current)
        term2_approx = term2_value + grad_term2 @ (x - x_current)
        
        # Term 3: Volume discount term
        term3_value = -gamma * np.sum(np.log(1 + beta2 * Ex_current))
        grad_term3 = -gamma * beta2 * E.T @ (1 / (1 + beta2 * Ex_current))
        term3_approx = term3_value + grad_term3 @ (x - x_current)
        
        # Add proximal term to stabilize convergence
        proximal_term = 0.5 * cp.sum_squares(x - x_current)
        
        # Linear cost term
        term1 = c @ x + alpha * p
        
        # Resource shortage penalty with element-wise approach
        shortage_components = []
        for r in range(m):
            shortage_components.append(cp.maximum(0, d[r] - K[r] @ x)**2)
        term4 = beta3 * cp.sum(cp.hstack(shortage_components))
        
        # CRITICAL: Add hard constraints to guarantee feasibility
        constraints = [
            K @ x >= d,  # Must satisfy demand (no penalty shortages)
            K @ x <= d + g  # Limit waste
        ]
        
        # Objective with approximated terms and proximal stabilization
        objective = cp.Minimize(term1 - term2_approx - term3_approx + term4 + proximal_term)
        
        problem = cp.Problem(objective, constraints)
        
        try:
            # Try SCS first, fall back to others if needed
            problem.solve(solver=cp.SCS)
            
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                try:
                    problem.solve(solver=cp.OSQP)
                except:
                    pass
            
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                # If we still don't have a solution, return best found so far
                if iter_num > 5:  # Give it a few iterations to find something
                    break
                continue
                
            # Extract solution and apply damping
            x_new = x.value
            
            # CRITICAL FIX: Apply adaptive damping to prevent oscillation
            x_damped = damping * x_current + (1 - damping) * x_new
            
            # Calculate true objective
            Ex_new = E @ x_damped
            Kx_new = K @ x_damped
            
            # True objective calculation
            term1_val = c @ x_damped + alpha * p
            term2_val = -alpha * np.sum(np.exp(-beta1 * Ex_new))
            term3_val = -gamma * np.sum(np.log(1 + beta2 * Ex_new))
            
            term4_val = 0
            for r in range(m):
                term4_val += beta3 * max(0, d[r] - Kx_new[r])**2
            
            new_obj = term1_val + term2_val - term3_val + term4_val
            
            # Check if this improved the objective
            if new_obj < best_obj:
                best_x = x_damped.copy()
                best_obj = new_obj
                no_improvement_count = 0
                # Reduce damping if we're making progress
                damping = max(0.1, damping * 0.9)
            else:
                no_improvement_count += 1
                # Increase damping if we're not improving
                damping = min(0.9, damping * 1.2)
            
            # Check convergence
            rel_change = np.linalg.norm(x_damped - x_current) / (np.linalg.norm(x_current) + 1e-10)
            print(f"Iteration {iter_num}: change = {rel_change:.6f}, obj = {new_obj:.4f}, damping = {damping:.2f}")
            
            if rel_change < tolerance:
                print(f"Converged after {iter_num+1} iterations.")
                break
                
            # Give up if we're not improving
            if no_improvement_count > 10:
                print("No improvement for 10 iterations, stopping early.")
                break
                
            # Update current point with damped version
            x_current = x_damped
            
        except Exception as e:
            print(f"Error in iteration {iter_num}: {e}")
            if iter_num > 5:  # Give it a few chances
                break
    
    # Return best solution found
    print(f"Best objective: {best_obj}")
    return best_x, best_obj

# ----- DATA COLLECTION AND PREPARATION -----

def fetch_aws_instance_data():
    """
    Simulate fetching AWS EC2 instance data
    Returns structured data for various instance types
    """
    # Simplified representation of common EC2 instance types
    instances = [
        {"name": "t3.micro", "vcpu": 2, "memory": 1, "storage": 0, "cost": 0.0104},
        {"name": "t3.small", "vcpu": 2, "memory": 2, "storage": 0, "cost": 0.0208},
        {"name": "t3.medium", "vcpu": 2, "memory": 4, "storage": 0, "cost": 0.0416},
        {"name": "m5.large", "vcpu": 2, "memory": 8, "storage": 0, "cost": 0.096},
        {"name": "m5.xlarge", "vcpu": 4, "memory": 16, "storage": 0, "cost": 0.192},
        {"name": "m5.2xlarge", "vcpu": 8, "memory": 32, "storage": 0, "cost": 0.384},
        {"name": "c5.large", "vcpu": 2, "memory": 4, "storage": 0, "cost": 0.085},
        {"name": "c5.xlarge", "vcpu": 4, "memory": 8, "storage": 0, "cost": 0.17},
        {"name": "c5.2xlarge", "vcpu": 8, "memory": 16, "storage": 0, "cost": 0.34},
        {"name": "r5.large", "vcpu": 2, "memory": 16, "storage": 0, "cost": 0.126},
    ]
    return pd.DataFrame(instances)

def fetch_azure_instance_data():
    """
    Simulate fetching Azure VM instance data
    Returns structured data for various instance types
    """
    # Simplified representation of common Azure VM sizes
    instances = [
        {"name": "B1s", "vcpu": 1, "memory": 1, "storage": 4, "cost": 0.0124},
        {"name": "B2s", "vcpu": 2, "memory": 4, "storage": 8, "cost": 0.0496},
        {"name": "B2ms", "vcpu": 2, "memory": 8, "storage": 16, "cost": 0.0832},
        {"name": "D2s_v3", "vcpu": 2, "memory": 8, "storage": 16, "cost": 0.096},
        {"name": "D4s_v3", "vcpu": 4, "memory": 16, "storage": 32, "cost": 0.192},
        {"name": "D8s_v3", "vcpu": 8, "memory": 32, "storage": 64, "cost": 0.384},
        {"name": "E2s_v3", "vcpu": 2, "memory": 16, "storage": 32, "cost": 0.126},
        {"name": "E4s_v3", "vcpu": 4, "memory": 32, "storage": 64, "cost": 0.252},
        {"name": "F2s_v2", "vcpu": 2, "memory": 4, "storage": 16, "cost": 0.085},
        {"name": "F4s_v2", "vcpu": 4, "memory": 8, "storage": 32, "cost": 0.17},
    ]
    return pd.DataFrame(instances)

def fetch_linode_instance_data():
    """
    Simulate fetching Linode instance data
    Returns structured data for various instance types
    """
    # Simplified representation of Linode instances
    instances = [
        {"name": "Nanode 1GB", "vcpu": 1, "memory": 1, "storage": 25, "cost": 0.0075},
        {"name": "Linode 2GB", "vcpu": 1, "memory": 2, "storage": 50, "cost": 0.015},
        {"name": "Linode 4GB", "vcpu": 2, "memory": 4, "storage": 80, "cost": 0.03},
        {"name": "Linode 8GB", "vcpu": 4, "memory": 8, "storage": 160, "cost": 0.06},
        {"name": "Linode 16GB", "vcpu": 6, "memory": 16, "storage": 320, "cost": 0.12},
        {"name": "Linode 32GB", "vcpu": 8, "memory": 32, "storage": 640, "cost": 0.24},
        {"name": "Linode 64GB", "vcpu": 16, "memory": 64, "storage": 1280, "cost": 0.48},
        {"name": "Linode 96GB", "vcpu": 20, "memory": 96, "storage": 1920, "cost": 0.72},
        {"name": "Linode 128GB", "vcpu": 24, "memory": 128, "storage": 2560, "cost": 0.96},
        {"name": "Linode 192GB", "vcpu": 32, "memory": 192, "storage": 3840, "cost": 1.44},
    ]
    return pd.DataFrame(instances)

def prepare_cloud_data():
    """
    Prepare the integrated dataset from all cloud providers
    """
    aws_data = fetch_aws_instance_data()
    aws_data['provider'] = 'AWS'
    
    azure_data = fetch_azure_instance_data()
    azure_data['provider'] = 'Azure'
    
    linode_data = fetch_linode_instance_data()
    linode_data['provider'] = 'Linode'
    
    # Combine all provider data
    all_data = pd.concat([aws_data, azure_data, linode_data], ignore_index=True)
    
    # Convert memory to GB for consistency (if needed)
    # all_data['memory'] = all_data['memory'].apply(lambda x: x if x >= 1 else x * 1024)
    
    return all_data

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
    
    # For DCP compliance, we'll reformulate the provider consolidation penalty 
    # Instead of using cp.exp, we'll use auxiliary variables and constraints
    z = cp.Variable(p, nonneg=True)  # Auxiliary variable for provider usage 
    
    # Resource shortage penalty - use element-wise maximum for DCP compliance
    Kx = K @ x
    shortage_components = []
    for r in range(m):
        shortage_components.append(cp.square(cp.maximum(0, demand_vector[r] - Kx[r])))
    shortage_penalty = beta3 * cp.sum(cp.hstack(shortage_components))
    
    # Full objective function - just using the linear and quadratic terms
    # which are DCP compliant
    objective = base_cost + alpha * p + shortage_penalty 
    
    # Constraints
    constraints = [
        Kx >= demand_vector - uncertainty_radius,  # Resource sufficiency with uncertainty
        Kx <= demand_vector + acceptable_waste,    # Limit on resource waste
        E @ x >= 0  # Provider usage is non-negative
    ]
    
    # Create and return the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem, x, K, E, c, providers

# ----- EVALUATION AND ANALYSIS -----

def analyze_allocation(x_values, cloud_data, K, E, providers, demand_vector):
    """
    Analyze the optimized allocation
    
    Parameters:
    x_values: solution values for the allocation variable
    cloud_data: DataFrame containing instance information
    K: resource composition matrix
    E: provider selector matrix
    providers: list of provider names
    demand_vector: original demand vector
    
    Returns:
    results: dictionary containing analysis results
    """
    # Round x values to integers
    x_rounded = np.round(x_values).astype(int)
    
    # Calculate allocated resources
    allocated_resources = K @ x_rounded
    
    # Calculate costs
    instance_costs = cloud_data['cost'].values
    total_cost = np.sum(instance_costs * x_rounded)
    
    # Calculate provider distribution
    provider_distribution = E @ x_rounded
    active_providers = np.sum(provider_distribution > 0)
    
    # Calculate resource utilization
    utilization = demand_vector / allocated_resources * 100
    
    # Create instance allocation table
    allocation_table = []
    for i, count in enumerate(x_rounded):
        if count > 0:
            instance = cloud_data.iloc[i]
            allocation_table.append({
                'Provider': instance['provider'],
                'Instance': instance['name'],
                'Count': count,
                'Cost': instance['cost'] * count
            })
    
    # Provider breakdown
    provider_breakdown = {}
    for i, provider in enumerate(providers):
        instances_from_provider = np.where(E[i, :] == 1)[0]
        count_from_provider = np.sum(x_rounded[instances_from_provider])
        cost_from_provider = np.sum(instance_costs[instances_from_provider] * x_rounded[instances_from_provider])
        
        provider_breakdown[provider] = {
            'instance_count': count_from_provider,
            'total_cost': cost_from_provider
        }
    
    # Create result dictionary
    results = {
        'allocated_resources': allocated_resources,
        'demand_vector': demand_vector,
        'total_cost': total_cost,
        'active_providers': active_providers,
        'utilization': utilization,
        'allocation_table': allocation_table,
        'provider_breakdown': provider_breakdown
    }
    
    return results

def format_allocation_results(results):
    """Format the allocation results for display"""
    output = []
    
    # Resource allocation summary
    output.append("Resource Allocation Summary:")
    resource_types = ["vCPU", "Memory (GB)", "Storage (GB)"]
    resource_data = []
    for i, resource in enumerate(resource_types):
        resource_data.append([
            resource, 
            results['demand_vector'][i], 
            results['allocated_resources'][i],
            f"{results['utilization'][i]:.2f}%"
        ])
    
    output.append(tabulate(resource_data, 
                           headers=["Resource", "Demanded", "Allocated", "Utilization"],
                           tablefmt="grid"))
    
    # Provider breakdown
    output.append("\nProvider Breakdown:")
    provider_data = []
    for provider, data in results['provider_breakdown'].items():
        if data['instance_count'] > 0:
            provider_data.append([
                provider, 
                data['instance_count'],
                f"${data['total_cost']:.2f}"
            ])
    
    output.append(tabulate(provider_data, 
                           headers=["Provider", "Instance Count", "Total Cost"],
                           tablefmt="grid"))
    
    # Instance allocation
    output.append("\nInstance Allocation:")
    allocation_data = []
    for alloc in results['allocation_table']:
        allocation_data.append([
            alloc['Provider'],
            alloc['Instance'],
            alloc['Count'],
            f"${alloc['Cost']:.2f}"
        ])
    
    output.append(tabulate(allocation_data, 
                          headers=["Provider", "Instance Type", "Count", "Cost"],
                          tablefmt="grid"))
    
    # Summary
    output.append(f"\nTotal Cost: ${results['total_cost']:.2f}")
    output.append(f"Active Providers: {results['active_providers']} out of {len(results['provider_breakdown'])}")
    
    return "\n".join(output)

# ----- COMPARISON WITH BASELINE APPROACHES -----

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
    
    Parameters:
    provider_idx: Index of the provider to use
    """
    m, n = K.shape  # m resources, n instance types
    
    # Filter to only use instances from the specified provider
    provider_mask = E[provider_idx, :] == 1
    K_provider = K[:, provider_mask]
    
    # Ensure we're only using rows of cloud_data where the mask is True
    provider_indices = np.where(provider_mask)[0]
    costs_provider = cloud_data.iloc[provider_indices]['cost'].values
    
    # If no instances from this provider, return zeros
    if len(costs_provider) == 0 or K_provider.shape[1] == 0:
        return np.zeros(n)
    
    # Solve using linear programming for minimum cost
    # min c^T x subject to K_provider * x >= demand
    c = costs_provider
    A_ub = -K_provider  # Convert to <= form with negative coefficients
    b_ub = -demand_vector
    
    # Ensure dimensions are correct
    print(f"Provider {provider_idx}: A_ub shape: {A_ub.shape}, c shape: {c.shape}, b_ub shape: {b_ub.shape}")
    
    try:
        # Reshape/transpose A_ub if needed to match expected dimensions
        if A_ub.shape[0] != len(b_ub) or A_ub.shape[1] != len(c):
            A_ub = A_ub.T
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
        
        if result.success:
            x_provider = np.round(result.x).astype(int)
            
            # Convert back to full allocation vector
            x_full = np.zeros(n)
            x_full[provider_mask] = x_provider
            
            return x_full
        else:
            print(f"Linprog failed for provider {provider_idx}: {result.message}")
            return np.zeros(n)
    except Exception as e:
        print(f"Error in linear programming for provider {provider_idx}: {e}")
        # Fall back to a simple allocation
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

# ----- EXPERIMENT EXECUTION -----

def run_experiments(scenarios, method='direct'):
    """
    Run optimization experiments for different demand scenarios
    
    Parameters:
    scenarios: list of demand scenarios to test
    method: optimization method to use ('direct', 'successive', or 'both')
    
    Returns:
    results: dictionary of results for each scenario
    """
    cloud_data = prepare_cloud_data()
    all_results = {}
    
    for scenario_name, demand in scenarios.items():
        print(f"Running scenario: {scenario_name}")
        
        # Set up the demand vectors
        demand_vector = np.array(demand['demand'])
        uncertainty_radius = np.array(demand['uncertainty'])
        acceptable_waste = np.array(demand['waste'])
        
        # Set up and solve the convex optimization problem
        problem, x, K, E, c, providers = setup_optimization_problem(
            cloud_data, 
            demand_vector, 
            uncertainty_radius, 
            acceptable_waste,
            alpha=demand.get('alpha', 1.0),
            beta1=demand.get('beta1', 0.1),
            beta2=demand.get('beta2', 0.01), 
            beta3=demand.get('beta3', 10.0),
            gamma=demand.get('gamma', 0.5)
        )
        
        # Extract parameters for the optimization methods
        alpha = demand.get('alpha', 1.0)
        beta1 = demand.get('beta1', 0.1)
        beta2 = demand.get('beta2', 0.01)
        beta3 = demand.get('beta3', 10.0)
        gamma = demand.get('gamma', 0.5)
        n = len(cloud_data)
        m = len(demand_vector)
        p = len(providers)
        
        results_dict = {}
        
        # Direct method (original approach)
        if method in ['direct', 'both']:
            try:
                print(f"Using direct optimization method...")
                # Try SCS first
                problem.solve(solver=cp.SCS)
                
                # If SCS doesn't converge well, try ECOS as fallback
                if problem.status not in ['optimal', 'optimal_inaccurate']:
                    print(f"SCS solver status: {problem.status}. Trying ECOS...")
                    try:
                        problem.solve(solver=cp.ECOS)
                    except Exception as e:
                        print(f"ECOS solver not available: {e}")
                        try:
                            problem.solve(solver=cp.OSQP)
                        except Exception as e:
                            print(f"OSQP solver not available: {e}")
                
                if problem.status == 'optimal' or problem.status == 'optimal_inaccurate':
                    # Get the solution
                    x_values = x.value
                    
                    # Analyze the optimized allocation
                    opt_results = analyze_allocation(x_values, cloud_data, K, E, providers, demand_vector)
                    opt_results['status'] = problem.status
                    opt_results['objective_value'] = problem.value
                    opt_results['method'] = 'Direct Optimization'
                    
                    results_dict['direct'] = opt_results
                else:
                    print(f"Direct method failed. Status: {problem.status}")
                    results_dict['direct'] = {'status': problem.status}
            except Exception as e:
                print(f"Error with direct method: {e}")
                results_dict['direct'] = {'status': 'error', 'message': str(e)}
        
        # Successive approximation method (new approach)
        if method in ['successive', 'both']:
            try:
                print(f"Using successive approximation method...")
                x_opt, f_opt = optimize_mixed_problem(
                    c=c, 
                    alpha=alpha, 
                    p=p, 
                    beta1=beta1, 
                    beta2=beta2, 
                    beta3=beta3, 
                    gamma=gamma, 
                    E=E, 
                    K=K, 
                    d=demand_vector, 
                    mu=uncertainty_radius, 
                    g=acceptable_waste, 
                    n=n, 
                    m=m,
                    max_iterations=50,
                    tolerance=1e-4
                )
                
                if x_opt is not None:
                    # Analyze the optimized allocation
                    opt_results = analyze_allocation(x_opt, cloud_data, K, E, providers, demand_vector)
                    opt_results['status'] = 'optimal'
                    opt_results['objective_value'] = f_opt
                    opt_results['method'] = 'Successive Approximation'
                    
                    results_dict['successive'] = opt_results
                else:
                    print(f"Successive approximation method failed.")
                    results_dict['successive'] = {'status': 'error'}
            except Exception as e:
                print(f"Error with successive approximation method: {e}")
                results_dict['successive'] = {'status': 'error', 'message': str(e)}
        
        # Select the best result to use for baseline comparisons
        best_method = None
        if 'successive' in results_dict and results_dict['successive'].get('status') == 'optimal':
            best_method = 'successive'
        elif 'direct' in results_dict and results_dict['direct'].get('status') in ['optimal', 'optimal_inaccurate']:
            best_method = 'direct'
        
        # Store the results for this scenario
        scenario_results = {
            'cloud_data': cloud_data,
            'problem_setup': {
                'demand_vector': demand_vector,
                'uncertainty_radius': uncertainty_radius,
                'acceptable_waste': acceptable_waste,
                'K': K,
                'E': E,
                'providers': providers
            }
        }
        
        # If a successful optimization method was found, do baseline comparisons
        if best_method is not None:
            # Use the selected method for baseline comparisons
            opt_results = results_dict[best_method]
            
            # Baseline comparisons
            greedy_alloc = greedy_allocation(cloud_data, demand_vector, K)
            greedy_results = analyze_allocation(greedy_alloc, cloud_data, K, E, providers, demand_vector)
            greedy_results['name'] = 'Greedy Allocation'
            
            single_prov_alloc = lowest_cost_provider_allocation(cloud_data, demand_vector, K, E, providers)
            single_prov_results = analyze_allocation(single_prov_alloc, cloud_data, K, E, providers, demand_vector)
            single_prov_results['name'] = 'Single Provider'
            
            # Add optimal and baseline results
            scenario_results['optimal'] = opt_results
            scenario_results['greedy'] = greedy_results
            scenario_results['single_provider'] = single_prov_results
            scenario_results['status'] = 'optimal'
        else:
            # If all methods failed, store the error information
            scenario_results['status'] = 'failed'
            scenario_results['error_info'] = results_dict
        
        # If both methods were used, store the results of each method separately
        if method == 'both' and 'direct' in results_dict and 'successive' in results_dict:
            scenario_results['direct_method'] = results_dict['direct']
            scenario_results['successive_method'] = results_dict['successive']
        
        # Store all results for this scenario
        all_results[scenario_name] = scenario_results
    
    return all_results

# ----- VISUALIZATION FUNCTIONS -----

def plot_resource_allocations(results, scenario_name):
    """Plot comparison of resource allocations across methods"""
    methods = ['optimal', 'greedy', 'single_provider']
    resource_types = ["vCPU", "Memory (GB)", "Storage (GB)"]
    
    # Extract data
    data = []
    for method in methods:
        method_results = results[scenario_name][method]
        for i, resource in enumerate(resource_types):
            data.append({
                'Method': method_results.get('name', method.capitalize()),
                'Resource': resource,
                'Demanded': results[scenario_name]['problem_setup']['demand_vector'][i],
                'Allocated': method_results['allocated_resources'][i],
                'Utilization': method_results['utilization'][i]
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define bar positions
    x = np.arange(len(resource_types))
    width = 0.25
    
    # Plot bars for each method
    for i, method in enumerate(df['Method'].unique()):
        df_method = df[df['Method'] == method]
        demanded = df_method['Demanded'].values
        allocated = df_method['Allocated'].values
        
        ax.bar(x + width * (i - 1), allocated, width, label=f'{method} (Allocated)')
        
        # Add utilization percentage as text
        for j, util in enumerate(df_method['Utilization'].values):
            ax.text(x[j] + width * (i - 1), allocated[j] + 0.5, f"{util:.1f}%", 
                    ha='center', va='bottom', fontsize=9)
    
    # Add demand as dashed line
    for i, demand in enumerate(demanded):
        ax.plot([x[i] - width, x[i] + width * 2], [demand, demand], 'r--', linewidth=1)
        ax.text(x[i] + width * 2, demand, 'Demand', va='bottom', fontsize=9, color='red')
    
    # Customize the plot
    ax.set_ylabel('Resource Quantity')
    ax.set_title(f'Resource Allocation Comparison - {scenario_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(resource_types)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    return fig

def plot_cost_comparison(results, scenario_name):
    """Plot cost comparison between different allocation methods"""
    methods = ['optimal', 'greedy', 'single_provider']
    
    # Extract data
    costs = []
    method_names = []
    providers_used = []
    
    for method in methods:
        method_results = results[scenario_name][method]
        costs.append(method_results['total_cost'])
        method_names.append(method_results.get('name', method.capitalize()))
        providers_used.append(method_results['active_providers'])
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot cost bars
    bar_positions = np.arange(len(methods))
    bars = ax1.bar(bar_positions, costs, 0.4, label='Total Cost')
    
    # Add cost labels on bars
    for i, v in enumerate(costs):
        ax1.text(i, v + 0.1, f"${v:.2f}", ha='center', fontsize=10)
    
    # Customize primary axis
    ax1.set_ylabel('Total Cost ($)')
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(method_names)
    
    # Create secondary axis for provider count
    ax2 = ax1.twinx()
    ax2.plot(bar_positions, providers_used, 'ro-', label='Providers Used')
    
    # Add provider count labels
    for i, v in enumerate(providers_used):
        ax2.text(i, v + 0.1, str(v), ha='center', fontsize=10, color='red')
    
    # Customize secondary axis
    ax2.set_ylabel('Number of Providers Used', color='red')
    ax2.tick_params(axis='y', colors='red')
    
    # Add title and make it look good
    plt.title(f'Cost and Provider Comparison - {scenario_name}')
    fig.tight_layout()
    
    # Add legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    
    return fig

def plot_provider_breakdown(results, scenario_name):
    """Plot breakdown of allocation by provider"""
    # Extract data for all methods
    methods = ['optimal', 'greedy', 'single_provider']
    all_data = []
    
    for method in methods:
        method_results = results[scenario_name][method]
        method_name = method_results.get('name', method.capitalize())
        
        # Get provider breakdown
        for provider, data in method_results['provider_breakdown'].items():
            if data['instance_count'] > 0:
                all_data.append({
                    'Method': method_name,
                    'Provider': provider,
                    'Cost': data['total_cost'],
                    'Instance Count': data['instance_count']
                })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot stacked bars for cost by provider
    pivot_df = df.pivot_table(index='Method', columns='Provider', values='Cost', fill_value=0)
    pivot_df.plot(kind='bar', stacked=True, ax=ax)
    
    # Add total cost annotations
    for i, method in enumerate(pivot_df.index):
        total = pivot_df.loc[method].sum()
        ax.text(i, total + 0.1, f"${total:.2f}", ha='center')
    
    # Customize plot
    ax.set_ylabel('Cost ($)')
    ax.set_title(f'Cost Breakdown by Provider - {scenario_name}')
    plt.legend(title='Provider')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def generate_simulation_scenarios():
    """Generate a set of experimental scenarios"""
    scenarios = {
        'Small Web Application': {
            'demand': [8, 16, 100],  # vCPU, Memory GB, Storage GB
            'uncertainty': [1, 2, 10],
            'waste': [4, 8, 50],
            'alpha': 1.0,   # Provider consolidation weight
            'beta1': 0.5,   # Indicator function approximation parameter
            'beta2': 0.01,  # Volume discount parameter
            'beta3': 10.0,  # Resource shortage penalty
            'gamma': 0.5    # Volume discount weight
        },
        'Medium Data Processing': {
            'demand': [16, 64, 500],
            'uncertainty': [2, 8, 50],
            'waste': [8, 16, 100],
            'alpha': 2.0,
            'beta1': 0.5,
            'beta2': 0.02,
            'beta3': 8.0,
            'gamma': 0.6
        },
        'Large Analytics Cluster': {
            'demand': [64, 256, 2000],
            'uncertainty': [8, 32, 200],
            'waste': [16, 64, 400],
            'alpha': 3.0,
            'beta1': 0.8,
            'beta2': 0.03,
            'beta3': 5.0,
            'gamma': 0.7
        },
        'CPU Intensive Workload': {
            'demand': [32, 32, 200],
            'uncertainty': [4, 4, 20],
            'waste': [8, 16, 100],
            'alpha': 1.5,
            'beta1': 0.6,
            'beta2': 0.02,
            'beta3': 12.0,
            'gamma': 0.4
        },
        'Memory Intensive Workload': {
            'demand': [16, 128, 200],
            'uncertainty': [2, 16, 20],
            'waste': [4, 32, 100],
            'alpha': 1.5,
            'beta1': 0.6,
            'beta2': 0.02,
            'beta3': 12.0,
            'gamma': 0.4
        }
    }
    
    return scenarios

# ----- MAIN FUNCTION -----

def main(method='both'):
    """
    Main function to run experiments and generate results
    
    Parameters:
    method: optimization method to use ('direct', 'successive', or 'both')
    """
    # Generate scenarios
    scenarios = generate_simulation_scenarios()
    
    # Run experiments
    results = run_experiments(scenarios, method=method)
    
    # Print and visualize results
    for scenario_name, scenario_results in results.items():
        print("\n" + "="*80)
        print(f"RESULTS FOR SCENARIO: {scenario_name}")
        print("="*80)
        
        # Check if optimization was successful
        if scenario_results.get('status') == 'optimal' and 'optimal' in scenario_results:
            # Print optimal allocation results
            print("\nOPTIMAL ALLOCATION:")
            print(format_allocation_results(scenario_results['optimal']))
            
            # Print method used
            method_used = scenario_results['optimal'].get('method', 'Direct Optimization')
            print(f"\nMethod used: {method_used}")
            
            # Print greedy allocation results
            print("\nGREEDY ALLOCATION:")
            print(format_allocation_results(scenario_results['greedy']))
            
            # Print single provider allocation results
            print("\nSINGLE PROVIDER ALLOCATION:")
            print(format_allocation_results(scenario_results['single_provider']))
            
            # If both methods were used, compare them
            if 'direct_method' in scenario_results and 'successive_method' in scenario_results:
                print("\nOPTIMIZATION METHOD COMPARISON:")
                direct_cost = scenario_results['direct_method'].get('total_cost', float('inf'))
                successive_cost = scenario_results['successive_method'].get('total_cost', float('inf'))
                
                if direct_cost != float('inf') and successive_cost != float('inf'):
                    cost_diff = abs(direct_cost - successive_cost)
                    cost_diff_percent = (cost_diff / max(direct_cost, successive_cost)) * 100
                    print(f"Direct method cost: ${direct_cost:.2f}")
                    print(f"Successive method cost: ${successive_cost:.2f}")
                    print(f"Cost difference: ${cost_diff:.2f} ({cost_diff_percent:.2f}%)")
                    
                    # Print which method performed better
                    if direct_cost < successive_cost:
                        print("Direct method performed better")
                    elif successive_cost < direct_cost:
                        print("Successive approximation method performed better")
                    else:
                        print("Both methods achieved the same cost")
            
            # Generate and save visualizations
            print("\nGenerating visualizations...")
            
            try:
                # Plot resource allocations
                fig1 = plot_resource_allocations(results, scenario_name)
                fig1.savefig(f"results/{scenario_name.replace(' ', '_')}_resources.png")
                
                # Plot cost comparison
                fig2 = plot_cost_comparison(results, scenario_name)
                fig2.savefig(f"results/{scenario_name.replace(' ', '_')}_costs.png")
                
                # Plot provider breakdown
                fig3 = plot_provider_breakdown(results, scenario_name)
                fig3.savefig(f"results/{scenario_name.replace(' ', '_')}_providers.png")
                
                plt.close('all')
                
                print(f"Visualizations saved to results directory.")
            except Exception as e:
                print(f"Error generating visualizations: {e}")
            
            # Calculate improvement metrics
            opt_cost = scenario_results['optimal']['total_cost']
            greedy_cost = scenario_results['greedy']['total_cost']
            single_cost = scenario_results['single_provider']['total_cost']
            
            print("\nIMPROVEMENT METRICS:")
            print(f"Optimization vs Greedy: {((greedy_cost - opt_cost) / greedy_cost * 100):.2f}% cost reduction")
            print(f"Optimization vs Single Provider: {((single_cost - opt_cost) / single_cost * 100):.2f}% cost reduction")
        else:
            print(f"\nScenario {scenario_name} could not be solved optimally.")
            if 'error_info' in scenario_results:
                for method, error in scenario_results['error_info'].items():
                    print(f"\n{method.capitalize()} method status: {error.get('status', 'unknown')}")
                    if 'message' in error:
                        print(f"Message: {error['message']}")
            
            # Try to still run baseline methods if possible
            try:
                # Set up problem data
                cloud_data = scenario_results['cloud_data']
                K = scenario_results['problem_setup']['K']
                E = scenario_results['problem_setup']['E']
                providers = scenario_results['problem_setup']['providers']
                demand_vector = scenario_results['problem_setup']['demand_vector']
                
                print("\nAttempting to run baseline methods even though optimization failed...")
                
                # Run greedy allocation
                greedy_alloc = greedy_allocation(cloud_data, demand_vector, K)
                greedy_results = analyze_allocation(greedy_alloc, cloud_data, K, E, providers, demand_vector)
                
                # Print greedy allocation results
                print("\nGREEDY ALLOCATION:")
                print(format_allocation_results(greedy_results))
                
                # Run single provider allocation
                single_prov_alloc = lowest_cost_provider_allocation(cloud_data, demand_vector, K, E, providers)
                single_prov_results = analyze_allocation(single_prov_alloc, cloud_data, K, E, providers, demand_vector)
                
                # Print single provider allocation results
                print("\nSINGLE PROVIDER ALLOCATION:")
                print(format_allocation_results(single_prov_results))
                
            except Exception as e:
                print(f"Could not run baseline methods: {e}")
        
        # Print divider
        print("\n" + "-"*80)
    
    print("\nAll experiments completed.")

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Run the main function with both optimization methods
    print("Running with both optimization methods for best results...")
    main(method='both')