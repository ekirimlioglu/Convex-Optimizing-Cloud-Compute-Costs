#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from non_optimized import kubernetes_autoscaler_allocation
from optimized import convex_optimize_allocation  # Import optimized model
from data_collection import prepare_cloud_data
import os

def load_cloud_instance_data():
    """
    Load real cloud instance data from the data collection module
    
    Returns:
    cloud_data: DataFrame with instance information
    K: Resource matrix (resources × instances)
    E: Provider matrix (providers × instances)
    providers: List of provider names
    resource_names: List of resource names
    """
    # Load real cloud provider data
    print("Loading real cloud provider data...")
    cloud_data = prepare_cloud_data()
    
    # Extract list of providers
    providers = cloud_data['provider'].unique().tolist()
    print(f"Found {len(providers)} providers: {providers}")
    
    # Add missing resource columns
    if 'Network' not in cloud_data.columns:
        # Estimate network capacity based on instance size
        cloud_data['Network'] = (cloud_data['vcpu'] / 2).apply(np.ceil)
    
    # Normalize column names - ensure consistency
    column_mapping = {
        'vcpu': 'CPU',
        'memory': 'Memory',
        'storage': 'Storage'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in cloud_data.columns and new_name not in cloud_data.columns:
            cloud_data[new_name] = cloud_data[old_name]
    
    # Define resource types to use in our model (removing GPU)
    resource_names = ["CPU", "Memory", "Network", "Storage"]
    
    # Validate that all required resources exist in the data
    for resource in resource_names:
        if resource not in cloud_data.columns:
            print(f"Warning: Resource '{resource}' not found in data, adding as zeros")
            cloud_data[resource] = 0
    
    # Create resource matrix K
    m = len(resource_names)  # Number of resources
    n = len(cloud_data)      # Number of instance types
    K = np.zeros((m, n))
    
    for i, resource in enumerate(resource_names):
        K[i, :] = cloud_data[resource].values
    
    # Create provider matrix E
    p = len(providers)  # Number of providers
    E = np.zeros((p, n))
    
    for i, provider in enumerate(providers):
        E[i, :] = (cloud_data["provider"] == provider).astype(int)
    
    # Print some statistics about the data
    print(f"Loaded {n} instance types across {p} providers")
    print(f"Resource dimensions: {m} ({', '.join(resource_names)})")
    
    return cloud_data, K, E, providers, resource_names

def get_instance_index(cloud_data, instance_name, provider=None):
    """
    Get the index of an instance by name
    
    Parameters:
    cloud_data: DataFrame with instance data
    instance_name: Name of the instance to find
    provider: Optional provider name to disambiguate
    
    Returns:
    Index of the instance (or -1 if not found)
    """
    if provider:
        mask = (cloud_data['name'] == instance_name) & (cloud_data['provider'] == provider)
    else:
        mask = cloud_data['name'] == instance_name
    
    if not mask.any():
        print(f"Warning: Instance {instance_name} not found")
        return -1
    
    return cloud_data[mask].index[0]

def create_node_pools(cloud_data, providers, pool_configs):
    """
    Create node pools based on configuration
    
    Parameters:
    cloud_data: DataFrame with instance data
    providers: List of provider names
    pool_configs: List of tuples (provider_name, instance_name)
    
    Returns:
    node_pools: List of (provider_idx, instance_idx) tuples
    """
    node_pools = []
    for provider_name, instance_name in pool_configs:
        provider_idx = providers.index(provider_name) if provider_name in providers else -1
        if provider_idx == -1:
            print(f"Warning: Provider {provider_name} not found, skipping node pool")
            continue
            
        instance_idx = get_instance_index(cloud_data, instance_name, provider_name)
        if instance_idx == -1:
            continue
            
        node_pools.append((provider_idx, instance_idx))
    
    return node_pools

def create_existing_allocation(cloud_data, allocation_config):
    """
    Create existing allocation vector
    
    Parameters:
    cloud_data: DataFrame with instance data
    allocation_config: Dict mapping (provider, instance_name) to counts
    
    Returns:
    existing_allocation: Vector of current instance counts
    """
    existing_allocation = np.zeros(len(cloud_data), dtype=int)
    
    for (provider, instance_name), count in allocation_config.items():
        idx = get_instance_index(cloud_data, instance_name, provider)
        if idx != -1:
            existing_allocation[idx] = count
    
    return existing_allocation

def run_allocation_comparison(cloud_data, K, E, providers, resource_names, 
                             demand_vector, scenario_name, existing_allocation=None, 
                             node_pools=None, description=None):
    """
    Run a comparison between non-optimized Kubernetes autoscaler and optimized convex allocation
    
    Parameters:
    cloud_data: DataFrame with instance information
    K: Resource matrix
    E: Provider matrix
    providers: List of provider names
    resource_names: List of resource names
    demand_vector: Vector of resource demands
    scenario_name: Name of the scenario for display
    existing_allocation: Optional existing allocation vector
    node_pools: Optional list of allowed node pools
    description: Optional scenario description
    
    Returns:
    Dictionary containing results from both methods
    """
    print("\n" + "="*80)
    print(f"SCENARIO: {scenario_name}")
    print("="*80)
    
    if description:
        print(f"\n{description}\n")
    
    # Print demand vector
    print("Resource demands:")
    for i, resource in enumerate(resource_names):
        print(f"  {resource}: {demand_vector[i]}")
    
    # Print existing allocation if available
    if existing_allocation is not None:
        current_resources = K @ existing_allocation
        total_existing_cost = np.sum(existing_allocation * cloud_data['cost'].values)
        
        print("\nExisting allocation:")
        for i in range(len(existing_allocation)):
            if existing_allocation[i] > 0:
                print(f"  {existing_allocation[i]} × {cloud_data.iloc[i]['name']} ({cloud_data.iloc[i]['provider']}, ${cloud_data.iloc[i]['cost']}/hr)")
        
        print("\nExisting resources:")
        for i, resource in enumerate(resource_names):
            print(f"  {resource}: {current_resources[i]}")
        
        print(f"\nExisting cost: ${total_existing_cost:.4f}/hr")
    
    # Print node pools if available
    if node_pools is not None:
        print("\nAvailable node pools:")
        for provider_idx, instance_idx in node_pools:
            provider_name = providers[provider_idx]
            instance_name = cloud_data.iloc[instance_idx]['name']
            cost = cloud_data.iloc[instance_idx]['cost']
            print(f"  {provider_name}: {instance_name} (${cost:.4f}/hr)")
    
    # Initialize results dictionary
    results = {
        'scenario_name': scenario_name,
        'demand': demand_vector,
        'providers': providers,
        'resource_names': resource_names,
        'cloud_data': cloud_data
    }
    
    # Run NON-OPTIMIZED Kubernetes Cluster Autoscaler
    print("\n----- NON-OPTIMIZED: Kubernetes Cluster Autoscaler -----")
    
    k8s_allocation = kubernetes_autoscaler_allocation(
        cloud_data, demand_vector, K, E, providers, 
        existing_allocation=existing_allocation, 
        node_pools=node_pools
    )
    
    # Calculate resources and cost for Kubernetes approach
    k8s_resources = K @ k8s_allocation
    k8s_cost = np.sum(k8s_allocation * cloud_data['cost'].values)
    
    # Print Kubernetes results
    print("\nKubernetes AutoScaler allocation:")
    for i in range(len(k8s_allocation)):
        if k8s_allocation[i] > 0:
            instance = cloud_data.iloc[i]
            print(f"  {k8s_allocation[i]} × {instance['name']} ({instance['provider']}, ${instance['cost']:.4f}/hr)")
    
    print("\nResources provided by Kubernetes AutoScaler:")
    for i, resource in enumerate(resource_names):
        satisfaction = "✓" if k8s_resources[i] >= demand_vector[i] else "✗"
        print(f"  {resource}: {k8s_resources[i]} (demanded: {demand_vector[i]}) {satisfaction}")
    
    print(f"\nKubernetes AutoScaler cost: ${k8s_cost:.4f}/hr")
    
    # Store Kubernetes results
    results['kubernetes'] = {
        'allocation': k8s_allocation,
        'resources': k8s_resources,
        'cost': k8s_cost,
        'satisfied': np.all(k8s_resources >= demand_vector)
    }
    
    # Run OPTIMIZED convex optimization allocation
    print("\n----- OPTIMIZED: Convex Optimization Model -----")
    
    # Call the optimized allocation function
    optimal_allocation = convex_optimize_allocation(
        cloud_data, demand_vector, K, E, providers
    )
    
    # Calculate resources and cost for optimal approach
    optimal_resources = K @ optimal_allocation
    optimal_cost = np.sum(optimal_allocation * cloud_data['cost'].values)
    
    # Print optimal results
    print("\nOptimal allocation:")
    for i in range(len(optimal_allocation)):
        if optimal_allocation[i] > 0:
            instance = cloud_data.iloc[i]
            print(f"  {optimal_allocation[i]} × {instance['name']} ({instance['provider']}, ${instance['cost']:.4f}/hr)")
    
    print("\nResources provided by optimal allocation:")
    for i, resource in enumerate(resource_names):
        satisfaction = "✓" if optimal_resources[i] >= demand_vector[i] else "✗"
        print(f"  {resource}: {optimal_resources[i]} (demanded: {demand_vector[i]}) {satisfaction}")
    
    print(f"\nOptimal allocation cost: ${optimal_cost:.4f}/hr")
    
    # Calculate cost difference
    cost_diff = k8s_cost - optimal_cost
    percent_diff = (cost_diff / optimal_cost * 100) if optimal_cost > 0 else 0
    print(f"\nCost comparison: Kubernetes is ${cost_diff:.4f}/hr ({percent_diff:.2f}%) more expensive than optimal")
    
    # Store optimal results
    results['optimal'] = {
        'allocation': optimal_allocation,
        'resources': optimal_resources,
        'cost': optimal_cost,
        'satisfied': np.all(optimal_resources >= demand_vector)
    }
    
    # Return all results for visualization
    return results

def main():
    """
    Main function to compare Kubernetes autoscaler with optimized convex allocation
    
    This function:
    1. Loads real cloud instance data from data_collection module
    2. Creates several realistic scenarios for resource allocation
    3. Compares non-optimized kubernetes_autoscaler_allocation with optimized convex allocation
    4. Analyzes and visualizes the results
    """
    # Create output directory for results if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Load real cloud instance data
    cloud_data, K, E, providers, resource_names = load_cloud_instance_data()
    
    if cloud_data is None or len(cloud_data) == 0:
        raise Exception("Failed to load cloud provider data from data_collection module")
    
    # Dictionary to store results from all scenarios
    results = {}
    
    # =====================================================================
    # Scenario 1: Basic Web Application (No Existing Infrastructure)
    # =====================================================================
    demand_vector = np.array([
        8,     # CPU cores
        16,    # GB Memory
        4,     # Network units
        100    # GB Storage
    ])
    
    results['scenario1'] = run_allocation_comparison(
        cloud_data, K, E, providers, resource_names,
        demand_vector,
        "New Web Application Deployment",
        description="A new web application deployment with no existing infrastructure, allowing the autoscaler to choose optimal instances."
    )
    
    # =====================================================================
    # Scenario 2: Scaling Up with Existing Infrastructure
    # =====================================================================
    # Get the most common providers from our data
    available_providers = cloud_data['provider'].unique()
    
    # Create an existing allocation with small instances
    existing_allocation_config = {}
    
    # Select small instances from available providers for our existing allocation
    for provider in available_providers[:2]:  # Use up to 2 providers
        provider_instances = cloud_data[cloud_data['provider'] == provider]
        
        # Find small instances (2-4 CPU cores)
        small_instances = provider_instances[
            (provider_instances['CPU'] >= 2) & 
            (provider_instances['CPU'] <= 4)
        ]
        
        if len(small_instances) > 0:
            # Get the first suitable instance
            instance = small_instances.iloc[0]
            # Add 1-2 instances to our existing allocation
            count = min(2, len(small_instances))
            existing_allocation_config[(provider, instance['name'])] = count
    
    existing_allocation = create_existing_allocation(cloud_data, existing_allocation_config)
    
    # Higher demand for a growing application
    demand_vector = np.array([
        16,    # CPU cores
        32,    # GB Memory
        8,     # Network units
        200    # GB Storage
    ])
    
    results['scenario2'] = run_allocation_comparison(
        cloud_data, K, E, providers, resource_names,
        demand_vector,
        "Scaling Up Existing Web Application",
        existing_allocation=existing_allocation,
        description="Traffic has increased and the existing web application needs more resources. The autoscaler will add to the existing instances."
    )
    
    # =====================================================================
    # Scenario 3: Enterprise Environment with Fixed Node Pools
    # =====================================================================
    node_pool_configs = []
    
    # Create enterprise node pools with a mix of instance types from available providers
    for provider in available_providers[:3]:  # Use up to 3 providers
        provider_instances = cloud_data[cloud_data['provider'] == provider]
        
        # Add small instances (2-4 cores)
        small_instances = provider_instances[
            (provider_instances['CPU'] >= 2) & 
            (provider_instances['CPU'] <= 4)
        ].head(2)  # Take up to 2 small instance types
        
        # Add medium instances (4-8 cores)
        medium_instances = provider_instances[
            (provider_instances['CPU'] > 4) & 
            (provider_instances['CPU'] <= 8)
        ].head(2)  # Take up to 2 medium instance types
        
        # Add large instances (8+ cores)
        large_instances = provider_instances[
            provider_instances['CPU'] > 8
        ].head(1)  # Take up to 1 large instance type
        
        # Add instances to node pools
        for instances in [small_instances, medium_instances, large_instances]:
            for _, instance in instances.iterrows():
                node_pool_configs.append((provider, instance['name']))
    
    # Create the node pools
    node_pools = create_node_pools(cloud_data, providers, node_pool_configs)
    
    # Higher demand for an enterprise application
    demand_vector = np.array([
        24,    # CPU cores
        64,    # GB Memory
        12,    # Network units
        300    # GB Storage
    ])
    
    results['scenario3'] = run_allocation_comparison(
        cloud_data, K, E, providers, resource_names,
        demand_vector,
        "Enterprise Environment with Fixed Node Pools",
        node_pools=node_pools,
        description="An enterprise environment with predefined node pool constraints, demonstrating how autoscalers are limited to specific instance types."
    )
    
    # =====================================================================
    # Scenario 4: Memory-Intensive Data Processing Workload
    # =====================================================================
    node_pool_configs = []
    
    # Create node pools with high-memory instances
    for provider in available_providers[:3]:  # Use up to 3 providers
        provider_instances = cloud_data[cloud_data['provider'] == provider]
        
        # Find high-memory instances (memory >= 16GB)
        high_mem_instances = provider_instances[
            provider_instances['Memory'] >= 16
        ].head(3)  # Take up to 3 high-memory instances
        
        # Add to node pools
        for _, instance in high_mem_instances.iterrows():
            node_pool_configs.append((provider, instance['name']))
    
    # Create the node pools
    node_pools = create_node_pools(cloud_data, providers, node_pool_configs)
    
    # Create an existing allocation with some high-memory instances
    existing_allocation_config = {}
    
    # Add one high-memory instance from each provider to existing allocation
    for provider in available_providers[:2]:  # Use up to 2 providers
        provider_instances = cloud_data[cloud_data['provider'] == provider]
        high_mem_instances = provider_instances[provider_instances['Memory'] >= 16]
        
        if len(high_mem_instances) > 0:
            instance = high_mem_instances.iloc[0]
            existing_allocation_config[(provider, instance['name'])] = 1
    
    existing_allocation = create_existing_allocation(cloud_data, existing_allocation_config)
    
    # Very high memory demand
    demand_vector = np.array([
        32,    # CPU cores
        128,   # GB Memory
        12,    # Network units
        500    # GB Storage
    ])
    
    results['scenario4'] = run_allocation_comparison(
        cloud_data, K, E, providers, resource_names,
        demand_vector,
        "Data Processing Workload with High Memory Requirements",
        existing_allocation=existing_allocation,
        node_pools=node_pools,
        description="A memory-intensive data processing workload requiring significant memory resources. Shows how the autoscaler handles specialized workloads."
    )
    
    # =====================================================================
    # Scenario 5: Resource Constraints with Limited Node Pools
    # =====================================================================
    node_pool_configs = []
    
    # Create very limited node pools with only small instances
    for provider in available_providers[:3]:  # Use up to 3 providers
        provider_instances = cloud_data[cloud_data['provider'] == provider]
        
        # Find only small instances (CPU <= 2)
        small_instances = provider_instances[
            provider_instances['CPU'] <= 2
        ].head(2)  # Take up to 2 small instances
        
        # Add to node pools
        for _, instance in small_instances.iterrows():
            node_pool_configs.append((provider, instance['name']))
    
    # Create the node pools
    node_pools = create_node_pools(cloud_data, providers, node_pool_configs)
    
    # High resource demands that will be challenging with the limited pools
    demand_vector = np.array([
        32,    # CPU cores
        64,    # GB Memory
        12,    # Network units
        300    # GB Storage
    ])
    
    results['scenario5'] = run_allocation_comparison(
        cloud_data, K, E, providers, resource_names,
        demand_vector,
        "Resource-Intensive Workload with Limited Node Pools",
        node_pools=node_pools,
        description="A resource-intensive workload with limited node pool options, demonstrating the limitations of fixed node pools in Kubernetes environments."
    )
    
    # =====================================================================
    # Summary and Visualizations
    # =====================================================================
    print("\n" + "="*80)
    print("SUMMARY OF ALL SCENARIOS")
    print("="*80)
    
    # Extract comparison metrics
    scenario_names = []
    k8s_costs = []
    optimal_costs = []
    savings = []
    
    for name, result in results.items():
        if 'kubernetes' in result and 'optimal' in result:
            scenario_names.append(name)
            k8s_cost = result['kubernetes']['cost']
            optimal_cost = result['optimal']['cost']
            saving = k8s_cost - optimal_cost
            
            k8s_costs.append(k8s_cost)
            optimal_costs.append(optimal_cost)
            savings.append(saving)
            
            # Print summary
            print(f"\n{name}:")
            print(f"  Kubernetes Cost: ${k8s_cost:.4f}/hr")
            print(f"  Optimal Cost: ${optimal_cost:.4f}/hr")
            print(f"  Potential Saving: ${saving:.4f}/hr ({saving/k8s_cost*100:.2f}%)")
    
    # Import visualization functions from visualize.py
    from visualize import create_paper_visualizations
    
    # Generate advanced visualizations for the research paper
    print("\nGenerating research paper visualizations...")
    paper_viz_dir = 'results/paper'
    paper_visualizations = create_paper_visualizations(results, resource_names, providers, paper_viz_dir)
    print(f"Created {len(paper_visualizations)} paper visualizations in {paper_viz_dir}")
    
    print("\nComparison analysis complete!")

if __name__ == "__main__":
    main()
