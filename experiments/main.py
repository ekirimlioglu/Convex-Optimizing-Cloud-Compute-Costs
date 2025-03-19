#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from non_optimized import (
    greedy_allocation,
    single_provider_allocation,
    lowest_cost_provider_allocation,
    multi_provider_allocation_cvxpy,
    kubernetes_autoscaler_allocation,
    kubernetes_resource_based_autoscaler
)

def load_cloud_instance_data():
    """
    Generate sample cloud instance data for demo purposes
    
    In a real scenario, you would load this from a file or API
    
    Returns:
    cloud_data: DataFrame with instance information
    K: Resource matrix (resources × instances)
    E: Provider matrix (providers × instances)
    providers: List of provider names
    resource_names: List of resource names
    """
    # Define providers
    providers = ["AWS", "GCP", "Azure"]
    
    # Define resource types
    resource_names = ["CPU", "Memory", "GPU", "Network", "Storage"]
    
    # Create instance data
    instance_data = [
        # Format: name, provider, cost, cpu, memory, gpu, network, storage
        ("t2.micro", "AWS", 0.0116, 1, 1, 0, 1, 20),
        ("t2.small", "AWS", 0.023, 1, 2, 0, 1, 20),
        ("t2.medium", "AWS", 0.0464, 2, 4, 0, 1, 20),
        ("t2.large", "AWS", 0.0928, 2, 8, 0, 2, 40),
        ("m5.large", "AWS", 0.096, 2, 8, 0, 3, 40),
        ("m5.xlarge", "AWS", 0.192, 4, 16, 0, 4, 80),
        ("m5.2xlarge", "AWS", 0.384, 8, 32, 0, 5, 100),
        ("g4dn.xlarge", "AWS", 0.526, 4, 16, 1, 5, 125),
        
        ("e2-micro", "GCP", 0.008, 0.5, 1, 0, 1, 10),
        ("e2-small", "GCP", 0.017, 1, 2, 0, 1, 10),
        ("e2-medium", "GCP", 0.034, 1, 4, 0, 2, 10),
        ("e2-standard-2", "GCP", 0.067, 2, 8, 0, 2, 20),
        ("e2-standard-4", "GCP", 0.134, 4, 16, 0, 3, 40),
        ("n1-standard-4", "GCP", 0.19, 4, 15, 0, 4, 40),
        ("n1-standard-8", "GCP", 0.38, 8, 30, 0, 5, 80),
        ("n1-standard-16", "GCP", 0.76, 16, 60, 0, 6, 100),
        ("a2-highgpu-1g", "GCP", 3.67, 12, 85, 1, 7, 200),
        
        ("B1s", "Azure", 0.0124, 1, 1, 0, 1, 4),
        ("B1ms", "Azure", 0.0248, 1, 2, 0, 1, 4),
        ("B2s", "Azure", 0.0496, 2, 4, 0, 2, 8),
        ("B2ms", "Azure", 0.0992, 2, 8, 0, 2, 16),
        ("D2s_v3", "Azure", 0.11, 2, 8, 0, 2, 16),
        ("D4s_v3", "Azure", 0.22, 4, 16, 0, 3, 32),
        ("D8s_v3", "Azure", 0.44, 8, 32, 0, 4, 64),
        ("NC6s_v3", "Azure", 3.06, 6, 112, 1, 6, 336)
    ]
    
    # Create DataFrame
    cloud_data = pd.DataFrame(
        instance_data, 
        columns=["name", "provider", "cost", "CPU", "Memory", "GPU", "Network", "Storage"]
    )
    
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
    
    return cloud_data, K, E, providers, resource_names

def compare_allocations(cloud_data, demand_vector, K, E, providers, resource_names):
    """
    Compare different allocation strategies
    
    Parameters:
    cloud_data: DataFrame with instance information
    demand_vector: Vector of resource demands
    K: Resource matrix
    E: Provider matrix
    providers: List of provider names
    resource_names: List of resource names
    """
    print(f"\nDemand vector: {dict(zip(resource_names, demand_vector))}")
    
    # Define allocation strategies
    strategies = {
        "Greedy": lambda: greedy_allocation(cloud_data, demand_vector, K),
        "Lowest Cost Provider": lambda: lowest_cost_provider_allocation(cloud_data, demand_vector, K, E, providers),
        "Multi-provider Optimal": lambda: multi_provider_allocation_cvxpy(cloud_data, demand_vector, K, E, providers),
        "Kubernetes CA (CPU-focused)": lambda: kubernetes_resource_based_autoscaler(
            cloud_data, demand_vector, K, E, providers, resource_idx=0
        ),
        "Kubernetes CA (Memory-focused)": lambda: kubernetes_resource_based_autoscaler(
            cloud_data, demand_vector, K, E, providers, resource_idx=1
        ),
        "Kubernetes Node Pool CA": lambda: kubernetes_autoscaler_allocation(
            cloud_data, demand_vector, K, E, providers
        )
    }
    
    results = {}
    
    # Run each strategy
    for name, strategy_func in strategies.items():
        print(f"\nRunning {name} strategy...")
        allocation = strategy_func()
        
        # Calculate resources and cost
        resources = K @ allocation
        cost = np.sum(allocation * cloud_data['cost'].values)
        
        # Print summary
        print(f"Total cost: ${cost:.2f}")
        print("Resources provided:")
        for i, resource in enumerate(resource_names):
            print(f"  {resource}: {resources[i]} (demanded: {demand_vector[i]})")
        
        # Print allocation
        print("Instances allocated:")
        for i in range(len(allocation)):
            if allocation[i] > 0:
                print(f"  {allocation[i]} × {cloud_data.iloc[i]['name']} (${cloud_data.iloc[i]['cost']}/hr)")
        
        results[name] = {
            'allocation': allocation,
            'resources': resources,
            'cost': cost
        }
    
    return results

def plot_cost_comparison(results):
    """Plot cost comparison between different strategies"""
    strategies = list(results.keys())
    costs = [results[s]['cost'] for s in strategies]
    
    plt.figure(figsize=(12, 6))
    bar_plot = plt.bar(strategies, costs)
    
    # Add cost values on top of bars
    for bar, cost in zip(bar_plot, costs):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.05,
            f'${cost:.2f}',
            ha='center',
            fontweight='bold'
        )
    
    plt.title('Cost Comparison of Different Allocation Strategies')
    plt.ylabel('Hourly Cost ($)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('cost_comparison.png')
    plt.close()
    
    print("\nCost comparison plot saved as 'cost_comparison.png'")

def plot_resource_efficiency(results, demand_vector, resource_names):
    """Plot resource efficiency (resource provided / resource demanded)"""
    strategies = list(results.keys())
    resources = np.array([results[s]['resources'] for s in strategies])
    
    # Calculate efficiency (resources provided / resources demanded)
    efficiency = resources / demand_vector
    
    # Plot
    plt.figure(figsize=(14, 8))
    
    bar_width = 0.15
    x = np.arange(len(resource_names))
    
    for i, strategy in enumerate(strategies):
        plt.bar(
            x + i*bar_width - (len(strategies)-1)*bar_width/2,
            efficiency[i],
            width=bar_width,
            label=strategy
        )
    
    plt.axhline(y=1.0, color='r', linestyle='--', label='Exact demand')
    plt.xlabel('Resource Type')
    plt.ylabel('Resource Efficiency (Provided/Demanded)')
    plt.title('Resource Efficiency Comparison')
    plt.xticks(x, resource_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig('resource_efficiency.png')
    plt.close()
    
    print("Resource efficiency plot saved as 'resource_efficiency.png'")

def main():
    """Main function to run the experiments"""
    print("Loading cloud instance data...")
    cloud_data, K, E, providers, resource_names = load_cloud_instance_data()
    
    print("\nCloud providers:", providers)
    print("Resource types:", resource_names)
    print(f"Instance types: {len(cloud_data)} total across all providers")
    
    # Define a workload with diverse resource requirements
    # This example has high CPU, memory and network demands, but also requires some GPU
    demand_vector = np.array([
        32,    # CPU cores
        64,    # GB Memory
        2,     # GPUs
        10,    # Network units
        200    # GB Storage
    ])
    
    # Compare allocation strategies
    results = compare_allocations(cloud_data, demand_vector, K, E, providers, resource_names)
    
    # Plot results
    plot_cost_comparison(results)
    plot_resource_efficiency(results, demand_vector, resource_names)
    
    # Try a different workload - more balanced
    print("\n\n=========================================")
    print("Testing with a more balanced workload")
    demand_vector = np.array([
        16,    # CPU cores
        32,    # GB Memory
        0,     # GPUs
        8,     # Network units
        100    # GB Storage
    ])
    
    results = compare_allocations(cloud_data, demand_vector, K, E, providers, resource_names)
    
    # Try a GPU-heavy workload
    print("\n\n=========================================")
    print("Testing with a GPU-heavy workload")
    demand_vector = np.array([
        8,     # CPU cores
        32,    # GB Memory
        3,     # GPUs
        5,     # Network units
        50     # GB Storage
    ])
    
    results = compare_allocations(cloud_data, demand_vector, K, E, providers, resource_names)

if __name__ == "__main__":
    main()
