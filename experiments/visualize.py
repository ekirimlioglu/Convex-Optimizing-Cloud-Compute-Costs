import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# ---- KUBERNETES AUTOSCALER VISUALIZATION FUNCTIONS ----

def plot_k8s_costs(results, output_path):
    """
    Create a bar chart comparing costs across Kubernetes autoscaler scenarios
    
    Parameters:
    results: Dictionary of results from all scenarios
    output_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 6))
    
    # Get scenario names and costs for valid scenarios
    valid_scenarios = []
    costs = []
    
    for scenario_name, scenario_results in results.items():
        if 'cost' in scenario_results and scenario_results['cost'] > 0:
            valid_scenarios.append(scenario_name)
            costs.append(scenario_results['cost'])
    
    if not valid_scenarios:
        print("Warning: No valid cost data to visualize")
        return None
    
    # Create bar chart
    bars = plt.bar(valid_scenarios, costs)
    
    # Add cost labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'${height:.4f}/hr', ha='center', va='bottom', fontsize=9)
    
    # Add chart elements
    plt.title('Hourly Cost Comparison Across Kubernetes Autoscaler Scenarios', fontsize=14)
    plt.ylabel('Hourly Cost ($)', fontsize=12)
    plt.xlabel('Scenario', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save chart
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return True

def plot_k8s_resource_utilization(results, resource_names, output_path):
    """
    Create a visualization of resource utilization efficiency across Kubernetes autoscaler scenarios
    
    Parameters:
    results: Dictionary of results from all scenarios
    resource_names: List of resource names
    output_path: Path to save the visualization
    """
    plt.figure(figsize=(14, 8))
    
    # Collect data on resource utilization
    valid_scenarios = []
    utilization_data = []
    
    for scenario_name, scenario_results in results.items():
        if 'resources' in scenario_results and 'demand' in scenario_results:
            resources = scenario_results['resources']
            demands = scenario_results['demand']
            
            # Only include scenarios with valid resource data
            if len(resources) == len(demands) and np.sum(demands) > 0:
                valid_scenarios.append(scenario_name)
                
                # Calculate utilization percentages (resources / demand)
                utilization = []
                for i, resource in enumerate(resource_names):
                    if demands[i] > 0:
                        # Cap utilization at 200% to prevent extreme values
                        util_percent = min(resources[i] / demands[i] * 100, 200)
                        utilization.append(util_percent)
                    else:
                        utilization.append(100)  # If no demand, assume 100% utilization
                
                utilization_data.append(utilization)
    
    if not valid_scenarios:
        print("Warning: No valid resource data to visualize")
        return None
    
    # Convert to numpy array for easier manipulation
    utilization_data = np.array(utilization_data)
    
    # Set up plot
    x = np.arange(len(valid_scenarios))
    width = 0.15
    multiplier = 0
    
    # Plot bars for each resource
    for i, resource in enumerate(resource_names):
        offset = width * multiplier
        plt.bar(x + offset, utilization_data[:, i], width, label=resource)
        multiplier += 1
    
    # Add chart elements
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.7, label='Optimal Utilization')
    plt.title('Resource Utilization Efficiency Across Kubernetes Autoscaler Scenarios', fontsize=14)
    plt.ylabel('Utilization Percentage (%)', fontsize=12)
    plt.xlabel('Scenario', fontsize=12)
    plt.xticks(x + width * (len(resource_names) - 1) / 2, valid_scenarios, rotation=45, ha='right')
    plt.legend(title='Resource Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save chart
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return True

def plot_k8s_instance_count(results, cloud_data, output_path):
    """
    Create a visualization of instance type distribution for each Kubernetes autoscaler scenario
    
    Parameters:
    results: Dictionary of results from all scenarios
    cloud_data: DataFrame with instance information
    output_path: Path to save the visualization
    """
    # Find scenarios with valid allocation data
    valid_scenarios = []
    
    for scenario_name, scenario_results in results.items():
        if 'allocation' in scenario_results and np.sum(scenario_results['allocation']) > 0:
            valid_scenarios.append(scenario_name)
    
    if not valid_scenarios:
        print("Warning: No valid allocation data to visualize")
        return None
    
    # Create a figure with subplots for each scenario
    fig, axes = plt.subplots(len(valid_scenarios), 1, figsize=(12, 4 * len(valid_scenarios)))
    
    # If only one scenario, wrap axes in a list for consistent iteration
    if len(valid_scenarios) == 1:
        axes = [axes]
    
    # Plot instance distribution for each scenario
    for i, scenario_name in enumerate(valid_scenarios):
        ax = axes[i]
        allocation = results[scenario_name]['allocation']
        
        # Get instance counts (filtering out zeros)
        instance_indices = np.where(allocation > 0)[0]
        instance_counts = allocation[instance_indices]
        
        # Get instance names and providers
        instance_names = []
        colors = []
        
        for idx in instance_indices:
            instance = cloud_data.iloc[idx]
            instance_names.append(f"{instance['name']} ({instance['provider']})")
            
            # Use different colors for different providers
            if instance['provider'] == 'Azure':
                colors.append('skyblue')
            elif instance['provider'] == 'Linode':
                colors.append('lightgreen')
            else:
                colors.append('lightgray')
        
        # Create horizontal bar chart
        bars = ax.barh(instance_names, instance_counts, color=colors)
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{width:.0f}', ha='left', va='center')
        
        # Add chart elements
        ax.set_title(f'Instance Distribution for {scenario_name}', fontsize=12)
        ax.set_xlabel('Number of Instances', fontsize=10)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save chart
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return True

def plot_k8s_scenario_summary(results, scenario_name, cloud_data, output_path):
    """
    Create a comprehensive visualization summary for a Kubernetes autoscaler scenario
    
    Parameters:
    results: Dictionary with results for the scenario
    scenario_name: Name of the scenario
    cloud_data: DataFrame with instance information
    output_path: Path to save the visualization
    """
    # Check if we have valid data
    if ('allocation' not in results or 
        'resources' not in results or 
        'demand' not in results or
        'cost' not in results):
        print(f"Warning: Invalid data for scenario {scenario_name}")
        return None
    
    # Extract data
    allocation = results['allocation']
    resources = results['resources']
    demand = results['demand']
    cost = results['cost']
    
    # Create 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Resource allocation vs demand
    ax1 = axes[0, 0]
    resource_names = ["CPU", "Memory", "Network", "Storage"]
    x = np.arange(len(resource_names))
    width = 0.35
    
    # Calculate utilization percentages for color coding
    utilization = []
    for i in range(len(demand)):
        if demand[i] > 0:
            util = resources[i] / demand[i] * 100
        else:
            util = 100
        utilization.append(util)
    
    # Create colors based on utilization (red for under-provisioned, yellow for 100-120%, green for 120-150%, blue for over-provisioned)
    colors = []
    for util in utilization:
        if util < 100:
            colors.append('red')
        elif util <= 120:
            colors.append('green')
        elif util <= 150:
            colors.append('limegreen')
        else:
            colors.append('royalblue')
    
    # Plot allocated resources
    bars = ax1.bar(x, resources, width, label='Allocated', color=colors)
    
    # Plot demanded resources
    ax1.bar(x + width, demand, width, label='Demanded', color='lightgray', alpha=0.6)
    
    # Add utilization percentage as text
    for i, (util, res) in enumerate(zip(utilization, resources)):
        ax1.text(i, res + 0.5, f"{util:.1f}%", ha='center', va='bottom', fontsize=9)
    
    # Customize the resource allocation plot
    ax1.set_ylabel('Resource Quantity')
    ax1.set_title('Resource Allocation vs Demand')
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels(resource_names)
    ax1.legend()
    
    # 2. Instance type distribution
    ax2 = axes[0, 1]
    
    # Get instance counts (filtering out zeros)
    instance_indices = np.where(allocation > 0)[0]
    instance_counts = allocation[instance_indices]
    
    # Get instance names and providers
    instance_names = []
    colors = []
    
    if len(instance_indices) > 0:
        for idx in instance_indices:
            instance = cloud_data.iloc[idx]
            instance_names.append(f"{instance['name']} ({instance['provider']})")
            
            # Use different colors for different providers
            if instance['provider'] == 'Azure':
                colors.append('skyblue')
            elif instance['provider'] == 'Linode':
                colors.append('lightgreen')
            else:
                colors.append('lightgray')
        
        # Create horizontal bar chart
        bars = ax2.barh(instance_names, instance_counts, color=colors)
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{width:.0f}', ha='left', va='center')
    
    # Customize the instance distribution plot
    ax2.set_title('Instance Distribution')
    ax2.set_xlabel('Number of Instances')
    
    # 3. Cost per provider
    ax3 = axes[1, 0]
    
    # Calculate cost per provider
    provider_costs = {}
    
    for i, count in enumerate(allocation):
        if count > 0:
            provider = cloud_data.iloc[i]['provider']
            instance_cost = cloud_data.iloc[i]['cost'] * count
            
            if provider not in provider_costs:
                provider_costs[provider] = 0
            
            provider_costs[provider] += instance_cost
    
    if provider_costs:
        # Create pie chart of costs per provider
        providers = list(provider_costs.keys())
        costs = list(provider_costs.values())
        
        # Define colors for providers
        provider_colors = {
            'Azure': 'skyblue',
            'Linode': 'lightgreen',
            'AWS': 'orange',
            'GCP': 'red'
        }
        
        colors = [provider_colors.get(p, 'lightgray') for p in providers]
        
        # Plot pie chart
        wedges, texts, autotexts = ax3.pie(costs, labels=providers, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        
        # Make the percentage text more readable
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(9)
    
    # Customize the cost per provider plot
    ax3.set_title('Cost Distribution by Provider')
    ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # 4. Text summary
    ax4 = axes[1, 1]
    ax4.axis('off')  # Turn off axis
    
    # Create text summary with box
    summary_text = (
        f"Scenario: {scenario_name}\n\n"
        f"Total Cost: ${cost:.4f}/hr\n\n"
        f"Resource Utilization:\n"
    )
    
    for i, resource in enumerate(resource_names):
        summary_text += f"  {resource}: {utilization[i]:.1f}%\n"
    
    summary_text += f"\nProvider Distribution:\n"
    
    for provider, prov_cost in provider_costs.items():
        percentage = (prov_cost / cost) * 100 if cost > 0 else 0
        summary_text += f"  {provider}: ${prov_cost:.4f}/hr ({percentage:.1f}%)\n"
    
    # Add the text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Add overall title for the figure
    plt.suptitle(f'Kubernetes Autoscaler Scenario: {scenario_name}', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for title
    
    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return True
