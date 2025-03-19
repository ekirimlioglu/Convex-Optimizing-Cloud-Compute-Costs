import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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

# ---- PAPER VISUALIZATIONS ----

def create_cost_comparison_barchart(results, output_path):
    """
    Create a bar chart comparing Kubernetes vs Optimal costs across scenarios
    
    Parameters:
    results: Dictionary containing results from all scenarios
    output_path: Path to save the visualization
    """
    # Set up the figure with larger size for better readability
    plt.figure(figsize=(12, 7))
    
    # Extract data for plotting
    scenario_names = []
    k8s_costs = []
    optimal_costs = []
    savings_pcts = []
    
    for name, result in results.items():
        if 'kubernetes' in result and 'optimal' in result:
            scenario_names.append(name)
            k8s_cost = result['kubernetes']['cost']
            optimal_cost = result['optimal']['cost']
            
            k8s_costs.append(k8s_cost)
            optimal_costs.append(optimal_cost)
            
            # Calculate savings percentage
            if k8s_cost > 0:
                savings_pct = (k8s_cost - optimal_cost) / k8s_cost * 100
            else:
                savings_pct = 0
            savings_pcts.append(savings_pct)
    
    # Set up x-axis
    x = np.arange(len(scenario_names))
    width = 0.35
    
    # Create the bar chart
    ax = plt.gca()
    k8s_bars = ax.bar(x - width/2, k8s_costs, width, label='Kubernetes', color='#3498db')
    opt_bars = ax.bar(x + width/2, optimal_costs, width, label='Optimal', color='#2ecc71')
    
    # Add cost labels on the bars with better visibility
    for i, (k8s_bar, opt_bar) in enumerate(zip(k8s_bars, opt_bars)):
        # Create backgrounds for better visibility
        k8s_height = k8s_bar.get_height()
        opt_height = opt_bar.get_height()
        
        # Add cost labels with white backgrounds for contrast
        ax.text(k8s_bar.get_x() + k8s_bar.get_width()/2, k8s_height * 0.9,
                f"${k8s_costs[i]:.2f}", ha='center', va='center', color='white', fontsize=9,
                fontweight='bold', bbox=dict(facecolor='#3498db', alpha=0.8, pad=0.2, boxstyle="round"))
                
        ax.text(opt_bar.get_x() + opt_bar.get_width()/2, opt_height * 0.9,
                f"${optimal_costs[i]:.2f}", ha='center', va='center', color='white', fontsize=9,
                fontweight='bold', bbox=dict(facecolor='#2ecc71', alpha=0.8, pad=0.2, boxstyle="round"))
        
        # Add savings percentage above the bars with sufficient vertical space
        # Determine position - should be above the highest bar with a buffer
        max_height = max(k8s_height, opt_height)
        y_pos = max_height + (max(k8s_costs + optimal_costs) * 0.08)  # Add buffer space
        
        # Create text box with background for better visibility
        ax.text(x[i], y_pos, f"{savings_pcts[i]:.1f}% savings",
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='#f39c12', alpha=0.7, edgecolor='black', 
                          boxstyle='round,pad=0.3', linewidth=0.5))
    
    # Customize the plot
    plt.xlabel('Scenarios', fontsize=12, labelpad=10)
    plt.ylabel('Hourly Cost ($)', fontsize=12, labelpad=10)
    plt.title('Cost Comparison: Kubernetes vs. Optimal Allocation', fontsize=14, pad=20)
    
    # Improve x-tick labels for better readability
    plt.xticks(x, [name.replace('_', ' ').title() for name in scenario_names], rotation=45, ha='right')
    
    # Extend y-axis slightly to make room for savings percentage
    y_max = max(k8s_costs + optimal_costs) * 1.25  # Add 25% headroom
    plt.ylim(0, y_max)
    
    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend with better positioning
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    # Add descriptive note
    plt.figtext(0.5, 0.01, 
                "Figure 1: Cost comparison between Kubernetes Cluster Autoscaler and Optimal resource allocation.",
                ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])  # Adjust for the note
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True

def create_resource_radar_chart(results, scenario_name, resource_names, output_path):
    """
    Create a radar/spider chart showing resource utilization efficiency
    
    Parameters:
    results: Dictionary containing results
    scenario_name: Name of the scenario to visualize
    resource_names: List of resource names
    output_path: Path to save the visualization
    """
    # Check if we have valid data
    result = results[scenario_name]
    if 'kubernetes' not in result or 'optimal' not in result:
        print(f"Warning: Missing data for scenario {scenario_name}")
        return False
    
    # Extract data
    demand = result['demand']
    k8s_resources = result['kubernetes']['resources']
    opt_resources = result['optimal']['resources']
    
    # Calculate utilization ratios (provisioned/demanded)
    k8s_util = [k8s_resources[i]/demand[i] if demand[i] > 0 else 1 for i in range(len(resource_names))]
    opt_util = [opt_resources[i]/demand[i] if demand[i] > 0 else 1 for i in range(len(resource_names))]
    
    # Number of variables
    N = len(resource_names)
    
    # Create angles for each resource (evenly spaced)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Make the plot circular by appending the first value to the end
    angles += angles[:1]
    k8s_util += k8s_util[:1]
    opt_util += opt_util[:1]
    resource_names_plot = resource_names + [resource_names[0]]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # Plot Kubernetes utilization
    ax.plot(angles, k8s_util, 'b-', linewidth=2, label='Kubernetes AutoScaler')
    ax.fill(angles, k8s_util, 'b', alpha=0.1)
    
    # Plot Optimized utilization
    ax.plot(angles, opt_util, 'r-', linewidth=2, label='Convex Optimization')
    ax.fill(angles, opt_util, 'r', alpha=0.1)
    
    # Draw ideal utilization (exactly matching demand)
    ax.plot(angles, [1]*len(angles), 'g--', linewidth=1, label='Ideal Utilization')
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(resource_names, fontsize=12, fontweight='bold')
    
    # Set y ticks with logarithmic scale
    ax.set_yscale('log')
    ax.set_yticks([0.5, 1, 2, 4, 8, 16])
    ax.set_yticklabels(['50%', '100%', '200%', '400%', '800%', '1600%'], fontsize=10)
    
    # Set y-axis limits to accommodate the data
    y_min = min(min(k8s_util), min(opt_util), 0.5)  # Start at 50% utilization
    y_max = max(max(k8s_util), max(opt_util), 16)   # Cap at 1600% utilization
    ax.set_ylim(y_min, y_max)
    
    # Add utilization labels with clear separation to avoid overlap
    for i in range(N):
        # Calculate positions that avoid overlap
        angle_rad = angles[i]
        
        # For Kubernetes (blue) labels - place them outside the circle
        k8s_label_distance = 1.05 * max(k8s_util[i], 0.2) + 0.3
        x_k8s = k8s_label_distance * np.cos(angle_rad)
        y_k8s = k8s_label_distance * np.sin(angle_rad)
        
        # For Optimized (red) labels - place them inside the circle or at different angle
        # If utilization is close, adjust position to avoid overlap
        if abs(k8s_util[i] - opt_util[i]) < 0.2:
            # If values are very close, shift the optimized label to a slightly different angle
            offset_angle = angle_rad + 0.2  # Small angular offset
            opt_label_distance = opt_util[i]
            x_opt = opt_label_distance * np.cos(offset_angle)
            y_opt = opt_label_distance * np.sin(offset_angle)
        else:
            opt_label_distance = max(min(opt_util[i], 1.8), 0.2)
            x_opt = opt_label_distance * np.cos(angle_rad)
            y_opt = opt_label_distance * np.sin(angle_rad)
        
        # Add labels in Cartesian coordinates for better control
        # Use distinctive background colors to help differentiate
        ax.text(x_k8s, y_k8s, f"{k8s_util[i]*100:.0f}%", 
                color='white', ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='blue', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
        
        ax.text(x_opt, y_opt, f"{opt_util[i]*100:.0f}%", 
                color='white', ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='red', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    
    # Add title and legend with better positioning
    plt.title(f'Resource Utilization Efficiency - {scenario_name.replace("_", " ").title()}', 
              fontsize=14, pad=20)
    
    # Move legend outside of plot for clarity
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=10)
    
    # Add descriptive note
    plt.figtext(0.5, 0.01, 
                "Figure 2: Resource utilization efficiency comparison showing how closely allocated resources match demands.",
                ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.93])  # Adjust to make room for note and legend
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True

def create_provider_distribution_chart(results, scenario_name, providers, output_path):
    """
    Create a visualization showing provider distribution between approaches
    
    Parameters:
    results: Dictionary containing results
    scenario_name: Name of the scenario to visualize
    providers: List of provider names
    output_path: Path to save the visualization
    """
    # Check if we have valid data
    result = results[scenario_name]
    if 'kubernetes' not in result or 'optimal' not in result:
        print(f"Warning: Missing data for scenario {scenario_name}")
        return False
    
    # Extract allocation data
    k8s_allocation = result['kubernetes']['allocation']
    opt_allocation = result['optimal']['allocation']
    cloud_data = result['cloud_data']
    
    # Calculate provider totals
    k8s_provider_totals = {}
    opt_provider_totals = {}
    
    for i in range(len(k8s_allocation)):
        if k8s_allocation[i] > 0:
            provider = cloud_data.iloc[i]['provider']
            cost = cloud_data.iloc[i]['cost'] * k8s_allocation[i]
            k8s_provider_totals[provider] = k8s_provider_totals.get(provider, 0) + cost
    
    for i in range(len(opt_allocation)):
        if opt_allocation[i] > 0:
            provider = cloud_data.iloc[i]['provider']
            cost = cloud_data.iloc[i]['cost'] * opt_allocation[i]
            opt_provider_totals[provider] = opt_provider_totals.get(provider, 0) + cost
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Use distinct colors that are easier to distinguish
    custom_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
    
    # Kubernetes Provider Distribution - First Pie Chart
    k8s_providers = list(k8s_provider_totals.keys())
    k8s_values = list(k8s_provider_totals.values())
    
    if k8s_values:
        # Check if we have too many providers which might cause overlap
        if len(k8s_providers) > 5:
            # Use a better technique for many segments
            wedges, texts = ax1.pie(
                k8s_values, 
                labels=None,  # No direct labels
                startangle=90,
                colors=custom_colors[:len(k8s_providers)],
                wedgeprops={'edgecolor': 'white', 'linewidth': 1}
            )
            
            # Add percentage annotations inside wedges for larger segments only
            for i, wedge in enumerate(wedges):
                ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
                percentage = k8s_values[i] / sum(k8s_values) * 100
                
                # Only add text for segments larger than 5%
                if percentage > 5:
                    x = np.cos(np.deg2rad(ang))
                    y = np.sin(np.deg2rad(ang))
                    ax1.text(x * 0.7, y * 0.7, f"{percentage:.1f}%", 
                            ha='center', va='center', fontsize=9, fontweight='bold',
                            color='white', bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))
            
            # Add legend outside the pie
            ax1.legend(wedges, k8s_providers, title="Providers", 
                      loc="center left", bbox_to_anchor=(0.9, 0.5), fontsize=9)
        else:
            # For fewer providers, we can use direct labels with better placement
            wedges, texts, autotexts = ax1.pie(
                k8s_values, 
                labels=k8s_providers, 
                autopct='%1.1f%%', 
                pctdistance=0.85,  # Move percentage labels inside
                labeldistance=1.1,  # Move provider labels outside
                startangle=90,
                colors=custom_colors[:len(k8s_providers)],
                wedgeprops={'edgecolor': 'white', 'linewidth': 1}
            )
            
            # Enhance text readability
            for text in texts:
                text.set_fontsize(10)
                text.set_fontweight('bold')
            
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                
        ax1.set_title('Kubernetes Provider Distribution', fontsize=14, pad=20)
    else:
        ax1.text(0.5, 0.5, "No provider data available", ha='center', va='center')
    
    # Optimized Provider Distribution - Second Pie Chart
    opt_providers = list(opt_provider_totals.keys())
    opt_values = list(opt_provider_totals.values())
    
    if opt_values:
        # Check if we have too many providers which might cause overlap
        if len(opt_providers) > 5:
            # Use a better technique for many segments
            wedges, texts = ax2.pie(
                opt_values, 
                labels=None,  # No direct labels
                startangle=90,
                colors=custom_colors[:len(opt_providers)],
                wedgeprops={'edgecolor': 'white', 'linewidth': 1}
            )
            
            # Add percentage annotations inside wedges for larger segments only
            for i, wedge in enumerate(wedges):
                ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
                percentage = opt_values[i] / sum(opt_values) * 100
                
                # Only add text for segments larger than 5%
                if percentage > 5:
                    x = np.cos(np.deg2rad(ang))
                    y = np.sin(np.deg2rad(ang))
                    ax2.text(x * 0.7, y * 0.7, f"{percentage:.1f}%", 
                            ha='center', va='center', fontsize=9, fontweight='bold',
                            color='white', bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))
            
            # Add legend outside the pie
            ax2.legend(wedges, opt_providers, title="Providers", 
                      loc="center right", bbox_to_anchor=(0.1, 0.5), fontsize=9)
        else:
            # For fewer providers, we can use direct labels with better placement
            wedges, texts, autotexts = ax2.pie(
                opt_values, 
                labels=opt_providers, 
                autopct='%1.1f%%', 
                pctdistance=0.85,  # Move percentage labels inside
                labeldistance=1.1,  # Move provider labels outside
                startangle=90,
                colors=custom_colors[:len(opt_providers)],
                wedgeprops={'edgecolor': 'white', 'linewidth': 1}
            )
            
            # Enhance text readability
            for text in texts:
                text.set_fontsize(10)
                text.set_fontweight('bold')
            
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
        ax2.set_title('Convex Optimization Provider Distribution', fontsize=14, pad=20)
    else:
        ax2.text(0.5, 0.5, "No provider data available", ha='center', va='center')
    
    # Count providers used
    k8s_provider_count = len(k8s_provider_totals)
    opt_provider_count = len(opt_provider_totals)
    
    # Add detailed provider cost table below the charts for clearer information
    plt.suptitle(f'Provider Distribution Comparison - {scenario_name.replace("_", " ").title()}', fontsize=16, y=0.98)
    
    # Add a text table with provider costs below the charts
    table_text = "Provider Cost Breakdown:\n"
    table_text += "-" * 50 + "\n"
    table_text += f"{'Provider':<15}{'Kubernetes ($)':<18}{'Optimal ($)':<18}{'Savings ($)':<15}\n"
    table_text += "-" * 50 + "\n"
    
    # Collect all providers
    all_providers = sorted(set(list(k8s_provider_totals.keys()) + list(opt_provider_totals.keys())))
    
    for provider in all_providers:
        k8s_cost = k8s_provider_totals.get(provider, 0)
        opt_cost = opt_provider_totals.get(provider, 0)
        savings = k8s_cost - opt_cost
        
        table_text += f"{provider:<15}{k8s_cost:<18.2f}{opt_cost:<18.2f}{savings:<15.2f}\n"
    
    plt.figtext(0.5, 0.05, table_text, ha='center', va='top', 
                fontsize=9, fontfamily='monospace', 
                bbox=dict(facecolor='#f8f9fa', alpha=0.8, boxstyle='round,pad=0.5', edgecolor='#dddddd'))
    
    # Add descriptive note
    plt.figtext(0.5, 0.01, 
                f"Figure 3: Provider distribution comparison. Kubernetes uses {k8s_provider_count} providers, "
                f"while our optimization approach uses {opt_provider_count} providers.",
                ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.93])  # Adjust to make room for suptitle, table and note
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True

def create_instance_type_visualization(results, scenario_name, output_path):
    """
    Create a comparison of instance types used between approaches
    
    Parameters:
    results: Dictionary containing results
    scenario_name: Name of the scenario to visualize
    output_path: Path to save the visualization
    """
    # Check if we have valid data
    result = results[scenario_name]
    if 'kubernetes' not in result or 'optimal' not in result:
        print(f"Warning: Missing data for scenario {scenario_name}")
        return False
    
    # Extract data
    k8s_allocation = result['kubernetes']['allocation']
    opt_allocation = result['optimal']['allocation']
    cloud_data = result['cloud_data']
    
    # Collect instance types data
    k8s_instances = []
    opt_instances = []
    
    for i in range(len(k8s_allocation)):
        if k8s_allocation[i] > 0:
            instance = cloud_data.iloc[i]
            k8s_instances.append({
                'Provider': instance['provider'],
                'Instance': instance['name'],
                'Count': k8s_allocation[i],
                'Cost': instance['cost'] * k8s_allocation[i],
                'CPU': instance['CPU'] * k8s_allocation[i],
                'Memory': instance['Memory'] * k8s_allocation[i]
            })
    
    for i in range(len(opt_allocation)):
        if opt_allocation[i] > 0:
            instance = cloud_data.iloc[i]
            opt_instances.append({
                'Provider': instance['provider'],
                'Instance': instance['name'],
                'Count': opt_allocation[i],
                'Cost': instance['cost'] * opt_allocation[i],
                'CPU': instance['CPU'] * opt_allocation[i],
                'Memory': instance['Memory'] * opt_allocation[i]
            })
    
    # Sort instances by cost
    k8s_instances.sort(key=lambda x: x['Cost'], reverse=True)
    opt_instances.sort(key=lambda x: x['Cost'], reverse=True)
    
    # Create figure with two subplots - stacked for better use of space
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'hspace': 0.4})
    
    # Custom colors by provider with better contrast
    provider_colors = {
        'Azure': '#3498db',  # Blue
        'Linode': '#e74c3c',  # Red
        'AWS': '#2ecc71',    # Green
        'GCP': '#9b59b6'     # Purple
    }
    
    def format_instance_name(provider, instance):
        """Format instance name to fit better in plot labels"""
        # Abbreviate provider names
        provider_short = {
            'Azure': 'Az',
            'Linode': 'Ln',
            'AWS': 'AWS',
            'GCP': 'GCP'
        }.get(provider, provider[:2])
        
        # Truncate long instance names
        if len(instance) > 15:
            instance = instance[:12] + '...'
            
        return f"{instance} ({provider_short})"
    
    # Kubernetes Instance Types
    if k8s_instances:
        # Format instance names for better display
        instance_names = [format_instance_name(i['Provider'], i['Instance']) for i in k8s_instances]
        instance_counts = [i['Count'] for i in k8s_instances]
        instance_costs = [i['Cost'] for i in k8s_instances]
        
        # Get colors based on provider
        bar_colors = [provider_colors.get(i['Provider'], '#CCCCCC') for i in k8s_instances]
        
        # Determine y-axis limits for consistent scale between plots
        max_count = max(instance_counts) if instance_counts else 0
        
        # Create horizontal bar chart with improved spacing
        bars = ax1.barh(instance_names, instance_counts, color=bar_colors, height=0.6)
        
        # Add count and cost annotations with better positioning
        for i, bar in enumerate(bars):
            width = bar.get_width()
            
            # Add count inside bar for larger bars
            if width > max_count * 0.2:  # Only for bars that are large enough
                # Place count inside the bar
                ax1.text(width * 0.5, bar.get_y() + bar.get_height()/2, 
                        f"{int(instance_counts[i])}",
                        ha='center', va='center', color='white', fontsize=9, fontweight='bold')
            else:
                # Place count to the right of the bar
                ax1.text(width + max_count * 0.02, bar.get_y() + bar.get_height()/2, 
                        f"{int(instance_counts[i])}",
                        ha='left', va='center', color='black', fontsize=9)
            
            # Add cost to the right with background for better visibility
            cost = instance_costs[i]
            ax1.text(max_count * 1.1, bar.get_y() + bar.get_height()/2, 
                    f"${cost:.2f}",
                    ha='left', va='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#f8f9fa', ec='#dddddd'))
        
        # Set axis limits to ensure consistent scale and enough room for annotations
        ax1.set_xlim(0, max_count * 1.3)
        
        # Customize appearance
        ax1.set_title('Kubernetes Instance Allocation', fontsize=14, pad=15)
        ax1.set_xlabel('Number of Instances', fontsize=11)
        ax1.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Add cost info to y-axis label for clarity
        ax1.set_ylabel('Instance Types (Provider)', fontsize=11)
    else:
        ax1.text(0.5, 0.5, "No instance data available", ha='center', va='center')
    
    # Optimized Instance Types
    if opt_instances:
        # Format instance names for better display
        instance_names = [format_instance_name(i['Provider'], i['Instance']) for i in opt_instances]
        instance_counts = [i['Count'] for i in opt_instances]
        instance_costs = [i['Cost'] for i in opt_instances]
        
        # Get colors based on provider
        bar_colors = [provider_colors.get(i['Provider'], '#CCCCCC') for i in opt_instances]
        
        # Determine max count for consistent scale
        k8s_max = max([i['Count'] for i in k8s_instances]) if k8s_instances else 0
        opt_max = max(instance_counts) if instance_counts else 0
        max_count = max(k8s_max, opt_max)
        
        # Create horizontal bar chart
        bars = ax2.barh(instance_names, instance_counts, color=bar_colors, height=0.6)
        
        # Add count and cost annotations with better positioning
        for i, bar in enumerate(bars):
            width = bar.get_width()
            
            # Add count inside bar for larger bars
            if width > max_count * 0.2:  # Only for bars that are large enough
                # Place count inside the bar
                ax2.text(width * 0.5, bar.get_y() + bar.get_height()/2, 
                        f"{int(instance_counts[i])}",
                        ha='center', va='center', color='white', fontsize=9, fontweight='bold')
            else:
                # Place count to the right of the bar
                ax2.text(width + max_count * 0.02, bar.get_y() + bar.get_height()/2, 
                        f"{int(instance_counts[i])}",
                        ha='left', va='center', color='black', fontsize=9)
            
            # Add cost to the right with background for better visibility
            cost = instance_costs[i]
            ax2.text(max_count * 1.1, bar.get_y() + bar.get_height()/2, 
                    f"${cost:.2f}",
                    ha='left', va='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#f8f9fa', ec='#dddddd'))
        
        # Set axis limits to ensure consistent scale and enough room for annotations
        ax2.set_xlim(0, max_count * 1.3)
        
        # Customize appearance
        ax2.set_title('Convex Optimization Instance Allocation', fontsize=14, pad=15)
        ax2.set_xlabel('Number of Instances', fontsize=11)
        ax2.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Add cost info to y-axis label for clarity
        ax2.set_ylabel('Instance Types (Provider)', fontsize=11)
    else:
        ax2.text(0.5, 0.5, "No instance data available", ha='center', va='center')
    
    # Add provider legend with colored boxes
    all_providers = sorted(set([i['Provider'] for i in k8s_instances + opt_instances]))
    legend_handles = [plt.Rectangle((0,0),1,1, color=provider_colors.get(p, '#CCCCCC')) for p in all_providers]
    
    # Add legend at bottom of figure
    fig.legend(legend_handles, all_providers, 
              loc='lower center', ncol=len(all_providers),
              bbox_to_anchor=(0.5, 0.02), fontsize=11)
    
    # Add overall title
    plt.suptitle(f'Instance Type Allocation - {scenario_name.replace("_", " ").title()}', 
                fontsize=16, y=0.98)
    
    # Add descriptive note below the legend
    plt.figtext(0.5, 0.01, 
                "Figure 4: Comparison of instance types allocated by each approach, showing count and hourly cost for each instance type.",
                ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])  # Adjust to make room for title, legend and note
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True

def create_scaling_behavior_chart(results, output_path):
    """
    Create visualization showing how cost scales with demand
    
    Parameters:
    results: Dictionary containing results from all scenarios
    output_path: Path to save the visualization
    """
    # Extract scenario names and sort by total resource demand
    scenarios = []
    
    for name, result in results.items():
        if 'kubernetes' in result and 'optimal' in result and 'demand' in result:
            # Calculate total demand as sum of all resources
            total_demand = np.sum(result['demand'])
            scenarios.append((name, total_demand, result))
    
    # Sort scenarios by total demand
    scenarios.sort(key=lambda x: x[1])
    
    # Extract data in order of increasing demand
    scenario_names = [s[0] for s in scenarios]
    total_demands = [s[1] for s in scenarios]
    k8s_costs = [s[2]['kubernetes']['cost'] for s in scenarios]
    opt_costs = [s[2]['optimal']['cost'] for s in scenarios]
    
    # Calculate overprovisioning percentage
    k8s_over = []
    opt_over = []
    
    for s in scenarios:
        result = s[2]
        demand = result['demand']
        k8s_resources = result['kubernetes']['resources']
        opt_resources = result['optimal']['resources']
        
        # Calculate average overprovisioning across all resources
        k8s_over_pct = np.mean([(k8s_resources[i] / demand[i] - 1) * 100 if demand[i] > 0 else 0 
                              for i in range(len(demand))])
        opt_over_pct = np.mean([(opt_resources[i] / demand[i] - 1) * 100 if demand[i] > 0 else 0 
                              for i in range(len(demand))])
        
        k8s_over.append(k8s_over_pct)
        opt_over.append(opt_over_pct)
    
    # Create figure with two subplots with more space between them
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'wspace': 0.3})
    
    # Plot costs vs total demand with improved styling
    ax1.plot(total_demands, k8s_costs, 'bo-', label='Kubernetes AutoScaler', linewidth=3, markersize=8)
    ax1.plot(total_demands, opt_costs, 'ro-', label='Convex Optimization', linewidth=3, markersize=8)
    
    # Fill area between the curves to highlight savings
    ax1.fill_between(total_demands, k8s_costs, opt_costs, alpha=0.2, color='green')
    
    # Add data labels with improved positioning to prevent overlap
    for i in range(len(total_demands)):
        # Place labels to avoid overlap - use alternating positions for close data points
        label_offset = max(k8s_costs) * 0.04 * (1 + (i % 2))  # Alternate offsets
        
        # For Kubernetes cost labels - place them above points
        ax1.annotate(f"${k8s_costs[i]:.2f}", 
                    xy=(total_demands[i], k8s_costs[i]),
                    xytext=(0, 10 + (i % 2) * 10),  # Vertical offset alternating
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='blue', alpha=0.8))
        
        # For optimized cost labels - place them below points
        ax1.annotate(f"${opt_costs[i]:.2f}", 
                    xy=(total_demands[i], opt_costs[i]),
                    xytext=(0, -20 - (i % 2) * 10),  # Vertical offset alternating
                    textcoords="offset points",
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='red', alpha=0.8))
        
        # Add savings percentage between curves - for odd indices, shift horizontally
        savings_pct = (k8s_costs[i] - opt_costs[i]) / k8s_costs[i] * 100
        savings_amt = k8s_costs[i] - opt_costs[i]
        
        # Position savings text based on index (odd/even) to avoid overlap
        y_pos = (k8s_costs[i] + opt_costs[i]) / 2
        x_pos = total_demands[i]
        if i > 0 and abs(total_demands[i] - total_demands[i-1]) < total_demands[-1] * 0.1:
            # If points are too close, offset this one horizontally
            x_pos = x_pos + (total_demands[-1] - total_demands[0]) * 0.03
        
        ax1.annotate(f"{savings_pct:.1f}%\n(${savings_amt:.2f})", 
                    xy=(x_pos, y_pos),
                    ha='center', va='center', fontsize=8, fontweight='bold', color='green',
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='green', alpha=0.9))
    
    # Customize cost plot for better readability
    ax1.set_title('Cost Scaling with Increasing Demand', fontsize=13, pad=15)
    ax1.set_xlabel('Total Resource Demand', fontsize=11, labelpad=8)
    ax1.set_ylabel('Hourly Cost ($)', fontsize=11, labelpad=8)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(axis='both', linestyle='--', alpha=0.3)
    
    # Add scenario labels at the bottom if there are 5 or fewer scenarios
    if len(scenario_names) <= 5:
        ax1.set_xticks(total_demands)
        ax1.set_xticklabels([name.replace('_', ' ').title() for name in scenario_names], 
                          rotation=45, ha='right', fontsize=8)
    
    # Ensure y-axis has enough room for labels
    y_max = max(k8s_costs) * 1.3
    ax1.set_ylim(0, y_max)
    
    # Plot overprovisioning vs total demand with improved styling
    ax2.plot(total_demands, k8s_over, 'bo-', label='Kubernetes AutoScaler', linewidth=3, markersize=8)
    ax2.plot(total_demands, opt_over, 'ro-', label='Convex Optimization', linewidth=3, markersize=8)
    
    # Fill area between the curves to highlight efficiency difference
    ax2.fill_between(total_demands, k8s_over, opt_over, alpha=0.2, color='purple')
    
    # Add data labels with improved positioning to prevent overlap
    for i in range(len(total_demands)):
        # Alternate label positions to avoid overlap
        k8s_offset_y = 15 + (i % 2) * 10  # Alternate vertical offset
        
        # For Kubernetes overprovisioning
        ax2.annotate(f"{k8s_over[i]:.1f}%", 
                    xy=(total_demands[i], k8s_over[i]),
                    xytext=(0, k8s_offset_y),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='blue', alpha=0.8))
        
        # For optimized overprovisioning
        opt_offset_y = -15 - (i % 2) * 10  # Alternate vertical offset
        ax2.annotate(f"{opt_over[i]:.1f}%", 
                    xy=(total_demands[i], opt_over[i]),
                    xytext=(0, opt_offset_y),
                    textcoords="offset points",
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='red', alpha=0.8))
    
    # Customize overprovisioning plot
    ax2.set_title('Resource Overprovisioning with Increasing Demand', fontsize=13, pad=15)
    ax2.set_xlabel('Total Resource Demand', fontsize=11, labelpad=8)
    ax2.set_ylabel('Average Overprovisioning (%)', fontsize=11, labelpad=8)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(axis='both', linestyle='--', alpha=0.3)
    
    # Add scenario labels at the bottom if there are 5 or fewer scenarios
    if len(scenario_names) <= 5:
        ax2.set_xticks(total_demands)
        ax2.set_xticklabels([name.replace('_', ' ').title() for name in scenario_names], 
                          rotation=45, ha='right', fontsize=8)
                          
    # Ensure y-axis has enough room for labels
    y_max = max(k8s_over) * 1.2
    ax2.set_ylim(0, y_max)
    
    # Add overall title
    plt.suptitle('Scaling Behavior Comparison: Cost and Resource Efficiency', fontsize=16, y=0.98)
    
    # Add descriptive note
    plt.figtext(0.5, 0.01, 
                "Figure 5: As resource demand increases, the gap between Kubernetes and optimal allocation widens, showing greater savings at scale.",
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])  # Adjust to make room for suptitle and note
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True

def create_paper_visualizations(results, resource_names, providers, output_dir):
    """
    Create all visualizations for the research paper
    
    Parameters:
    results: Dictionary containing results from all scenarios
    resource_names: List of resource names
    providers: List of provider names
    output_dir: Directory to save visualizations
    
    Returns:
    List of created visualization paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    created_files = []
    
    # 1. Create cost comparison chart across all scenarios
    cost_chart_path = os.path.join(output_dir, 'paper_cost_comparison.png')
    if create_cost_comparison_barchart(results, cost_chart_path):
        created_files.append(cost_chart_path)
    
    # 2. Create resource radar charts for each scenario
    for scenario_name in results.keys():
        radar_chart_path = os.path.join(output_dir, f'paper_{scenario_name}_resource_radar.png')
        if create_resource_radar_chart(results, scenario_name, resource_names, radar_chart_path):
            created_files.append(radar_chart_path)
    
    # 3. Create provider distribution charts for each scenario
    for scenario_name in results.keys():
        provider_chart_path = os.path.join(output_dir, f'paper_{scenario_name}_provider_distribution.png')
        if create_provider_distribution_chart(results, scenario_name, providers, provider_chart_path):
            created_files.append(provider_chart_path)
    
    # 4. Create instance type visualization for each scenario
    for scenario_name in results.keys():
        instance_chart_path = os.path.join(output_dir, f'paper_{scenario_name}_instance_types.png')
        if create_instance_type_visualization(results, scenario_name, instance_chart_path):
            created_files.append(instance_chart_path)
    
    # 5. Create scaling behavior chart
    scaling_chart_path = os.path.join(output_dir, 'paper_scaling_behavior.png')
    if create_scaling_behavior_chart(results, scaling_chart_path):
        created_files.append(scaling_chart_path)
    
    return created_files
