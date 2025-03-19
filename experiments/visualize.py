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
