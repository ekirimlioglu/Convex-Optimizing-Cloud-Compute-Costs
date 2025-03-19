#!/usr/bin/env python3
"""
Azure Instance Parser

A standalone script to parse Azure VM instance data from HTML and display it in a formatted way.
"""

import os
import sys
import pandas as pd
import argparse
from bs4 import BeautifulSoup
import json


def parse_azure_instance_data(html_file):
    """
    Parse Azure instance data from HTML file
    Returns structured data for various instance types
    """
    print(f"Parsing Azure instance data from {html_file}...")
    
    try:
        # Check if the file exists
        if not os.path.exists(html_file):
            raise FileNotFoundError(f"Azure HTML file not found at {html_file}")
        
        # Read the HTML file
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all rows in the pricing table
        rows = soup.select('table.data-table__table--pricing tbody tr')
        
        instances = []
        error_count = 0
        skipped_no_price_count = 0
        
        # Debug specific instances
        debug_instances = ["G3", "G4", "L16s", "NP20s"]
        
        # List to collect skipped instances
        skipped_instances = []
        
        # Extract information from each row
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 6:  # Ensure we have enough cells (some rows have more columns)
                try:
                    # Extract instance name
                    name = cells[0].get_text().strip()
                    
                    # Debug for specific instances
                    debug_mode = name in debug_instances
                    if debug_mode:
                        print(f"\nDEBUG - Found instance: {name}")
                        print(f"  Number of cells: {len(cells)}")
                        print(f"  Raw HTML: {row}")
                        for i, cell in enumerate(cells):
                            print(f"  Cell {i} text: {cell.get_text().strip()}")
                    
                    # Extract vCPU count - handle "X / Y" format
                    vcpu_text = cells[1].get_text().strip()
                    if '/' in vcpu_text:
                        # If format is "X / Y", use X as the vCPU count
                        vcpu = int(vcpu_text.split('/')[0].strip())
                    else:
                        vcpu = int(vcpu_text)
                    
                    # Extract RAM (convert to GB)
                    ram_text = cells[2].get_text().strip()
                    ram_parts = ram_text.split()
                    if len(ram_parts) >= 2:
                        ram_value = float(ram_parts[0])
                        ram_unit = ram_parts[1]
                        # Convert to GB if in different unit
                        memory = ram_value if 'GiB' in ram_unit else ram_value / 1024
                    else:
                        memory = 0.0
                    
                    # Extract temporary storage
                    storage_text = cells[3].get_text().strip()
                    storage_parts = storage_text.split()
                    if len(storage_parts) >= 2:
                        try:
                            # Remove commas from the storage value before converting to float
                            storage_value = storage_parts[0].replace(',', '')
                            storage = float(storage_value)
                        except ValueError:
                            storage = 0.0
                    else:
                        storage = 0.0
                    
                    # Try to find the price span in multiple columns
                    price_span = None
                    price_cell_index = None
                    
                    # Check multiple columns for price information
                    for idx in range(4, min(9, len(cells))):
                        test_span = cells[idx].select_one('span.price-value')
                        if test_span:
                            price_text = test_span.get_text().strip()
                            if "$" in price_text:
                                price_span = test_span
                                price_cell_index = idx
                                break
                    
                    if debug_mode:
                        if price_span:
                            print(f"  Found price in cell {price_cell_index}: {price_span}")
                        else:
                            print("  No price span found in any cell")
                            print("  Skipping instance with no pricing information")
                    
                    if not price_span:
                        # Skip instances with no pricing information
                        skipped_no_price_count += 1
                        skipped_instances.append(name)
                        continue
                    
                    price_text = price_span.get_text().strip()
                    # Remove $, /month, and commas, then convert to float
                    price_text = price_text.replace('$', '').replace('/month', '').replace(',', '')
                    try:
                        cost = float(price_text)
                    except ValueError:
                        # If we can't parse the price, skip this instance
                        error_count += 1
                        continue
                    
                    instance = {
                        "name": name,
                        "vcpu": vcpu,
                        "memory": memory,
                        "storage": storage,
                        "cost_monthly": cost,
                        "cost_hourly": cost / 730.0  # Convert monthly cost to hourly (730 hours per month)
                    }
                    instances.append(instance)
                    
                except Exception as e:
                    error_count += 1
                    if debug_mode:
                        print(f"  Error processing instance: {e}")
                    continue
        
        # If we couldn't get data, raise an exception
        if len(instances) == 0:
            raise Exception("No instance data found in Azure HTML")
        
        print(f"Successfully parsed {len(instances)} Azure instances (skipped {error_count} with parsing errors)")
        print(f"Skipped {skipped_no_price_count} instances with no pricing: {', '.join(skipped_instances)}")
        
        return instances
        
    except Exception as e:
        print(f"Error parsing Azure data: {e}")
        return []


def filter_instances(instances, min_vcpu=None, max_vcpu=None, min_memory=None, max_memory=None, 
                    min_cost=None, max_cost=None, sort_by='cost_monthly'):
    """Filter instances based on provided criteria"""
    
    filtered_instances = instances.copy()
    
    # Apply filters
    if min_vcpu is not None:
        filtered_instances = [i for i in filtered_instances if i['vcpu'] >= min_vcpu]
    if max_vcpu is not None:
        filtered_instances = [i for i in filtered_instances if i['vcpu'] <= max_vcpu]
    if min_memory is not None:
        filtered_instances = [i for i in filtered_instances if i['memory'] >= min_memory]
    if max_memory is not None:
        filtered_instances = [i for i in filtered_instances if i['memory'] <= max_memory]
    if min_cost is not None:
        filtered_instances = [i for i in filtered_instances if i['cost_monthly'] >= min_cost]
    if max_cost is not None:
        filtered_instances = [i for i in filtered_instances if i['cost_monthly'] <= max_cost]
    
    # Sort instances
    if sort_by in ['name', 'vcpu', 'memory', 'storage', 'cost_monthly', 'cost_hourly']:
        filtered_instances.sort(key=lambda x: x[sort_by])
    
    return filtered_instances


def display_instances(instances, limit=None, output_format='table'):
    """Display instance information in the specified format"""
    
    if limit is not None:
        instances = instances[:limit]
    
    if output_format == 'json':
        print(json.dumps(instances, indent=2))
    
    elif output_format == 'csv':
        if instances:
            # Print header
            print(','.join(instances[0].keys()))
            # Print values
            for instance in instances:
                print(','.join(str(val) for val in instance.values()))
    
    else:  # Default to table format
        # Convert to DataFrame for pretty printing
        df = pd.DataFrame(instances)
        
        # Format DataFrame columns for better display
        if 'cost_monthly' in df.columns:
            df['cost_monthly'] = df['cost_monthly'].apply(lambda x: f"${x:.2f}")
        if 'cost_hourly' in df.columns:
            df['cost_hourly'] = df['cost_hourly'].apply(lambda x: f"${x:.4f}")
        if 'memory' in df.columns:
            df['memory'] = df['memory'].apply(lambda x: f"{x:.1f} GB")
        if 'storage' in df.columns:
            df['storage'] = df['storage'].apply(lambda x: f"{x:.0f} GB")
            
        print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Parse and display Azure VM instance information')
    parser.add_argument('html_file', help='Path to the Azure HTML file')
    parser.add_argument('--min-vcpu', type=int, help='Minimum vCPU count')
    parser.add_argument('--max-vcpu', type=int, help='Maximum vCPU count')
    parser.add_argument('--min-memory', type=float, help='Minimum memory in GB')
    parser.add_argument('--max-memory', type=float, help='Maximum memory in GB')
    parser.add_argument('--min-cost', type=float, help='Minimum monthly cost in USD')
    parser.add_argument('--max-cost', type=float, help='Maximum monthly cost in USD')
    parser.add_argument('--sort-by', choices=['name', 'vcpu', 'memory', 'storage', 'cost_monthly', 'cost_hourly'], 
                        default='cost_monthly', help='Sort results by this field')
    parser.add_argument('--output', choices=['table', 'json', 'csv'], default='table', 
                        help='Output format')
    parser.add_argument('--limit', type=int, help='Limit number of results displayed')
    
    args = parser.parse_args()
    
    # Parse instance data
    instances = parse_azure_instance_data(args.html_file)
    
    if not instances:
        sys.exit(1)
    
    # Filter instances
    filtered_instances = filter_instances(
        instances,
        min_vcpu=args.min_vcpu,
        max_vcpu=args.max_vcpu,
        min_memory=args.min_memory,
        max_memory=args.max_memory,
        min_cost=args.min_cost,
        max_cost=args.max_cost,
        sort_by=args.sort_by
    )
    
    # Display results
    print(f"\nDisplaying {len(filtered_instances)} instances:")
    display_instances(filtered_instances, limit=args.limit, output_format=args.output)


if __name__ == "__main__":
    main() 