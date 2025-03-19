# Convex Optimizing Cloud Compute Costs

This repository contains tools and algorithms for optimizing cloud compute costs across multiple providers.

## Setup

1. Create and activate a virtual environment:
```bash
uv sync
source .venv/bin/activate
```

## Tools

### Azure VM Instance Parser

The `azure_parser.py` tool allows you to parse Azure VM instance data from an HTML file (such as the Azure pricing page) and filter/display it in various formats.

#### Usage

```bash
./experiments/azure_parser.py /path/to/azure.html [options]
```

#### Options

- `--min-vcpu INT`: Filter instances with at least this many vCPUs
- `--max-vcpu INT`: Filter instances with at most this many vCPUs
- `--min-memory FLOAT`: Filter instances with at least this much memory (GB)
- `--max-memory FLOAT`: Filter instances with at most this much memory (GB)
- `--min-cost FLOAT`: Filter instances with at least this monthly cost (USD)
- `--max-cost FLOAT`: Filter instances with at most this monthly cost (USD)
- `--sort-by FIELD`: Sort results by this field (name, vcpu, memory, storage, cost_monthly, cost_hourly)
- `--output FORMAT`: Output format (table, json, csv)
- `--limit INT`: Limit number of results displayed

#### Examples

Display small instances (2-4 vCPUs, max $100/month):
```bash
./experiments/azure_parser.py ./experiments/azure.html --min-vcpu 2 --max-vcpu 4 --max-cost 100 --limit 10
```

Find instances with 16-32 GB memory, sorted by vCPU count (JSON output):
```bash
./experiments/azure_parser.py ./experiments/azure.html --min-memory 16 --max-memory 32 --sort-by vcpu --output json
```

Find larger instances (8-16 vCPUs, min 64 GB memory), sorted by cost (CSV output):
```bash
./experiments/azure_parser.py ./experiments/azure.html --min-vcpu 8 --max-vcpu 16 --min-memory 64 --sort-by cost_monthly --output csv
```

### Multi-Cloud Data Collection

The `data_collection.py` script collects and combines instance data from multiple cloud providers:

- Azure
- Linode

Run it to generate a combined dataset:
```bash
python -m experiments.data_collection
```

This will create pickle files in the `data/` directory for each provider and a combined file.

## File Structure

- `data/`: Cached instance data (pickle files)
- `experiments/`: Scripts and tools for data collection and analysis
  - `data_collection.py`: Core data collection script
  - `azure_parser.py`: Standalone Azure instance parser
  - `azure.html`: HTML file containing Azure pricing data
- `requirements.txt`: Python dependencies

## Requirements

- Python 3.6+
- pandas
- requests
- beautifulsoup4

## License

MIT