# vshyde: Automated HyDe Analysis Pipeline

A Python package for automating Hybridization Detection (HyDe) analysis, including RY-coding, sensitivity analysis, and visualization.

## Features

- **Automated Workflow**: One command to run HyDe, Sensitivity Analysis, and Visualization.
- **RY-Coding Support**: Automatically converts sequences to RY-coding to reduce composition bias.
- **Visualization**: Generates high-quality heatmaps and phylogenetic mappings.
- **Easy Installation**: Installs all dependencies automatically.

## Prerequisites

- **C++ Compiler**: Required to compile the underlying `HyDe` C++ extensions (e.g., `g++` on Linux/Mac, or MSVC on Windows).
- **Python**: 3.7 or higher.

## Installation

You can install `vshyde` directly from the source. The installation process will automatically fetch and compile the core `HyDe` engine.

```bash
# Clone the repository
git clone https://github.com/yourusername/vshyde.git
cd vshyde

# Install with pip
pip install .
```

*Note: The installation may take a few minutes as it compiles `phyde`.*

## Usage

After installation, the `vshyde` command is available system-wide.

### Quick Start

```bash
vshyde -i data.fasta -o OutgroupName -t tree.nwk
```

### Full Options

```text
usage: vshyde [-h] -i INPUT -o OUTGROUP -t TREE [-r RESULTS] [--run-mode {std,all}] [--pthresh PTHRESH] [--gdiff GDIFF] [-j THREADS] [-f]

Required Inputs:
  -i INPUT, --input INPUT
                        Input concatenated CDS fasta file
  -o OUTGROUP, --outgroup OUTGROUP
                        Name of the outgroup species
  -t TREE, --tree TREE  Phylogenetic tree file (Newick)

Output Control:
  -r RESULTS, --results RESULTS
                        Root directory for all results (default: pipeline_results)

Execution Modes:
  --run-mode {std,all}  Analysis mode:
                        std: Standard HyDe + Sensitivity + Viz
                        all: Run everything (Standard + RY + Diff) (default)
...
```

## Output Structure

```text
pipeline_results/
├── hyde_standard/       # Raw HyDe results
├── hyde_ry/             # RY-coded HyDe results
├── processed_std/       # Processed matrices (Standard)
├── processed_ry/        # Processed matrices (RY)
├── visualizations/      # Final Heatmaps & Plots
└── pipeline.log         # Execution Log
```

## License

MIT License.
