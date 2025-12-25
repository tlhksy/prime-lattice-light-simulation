# Prime Lattice Light Simulation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Computational framework for analyzing light transmission performance of prime number-based perforation patterns. This repository accompanies the research paper:

> **Spatializing the Integer: A Generative Lattice Model for Prime Number Visualization with Applications in Architectural Light Modulation**
> 
> Talha Aksoy, Department of Landscape Architecture, Kırklareli University, Türkiye

## Overview

This project investigates whether prime number-derived perforation patterns offer measurable advantages over conventional distributions (regular grid, random, Fibonacci) for architectural light modulation applications.

### Key Findings

| Pattern | CV (↓) | Glare Index (↓) | Moiré Risk (↓) | Coverage (↑) |
|---------|--------|-----------------|----------------|--------------|
| **Prime-based** | 0.479 | 3.64 | 0.436 | **33.1%** |
| Regular Grid | **0.433** | 3.70 | 1.000 | 31.1% |
| Random | 0.525 | 3.83 | **0.027** | 25.4% |
| Fibonacci | 0.456 | **3.45** | 0.045 | 26.6% |

Prime-based patterns achieve **24-30% higher ground coverage** than alternatives while maintaining significantly lower moiré interference than regular grids.

## Installation

```bash
git clone https://github.com/tlhksy/prime-lattice-light-simulation.git
cd prime-lattice-light-simulation
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from src.light_simulation import run_full_simulation, create_visualizations

# Run simulation with default parameters
patterns, ground_maps, metrics, sun_angles = run_full_simulation(
    size=50,           # Pattern grid size
    target_porosity=0.3 # Target hole ratio (~30%)
)

# Generate visualization figures
create_visualizations(patterns, ground_maps, metrics, output_path='./results')
```

### Command Line

```bash
python src/light_simulation.py
```

This generates:
- `fig1_pattern_comparison.png` - Binary patterns and ground illumination maps
- `fig2_metrics_comparison.png` - Performance metrics bar charts
- `fig3_fft_analysis.png` - Frequency analysis for moiré detection
- `fig4_illumination_analysis.png` - Light distribution profiles
- `fig5_summary_table.png` - Results summary
- `latex_table.tex` - LaTeX-formatted results table

### Prime Exponent Lattice Visualization

```python
from src.prime_lattice import (
    generate_prime_pattern,
    visualize_3d_lattice,
    analyze_porosity
)

# Generate prime-based pattern
binary_pattern, continuous_pattern = generate_prime_pattern(size=50)

# Visualize 3D lattice structure
visualize_3d_lattice(max_n=100, output_path='./results')

# Analyze porosity trend
analyze_porosity(n_max=100000, output_path='./results')
```

## Methodology

### Pattern Generation

1. **Prime-based**: Holes positioned at prime indices; opacity modulated by Ω(n) (number of prime factors with multiplicity)
2. **Regular Grid**: Equidistant hole spacing
3. **Random**: Stochastic distribution (fixed seed for reproducibility)
4. **Fibonacci**: Golden angle spiral distribution

### Light Simulation

Simplified 2D raytracing model simulating light transmission through perforated panels:
- 7 solar positions (altitude 15°–60°, azimuth -60° to +60°)
- Ground illumination mapping
- Penumbra modeling with Gaussian falloff

### Performance Metrics

- **Coefficient of Variation (CV)**: σ/μ of illumination (lower = more uniform)
- **Glare Index**: max/mean ratio (lower = reduced hotspots)
- **Moiré Risk**: Peak-to-mean ratio in FFT spectrum (lower = less periodic interference)
- **Coverage**: Percentage of ground receiving significant illumination

## Project Structure

```
prime-lattice-light-simulation/
├── src/
│   ├── light_simulation.py    # Main simulation code
│   ├── prime_lattice.py       # Prime lattice generation & visualization
│   └── utils.py               # Helper functions
├── results/                   # Output figures and data
├── docs/                      # Additional documentation
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SciPy

## Citation

If you use this code in your research, please cite:

```bibtex
@article{aksoy2025prime,
  title={Spatializing the Integer: A Generative Lattice Model for Prime Number 
         Visualization with Applications in Architectural Light Modulation},
  author={Aksoy, Talha},
  journal={Architectural Science Review},
  year={2025},
  publisher={Taylor \& Francis}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Talha Aksoy**  
Department of Landscape Architecture  
Kırklareli University, Türkiye  
Email: talha.aksoy@klu.edu.tr

## Acknowledgments

- Schroeder, M. R. (1975) for foundational work on prime-based acoustic diffusion
- Hardy & Wright for mathematical foundations in *An Introduction to the Theory of Numbers*

## AI Assistance Disclosure

This project was developed with assistance from **Claude** (Anthropic), a large language model. Claude contributed to:
- Code development and optimization
- Documentation writing
- Visualization design
- Statistical analysis methodology

The author maintains full responsibility for the scientific validity and interpretation of results. All code has been reviewed and validated by the author.
