# Prime Lattice Light Simulation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Validation: Passed](https://img.shields.io/badge/Validation-PASSED-green.svg)](#validation)

A validated Monte Carlo ray-tracing simulation comparing prime number-based architectural perforation patterns against grid, random, and Fibonacci distributions for light modulation applications.

## 📊 Key Results

| Pattern | CV | Uniformity | Glare | Moiré | DF% | UDI% |
|---------|-----|------------|-------|-------|-----|------|
| **Prime** | 0.292 | 0.849 | **0.12** | 0.285 | **1.94** | 87.3 |
| Grid | 0.303 | 0.859 | 0.14 | 1.000 | 1.23 | 100.0 |
| Random | 0.368 | 0.883 | 0.27 | 0.107 | 0.64 | 100.0 |
| Fibonacci | **0.173** | **0.970** | 0.21 | **0.101** | 0.72 | 100.0 |

### Key Findings

- **Prime pattern** achieves highest daylight factor (1.94%) and lowest glare (0.12)
- **72% moiré reduction** vs regular grids while maintaining deterministic reproducibility
- Validated against analytical solution: **RMSE = 0.022**, Correlation = 0.93

## 🔬 Validation Status

```
✓ RMSE:        0.0218  (threshold < 0.05)
✓ Correlation: 0.9328  (threshold > 0.90)
✓ Overlap:     0.8776  (threshold > 0.70)
✓ GCI:         1.00%   (threshold < 5%)

Status: PASSED - Publication Ready
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/tlhksy/prime-lattice-light-simulation.git
cd prime-lattice-light-simulation
pip install -r requirements.txt
```

### Run Simulation

```bash
python light_simulation_v4.5_final.py
```

### Output

```
./results_v45/
├── validation_report.png   # Validation summary
├── fig1_patterns.png       # Pattern comparison
├── fig2_metrics.png        # Performance metrics
├── fig3_profiles.png       # Cross-sections
└── fig4_fft.png            # Moiré analysis
```

## 📐 Physical Model

```
                    ☀️ SUN
                   ↙️ ↓ ↘️
                  rays
    ┌──────────────────────────────┐
    │  ○   ○       ○   ○   ○      │  ← PERFORATED PANEL
    │     ○   ○ ○       ○     ○   │    (prime positions)
    └──────────────────────────────┘
                   ↓ ↓ ↓
              transmitted light
    ═══════════════════════════════  ← GROUND PLANE
         illumination pattern
```

### Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Solar constant | 1361 W/m² | SORCE satellite |
| Atmospheric turbidity | 2.5 (Linke) | Clear sky |
| Air mass formula | Kasten-Young (1989) | ISO standard |
| Panel size | 1.0 m × 1.0 m | - |
| Panel height | 2.5 m | - |
| Hole diameter | 25 mm | - |
| Location | 41°N (Istanbul) | - |

## 📁 Project Structure

```
prime-lattice-light-simulation/
├── light_simulation_v4.5_final.py  # Main simulation code
├── fix_png_for_latex.py            # PNG utility for LaTeX
├── paper_v3_validated.tex          # LaTeX manuscript
├── requirements.txt                # Dependencies
├── README.md                       # This file
└── results_v45/                    # Output figures
```

## 🔧 Pattern Generation

### Prime Pattern
```python
# Holes at prime number positions in row-major order
n = row × width + column + 1
if is_prime(n):
    place_hole(row, column)
```

### Metrics Calculated

- **CV**: Coefficient of Variation (σ/μ)
- **Uniformity**: E_min / E_mean
- **Glare Probability**: Normalized contrast ratio
- **Moiré Risk**: FFT peak-to-mean ratio
- **Daylight Factor (DF)**: Interior/exterior illuminance ratio
- **UDI**: Useful Daylight Illuminance (100-3000 lux range)

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@misc{aksoy2024prime,
  title={Spatializing the Integer: A Generative Lattice Model for Prime 
         Number Visualization with Applications in Architectural Light Modulation},
  author={Aksoy, Talha},
  year={2024},
  note={Manuscript in preparation}
}
```

*Note: Full citation will be updated upon publication.*

## 📚 References

- Schroeder, M. R. (1975). Diffuse sound reflection by maximum-length sequences. *JASA*, 57(1), 149-150.
- Ward, G. J. (1994). The RADIANCE lighting simulation and rendering system. *SIGGRAPH '94*.
- Nabil, A., & Mardaljevic, J. (2006). Useful daylight illuminances. *Energy and Buildings*, 38(7), 905-913.
- Kasten, F., & Young, A. T. (1989). Revised optical air mass tables. *Applied Optics*, 28(22), 4735-4738.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Talha Aksoy**  
Department of Landscape Architecture  
Kırklareli University, Türkiye  
📧 talha.aksoy@klu.edu.tr

---

### AI Assistance Disclosure

This research was conducted with assistance from Claude (Anthropic) and Grok (xAI) for code development, debugging, and visualization design. The author maintains full responsibility for scientific validity and interpretation of results.
