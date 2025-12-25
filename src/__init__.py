"""
Prime Lattice Light Simulation Package

Computational framework for analyzing light transmission performance
of prime number-based perforation patterns.

Author: Talha Aksoy
Affiliation: Department of Landscape Architecture, Kırklareli University, Türkiye
"""

from .light_simulation import (
    run_full_simulation,
    create_visualizations,
    generate_prime_pattern,
    generate_grid_pattern,
    generate_random_pattern,
    generate_fibonacci_pattern,
    simulate_light_transmission,
    calculate_metrics,
    generate_latex_table,
)

from .prime_lattice import (
    is_prime,
    generate_primes,
    prime_factorization,
    integer_to_exponent_vector,
    analyze_porosity,
    visualize_dimension_growth,
    visualize_3d_lattice,
    visualize_pca_projection,
    visualize_porosity,
    visualize_generative_pattern,
    generate_all_lattice_figures,
)

__version__ = '1.0.0'
__author__ = 'Talha Aksoy'
__email__ = 'talha.aksoy@klu.edu.tr'
