"""
Light Transmission Simulation for Architectural Perforation Patterns

This module compares the light transmission performance of four perforation patterns:
- Prime-based (holes at prime positions, opacity by Ω(n))
- Regular Grid (equidistant spacing)
- Random (stochastic distribution)
- Fibonacci (golden angle spiral)

Author: Talha Aksoy
Affiliation: Department of Landscape Architecture, Kırklareli University, Türkiye
Email: talha.aksoy@klu.edu.tr

License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import fft
from scipy.ndimage import uniform_filter
import os
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PRIME NUMBER UTILITIES
# =============================================================================

def is_prime(n: int) -> bool:
    """
    Check if n is a prime number.
    
    Parameters
    ----------
    n : int
        Integer to check for primality.
        
    Returns
    -------
    bool
        True if n is prime, False otherwise.
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def count_prime_factors(n: int) -> int:
    """
    Count total prime factors with multiplicity (Big Omega function Ω(n)).
    
    For example:
    - Ω(12) = Ω(2²×3) = 3
    - Ω(7) = 1 (prime)
    - Ω(1) = 0
    
    Parameters
    ----------
    n : int
        Positive integer.
        
    Returns
    -------
    int
        Number of prime factors counted with multiplicity.
    """
    if n < 2:
        return 0
    count = 0
    d = 2
    temp = n
    while d * d <= temp:
        while temp % d == 0:
            count += 1
            temp //= d
        d += 1
    if temp > 1:
        count += 1
    return count


# =============================================================================
# PATTERN GENERATION
# =============================================================================

def generate_prime_pattern(size: int, target_porosity: float = 0.3) -> tuple:
    """
    Generate prime-based perforation pattern.
    
    Primes map to holes (value=1), composites to solid (value=0).
    Opacity is modulated by Ω(n) - fewer factors = more transparent.
    
    Parameters
    ----------
    size : int
        Grid dimension (size × size pattern).
    target_porosity : float
        Target ratio of holes to total area (0 to 1).
        
    Returns
    -------
    tuple
        (binary_pattern, continuous_pattern) as numpy arrays.
    """
    pattern = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            n = i * size + j + 1
            if is_prime(n):
                pattern[i, j] = 1  # Hole
            else:
                omega = count_prime_factors(n)
                # Lower omega = more transparent
                pattern[i, j] = max(0, 1 - omega / 6)
    
    # Normalize to target porosity
    threshold = np.percentile(pattern, (1 - target_porosity) * 100)
    binary_pattern = (pattern >= threshold).astype(float)
    
    return binary_pattern, pattern


def generate_grid_pattern(size: int, target_porosity: float = 0.3) -> tuple:
    """
    Generate regular grid perforation pattern with equidistant holes.
    
    Parameters
    ----------
    size : int
        Grid dimension (size × size pattern).
    target_porosity : float
        Target ratio of holes to total area.
        
    Returns
    -------
    tuple
        (binary_pattern, continuous_pattern) as numpy arrays.
    """
    pattern = np.zeros((size, size))
    
    # Calculate spacing to achieve target porosity
    hole_spacing = int(np.sqrt(1 / target_porosity))
    if hole_spacing < 2:
        hole_spacing = 2
    
    for i in range(0, size, hole_spacing):
        for j in range(0, size, hole_spacing):
            if i < size and j < size:
                pattern[i, j] = 1
    
    return pattern, pattern


def generate_random_pattern(size: int, target_porosity: float = 0.3, 
                           seed: int = 42) -> tuple:
    """
    Generate random perforation pattern.
    
    Parameters
    ----------
    size : int
        Grid dimension (size × size pattern).
    target_porosity : float
        Target ratio of holes to total area.
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    tuple
        (binary_pattern, continuous_pattern) as numpy arrays.
    """
    np.random.seed(seed)
    pattern = np.random.random((size, size))
    threshold = 1 - target_porosity
    binary_pattern = (pattern >= threshold).astype(float)
    
    return binary_pattern, pattern


def generate_fibonacci_pattern(size: int, target_porosity: float = 0.3) -> tuple:
    """
    Generate Fibonacci/Golden spiral based perforation pattern.
    
    Uses the golden angle (137.5°) to place holes in a phyllotactic pattern,
    similar to sunflower seed arrangements.
    
    Parameters
    ----------
    size : int
        Grid dimension (size × size pattern).
    target_porosity : float
        Target ratio of holes to total area.
        
    Returns
    -------
    tuple
        (binary_pattern, continuous_pattern) as numpy arrays.
    """
    pattern = np.zeros((size, size))
    center = size // 2
    
    # Golden angle in radians
    golden_angle = np.pi * (3 - np.sqrt(5))  # ~137.5 degrees
    
    # Number of points based on target porosity
    n_points = int(size * size * target_porosity)
    
    for k in range(n_points):
        r = np.sqrt(k) * (size / (2 * np.sqrt(n_points)))
        theta = k * golden_angle
        
        x = int(center + r * np.cos(theta))
        y = int(center + r * np.sin(theta))
        
        if 0 <= x < size and 0 <= y < size:
            pattern[x, y] = 1
    
    return pattern, pattern


# =============================================================================
# LIGHT TRANSMISSION SIMULATION
# =============================================================================

def simulate_light_transmission(pattern: np.ndarray, 
                                sun_angles: list,
                                panel_height: float = 2.0, 
                                resolution: int = 200) -> np.ndarray:
    """
    Simulate light transmission through a perforated panel.
    
    Uses simplified 2D raytracing to model how light passes through
    holes in a panel and illuminates a ground plane below.
    
    Parameters
    ----------
    pattern : np.ndarray
        2D binary array where 1=hole, 0=solid.
    sun_angles : list
        List of (altitude, azimuth) tuples in degrees.
    panel_height : float
        Height of panel above ground plane in meters.
    resolution : int
        Resolution of ground illumination map.
        
    Returns
    -------
    np.ndarray
        2D array of accumulated ground illumination.
    """
    pattern_size = pattern.shape[0]
    ground = np.zeros((resolution, resolution))
    
    for altitude, azimuth in sun_angles:
        # Convert to radians
        alt_rad = np.radians(altitude)
        az_rad = np.radians(azimuth)
        
        # Sun direction vector
        sun_dx = np.cos(alt_rad) * np.sin(az_rad)
        sun_dy = np.cos(alt_rad) * np.cos(az_rad)
        sun_dz = np.sin(alt_rad)
        
        # For each hole in pattern, calculate light cone on ground
        for i in range(pattern_size):
            for j in range(pattern_size):
                if pattern[i, j] > 0.5:  # Hole
                    # Panel position (normalized to ground resolution)
                    panel_x = (i / pattern_size) * resolution
                    panel_y = (j / pattern_size) * resolution
                    
                    # Ground intersection point
                    if sun_dz > 0.1:  # Sun above horizon
                        t = panel_height / sun_dz
                        ground_x = int(panel_x - sun_dx * t * (resolution / panel_height))
                        ground_y = int(panel_y - sun_dy * t * (resolution / panel_height))
                        
                        # Intensity based on sun angle
                        intensity = np.sin(alt_rad)
                        
                        # Spread light in small area (penumbra)
                        spread = 3
                        for dx in range(-spread, spread + 1):
                            for dy in range(-spread, spread + 1):
                                gx, gy = ground_x + dx, ground_y + dy
                                if 0 <= gx < resolution and 0 <= gy < resolution:
                                    dist = np.sqrt(dx**2 + dy**2)
                                    falloff = np.exp(-dist / spread)
                                    ground[gx, gy] += intensity * falloff * pattern[i, j]
    
    return ground


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def calculate_metrics(ground_illumination: np.ndarray, 
                     pattern: np.ndarray) -> dict:
    """
    Calculate performance metrics for light distribution.
    
    Parameters
    ----------
    ground_illumination : np.ndarray
        2D array of ground illumination values.
    pattern : np.ndarray
        2D binary pattern array.
        
    Returns
    -------
    dict
        Dictionary containing:
        - homogeneity_std: Standard deviation of illumination
        - homogeneity_cv: Coefficient of variation
        - glare_index: Max/mean ratio
        - moire_risk: Periodicity measure from FFT
        - coverage: Fraction of ground receiving light
    """
    # Flatten for statistics
    illum = ground_illumination.flatten()
    illum_nonzero = illum[illum > 0]
    
    if len(illum_nonzero) == 0:
        return {
            'homogeneity_std': np.inf,
            'homogeneity_cv': np.inf,
            'glare_index': np.inf,
            'moire_risk': 1.0,
            'coverage': 0.0
        }
    
    # 1. Homogeneity (lower std = more uniform)
    homogeneity_std = np.std(illum_nonzero)
    homogeneity_cv = homogeneity_std / np.mean(illum_nonzero) if np.mean(illum_nonzero) > 0 else np.inf
    
    # 2. Glare Index (max/mean ratio, lower = better)
    glare_index = np.max(illum_nonzero) / np.mean(illum_nonzero) if np.mean(illum_nonzero) > 0 else np.inf
    
    # 3. Moiré Risk (periodicity detection via FFT)
    fft_2d = np.abs(fft.fft2(pattern))
    fft_shifted = fft.fftshift(fft_2d)
    center = pattern.shape[0] // 2
    
    # Exclude DC component
    fft_shifted[center-2:center+3, center-2:center+3] = 0
    moire_risk = np.max(fft_shifted) / np.mean(fft_shifted) if np.mean(fft_shifted) > 0 else 0
    moire_risk = min(1.0, moire_risk / 100)  # Normalize to 0-1
    
    # 4. Coverage (percentage of ground receiving light)
    coverage = np.sum(illum > np.max(illum) * 0.1) / len(illum)
    
    return {
        'homogeneity_std': homogeneity_std,
        'homogeneity_cv': homogeneity_cv,
        'glare_index': glare_index,
        'moire_risk': moire_risk,
        'coverage': coverage
    }


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_full_simulation(size: int = 50, 
                       target_porosity: float = 0.3,
                       verbose: bool = True) -> tuple:
    """
    Run complete comparative simulation for all pattern types.
    
    Parameters
    ----------
    size : int
        Pattern grid size.
    target_porosity : float
        Target hole ratio.
    verbose : bool
        If True, print progress messages.
        
    Returns
    -------
    tuple
        (patterns, ground_maps, metrics, sun_angles)
    """
    if verbose:
        print("=" * 60)
        print("LIGHT TRANSMISSION SIMULATION")
        print("Comparing Perforation Pattern Performance")
        print("=" * 60)
    
    # Generate patterns
    if verbose:
        print("\n[1/4] Generating patterns...")
    
    patterns = {}
    patterns['Prime-based'] = generate_prime_pattern(size, target_porosity)
    patterns['Regular Grid'] = generate_grid_pattern(size, target_porosity)
    patterns['Random'] = generate_random_pattern(size, target_porosity)
    patterns['Fibonacci'] = generate_fibonacci_pattern(size, target_porosity)
    
    # Check actual porosity
    if verbose:
        print("\n[2/4] Pattern porosity check:")
        for name, (binary, _) in patterns.items():
            actual_porosity = np.mean(binary)
            print(f"  {name}: {actual_porosity:.1%}")
    
    # Sun angles throughout day (altitude, azimuth) in degrees
    sun_angles = [
        (15, -60),   # Early morning
        (30, -45),   # Mid morning
        (45, -20),   # Late morning
        (60, 0),     # Noon
        (45, 20),    # Early afternoon
        (30, 45),    # Mid afternoon
        (15, 60),    # Late afternoon
    ]
    
    # Simulate light transmission
    if verbose:
        print("\n[3/4] Simulating light transmission...")
    
    ground_maps = {}
    metrics = {}
    
    for name, (binary, continuous) in patterns.items():
        if verbose:
            print(f"  Processing {name}...")
        ground = simulate_light_transmission(binary, sun_angles)
        ground_maps[name] = ground
        metrics[name] = calculate_metrics(ground, binary)
    
    # Results summary
    if verbose:
        print("\n[4/4] Calculating metrics...")
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"\n{'Pattern':<15} {'CV':<10} {'Glare':<10} {'Moiré':<10} {'Coverage':<10}")
        print("-" * 55)
        for name in patterns.keys():
            m = metrics[name]
            print(f"{name:<15} {m['homogeneity_cv']:<10.3f} {m['glare_index']:<10.2f} "
                  f"{m['moire_risk']:<10.3f} {m['coverage']:<10.1%}")
    
    return patterns, ground_maps, metrics, sun_angles


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(patterns: dict, 
                         ground_maps: dict, 
                         metrics: dict, 
                         output_path: str = './results') -> None:
    """
    Create comprehensive visualization figures.
    
    Parameters
    ----------
    patterns : dict
        Dictionary of pattern name -> (binary, continuous) arrays.
    ground_maps : dict
        Dictionary of pattern name -> ground illumination array.
    metrics : dict
        Dictionary of pattern name -> metrics dict.
    output_path : str
        Output directory for figures.
    """
    os.makedirs(output_path, exist_ok=True)
    pattern_names = list(patterns.keys())
    
    # Figure 1: Pattern Comparison
    fig1, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for idx, name in enumerate(pattern_names):
        binary, _ = patterns[name]
        
        # Top row: Binary patterns
        axes[0, idx].imshow(binary, cmap='Greys', interpolation='nearest')
        axes[0, idx].set_title(f'{name}\n(Binary)', fontsize=11)
        axes[0, idx].axis('off')
        
        # Bottom row: Ground illumination
        ground = ground_maps[name]
        axes[1, idx].imshow(ground, cmap='hot', interpolation='bilinear')
        axes[1, idx].set_title('Ground Illumination', fontsize=11)
        axes[1, idx].axis('off')
    
    plt.suptitle('Perforation Pattern Comparison: Structure and Light Distribution', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_path}/fig1_pattern_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Metrics Comparison
    fig2, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    metric_names = ['homogeneity_cv', 'glare_index', 'moire_risk', 'coverage']
    metric_labels = ['Coefficient of Variation\n(lower = more uniform)', 
                     'Glare Index\n(lower = less glare)',
                     'Moiré Risk\n(lower = better)',
                     'Light Coverage\n(higher = better)']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        values = [metrics[name][metric] for name in pattern_names]
        bars = axes[idx].bar(pattern_names, values, color=colors[idx], alpha=0.8, edgecolor='black')
        axes[idx].set_ylabel(label, fontsize=10)
        axes[idx].set_xticklabels(pattern_names, rotation=45, ha='right', fontsize=9)
        
        # Highlight best performer
        if metric == 'coverage':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
    
    plt.suptitle('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_path}/fig2_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: FFT Analysis
    fig3, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for idx, name in enumerate(pattern_names):
        binary, _ = patterns[name]
        fft_2d = np.abs(fft.fft2(binary))
        fft_shifted = fft.fftshift(fft_2d)
        fft_log = np.log1p(fft_shifted)
        
        axes[idx].imshow(fft_log, cmap='viridis')
        axes[idx].set_title(f'{name}\nFFT Spectrum', fontsize=11)
        axes[idx].axis('off')
    
    plt.suptitle('Frequency Analysis: Detecting Periodic Structures (Moiré Risk)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_path}/fig3_fft_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Illumination Profiles
    fig4, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    mid = ground_maps['Prime-based'].shape[0] // 2
    
    for name in pattern_names:
        ground = ground_maps[name]
        profile = ground[mid, :]
        axes[0].plot(profile, label=name, linewidth=2, alpha=0.8)
    
    axes[0].set_xlabel('Position', fontsize=11)
    axes[0].set_ylabel('Light Intensity', fontsize=11)
    axes[0].set_title('Horizontal Cross-Section (y = center)', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for name in pattern_names:
        ground = ground_maps[name]
        values = ground.flatten()
        values = values[values > 0]
        axes[1].hist(values, bins=50, alpha=0.5, label=name, density=True)
    
    axes[1].set_xlabel('Light Intensity', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title('Distribution of Ground Illumination', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Light Distribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_path}/fig4_illumination_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 5: Summary Table
    fig5, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    headers = ['Pattern', 'CV (↓)', 'Glare (↓)', 'Moiré (↓)', 'Coverage (↑)', 'Overall Rank']
    
    # Calculate ranks
    ranks = {}
    for name in pattern_names:
        rank_cv = sorted(pattern_names, key=lambda x: metrics[x]['homogeneity_cv']).index(name) + 1
        rank_glare = sorted(pattern_names, key=lambda x: metrics[x]['glare_index']).index(name) + 1
        rank_moire = sorted(pattern_names, key=lambda x: metrics[x]['moire_risk']).index(name) + 1
        rank_coverage = sorted(pattern_names, key=lambda x: -metrics[x]['coverage']).index(name) + 1
        ranks[name] = (rank_cv + rank_glare + rank_moire + rank_coverage) / 4
    
    table_data = []
    for name in pattern_names:
        m = metrics[name]
        table_data.append([
            name,
            f"{m['homogeneity_cv']:.3f}",
            f"{m['glare_index']:.2f}",
            f"{m['moire_risk']:.3f}",
            f"{m['coverage']:.1%}",
            f"{ranks[name]:.2f}"
        ])
    
    table_data.sort(key=lambda x: float(x[-1]))
    
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0'] * len(headers)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    for j in range(len(headers)):
        table[(1, j)].set_facecolor('#d5f5e3')
    
    plt.title('Performance Summary\n(↓ = lower is better, ↑ = higher is better)\n', 
              fontsize=14, fontweight='bold')
    plt.savefig(f'{output_path}/fig5_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nFigures saved to {output_path}/")


def generate_latex_table(metrics: dict, patterns: dict) -> str:
    """
    Generate LaTeX table for academic paper.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metrics by pattern name.
    patterns : dict
        Dictionary of patterns.
        
    Returns
    -------
    str
        LaTeX table code.
    """
    pattern_names = list(patterns.keys())
    
    latex = r"""
\begin{table}[H]
\centering
\caption{Light Transmission Performance Metrics for Different Perforation Patterns}
\label{tab:light_metrics}
\begin{tabular}{@{}l@{\hspace{1.5em}}c@{\hspace{1.5em}}c@{\hspace{1.5em}}c@{\hspace{1.5em}}c@{}}
\toprule
\textbf{Pattern} & \textbf{CV} $\downarrow$ & \textbf{Glare} $\downarrow$ & \textbf{Moiré} $\downarrow$ & \textbf{Coverage} $\uparrow$ \\
\midrule
"""
    
    for name in pattern_names:
        m = metrics[name]
        latex += f"{name:<12} & {m['homogeneity_cv']:.3f} & {m['glare_index']:.2f} & "
        latex += f"{m['moire_risk']:.3f} & {m['coverage']:.1%} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item CV = Coefficient of Variation. $\downarrow$ = lower is better; $\uparrow$ = higher is better.
\end{tablenotes}
\end{table}
"""
    
    return latex


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Create output directory
    output_path = "./results"
    os.makedirs(output_path, exist_ok=True)
    
    # Run simulation
    patterns, ground_maps, metrics, sun_angles = run_full_simulation(
        size=50, 
        target_porosity=0.3
    )
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(patterns, ground_maps, metrics, output_path)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(metrics, patterns)
    with open(f'{output_path}/latex_table.tex', 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to {output_path}/latex_table.tex")
    
    # Print interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    best_cv = min(metrics.keys(), key=lambda x: metrics[x]['homogeneity_cv'])
    best_glare = min(metrics.keys(), key=lambda x: metrics[x]['glare_index'])
    best_moire = min(metrics.keys(), key=lambda x: metrics[x]['moire_risk'])
    best_coverage = max(metrics.keys(), key=lambda x: metrics[x]['coverage'])
    
    print(f"\nBest Homogeneity (CV): {best_cv}")
    print(f"Best Glare Control: {best_glare}")
    print(f"Best Moiré Resistance: {best_moire}")
    print(f"Best Coverage: {best_coverage}")
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
