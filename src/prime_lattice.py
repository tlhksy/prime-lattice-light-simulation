"""
Prime Exponent Lattice Visualization

This module generates visualizations of the prime exponent lattice structure,
including 3D lattice projections, PCA analysis, and porosity trends.

The Prime Exponent Lattice maps integers to high-dimensional space where:
- Each prime number defines an orthogonal axis (basis vector)
- Composite numbers are linear combinations of basis vectors
- The mapping Φ(n) = (e₁, e₂, e₃, ...) where n = Π pᵢ^eᵢ

Author: Talha Aksoy
Affiliation: Department of Landscape Architecture, Kırklareli University, Türkiye
Email: talha.aksoy@klu.edu.tr

License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PRIME NUMBER UTILITIES
# =============================================================================

def is_prime(n: int) -> bool:
    """Check if n is a prime number."""
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


def generate_primes(n_max: int) -> list:
    """Generate all prime numbers up to n_max."""
    return [i for i in range(2, n_max + 1) if is_prime(i)]


def prime_factorization(n: int) -> dict:
    """
    Compute prime factorization of n.
    
    Returns dict mapping prime -> exponent.
    Example: prime_factorization(12) = {2: 2, 3: 1}
    """
    if n < 2:
        return {}
    
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    
    return factors


def integer_to_exponent_vector(n: int, primes: list) -> np.ndarray:
    """
    Map integer n to exponent vector in prime lattice.
    
    Φ(n) = (e₁, e₂, e₃, ...) where n = Π pᵢ^eᵢ
    
    Parameters
    ----------
    n : int
        Positive integer to map.
    primes : list
        List of prime numbers defining the basis.
        
    Returns
    -------
    np.ndarray
        Exponent vector.
    """
    factors = prime_factorization(n)
    vector = np.zeros(len(primes))
    
    for i, p in enumerate(primes):
        if p in factors:
            vector[i] = factors[p]
    
    return vector


def prime_counting_function(n: int, primes: list) -> int:
    """Return π(n) - count of primes ≤ n."""
    return sum(1 for p in primes if p <= n)


# =============================================================================
# LATTICE ANALYSIS
# =============================================================================

def compute_prime_gaps(primes: list) -> list:
    """
    Compute gaps between consecutive primes.
    
    Returns list of (prime, gap_size) tuples.
    """
    gaps = []
    for i in range(1, len(primes)):
        gap = primes[i] - primes[i-1]
        gaps.append((primes[i], gap))
    return gaps


def analyze_dimension_vs_gaps(n_max: int = 1000) -> tuple:
    """
    Analyze relationship between lattice dimension and prime gaps.
    
    Parameters
    ----------
    n_max : int
        Maximum integer to analyze.
        
    Returns
    -------
    tuple
        (n_values, dimensions, gap_positions, gap_sizes)
    """
    primes = generate_primes(n_max)
    gaps = compute_prime_gaps(primes)
    
    n_values = list(range(1, n_max + 1))
    dimensions = [prime_counting_function(n, primes) for n in n_values]
    
    gap_positions = [g[0] for g in gaps]
    gap_sizes = [g[1] for g in gaps]
    
    return n_values, dimensions, gap_positions, gap_sizes


def analyze_porosity(n_max: int = 100000, window: int = 500) -> tuple:
    """
    Analyze lattice porosity (gap size vs dimension) up to n_max.
    
    Parameters
    ----------
    n_max : int
        Maximum integer to analyze.
    window : int
        Moving average window size.
        
    Returns
    -------
    tuple
        (dimensions, gaps, moving_avg_x, moving_avg)
    """
    print(f"Generating primes up to {n_max:,}...")
    primes = generate_primes(n_max)
    print(f"Found {len(primes):,} primes")
    
    # Calculate gaps with dimension
    gaps_data = []
    for i in range(1, len(primes)):
        gap = primes[i] - primes[i-1]
        dimension = i  # π(pᵢ) = i
        gaps_data.append((dimension, gap))
    
    dimensions = [d[0] for d in gaps_data]
    gaps = [d[1] for d in gaps_data]
    
    # Moving average
    moving_avg = []
    moving_avg_x = []
    for i in range(window, len(gaps)):
        avg = np.mean(gaps[i-window:i])
        moving_avg.append(avg)
        moving_avg_x.append(dimensions[i])
    
    return dimensions, gaps, moving_avg_x, moving_avg


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_dimension_growth(n_max: int = 1000, 
                               output_path: str = './results') -> None:
    """
    Visualize dimension growth vs prime gaps.
    
    Creates Figure 1 for the paper: local view showing d(n) = π(n)
    and prime gap distribution.
    """
    os.makedirs(output_path, exist_ok=True)
    
    n_values, dimensions, gap_positions, gap_sizes = analyze_dimension_vs_gaps(n_max)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Dimension growth (left y-axis)
    ax1.plot(n_values, dimensions, 'b-', linewidth=1.5, label=r'$d(n) = \pi(n)$')
    ax1.set_xlabel('Integer $n$', fontsize=12)
    ax1.set_ylabel(r'Dimension $d(n) = \pi(n)$', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Prime gaps (right y-axis)
    ax2 = ax1.twinx()
    ax2.scatter(gap_positions, gap_sizes, c='red', s=15, alpha=0.6)
    ax2.set_ylabel('Prime Gap Size', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Annotate notable gaps
    for pos, size in zip(gap_positions, gap_sizes):
        if size == 20 and 850 <= pos <= 950:
            ax2.annotate(f'Gap {size}', xy=(pos, size), xytext=(pos-50, size+2),
                        fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))
            break
    
    plt.title(r'The Prime Exponent Lattice: Dimension Growth vs. Prime Gaps ($n=1{,}000$)', 
              fontsize=14, fontweight='bold')
    
    def thousands_formatter(x, pos):
        return f'{int(x):,}'
    ax1.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/fig1_dimension_growth.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}/fig1_dimension_growth.png")


def visualize_3d_lattice(max_exp: int = 4, output_path: str = './results') -> None:
    """
    Visualize 3D projection of prime exponent lattice.
    
    Shows integers composed only of factors 2, 3, and 5.
    """
    os.makedirs(output_path, exist_ok=True)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate 5-smooth numbers (only factors 2, 3, 5)
    smooth_numbers = []
    for e1 in range(max_exp + 1):
        for e2 in range(max_exp + 1):
            for e3 in range(max_exp + 1):
                n = (2**e1) * (3**e2) * (5**e3)
                smooth_numbers.append((n, e1, e2, e3))
    
    smooth_numbers.sort(key=lambda x: x[0])
    
    # Plot composite numbers
    for n, e1, e2, e3 in smooth_numbers:
        if e1 + e2 + e3 > 1:  # Not 1 or prime
            ax.scatter(e1, e2, e3, c='blue', s=30, alpha=0.6)
            ax.text(e1, e2, e3, f'{n}', fontsize=7, alpha=0.7)
    
    # Plot basis primes
    ax.scatter(1, 0, 0, c='red', s=100, marker='^', label='Basis Primes (2, 3, 5)')
    ax.scatter(0, 1, 0, c='red', s=100, marker='^')
    ax.scatter(0, 0, 1, c='red', s=100, marker='^')
    
    # Plot origin (n=1)
    ax.scatter(0, 0, 0, c='black', s=100, marker='*', label='Origin (n=1)')
    
    # Draw basis vectors
    ax.plot([0, 1], [0, 0], [0, 0], 'r--', alpha=0.5)
    ax.plot([0, 0], [0, 1], [0, 0], 'r--', alpha=0.5)
    ax.plot([0, 0], [0, 0], [0, 1], 'r--', alpha=0.5)
    
    ax.set_xlabel(r'Exponent of 2 ($e_1$)', fontsize=11, color='red')
    ax.set_ylabel(r'Exponent of 3 ($e_2$)', fontsize=11, color='red')
    ax.set_zlabel(r'Exponent of 5 ($e_3$)', fontsize=11, color='red')
    
    ax.set_title('3D Projection of the Prime Exponent Lattice\n'
                 '(Numbers composed only of factors 2, 3, and 5)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/fig2_3d_lattice.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}/fig2_3d_lattice.png")


def visualize_pca_projection(n_max: int = 100, output_path: str = './results') -> None:
    """
    Visualize PCA projection of high-dimensional integer lattice.
    """
    os.makedirs(output_path, exist_ok=True)
    
    primes = generate_primes(n_max)
    
    # Build exponent matrix
    vectors = []
    labels = []
    is_prime_list = []
    
    for n in range(2, n_max + 1):
        vec = integer_to_exponent_vector(n, primes)
        vectors.append(vec)
        labels.append(n)
        is_prime_list.append(is_prime(n))
    
    X = np.array(vectors)
    
    # PCA projection
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    explained_var = sum(pca.explained_variance_ratio_) * 100
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Composites
    composite_mask = [not p for p in is_prime_list]
    ax.scatter(X_pca[composite_mask, 0], X_pca[composite_mask, 1], 
              c='steelblue', s=50, alpha=0.6, label='Composites (Linear Combinations)')
    
    # Primes
    prime_mask = is_prime_list
    ax.scatter(X_pca[prime_mask, 0], X_pca[prime_mask, 1], 
              c='red', s=80, marker='^', alpha=0.8, label='Primes (Basis Vectors)')
    
    # Label some points
    for i, (x, y) in enumerate(X_pca):
        n = labels[i]
        if n <= 30 or is_prime(n):
            ax.annotate(str(n), (x, y), fontsize=8, alpha=0.7)
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.plot(0, 0, 'k+', markersize=15, markeredgewidth=2)
    
    ax.set_xlabel('Principal Component 1 (Direction of Max Variance)', fontsize=11)
    ax.set_ylabel('Principal Component 2', fontsize=11)
    ax.set_title(f'PCA Projection of the High-Dimensional Integer Lattice (n=2 to {n_max})\n'
                 f'(Projecting {len(primes)} Dimensions onto 2D, '
                 f'Explained Variance: {explained_var:.1f}%)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/fig3_pca_projection.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}/fig3_pca_projection.png")
    print(f"Explained variance: {explained_var:.1f}%")


def visualize_porosity(n_max: int = 100000, 
                       window: int = 500,
                       output_path: str = './results') -> None:
    """
    Visualize lattice porosity trend (global view).
    """
    os.makedirs(output_path, exist_ok=True)
    
    dimensions, gaps, moving_avg_x, moving_avg = analyze_porosity(n_max, window)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.scatter(dimensions, gaps, c='lightblue', s=3, alpha=0.4, label='Raw Gap Size Data')
    ax.plot(moving_avg_x, moving_avg, 'r-', linewidth=2.5, 
            label=f'Porosity Trend (Moving Avg, n={window})')
    
    ax.set_xlabel(r'Dimension $d(n)$ (Cumulative Count of Primes)', fontsize=12)
    ax.set_ylabel('Void Size (Prime Gap Length)', fontsize=12)
    ax.set_title(f'Experiment A: Lattice Porosity Analysis up to $N={n_max:,}$\n'
                 '(Demonstrating Non-Linear Void Growth)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    
    def thousands_formatter(x, pos):
        return f'{int(x):,}'
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    
    ax.set_xlim(0, max(dimensions) * 1.02)
    ax.set_ylim(0, max(gaps) * 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}/fig4_porosity_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}/fig4_porosity_analysis.png")


def visualize_generative_pattern(max_size: int = 6, 
                                 output_path: str = './results') -> None:
    """
    Visualize evolution of prime lattice pattern across grid sizes.
    """
    os.makedirs(output_path, exist_ok=True)
    
    sizes = range(2, max_size + 1)
    fig, axes = plt.subplots(1, len(list(sizes)), figsize=(16, 3.5))
    
    # Color scheme
    prime_color = '#f5f5dc'  # Light (voids)
    composite_colors = {
        2: '#6b8e23',  # Olive
        3: '#228b22',  # Forest green
        4: '#2e8b57',  # Sea green
        5: '#008080',  # Teal
        6: '#000000',  # Black
    }
    
    for idx, size in enumerate(range(2, max_size + 1)):
        ax = axes[idx]
        grid = np.zeros((size, size, 3))
        
        for i in range(size):
            for j in range(size):
                n = i * size + j + 1
                
                if is_prime(n):
                    # Prime - light color
                    grid[i, j] = [0.96, 0.96, 0.86]  # Beige
                elif n == 1:
                    # Origin - white
                    grid[i, j] = [1, 1, 1]
                else:
                    # Composite - color by Ω(n)
                    omega = 0
                    temp = n
                    d = 2
                    while d * d <= temp:
                        while temp % d == 0:
                            omega += 1
                            temp //= d
                        d += 1
                    if temp > 1:
                        omega += 1
                    
                    if omega == 2:
                        grid[i, j] = [0.42, 0.56, 0.14]
                    elif omega == 3:
                        grid[i, j] = [0.13, 0.55, 0.13]
                    elif omega == 4:
                        grid[i, j] = [0.18, 0.55, 0.34]
                    else:
                        grid[i, j] = [0, 0, 0]
        
        ax.imshow(grid, interpolation='nearest')
        
        # Add number labels
        for i in range(size):
            for j in range(size):
                n = i * size + j + 1
                color = 'red' if is_prime(n) else 'white' if n > 4 else 'black'
                ax.text(j, i, str(n), ha='center', va='center', 
                       fontsize=10, color=color, fontweight='bold')
        
        ax.set_title(f'{size}×{size} Grid\n(n=1 to {size*size})', fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Evolution of the Prime Lattice Pattern (Deterministic Growth)', 
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f'{output_path}/fig5_generative_pattern.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}/fig5_generative_pattern.png")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_all_lattice_figures(output_path: str = './results') -> None:
    """Generate all prime lattice visualization figures."""
    
    print("=" * 60)
    print("PRIME EXPONENT LATTICE VISUALIZATION")
    print("=" * 60)
    
    print("\n[1/5] Generating dimension growth figure...")
    visualize_dimension_growth(n_max=1000, output_path=output_path)
    
    print("\n[2/5] Generating 3D lattice projection...")
    visualize_3d_lattice(max_exp=4, output_path=output_path)
    
    print("\n[3/5] Generating PCA projection...")
    visualize_pca_projection(n_max=100, output_path=output_path)
    
    print("\n[4/5] Generating porosity analysis...")
    visualize_porosity(n_max=100000, output_path=output_path)
    
    print("\n[5/5] Generating generative pattern...")
    visualize_generative_pattern(max_size=6, output_path=output_path)
    
    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_lattice_figures('./results')
