import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_G_matrix(N=10, u_min=-10.0, u_max=10.0, x_min=-1.0, x_max=1.0):
    n_vars = 2 * N  
    n_constraints = 4 * N  
    
    G = np.zeros((n_constraints, n_vars))
    
    for i in range(N):
        G[4*i, N + i] = 1.0
        G[4*i + 1, N + i] = -1.0
        
        G[4*i + 2, i] = 1.0
        G[4*i + 3, i] = -1.0
    
    row_labels = []
    for i in range(N):
        row_labels.append(f'u{i} <= u_max')
        row_labels.append(f'-u{i} <= -u_min')
        row_labels.append(f'x{i+1}[pos] <= x_max')
        row_labels.append(f'-x{i+1}[pos] <= -x_min')
    
    col_labels = []
    for i in range(N):
        col_labels.append(f'x{i+1}[pos]')
    for i in range(N):
        col_labels.append(f'u{i}')
    
    plt.figure(figsize=(14, 12))
    ax = sns.heatmap(G, annot=False, cmap='coolwarm', center=0,
                     xticklabels=col_labels, yticklabels=row_labels,
                     cbar=True, cbar_kws={'label': 'Coefficient value'},
                     linewidths=0.3, linecolor='white')
    ax.set_title('G matrix')
    ax.set_xlabel('Decision variable index')
    ax.set_ylabel('Inequality constraint')
    plt.tight_layout()
    plt.savefig('plots/G_matrix_N10.png', dpi=300, bbox_inches='tight')
    plt.savefig('plots/G_matrix_N10.pdf', dpi=300, bbox_inches='tight')
    print("Saved: plots/G_matrix_N10.png")
    plt.close()

def plot_h_vector(N=10, u_min=-10.0, u_max=10.0, x_min=-1.0, x_max=1.0):
    h = []
    labels = []
    
    for i in range(N):
        h.append(u_max)
        labels.append(f'u{i}_max')
        
        h.append(-u_min)
        labels.append(f'u{i}_min')
        
        h.append(x_max)
        labels.append(f'x{i+1}_max')
        
        h.append(-x_min)
        labels.append(f'x{i+1}_min')
    
    h = np.array(h)
    
    plt.figure(figsize=(14, 5))
    bars = plt.bar(range(len(h)), h, color='#348ABD', edgecolor='black', linewidth=1)
    
    for bar, label, val in zip(bars, labels, h):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.title(f'Inequality constraint vector h ({len(h)})')
    plt.xlabel('Constraint index')
    plt.ylabel('h value')
    plt.xticks(range(len(h)), labels, rotation=45, ha='right', fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('plots/h_vector_N10.png', dpi=300, bbox_inches='tight')
    plt.savefig('plots/h_vector_N10.pdf', dpi=300, bbox_inches='tight')
    print("Saved: plots/h_vector_N10.png")
    plt.close()

def main():
    N = 10
    u_min, u_max = -10.0, 10.0
    x_min, x_max = -1.0, 1.0
    
    plot_G_matrix(N, u_min, u_max, x_min, x_max)
    plot_h_vector(N, u_min, u_max, x_min, x_max)
    
    print(f"\nQP约束可视化图已生成！")
    print(f"模型参数：N={N}, u_range=[{u_min}, {u_max}], x_range=[{x_min}, {x_max}]")

if __name__ == '__main__':
    main()