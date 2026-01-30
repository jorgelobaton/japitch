import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_evaluation_stats(csv_path='evaluation_results.csv', 
                          pie_output='evaluation_pies.png', 
                          hist_output='evaluation_histogram.png'):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run evaluate_jvs.py first.")
        return

    df = pd.read_csv(csv_path)
    
    # Calculate Tolerant Hits
    TOLERANCE = 0.1
    # A 'Tolerant Hit' is either a Strict Hit OR a Miss with deviation <= TOLERANCE
    df['tolerant_hit'] = df.apply(lambda row: True if row['hit'] else (row['deviation'] <= TOLERANCE), axis=1)
    
    # --- Figure 1: Pie Charts ---
    plt.figure(figsize=(12, 6))
    
    # 1. Strict Hit Rate Pie Chart
    plt.subplot(1, 2, 1)
    hit_counts = df['hit'].value_counts()
    plt.pie(hit_counts, labels=['Hit' if x else 'Miss' for x in hit_counts.index], 
            autopct='%1.1f%%', colors=['#66b3ff' if x else '#ff9999' for x in hit_counts.index], startangle=90)
    plt.title('Strict Hit Rate')

    # 2. Tolerant Hit Rate Pie Chart
    plt.subplot(1, 2, 2)
    tol_hit_counts = df['tolerant_hit'].value_counts()
    # Use f-string to ensure label matches the actual variable
    plt.pie(tol_hit_counts, labels=[f'Hit (Tolerance: {TOLERANCE}s)' if x else 'Miss' for x in tol_hit_counts.index], 
            autopct='%1.1f%%', colors=['#66b3ff' if x else '#ff9999' for x in tol_hit_counts.index], startangle=90)
    plt.title(f'Hit Rate (Tolerance {TOLERANCE}s)')

    plt.tight_layout()
    plt.savefig(pie_output)
    print(f"Pie charts saved to {pie_output}")
    plt.close() # Close figure to start fresh for the next one

    # --- Figure 2: Deviation Histogram ---
    plt.figure(figsize=(8, 6))
    misses = df[df['hit'] == False]
    
    if not misses.empty:
        # stat='percent' makes Y axis percentage
        sns.histplot(misses['deviation'], color='salmon', bins=40, stat="percent")
        plt.title('Time Deviation Distribution (Strict Misses)')
        plt.xlabel('Deviation (seconds)')
        plt.ylabel('Percentage of Misses')
        
        # Add vertical line for tolerance
        plt.axvline(x=TOLERANCE, color='green', linestyle='--', label=f'Tolerance {TOLERANCE}s')
        plt.legend()
        
        # Add a text box with summary stats
        mean_dev = misses['deviation'].mean()
        median_dev = misses['deviation'].median()
        stats_text = f"Mean: {mean_dev:.3f}s\nMedian: {median_dev:.3f}s"
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    else:
        plt.text(0.5, 0.5, "No Misses to Analyze!", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Time Deviation Distribution')

    plt.tight_layout()
    plt.savefig(hist_output)
    print(f"Histogram saved to {hist_output}")
    plt.close()

if __name__ == "__main__":
    plot_evaluation_stats()