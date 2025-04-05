import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def visualize_keystroke_dynamics(metrics_file):
    """
    Generate visualization for keystroke dynamics data.
    
    Parameters:
    metrics_file (str): Path to the CSV file containing keystroke metrics
    """
    print(f"Loading data from {metrics_file}...")
    
    # Load the data
    df = pd.read_csv(metrics_file)
    
    # Basic data check
    print(f"Loaded {len(df)} keystroke events")
    print(f"Columns in the data: {', '.join(df.columns)}")
    
    # Clean the data - remove any rows with missing values in key metrics
    df = df.dropna(subset=['pressDuration']).copy()
    
    # Extract just the key name from the code (e.g., "KeyA" -> "A")
    df['simple_key'] = df['key'].str.replace('Key', '').replace('Digit', '')
    
    # Create a directory for saving plots
    output_dir = "keystroke_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # ====== Overall Distribution Visualizations ======
    plt.figure(figsize=(12, 6))
    
    # Distribution of press durations
    plt.subplot(1, 2, 1)
    sns.histplot(df['pressDuration'], kde=True)
    plt.title('Distribution of Key Press Durations')
    plt.xlabel('Duration (ms)')
    plt.ylabel('Frequency')
    
    # Distribution of flight times
    plt.subplot(1, 2, 2)
    flight_times = df['flightTime'].dropna()
    sns.histplot(flight_times, kde=True)
    plt.title('Distribution of Flight Times')
    plt.xlabel('Flight Time (ms)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_distributions.png", dpi=300)
    plt.close()
    
    # ====== Key-specific Analysis ======
    
    # Only include keys that appear at least 3 times for meaningful analysis
    key_counts = df['simple_key'].value_counts()
    common_keys = key_counts[key_counts >= 3].index
    key_data = df[df['simple_key'].isin(common_keys)]
    
    # Get mean press duration and flight time for each key
    key_metrics = key_data.groupby('simple_key').agg({
        'pressDuration': 'mean',
        'flightTime': lambda x: x.dropna().mean() if not x.dropna().empty else np.nan
    }).reset_index()
    
    # Sort by key name for consistent display
    key_metrics = key_metrics.sort_values('simple_key')
    
    # ====== Bar Chart of Press Durations by Key ======
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    bars = plt.bar(key_metrics['simple_key'], key_metrics['pressDuration'], color='skyblue')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.title('Average Press Duration by Key')
    plt.xlabel('Key')
    plt.ylabel('Average Duration (ms)')
    plt.axhline(y=key_metrics['pressDuration'].mean(), color='r', linestyle='--', 
                label=f'Overall Average: {key_metrics["pressDuration"].mean():.1f} ms')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # ====== Bar Chart of Flight Times by Key ======
    plt.subplot(2, 1, 2)
    
    # Filter out NaN flight times
    flight_key_metrics = key_metrics.dropna(subset=['flightTime'])
    
    bars = plt.bar(flight_key_metrics['simple_key'], flight_key_metrics['flightTime'], color='lightgreen')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.title('Average Flight Time After Key')
    plt.xlabel('Key')
    plt.ylabel('Average Flight Time (ms)')
    
    flight_mean = flight_key_metrics['flightTime'].mean()
    plt.axhline(y=flight_mean, color='r', linestyle='--', 
                label=f'Overall Average: {flight_mean:.1f} ms')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/key_metrics.png", dpi=300)
    plt.close()
    
    # ====== Heatmap of most common key transitions and their flight times ======
    # Create a pivot table to analyze transitions between keys
    transitions = []
    
    for i in range(len(df) - 1):
        if pd.notna(df.iloc[i]['flightTime']):
            transitions.append({
                'from_key': df.iloc[i]['simple_key'],
                'to_key': df.iloc[i+1]['simple_key'],
                'flight_time': df.iloc[i]['flightTime']
            })
    
    if transitions:
        transitions_df = pd.DataFrame(transitions)
        
        # Group by transition pairs and get mean flight time
        transition_metrics = transitions_df.groupby(['from_key', 'to_key'])['flight_time'].agg(['mean', 'count']).reset_index()
        
        # Only include transitions that occur at least 2 times
        common_transitions = transition_metrics[transition_metrics['count'] >= 2]
        
        if len(common_transitions) > 0:
            # Create a pivot table for the heatmap
            pivot_table = common_transitions.pivot_table(
                index='from_key', 
                columns='to_key', 
                values='mean'
            )
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='viridis')
            plt.title('Average Flight Time Between Key Transitions (ms)')
            plt.xlabel('To Key')
            plt.ylabel('From Key')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/transition_heatmap.png", dpi=300)
            plt.close()
    
    print(f"Analysis complete! Images saved to {output_dir} directory.")
    
    return {
        'avg_press_duration': df['pressDuration'].mean(),
        'avg_flight_time': df['flightTime'].dropna().mean(),
        'total_keys': len(df),
        'unique_keys': len(df['simple_key'].unique())
    }

def analyze_consistency(metrics_file):
    """
    Analyze the consistency of keystroke patterns by looking at variability
    in press durations and flight times.
    
    Parameters:
    metrics_file (str): Path to the CSV file containing keystroke metrics
    """
    # Load the data
    df = pd.read_csv(metrics_file)
    
    # Clean the data
    df = df.dropna(subset=['pressDuration']).copy()
    df['simple_key'] = df['key'].str.replace('Key', '').replace('Digit', '')
    
    # Only analyze keys that appear at least 3 times
    key_counts = df['simple_key'].value_counts()
    common_keys = key_counts[key_counts >= 3].index
    key_data = df[df['simple_key'].isin(common_keys)]
    
    # Calculate coefficient of variation (CV) for press durations by key
    cv_data = key_data.groupby('simple_key').agg({
        'pressDuration': ['mean', 'std', lambda x: x.std() / x.mean() * 100 if x.mean() > 0 else np.nan]
    })
    
    cv_data.columns = ['mean', 'std', 'cv_percent']
    cv_data = cv_data.sort_values('cv_percent')
    
    # Create a directory for saving plots
    output_dir = "keystroke_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot CV for each key (lower CV = more consistent)
    plt.figure(figsize=(14, 8))
    bars = plt.bar(cv_data.index, cv_data['cv_percent'], color='coral')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.title('Keystroke Consistency by Key (Lower CV% = More Consistent)')
    plt.xlabel('Key')
    plt.ylabel('Coefficient of Variation (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/key_consistency.png", dpi=300)
    plt.close()
    
    print(f"Consistency analysis complete! Images saved to {output_dir} directory.")
    
    return cv_data

if __name__ == "__main__":
    # Provide the path to your metrics CSV file
    metrics_file = input("Enter the path to your keystroke metrics CSV file: ")
    
    # Run the visualizations
    results = visualize_keystroke_dynamics(metrics_file)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average press duration: {results['avg_press_duration']:.2f} ms")
    print(f"Average flight time: {results['avg_flight_time']:.2f} ms")
    print(f"Total keystrokes analyzed: {results['total_keys']}")
    print(f"Unique keys pressed: {results['unique_keys']}")
    
    # Ask if user wants to run consistency analysis
    run_consistency = input("\nWould you like to analyze keystroke consistency? (y/n): ")
    if run_consistency.lower() == 'y':
        consistency_results = analyze_consistency(metrics_file)
        
        # Display the 3 most consistent keys
        print("\nMost consistent keys (lowest variability):")
        print(consistency_results[['cv_percent']].head(3))
        
        # Display the 3 least consistent keys
        print("\nLeast consistent keys (highest variability):")
        print(consistency_results[['cv_percent']].tail(3))
