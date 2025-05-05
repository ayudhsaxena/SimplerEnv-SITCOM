import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Base directory containing all experiment results
base_dir = "/home/rishisha/SimplerEnv-SITCOM/results_scaling_laws"

# List of candidate values
candidates = [1, 5, 10, 15, 20, 25]

# Read and combine data
all_data = []
for num_candidates in candidates:
    exp_dir = f"openvla-7b+simpler_rlds+b6+lr-0.0005+lora-r16+dropout-0.0--image_aug-0.8_temp1.0_candidates{num_candidates}"
    csv_path = os.path.join(base_dir, exp_dir, "results.csv")
    
    try:
        df = pd.read_csv(csv_path)
        df['candidates'] = num_candidates
        all_data.append(df)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")

combined_df = pd.concat(all_data, ignore_index=True)

# Define scenarios
scenarios = [
    "put_spoon_on_tablecloth",
    "put_carrot_on_plate",
    "stack_green_block_on_yellow_block",
    "put_eggplant_in_basket"
]

# Pretty scenario names for plotting
pretty_names = {
    "put_spoon_on_tablecloth": "Put Spoon on Tablecloth",
    "put_carrot_on_plate": "Put Carrot on Plate",
    "stack_green_block_on_yellow_block": "Stack Green Block on Yellow Block",
    "put_eggplant_in_basket": "Put Eggplant in Basket"
}

# Create separate plots for partial and entire success rates
def create_scaling_plot(success_type, title_suffix):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    for scenario in scenarios:
        col = f"{scenario}/matching_{success_type}"
        values = []
        for num_candidates in candidates:
            row = combined_df[combined_df['candidates'] == num_candidates]
            if not row.empty:
                values.append(row[col].values[0])
            else:
                values.append(np.nan)
        
        ax.plot(candidates, values, marker='o', linewidth=3, markersize=10, 
                label=f'{pretty_names[scenario]} (SITCOM-EnvSim)')
    
    # Add SITCOM-dynamic line for put carrot on plate (only for entire success)
    if success_type == 'entire':
        # SITCOM-dynamic results for put carrot on plate
        sitcom_dynamic_values = [0.208, 0.417, 0.791, 0.791, 0.75, 0.9583]
        ax.plot(candidates, sitcom_dynamic_values, marker='^', linestyle=':', 
                linewidth=3, markersize=10, color='red',
                label='Put Carrot on Plate (SITCOM-Dynamics)')
    
    ax.set_xlabel('Number of Candidates', fontsize=16, labelpad=10)
    ax.set_ylabel('Success Rate', fontsize=16, labelpad=10)
    ax.set_title(f'{title_suffix} Success Rate vs Number of Candidates', fontsize=18, pad=20)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 26)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    ax.set_xticks(candidates)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f'{success_type}_success_scaling.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(base_dir, f'{success_type}_success_scaling.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

# Create plots
create_scaling_plot('partial', 'Partial')
create_scaling_plot('entire', 'Entire')

# Create a combined comparison plot with shaded areas
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))

for idx, scenario in enumerate(scenarios):
    partial_col = f"{scenario}/matching_partial"
    entire_col = f"{scenario}/matching_entire"
    
    partial_values = []
    entire_values = []
    
    for num_candidates in candidates:
        row = combined_df[combined_df['candidates'] == num_candidates]
        if not row.empty:
            partial_values.append(row[partial_col].values[0])
            entire_values.append(row[entire_col].values[0])
        else:
            partial_values.append(np.nan)
            entire_values.append(np.nan)
    
    # Plot partial with solid line
    ax.plot(candidates, partial_values, '-', color=colors[idx], linewidth=3, 
            marker='o', markersize=10, label=f'{pretty_names[scenario]} (Partial)')
    
    # Plot entire with dashed line
    ax.plot(candidates, entire_values, '--', color=colors[idx], linewidth=3, 
            marker='s', markersize=10, label=f'{pretty_names[scenario]} (Entire)')
    
    # Fill between
    ax.fill_between(candidates, entire_values, partial_values, color=colors[idx], alpha=0.1)

ax.set_xlabel('Number of Candidates', fontsize=16, labelpad=10)
ax.set_ylabel('Success Rate', fontsize=16, labelpad=10)
ax.set_title('Scaling Law Comparison: Partial vs Entire Success Rates', fontsize=18, pad=20)
ax.set_ylim(0, 1)
ax.set_xlim(0, 26)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xticks(candidates)
ax.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'combined_scaling_comparison.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(base_dir, 'combined_scaling_comparison.pdf'), dpi=300, bbox_inches='tight')
plt.show()

# Create table summary
summary_table = pd.DataFrame(columns=['Scenario', 'Candidates', 'Partial', 'Entire', 'Improvement_Partial', 'Improvement_Entire'])

rows = []
for scenario in scenarios:
    for i, num_candidates in enumerate(candidates):
        row = combined_df[combined_df['candidates'] == num_candidates]
        if not row.empty:
            partial_val = row[f'{scenario}/matching_partial'].values[0]
            entire_val = row[f'{scenario}/matching_entire'].values[0]
            
            # Calculate improvement from candidate=1
            if i == 0:
                base_partial = partial_val
                base_entire = entire_val
                improvement_partial = 0
                improvement_entire = 0
            else:
                improvement_partial = (partial_val - base_partial) / base_partial if base_partial > 0 else 0
                improvement_entire = (entire_val - base_entire) / base_entire if base_entire > 0 else 0
            
            rows.append({
                'Scenario': pretty_names[scenario],
                'Candidates': num_candidates,
                'Partial': partial_val,
                'Entire': entire_val,
                'Improvement_Partial': improvement_partial,
                'Improvement_Entire': improvement_entire
            })

summary_table = pd.DataFrame(rows)
summary_table.to_csv(os.path.join(base_dir, 'scaling_summary.csv'), index=False)

# Create heatmap for improvement
pivot_partial = summary_table.pivot(index='Scenario', columns='Candidates', values='Improvement_Partial')
pivot_entire = summary_table.pivot(index='Scenario', columns='Candidates', values='Improvement_Entire')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

sns.heatmap(pivot_partial, annot=True, fmt='.2%', cmap='YlGnBu', ax=ax1)
ax1.set_title('Partial Success Rate Improvement', fontsize=16)
ax1.set_xlabel('Number of Candidates', fontsize=14)
ax1.set_ylabel('Scenario', fontsize=14)

sns.heatmap(pivot_entire, annot=True, fmt='.2%', cmap='YlOrRd', ax=ax2)
ax2.set_title('Entire Success Rate Improvement', fontsize=16)
ax2.set_xlabel('Number of Candidates', fontsize=14)
ax2.set_ylabel('Scenario', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'improvement_heatmap.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(base_dir, 'improvement_heatmap.pdf'), dpi=300, bbox_inches='tight')
# plt.show()