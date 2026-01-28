import pandas as pd
import numpy as np
import os
import argparse
from ete3 import Tree

# ========================================
# Helper Function: Create leaf-level heatmap
# ========================================
def make_hotmap_table_gamma(leaf_labels, hybrid, dataframe, gamma_column='Gamma', p_thresh=0.05):
    leaf_labels_str = [str(l) for l in leaf_labels]
    hybrid_str = str(hybrid)
    
    # Filter for the specific hybrid and significant p-values
    # If Pvalue column doesn't exist, we skip p-filtering
    if 'Pvalue' in dataframe.columns:
        sub = dataframe[(dataframe['Hybrid'].astype(str) == hybrid_str) & (dataframe['Pvalue'] < p_thresh)].copy()
    else:
        sub = dataframe[dataframe['Hybrid'].astype(str) == hybrid_str].copy()
    
    gamma_map = {}
    for _, row in sub.iterrows():
        p1, p2, g = str(row['P1']), str(row['P2']), row[gamma_column]
        gamma_map[(p1, p2)] = g
        gamma_map[(p2, p1)] = 1.0 - g
        
    n = len(leaf_labels_str)
    res = np.full((n, n), np.nan)
    for i, p1 in enumerate(leaf_labels_str):
        for j, p2 in enumerate(leaf_labels_str):
            if i != j:
                res[i, j] = gamma_map.get((p1, p2), np.nan)
    return pd.DataFrame(res, index=leaf_labels_str, columns=leaf_labels_str)


# ========================================
# Core Logic: Calculate Ancestral Nodes (Consensus)
# ========================================
def calculate_all_node_heatmaps(tree, leaf_labels, dataframe, gamma_column='Gamma', 
                               p_thresh=0.05, g_diff=0.2, c_thresh=0.1, c_ratio=0.8):
    """
    Consensus algorithm to map leaf-level hybridization signals to internal nodes.
    """
    leaf_labels_str = [str(l) for l in leaf_labels]
    n_leaves = len(leaf_labels_str)
    label_to_idx = {lbl: i for i, lbl in enumerate(leaf_labels_str)}
    
    drafts = {}
    diffs = {}
    
    print("  Stage 1: Processing nodes (postorder)...")
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            name = str(node.name)
            if name in label_to_idx:
                # Get leaf-level data
                drafts[node] = make_hotmap_table_gamma(leaf_labels_str, name, dataframe, gamma_column, p_thresh).values
            else:
                drafts[node] = np.full((n_leaves, n_leaves), np.nan)
        else:
            drafts[node] = np.full((n_leaves, n_leaves), np.nan)
            diffs[node] = np.full((n_leaves, n_leaves), np.nan)
            # Only consider nodes with 2 children (binary tree)
            if len(node.children) == 2:
                c1, c2 = node.children
                m1, m2 = drafts[c1], drafts[c2]
                mask = np.isfinite(m1) & np.isfinite(m2)
                d = np.abs(m1 - m2)
                diffs[node][mask] = d[mask]
                # If difference is small, average them
                valid_draft = mask & (d < g_diff)
                drafts[node][valid_draft] = (m1[valid_draft] + m2[valid_draft]) / 2.0
                
    final_heatmaps = {}
    print("  Stage 2: Calculating consensus (preorder)...")
    for node in tree.traverse("preorder"):
        if node.is_leaf():
            final_heatmaps[node] = pd.DataFrame(drafts[node], index=leaf_labels_str, columns=leaf_labels_str)
            continue
        
        # Look at the subtree rooted at this node
        subtree_nodes = [n for n in node.traverse() if not n.is_leaf()]
        current_final = np.full((n_leaves, n_leaves), np.nan)
        
        # Aggregate differences across the subtree
        stack = []
        for n in subtree_nodes:
            if n in diffs:
                stack.append(diffs[n])
        
        if stack:
            stack_arr = np.array(stack)
            valid_mask = np.isfinite(stack_arr)
            count_valid = np.sum(valid_mask, axis=0)
            count_low = np.sum(stack_arr < c_thresh, axis=0)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = count_low / count_valid
            
            # Consensus: enough support and small deviation
            consensus_mask = (count_valid > 0) & (ratio > c_ratio)
            current_final[consensus_mask] = drafts[node][consensus_mask]
            
        final_heatmaps[node] = pd.DataFrame(current_final, index=leaf_labels_str, columns=leaf_labels_str)
        
    return final_heatmaps


def main():
    parser = argparse.ArgumentParser(description="Calculate Ancestral Node Gamma values using Consensus.")
    # Changed required=True to False to allow for diff mode which doesn't need input/tree
    parser.add_argument("-i", "--input", help="Input HyDe result file (CSV or TSV)")
    parser.add_argument("-t", "--tree", help="Phylogenetic tree file (Newick)")
    parser.add_argument("-o", "--output", help="Output directory for node CSVs")
    parser.add_argument("--column", default="Gamma", help="Name of the Gamma column in input (default: Gamma)")
    parser.add_argument("--pthresh", type=float, default=0.05, help="P-value threshold (default: 0.05)")
    parser.add_argument("--gdiff", type=float, default=0.2, help="Max allowed Gamma difference between children (default: 0.2)")
    parser.add_argument("--cthresh", type=float, default=0.1, help="Consensus threshold (default: 0.1)")
    parser.add_argument("--ratio", type=float, default=0.8, help="Minimum support ratio for consensus (default: 0.8)")
    
    # New arguments for Diff Mode
    parser.add_argument("--mode", choices=['process', 'diff'], default='process', 
                        help="Mode: 'process' to calc nodes from HyDe output, 'diff' to calc RY-Std difference.")
    parser.add_argument("--std-dir", help="Directory containing Standard CSVs (for diff mode)")
    parser.add_argument("--ry-dir", help="Directory containing RY CSVs (for diff mode)")
    
    args = parser.parse_args()

    # Manual Validation
    if args.mode == 'process':
        if not args.input or not args.tree or not args.output:
            parser.error("Mode 'process' requires --input, --tree, and --output.")
    elif args.mode == 'diff':
        if not args.std_dir or not args.ry_dir or not args.output:
            parser.error("Mode 'diff' requires --std-dir, --ry-dir, and --output.")
    
    if args.mode == 'diff':
        print(f"--- Running Difference Calculation (RY - Std) ---")
        if not os.path.exists(args.output):
            os.makedirs(args.output)
            
        # We need to process both 'nodes' and 'leaves' subdirectories if they exist, 
        # or just the root dir if flat. 
        # Based on previous output: output/node_data/std/nodes and output/node_data/std/leaves
        
        subdirs = ['nodes', 'leaves']
        total_processed = 0
        
        for subdir in subdirs:
            std_path = os.path.join(args.std_dir, subdir)
            ry_path = os.path.join(args.ry_dir, subdir)
            out_path = os.path.join(args.output, subdir)
            
            if not os.path.exists(std_path):
                print(f"Skipping {subdir}: {std_path} not found.")
                continue
            if not os.path.exists(ry_path):
                print(f"Skipping {subdir}: {ry_path} not found.")
                continue
                
            if not os.path.exists(out_path):
                os.makedirs(out_path)
                
            # List files
            std_files = {f for f in os.listdir(std_path) if f.endswith('.csv')}
            ry_files = {f for f in os.listdir(ry_path) if f.endswith('.csv')}
            
            common_files = std_files.intersection(ry_files)
            print(f"Processing {subdir}: {len(common_files)} common files found.")
            
            for fname in common_files:
                try:
                    df_std = pd.read_csv(os.path.join(std_path, fname), index_col=0)
                    df_ry = pd.read_csv(os.path.join(ry_path, fname), index_col=0)
                    
                    # Align Indices
                    if not df_std.index.equals(df_ry.index) or not df_std.columns.equals(df_ry.columns):
                        df_ry = df_ry.reindex(index=df_std.index, columns=df_std.columns)
                    
                    # Formula: |Std - 0.5| - |RY - 0.5|
                    # Positive = RY is closer to 0.5 (Improved/Significant)
                    # Negative = Standard is closer to 0.5
                    std_vals = df_std.values
                    ry_vals = df_ry.values
                    dist_std = np.abs(std_vals - 0.5)
                    dist_ry = np.abs(ry_vals - 0.5)
                    diff_vals = dist_std - dist_ry
                    
                    df_diff = pd.DataFrame(diff_vals, index=df_std.index, columns=df_std.columns)
                    
                    # Save
                    # Naming convention: The user might want distinct names, but keeping original name in a new dir is standard.
                    # Or we can append _diff. Let's keep original filename but in 'diff' folder.
                    df_diff.to_csv(os.path.join(out_path, fname))
                    total_processed += 1
                except Exception as e:
                    print(f"Error processing {fname}: {e}")
        
        print(f"Difference calculation complete. {total_processed} files saved to {args.output}")
        return

    # --- Existing 'process' Mode Logic ---
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # 1. Load data
    print(f"Loading data from {args.input}...")
    # Check if TSV or CSV
    if args.input.endswith('.tsv') or args.input.endswith('.txt'):
        df = pd.read_csv(args.input, sep="\t")
    else:
        df = pd.read_csv(args.input)
    
    # 2. Parse tree
    print(f"Parsing tree from {args.tree}...")
    tree = Tree(args.tree)
    leaf_labels = [str(l.name) for l in tree.get_leaves()]
    
    # 3. Calculate internal node heatmaps
    print(f"Calculating node values for column: {args.column}...")
    
    # We need to modify calculate_all_node_heatmaps to return leaf data as well,
    # or just access the 'drafts' if we move it out, or simply re-iterate for leaves.
    # To keep it clean, let's just do a dedicated loop for leaves here since we have the data and tree.
    
    # --- Part A: Internal Nodes ---
    node_heatmaps = calculate_all_node_heatmaps(
        tree, leaf_labels, df, args.column,
        p_thresh=args.pthresh, g_diff=args.gdiff, c_thresh=args.cthresh, c_ratio=args.ratio
    )
    
    # Create subdirectories
    nodes_dir = os.path.join(args.output, "nodes")
    leaves_dir = os.path.join(args.output, "leaves")
    if not os.path.exists(nodes_dir): os.makedirs(nodes_dir)
    if not os.path.exists(leaves_dir): os.makedirs(leaves_dir)

    # Save Internal Nodes
    print(f"Saving internal node heatmaps to {nodes_dir}...")
    internal_nodes = [n for n in tree.traverse('postorder') if not n.is_leaf() and not n.is_root()]
    
    for node_idx, node in enumerate(internal_nodes, 1):
        if node in node_heatmaps:
            heatmap = node_heatmaps[node]
            filename = f"Node_{node_idx}.csv"
            filepath = os.path.join(nodes_dir, filename)
            heatmap.to_csv(filepath)
            
    # --- Part B: Leaves ---
    print(f"Saving leaf heatmaps to {leaves_dir}...")
    # Hybrids in input dataframe
    unique_hybrids = set(df['Hybrid'].astype(str).unique())
    
    count_leaves = 0
    for leaf in tree.get_leaves():
        leaf_name = str(leaf.name)
        # Only generate heatmap if this leaf appears as a 'Hybrid' in the input data
        if leaf_name in unique_hybrids:
            heatmap = make_hotmap_table_gamma(leaf_labels, leaf_name, df, args.column, args.pthresh)
            # Check if heatmap has any valid data (optional, but good for cleanliness)
            if not heatmap.isnull().all().all():
                safe_name = leaf_name.replace("/", "_").replace(" ", "_")
                filename = f"{safe_name}.csv"
                filepath = os.path.join(leaves_dir, filename)
                heatmap.to_csv(filepath)
                count_leaves += 1
    
    print(f"\nDone! Saved {len(internal_nodes)} internal nodes and {count_leaves} leaves.")

if __name__ == "__main__":
    main()