import os
import io
import argparse
import warnings
import colorsys
from concurrent.futures import ProcessPoolExecutor

# --- Data Science & AI ---
import numpy as np
import pandas as pd
from PIL import Image

# --- Bioinformatics ---
from ete3 import Tree, TreeStyle, NodeStyle, faces, random_color, TextFace

# --- Visualization Pipeline ---
import matplotlib
matplotlib.use('Agg')
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase

# --- Environment Setup ---
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

Description = """
Visual HyDe (Pure Viz)
Generates heatmaps + tree visualizations from pre-calculated CSV matrices.
Input: A directory containing CSV files (node or leaf data).
Output: Visualization images for each CSV.
"""

def get_safe_leaf_name(name):
    return str(name).replace("/", "_").replace(" ", "_")

def get_automatic_clades(tree):
    clades = []
    def traverse(node):
        size = len(node.get_leaves())
        if size <= 4:
            clades.append([str(l.name) for l in node.get_leaves()])
        else:
            if node.is_leaf():
                clades.append([str(node.name)])
            else:
                for child in node.children:
                    traverse(child)
    traverse(tree)
    return clades

def parse_tree(tree_file):
    try:
        t = Tree(tree_file)
        clades = get_automatic_clades(t)
        leaf_names = [str(l.name) for l in t.get_leaves()]
        max_len = max(len(n) for n in leaf_names) if leaf_names else 10
        return t, leaf_names, clades, (0, max_len)
    except Exception as e:
        print(f"Error parsing tree: {e}")
        return None

def get_shared_resources(tree_file, name_len, clade_definitions, brightness=0.6, mode='std', limit=0.2):
    if mode == 'diff':
        # Diverging Colormap (Blue - LightGray/White - Red)
        colors = [
            (0.13, 0.40, 0.67), # Deep Blue
            (0.40, 0.66, 0.81), # Medium Blue
            (0.90, 0.90, 0.90), # Light Gray (Center)
            (0.93, 0.54, 0.38), # Medium Red
            (0.70, 0.09, 0.17)  # Deep Red
        ]
        cmap = LinearSegmentedColormap.from_list("BlueGrayRed", colors, N=256)
        cmap.set_bad(color='white', alpha=0)
        
        # Norm centered on 0
        norm = Normalize(vmin=-limit, vmax=limit)
        
        # Colorbar
        fig = plt.figure(figsize=(2, 10))
        ax = fig.add_axes([0.2, 0.05, 0.2, 0.9])
        cb = ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
        cb.ax.tick_params(labelsize=30)
        
    else:
        # Original Logic: Black/Blue -> Red (Gamma 0-1)
        cmap_list = []
        steps = 5000
        for i in range(steps): cmap_list.append([0, 0, 1.0 - i/steps, i/steps])
        for i in range(steps): cmap_list.append([i/steps, 0, 0, 1.0 - i/steps])
        original_cmap = ListedColormap(np.clip(cmap_list, 0, 1))
        original_cmap.set_bad(color='white', alpha=0)
        
        # Darker version for colorbar
        darker = []
        for r, g, b, a in original_cmap.colors:
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            nr, ng, nb = colorsys.hsv_to_rgb(h, s, max(0, min(1, v * brightness)))
            darker.append([nr, ng, nb, a])
        cbar_cmap = ListedColormap(darker)
        
        cmap = original_cmap
        norm = Normalize(vmin=0, vmax=1)
        
        fig = plt.figure(figsize=(2, 10))
        ax = fig.add_axes([0.2, 0.05, 0.2, 0.9])
        cb = ColorbarBase(ax, cmap=cbar_cmap, norm=norm, orientation='vertical')
        cb.ax.tick_params(labelsize=40)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return {
        'cmap': cmap,
        'cbar_img': Image.open(buf).copy(),
        'norm': norm,
        'outdir': None # Placeholder, will be set in main
    }

def generate_hyde_visualization(prefix, hyde_array, tree_file, node_label, highlight, clades, name_len, shared):
    cmap = shared['cmap']
    cbar_img = shared['cbar_img']
    norm = shared['norm']
    outdir = shared['outdir']
    
    # --- Reindexing Logic (Added to ensure alignment) ---
    if isinstance(hyde_array, pd.DataFrame):
        # Parse tree to get leaf order
        t_temp = Tree(tree_file)
        tree_leaves = [str(l.name) for l in t_temp.get_leaves()]
        
        # Reindex DataFrame to match Tree leaf order exactly
        hyde_array = hyde_array.reindex(index=tree_leaves, columns=tree_leaves)
        
        n_leaves = len(hyde_array.index)
        data_values = hyde_array.values
    else:
        n_leaves = hyde_array.shape[0]
        data_values = hyde_array

    if n_leaves == 0: return False

    # --- Apply Lower Triangle Mask (Hide Upper Right) ---
    # Matches visual_hyde_diff.py style for consistency
    mask = np.triu(np.ones((n_leaves, n_leaves), dtype=bool), k=1)
    data_values = data_values.astype(float)
    data_values[mask] = np.nan
    
    # --- Part 1: Draw Hotmap ---
    inches_per_leaf = 0.4
    base_inches, max_inches = 2.0, 30.0
    fig_inches = min(max_inches, base_inches + n_leaves * inches_per_leaf)
    
    fig_hotmap = plt.figure(figsize=(fig_inches, fig_inches))
    ax = fig_hotmap.add_axes([0.0001, 0.0001, 0.9998, 0.9998])
    # Use data_values (numpy array) which is now aligned and masked
    ax.imshow(data_values, norm=norm, cmap=cmap, interpolation='nearest', aspect='equal')
    
    # Restore grid lines
    grid_linewidth = max(0.5, min(2, fig_inches / 10.0))
    ax.set_xticks(np.arange(n_leaves + 1) - .5, minor=True)
    ax.set_yticks(np.arange(n_leaves + 1) - .5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=grid_linewidth)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xticklabels([]); ax.set_yticklabels([])
    
    hotmap_buf = io.BytesIO()
    plt.savefig(hotmap_buf, format='png', dpi=300)
    plt.close(fig_hotmap)
    hotmap_buf.seek(0)
    hotpic = Image.open(hotmap_buf)

    # --- Part 2: Draw Tree ---
    def node_layout(node, color=None):
        if node.is_leaf():
            node_name_str = str(node.name) if node.name is not None else ""
            # Adjusted padding to reduce excessive gap
            # Using (max - current + 2) ensures alignment without huge empty space
            padding_dots = "·" * (name_len[1] - len(node_name_str) + 2)
            descFace = faces.TextFace(node_name_str + padding_dots, fsize=30, fgcolor=color if color else 'black')
            descFace.margin_top = 2; descFace.margin_bottom = 2; node.add_face(descFace, column=1, position='aligned')
        ns = NodeStyle(); ns["hz_line_width"]=4; ns["vt_line_width"]=4; ns["size"]=0
        if color: ns["vt_line_color"]=color; ns["hz_line_color"]=color
        node.set_style(ns)

    t = Tree(tree_file)
    ts = TreeStyle(); ts.scale = 40; ts.draw_guiding_lines = True; ts.show_leaf_name = False; ts.force_topology = True; ts.show_scale = False
    
    # --- Determine Colors First ---
    node_colors = {}
    h = 0
    for each_subtree in clades:
        color = random_color(h, s=0.9, l=0.4); h += 0.58
        subtree_str = {str(name) for name in each_subtree if name}
        if not subtree_str: continue
        nodes_in_subtree = [l for l in t.get_leaves() if str(l.name) in subtree_str]
        if nodes_in_subtree:
            if len(nodes_in_subtree) == 1:
                node_colors[nodes_in_subtree[0]] = color
            else:
                ancestor = t.get_common_ancestor(nodes_in_subtree)
                for n in ancestor.traverse():
                    if not n.is_leaf() or str(n.name) in subtree_str:
                        node_colors[n] = color

    # --- Apply Styles and Labels Once ---
    for node in t.traverse():
        color = node_colors.get(node)
        node_layout(node, color)
        # Highlight logic (o marker)
        if node.is_leaf() and highlight and str(node.name) == highlight:
            node_face = TextFace("o", fsize=40); node_face.background.color = "red"
            node.add_face(node_face, column=0, position="branch-right")

    if node_label and isinstance(highlight, list):
        target_leaves = {str(name) for name in highlight}
        target_nodes = [l for l in t.get_leaves() if str(l.name) in target_leaves]
        if target_nodes:
            ancestor = t.get_common_ancestor(target_nodes)
            node_face = TextFace(str(node_label), fsize=40); node_face.background.color = "LightGreen"
            ancestor.add_face(node_face, column=0, position="branch-right")

    temp_tree = os.path.join(outdir, f"temp_{prefix}_{os.getpid()}_tree.png")
    try:
        # Render at the exact height of the hotmap to ensure 1:1 alignment and clarity
        t.render(temp_tree, h=hotpic.height, tree_style=ts, dpi=300)
        treepic = Image.open(temp_tree).copy()
    finally:
        if os.path.exists(temp_tree): os.remove(temp_tree)
    
    # --- Part 3: Combine ---
    # Render bottom tree at the exact width of hotmap (before rotation)
    temp_rot = os.path.join(outdir, f"temp_{prefix}_{os.getpid()}_rot.png")
    try:
        t.render(temp_rot, h=hotpic.width, tree_style=ts, dpi=300)
        treepic_rotate_orig = Image.open(temp_rot).rotate(90, expand=True)
    finally:
        if os.path.exists(temp_rot): os.remove(temp_rot)

    target_h, target_w = hotpic.height, hotpic.width
    
    new_tree_w = int(target_h * (treepic.width / treepic.height))
    treepic_resized = treepic.resize((new_tree_w, target_h), Image.LANCZOS)
    
    new_rot_h = int(target_w * (treepic_rotate_orig.height / treepic_rotate_orig.width))
    treepic_rotate = treepic_rotate_orig.resize((target_w, new_rot_h), Image.LANCZOS)
    
    new_cbar_w = int(target_h * (cbar_img.width / cbar_img.height))
    colorbarpic = cbar_img.resize((new_cbar_w, target_h), Image.LANCZOS)
    
    padding, p_between = 20, 10
    total_w = treepic_resized.width + target_w + colorbarpic.width + 4 * padding
    total_h = max(treepic_resized.height, target_h + p_between + treepic_rotate.height) + 2 * padding
    
    combine = Image.new("RGB", (total_w, total_h), "#FFFFFF")
    combine.paste(treepic_resized, (padding, padding))
    combine.paste(hotpic, (padding + treepic_resized.width + padding, padding))
    combine.paste(treepic_rotate, (padding + treepic_resized.width + padding, padding + target_h + p_between))
    combine.paste(colorbarpic, (padding + treepic_resized.width + padding + target_w + padding, padding))
    
    save_path = os.path.join(outdir, prefix + ".png")
    combine.save(save_path)
    return True

def file_worker(p):
    filepath, tree_file, clades, nlen, shared = p
    filename = os.path.basename(filepath)
    prefix = os.path.splitext(filename)[0]
    
    try:
        df = pd.read_csv(filepath, index_col=0)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return False
        
    # Determine if this is a Node or a Leaf based on filename or just treat generic?
    # User said: "Output one image per CSV".
    # Logic:
    # 1. If it's a Node (e.g. Node_1.csv), we want to highlight the subtree.
    # 2. If it's a Leaf (e.g. Species_Name.csv), we want to highlight the leaf.
    
    t = Tree(tree_file)
    
    node_label = ""
    highlight = None
    
    if prefix.startswith("Node_"):
        # Try to parse Node ID
        try:
            node_idx = int(prefix.split("_")[1])
            # Reconstruct tree to find the node
            # Assuming postorder index matches (1-based)
            internal_nodes = [n for n in t.traverse('postorder') if not n.is_leaf() and not n.is_root()]
            if 1 <= node_idx <= len(internal_nodes):
                target_node = internal_nodes[node_idx - 1]
                highlight = [str(l.name) for l in target_node.get_leaves()]
                node_label = f"Node_{node_idx}"
        except:
            pass
    else:
        # Assume filename is species name
        # Restore safe name to original might be hard, but usually safe name is enough to match
        # Actually, our visualizer uses safe names for filenames.
        # But tree leaves might have original names.
        # Let's try to match prefix to leaf names.
        
        # Simple heuristic: treat prefix as the highlight target (Leaf name)
        # Note: Filenames might have been sanitized (spaces -> underscores).
        # We might need to match fuzzily or assume tree uses underscores too.
        highlight = prefix # String
        
    return generate_hyde_visualization(prefix, df, tree_file, node_label, highlight, clades, nlen, shared)

def main():
    parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', '--data-dir', required=True, 
                        help='Directory containing CSV files to visualize.')
    parser.add_argument('-t', '--treefile', required=True, 
                        help='Path to phylogenetic species tree file.')
    parser.add_argument('-o', '--outdir', type=str, default="visualization_output", 
                        help='Output directory.')
    parser.add_argument('--mode', type=str, default='std', choices=['std', 'diff'],
                        help="Visualization mode: 'std' (0 to 1) or 'diff' (-limit to +limit). Default: std")
    parser.add_argument('--limit', type=float, default=0.2,
                        help="Limit for diff mode (e.g. 0.2 means range -0.2 to 0.2). Default: 0.2")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        
    # 1. Parse Tree
    t, leaves, clades, nlen = parse_tree(args.treefile)
    if not t: return
    
    # 2. Find CSVs
    files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".csv")]
    if not files:
        print(f"No CSV files found in {args.data_dir}")
        return
        
    print(f"Found {len(files)} CSV files to process in mode '{args.mode}'.")
    
    # 3. Shared Resources
    shared = get_shared_resources(args.treefile, nlen, clades, mode=args.mode, limit=args.limit)
    shared['outdir'] = args.outdir
    
    # 4. Run Parallel
    tasks = [(f, args.treefile, clades, nlen, shared) for f in files]
    
    with ProcessPoolExecutor() as exe:
        results = list(exe.map(file_worker, tasks))
        
    print(f"Done. Generated {sum(results)} images in {args.outdir}.")

if __name__ == "__main__":
    main()
