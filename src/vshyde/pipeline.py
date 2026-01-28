#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
import time
import shutil

"""
HyDe Analysis Pipeline
Automates the workflow: run_ry_hyde.py -> run_sensitivity_analysis.py -> visual_hyde.py
"""

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def run_command(cmd, cwd=None, log_file=None):
    cmd_str = ' '.join(cmd)
    print(f"Executing: {cmd_str}")
    try:
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"\n--- [{time.ctime()}] Executing: {cmd_str} ---\n")
                f.flush()
                # Run and pipe both stdout and stderr to the log file
                process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    f.write(line)
                    f.flush()
                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)
        else:
            subprocess.run(cmd, cwd=cwd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def is_complete(path, type="file"):
    """
    Check if a file or directory is complete.
    - file: non-empty and has header
    - dir: exists and has files
    """
    if not os.path.exists(path):
        return False
    
    if type == "file":
        if os.path.getsize(path) < 100: # Very small file is likely a failure or just header
            return False
        # Check for header
        try:
            with open(path, 'r') as f:
                first_line = f.readline()
                return "P1" in first_line and "P2" in first_line and "Gamma" in first_line
        except:
            return False
    elif type == "dir":
        return os.path.isdir(path) and len(os.listdir(path)) > 0
    return False

def main():
    parser = argparse.ArgumentParser(description="Master Pipeline for HyDe Analysis (Standard, RY, and Diff modes)")
    
    # Core Inputs
    group_input = parser.add_argument_group("Required Inputs")
    group_input.add_argument("-i", "--input", required=True, help="Input concatenated CDS fasta file")
    group_input.add_argument("-o", "--outgroup", required=True, help="Name of the outgroup species")
    group_input.add_argument("-t", "--tree", required=True, help="Phylogenetic tree file (Newick)")
    
    # Output Control
    group_output = parser.add_argument_group("Output Control")
    group_output.add_argument("-r", "--results", default="pipeline_results", help="Root directory for all results (default: pipeline_results)")
    
    # Execution Modes
    group_mode = parser.add_argument_group("Execution Modes")
    group_mode.add_argument("--run-mode", choices=['std', 'all'], default='all', 
                        help="Analysis mode: \n"
                             "std: Standard HyDe + Sensitivity + Viz\n"
                             "all: Run everything (Standard + RY + Diff) (default)")
    
    # Parameters
    group_params = parser.add_argument_group("Algorithm Parameters")
    group_params.add_argument("--pthresh", type=float, default=0.05, help="P-value threshold for sensitivity analysis (default: 0.05)")
    group_params.add_argument("--gdiff", type=float, default=0.2, help="Max allowed Gamma difference for node consensus (default: 0.2)")
    group_params.add_argument("--threads", type=str, default=str(os.cpu_count() or 4), help="Number of threads for HyDe analysis")
    group_params.add_argument("-f", "--force", action="store_true", help="Force re-run all steps even if results exist")
    
    args = parser.parse_args()

    # Paths to sub-scripts
    # Assumes structure:
    # vshyde/
    #   pipeline.py
    #   scripts/
    #     run_ry_hyde.py
    #     ...
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(base_dir, "scripts")
    
    run_ry_hyde_py = os.path.join(script_dir, "run_ry_hyde.py")
    sensitivity_py = os.path.join(script_dir, "run_sensitivity_analysis.py")
    visual_hyde_py = os.path.join(script_dir, "visual_hyde.py")

    # Input Validation
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at '{args.input}'")
        sys.exit(1)
    if not os.path.exists(args.tree):
        print(f"Error: Tree file not found at '{args.tree}'")
        sys.exit(1)

    # Convert to absolute paths
    input_file = os.path.abspath(args.input)
    tree_file = os.path.abspath(args.tree)

    # Verify script existence
    for s in [run_ry_hyde_py, sensitivity_py, visual_hyde_py]:
        if not os.path.exists(s):
            print(f"Critical Error: Script not found at {s}")
            sys.exit(1)

    # Directories
    root_dir = os.path.abspath(args.results)
    ensure_dir(root_dir)
    log_file = os.path.join(root_dir, "pipeline.log")
    
    # Sub-directories for organization
    hyde_std_dir = os.path.join(root_dir, "hyde_standard")
    hyde_ry_dir = os.path.join(root_dir, "hyde_ry")
    sens_std_dir = os.path.join(root_dir, "processed_std")
    sens_ry_dir = os.path.join(root_dir, "processed_ry")
    diff_dir = os.path.join(root_dir, "processed_diff")
    viz_dir = os.path.join(root_dir, "visualizations")

    print(f"--- HyDe Pipeline Started ---")
    print(f"Root Directory: {root_dir}")
    print(f"Log File: {log_file}")

    # Logic for what to run
    run_std = True # Standard is always part of std and all modes
    run_ry = args.run_mode == 'all'
    run_diff = args.run_mode == 'all'

    # --- Phase 1: Analysis & Processing ---
    # Standard Mode
    if run_std:
        print("\n[STEP 1/3] Running Standard Analysis...")
        input_prefix = os.path.splitext(os.path.basename(input_file))[0]
        hyde_txt = os.path.join(hyde_std_dir, f"{input_prefix}_analysis", f"{input_prefix}-out.txt")
        
        # 1.1 HyDe
        if not args.force and is_complete(hyde_txt, "file"):
            print(f"  [Skipped] HyDe results already exist and are complete: {hyde_txt}")
        else:
            run_command([sys.executable, run_ry_hyde_py, "-i", input_file, "-o", args.outgroup, "-r", hyde_std_dir, "-m", "std", "-j", args.threads], log_file=log_file)
            
        # 1.2 Sensitivity
        sens_nodes_dir = os.path.join(sens_std_dir, "nodes")
        if not args.force and is_complete(sens_nodes_dir, "dir"):
             print(f"  [Skipped] Sensitivity processed results already exist in {sens_std_dir}")
        else:
            if os.path.exists(hyde_txt):
                run_command([sys.executable, sensitivity_py, "-i", hyde_txt, "-t", tree_file, "-o", sens_std_dir, "--mode", "process", "--pthresh", str(args.pthresh), "--gdiff", str(args.gdiff)], log_file=log_file)
            else:
                print(f"Warning: Could not find HyDe output at {hyde_txt}")

    # RY Mode
    if run_ry:
        print("\n[STEP 1/3] Running RY Analysis...")
        input_prefix = os.path.splitext(os.path.basename(input_file))[0]
        ry_prefix = input_prefix + "_RY"
        hyde_ry_txt = os.path.join(hyde_ry_dir, f"{ry_prefix}_analysis", f"{ry_prefix}-out.txt")
        
        # 2.1 HyDe
        if not args.force and is_complete(hyde_ry_txt, "file"):
            print(f"  [Skipped] RY-HyDe results already exist and are complete: {hyde_ry_txt}")
        else:
            run_command([sys.executable, run_ry_hyde_py, "-i", input_file, "-o", args.outgroup, "-r", hyde_ry_dir, "-m", "ry", "-j", args.threads], log_file=log_file)
            
        # 2.2 Sensitivity
        sens_ry_nodes_dir = os.path.join(sens_ry_dir, "nodes")
        if not args.force and is_complete(sens_ry_nodes_dir, "dir"):
            print(f"  [Skipped] RY Sensitivity processed results already exist in {sens_ry_dir}")
        else:
            if os.path.exists(hyde_ry_txt):
                run_command([sys.executable, sensitivity_py, "-i", hyde_ry_txt, "-t", tree_file, "-o", sens_ry_dir, "--mode", "process", "--pthresh", str(args.pthresh), "--gdiff", str(args.gdiff)], log_file=log_file)
            else:
                print(f"Warning: Could not find HyDe output at {hyde_ry_txt}")

    # Diff Mode (Difference calculation)
    if run_diff:
        print("\n[STEP 2/3] Calculating Differences (RY - Std)...")
        diff_nodes_dir = os.path.join(diff_dir, "nodes")
        if not args.force and is_complete(diff_nodes_dir, "dir"):
            print(f"  [Skipped] Difference results already exist in {diff_dir}")
        else:
            if os.path.exists(sens_std_dir) and os.path.exists(sens_ry_dir):
                run_command([sys.executable, sensitivity_py, "--mode", "diff", "--std-dir", sens_std_dir, "--ry-dir", sens_ry_dir, "-o", diff_dir], log_file=log_file)
            else:
                print("Error: Difference calculation requires both Standard and RY processed results.")
    print("\n[STEP 3/3] Generating Visualizations...")
    ensure_dir(viz_dir)

    # Standard Viz (Including Nodes and Leaves)
    if run_std and os.path.exists(sens_std_dir):
        print("  Visualizing Standard results...")
        for sub in ["nodes", "leaves"]:
            data_path = os.path.join(sens_std_dir, sub)
            if os.path.exists(data_path) and os.listdir(data_path):
                run_command([sys.executable, visual_hyde_py, "-d", data_path, "-t", tree_file, "-o", os.path.join(viz_dir, "std", sub), "--mode", "std"], log_file=log_file)

    # RY Viz
    if run_ry and os.path.exists(sens_ry_dir):
        print("  Visualizing RY results...")
        for sub in ["nodes", "leaves"]:
            data_path = os.path.join(sens_ry_dir, sub)
            if os.path.exists(data_path) and os.listdir(data_path):
                run_command([sys.executable, visual_hyde_py, "-d", data_path, "-t", tree_file, "-o", os.path.join(viz_dir, "ry", sub), "--mode", "std"], log_file=log_file)

    # Diff Viz
    if run_diff and os.path.exists(diff_dir):
        print("  Visualizing Difference results...")
        for sub in ["nodes", "leaves"]:
            data_path = os.path.join(diff_dir, sub)
            if os.path.exists(data_path) and os.listdir(data_path):
                run_command([sys.executable, visual_hyde_py, "-d", data_path, "-t", tree_file, "-o", os.path.join(viz_dir, "diff", sub), "--mode", "diff"], log_file=log_file)

    print(f"\n--- Pipeline Finished! ---")
    print(f"All visualizations saved to: {viz_dir}")

if __name__ == "__main__":
    main()
