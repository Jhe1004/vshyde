# -*- coding: utf-8 -*-
import argparse
import sys
import os
import shutil

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def convert_fasta_to_ry(input_fasta, output_dir):
    """
    读取FASTA文件，转换为RY编码文件。
    Stream processing for speed and low memory usage.
    """
    base_name = os.path.basename(input_fasta)
    name_root = os.path.splitext(base_name)[0]
    output_fasta = os.path.join(output_dir, f"{name_root}_RY.fasta")
    
    print(f"正在读取并转换文件: {input_fasta}")
    
    # Pre-compute translation table
    # A->A, G->A (Purines -> R -> A)
    # C->T, T->T (Pyrimidines -> Y -> T)
    # We use A and T to represent R and Y for compatibility with standard tools.
    # Note: str.translate is very fast. 
    # We map upper and lower case.
    # Characters not in the map (like -, ?, N) are left as is.
    trans_table = str.maketrans("AGCTagct", "AATTaatt")
    
    with open(output_fasta, 'w') as f_out, open(input_fasta, 'r') as f_in:
        for line in f_in:
            if line.startswith(">"):
                f_out.write(line)
            else:
                # Direct stream translation
                # Keep the original line structure (wrapping)
                f_out.write(line.translate(trans_table))
                
    print(f"RY编码文件已生成: {output_fasta}")
    return output_fasta

def fasta2phy(fasta_file):
    """
    将FASTA转化为PHY格式 (Sequential) - Memory Efficient Version.
    Uses a two-pass approach to avoid loading the whole file into RAM.
    返回 (species_num, species_len, phy_filename)
    """
    print(f"Counting species and sites in {fasta_file}...")
    species_num = 0
    species_len = 0
    
    # Pass 1: Count species and get length of the first one
    with open(fasta_file, 'r') as f:
        first_seq_found = False
        current_len = 0
        for line in f:
            line = line.strip()
            if not line: continue
            
            if line.startswith(">"):
                species_num += 1
                if first_seq_found and species_num == 2:
                    # We have finished reading the first sequence
                    species_len = current_len
                first_seq_found = True
                current_len = 0 # Reset for next (though we only need the first one)
            elif species_num == 1:
                # Accumulate length only for the first sequence
                current_len += len(line)
        
        # If there was only 1 sequence, set length
        if species_num == 1:
            species_len = current_len

    print(f"Detected: {species_num} taxa, {species_len} sites.")
    
    phy_file = os.path.splitext(fasta_file)[0] + ".phy"
    
    # Pass 2: Write to PHY stream
    print(f"Writing PHY file: {phy_file}")
    with open(fasta_file, 'r') as f_in, open(phy_file, 'w') as f_out:
        f_out.write(f" {species_num} {species_len}\n")
        
        current_seq_written = False
        
        for line in f_in:
            line = line.strip()
            if not line: continue
            
            if line.startswith(">"):
                if current_seq_written:
                    f_out.write("\n") # Finish previous sequence
                
                # Write ID (Take first word, standard Phylip/HyDe expectation)
                name = line[1:].split()[0]
                f_out.write(f"{name} ")
                current_seq_written = True
            else:
                f_out.write(line)
        
        f_out.write("\n") # Finish last sequence
            
    return str(species_num), str(species_len), phy_file

def make_mapfile(fasta_file, outgroup_name, map_filename):
    """
    创建 map.txt 文件
    """
    with open(map_filename, "w") as write_file:
        with open(fasta_file, "r") as read_file:
            for each_line in read_file:
                if each_line.startswith(">"):
                    species_name = each_line.replace(">", "").strip()
                    if species_name == outgroup_name:
                        write_file.write(f"{species_name}\tout\n")
                    else:
                        write_file.write(f"{species_name}\t{species_name}\n")

def find_hyde_script():
    """
    查找 hyde 脚本路径 (Vendor 版本)
    """
    # Get the directory where this script (run_ry_hyde.py) is located
    # Expected: .../src/vshyde/scripts/run_ry_hyde.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to 'vshyde'
    vshyde_root = os.path.dirname(current_dir)
    
    # Path to vendored script: .../src/vshyde/vendor/run_hyde_mp.py
    vendor_script = os.path.join(vshyde_root, "vendor", "run_hyde_mp.py")
    
    if os.path.exists(vendor_script):
        return vendor_script
            
    return None

import subprocess

def run_hyde_analysis_with_threads(fasta_file, outgroup, output_dir, threads):
    """
    对单个FASTA文件执行完整的HyDe分析流程
    """
    base_name = os.path.splitext(os.path.basename(fasta_file))[0]
    work_dir = os.path.join(output_dir, base_name + "_analysis")
    ensure_dir(work_dir)
    
    print(f"\n[{base_name}] 开始HyDe分析...")
    print(f"工作目录: {work_dir}")
    
    # 1. 复制/移动 fasta 到工作目录
    local_fasta = os.path.join(work_dir, os.path.basename(fasta_file))
    shutil.copy2(fasta_file, local_fasta)
    
    # 2. 转换 PHY
    print(f"[{base_name}] 转换为PHY格式...")
    sp_num, sp_len, phy_file = fasta2phy(local_fasta)
    
    # 3. 创建 Map 文件
    map_file = os.path.join(work_dir, "map.txt")
    print(f"[{base_name}] 创建Map文件 (外群: {outgroup})...")
    make_mapfile(local_fasta, outgroup, map_file)
    
    # 4. 运行 HyDe
    hyde_script = find_hyde_script()
    if not hyde_script:
        print(f"错误: 找不到 run_hyde_mp.py 脚本。")
        return
    
    print(f"[{base_name}] 运行 HyDe ({os.path.basename(hyde_script)})...")
    
    # threads is now passed as an argument
    
    # Calculate HyDe root for PYTHONPATH
    hyde_pkg_path = os.path.abspath(os.path.join(os.path.dirname(hyde_script), ".."))
    
    # Prepare environment
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{hyde_pkg_path}:{current_pythonpath}"
    
    # Construct command list
    cmd = [
        sys.executable,
        hyde_script,
        "-i", os.path.abspath(phy_file),
        "-m", os.path.abspath(map_file),
        "-o", "out",
        "-n", sp_num,
        "-t", sp_num,
        "-s", sp_len,
        "--prefix", base_name,
        "--ignore_amb_sites"
    ]
    
    if "mp" in hyde_script:
        cmd.extend(["-j", threads])
        
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run subprocess
        result = subprocess.run(cmd, cwd=work_dir, env=env, check=True)
        print(f"[{base_name}] HyDe 完成.")
    except subprocess.CalledProcessError as e:
        print(f"[{base_name}] Error: HyDe failed with exit code {e.returncode}")

def main():
    parser = argparse.ArgumentParser(description="Run HyDe Analysis (Standard or RY-coded).")
    parser.add_argument("-i", "--input", required=True, help="Input concatenated CDS fasta file")
    parser.add_argument("-o", "--outgroup", required=True, help="Name of the outgroup species (in input Fasta)")
    parser.add_argument("-r", "--results", default="hyde_results", help="Directory for results")
    parser.add_argument("-m", "--mode", choices=['ry', 'std'], default='ry', help="Analysis mode: 'ry' (default) or 'std' (standard)")
    parser.add_argument("-j", "--threads", type=str, default=str(os.cpu_count() or 4), help="Number of threads for HyDe")
    
    args = parser.parse_args()
    
    input_file = os.path.abspath(args.input)
    results_dir = os.path.abspath(args.results)
    outgroup = args.outgroup
    
    ensure_dir(results_dir)
    
    if args.mode == 'ry':
        print("\n--- Running in RY Mode ---")
        # 1. Convert to RY
        ry_fasta = convert_fasta_to_ry(input_file, results_dir)
        # 2. Run HyDe
        run_hyde_analysis_with_threads(ry_fasta, outgroup, results_dir, args.threads)
    else:
        print("\n--- Running in Standard Mode ---")
        # Direct run
        run_hyde_analysis_with_threads(input_file, outgroup, results_dir, args.threads)
    
    print(f"\n所有 {args.mode.upper()}-HyDe 分析已完成。")

if __name__ == "__main__":
    main()
