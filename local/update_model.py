import os
import subprocess

# 版本管理
version_file = "../models/version.txt"
models_dir = "../models"
lora_path = os.path.join(models_dir, "latest_lora.safetensors")

def read_version():
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            ver = f.read().strip()
            # 解析版本号，如 v0.1 -> 0.1
            try:
                num = float(ver.replace("v", ""))
                return num, ver
            except:
                return 0.1, "v0.1"
    else:
        return 0.1, "v0.1"

def write_version(new_ver):
    with open(version_file, "w") as f:
        f.write(new_ver)

def main():
    if not os.path.exists(lora_path):
        print("[错误] 未找到 latest_lora.safetensors，请先运行训练！")
        return
    
    current_num, current_ver = read_version()
    new_num = current_num + 0.1
    new_ver = f"v{new_num:.1f}"
    
    print(f"[更新] 检测到新 LoRA 权重，正在合并模型...")
    
    # 确保 Modelfile 存在（从项目根目录）
    modelfile_path = "../Modelfile"
    if not os.path.exists(modelfile_path):
        print("[错误] Modelfile 不存在！")
        return
    
    # 执行 ollama create
    try:
        result = subprocess.run(
            ["ollama", "create", "phi3:evolving", "-f", modelfile_path],
            cwd="..",
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        write_version(new_ver)
        print(f"[成功] 模型已更新到 {new_ver}，能力提升！")
    except subprocess.CalledProcessError as e:
        print(f"[失败] Ollama 合并失败: {e.stderr}")
    except FileNotFoundError:
        print("[错误] 未找到 ollama 命令，请确保已安装并加入 PATH")

if __name__ == "__main__":
    main()