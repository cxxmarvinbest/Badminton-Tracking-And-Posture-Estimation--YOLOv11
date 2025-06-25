import argparse
import subprocess
import sys

def main():
    # 配置参数
    command = [
        sys.executable,  # 使用当前解释器路径
        "predict.py",
        "--video_file", "output.mp4",
        "--tracknet_file", "ckpts/TrackNet_best.pt",
        "--inpaintnet_file", "ckpts/InpaintNet_best.pt",
        "--save_dir", "prediction",
        "--batch_size", "4",
        "--output_video"
    ]
    
    # 执行命令
    result = subprocess.run(command, check=True)
    
    if result.returncode == 0:
        print("\n预测完成！结果保存在 prediction/ 目录")
    else:
        print("\n执行过程中出现错误！")

if __name__ == "__main__":
    main()