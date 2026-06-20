import json
import sys
import os
from pathlib import Path

# Path Fix
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet3d import UNet3D
from utils.layer_profiler import profile_model_layers
from simulator.acf_simulator import ACFSimulator

# 尝试导入 UNETR
try:
    from models.swin_unetr import SwinUNETR
except ImportError:
    SwinUNETR = None


def generate_roofline_data(output_dir='results/microarch'):
    print("⚙️  Generating Roofline Data (Dual-Model Analysis)...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sim = ACFSimulator('hardware_profile.json')
    roofline_combined = {'peak_flops_gflops': 0, 'peak_bandwidth_gbs': 0, 'layers': []}

    # 定义要对比的模型 (输入尺寸统一为 64^3)
    models_to_profile = [('3D U-Net (CNN)', UNet3D(n_channels=4, n_classes=4))]

    if SwinUNETR:
        print("   ✅ Swin-UNETR detected. Adding to analysis.")
        models_to_profile.append(('Swin-UNETR (Transformer)', SwinUNETR(img_size=64, in_channels=4, out_channels=4)))
    else:
        print("   ⚠️ Swin-UNETR not found (install monai to include).")

    for model_name, model in models_to_profile:
        print(f"   -> Profiling {model_name}...")
        try:
            df = profile_model_layers(model, input_size=(1, 4, 64, 64, 64))
            workload = df.to_dict('records')

            # 仿真 FP8_E5M2
            res = sim.roofline_analysis(workload, precision='FP8_E5M2')

            # 初始化峰值数据
            if roofline_combined['peak_flops_gflops'] == 0:
                roofline_combined['peak_flops_gflops'] = res['peak_flops_gflops']
                roofline_combined['peak_bandwidth_gbs'] = res['peak_bandwidth_gbs']

            # 标记模型类型
            for layer in res['layers']:
                layer['model_type'] = model_name  # 用于绘图区分颜色
                roofline_combined['layers'].append(layer)

        except Exception as e:
            print(f"   ⚠️ Failed to profile {model_name}: {e}")

    with open(f'{output_dir}/roofline_data.json', 'w') as f:
        json.dump(roofline_combined, f, indent=2)
    print(f"✅ Data saved. Figure 8 will now demonstrate Generality.")


if __name__ == '__main__':
    generate_roofline_data()