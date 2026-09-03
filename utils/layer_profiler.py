import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict


def profile_model_layers(model: nn.Module,
                         input_size: Tuple[int, ...] = (1, 4, 64, 64, 64),
                         device: str = None) -> pd.DataFrame:
    """
    [TCAD Enhanced] 支持 U-Net/Transformer 的全算子计算特征分析
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = 'cpu'

    # 保持模型原始状态，切换到 eval 避免 BN 统计量更新
    original_mode = model.training
    model.eval()
    model.to(device)

    trace_data: List[Dict] = []
    hooks = []

    def hook_fn(name: str):
        def hook(module, input_t, output_t):
            inp = input_t[0] if isinstance(input_t, tuple) else input_t
            out = output_t[0] if isinstance(output_t, tuple) else output_t

            num_params = sum(p.numel() for p in module.parameters())
            in_bytes = inp.numel() * 4
            out_bytes = out.numel() * 4
            macs = 0

            # --- 1. 卷积/反卷积 (关键修复) ---
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                # MACs ~= Output_Elements * Kernel_Vol * In_Channels / Groups
                out_vol = np.prod(out.shape[2:])
                kernel_vol = np.prod(module.kernel_size)
                # ConvTranspose 的逻辑略有不同，但 MACs 总量近似
                # 修正 in_channels 计算逻辑以适配 Transpose
                c_in = module.in_channels
                if isinstance(module, nn.ConvTranspose3d):
                    # TransposeConv: in_channels 是输出通道数对应方向，这里简化处理
                    # 重要的是捕捉计算规模：Input_Ch * Kernel * Output_Spatial
                    c_in = module.in_channels

                macs = int(out.numel() * kernel_vol * (c_in / module.groups))

            elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                kernel_area = np.prod(module.kernel_size)
                macs = int(out.numel() * kernel_area * (module.in_channels / module.groups))

            # --- 2. 线性层 ---
            elif isinstance(module, nn.Linear):
                macs = int(out.numel() * module.in_features)

            # --- 3. Attention ---
            elif isinstance(module, nn.MultiheadAttention):
                b_sz, seq_len, embed_dim = inp.shape[:3]
                # 粗略估算：4次投影 + 2次矩阵乘法
                macs = int(4 * b_sz * seq_len * (embed_dim ** 2) + 2 * b_sz * (seq_len ** 2) * embed_dim)

            # --- 4. Norm层 (内存密集型) ---
            elif isinstance(module, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm)):
                # 计算量虽小，但内存访问是瓶颈，记录下来供 Simulator 的 Roofline 模型判断
                macs = int(out.numel())

            if macs > 0 or num_params > 0:
                trace_data.append({
                    'name': name,
                    'type': module.__class__.__name__,
                    'macs': macs,
                    'params': num_params,
                    'in_bytes': in_bytes,
                    'out_bytes': out_bytes
                })

        return hook

    # 注册 Hooks
    for name, module in model.named_modules():
        # 仅 Hook 叶子节点
        if len(list(module.children())) == 0:
            if isinstance(module,
                          (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d, nn.Linear, nn.MultiheadAttention, nn.BatchNorm3d)):
                hooks.append(module.register_forward_hook(hook_fn(name)))

    # Dummy Forward
    dummy_input = torch.randn(input_size, device=device)
    try:
        with torch.no_grad():
            model(dummy_input)
    except Exception as e:
        print(f"⚠️ Profiler Warning: {e}")
    finally:
        for h in hooks: h.remove()
        model.train(original_mode)

    return pd.DataFrame(trace_data)


# 兼容接口: 提供 analyze_model_workload 别名，并自动转为 Dict 列表
def analyze_model_workload(model, input_shape):
    df = profile_model_layers(model, input_size=input_shape)
    return df.to_dict('records')