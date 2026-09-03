import torch
import torch.nn as nn
import inspect

# 尝试导入 MONAI
try:
    from monai.networks.nets import SwinUNETR as MonaiSwinUNETR

    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


class SwinUNETR(nn.Module):
    def __init__(self, img_size=64, in_channels=4, out_channels=4, feature_size=24):
        super().__init__()
        if not HAS_MONAI:
            raise ImportError("❌ MONAI 未安装。请运行: pip install monai")

        # [TCAD Critical Fix] 动态检测当前 MONAI 版本的参数签名
        sig = inspect.signature(MonaiSwinUNETR.__init__)

        # 构造全量参数字典
        all_possible_args = {
            'img_size': (img_size, img_size, img_size),  # 旧版参数
            'spatial_size': (img_size, img_size, img_size),  # 某些变体名
            'in_channels': in_channels,
            'out_channels': out_channels,
            'feature_size': feature_size,
            'use_checkpoint': True,  # 开启梯度检查点，节省显存
            'spatial_dims': 3
        }

        # 智能过滤：只传递当前版本支持的参数
        valid_kwargs = {k: v for k, v in all_possible_args.items() if k in sig.parameters}

        try:
            self.model = MonaiSwinUNETR(**valid_kwargs)
        except Exception as e:
            print(f"❌ MONAI Init Failed. Accepted args: {list(sig.parameters.keys())}")
            raise e

    def forward(self, x):
        return self.model(x)