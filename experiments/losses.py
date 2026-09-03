import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    [TCAD Requirement] BraTS 3D Segmentation Standard Loss.
    """

    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = (input_tensor == i).float()
            tensor_list.append(temp_prob.unsqueeze(1))
        return torch.cat(tensor_list, dim=1)

    def forward(self, inputs, target, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        # Target: (B, D, H, W) -> One-hot: (B, C, D, H, W)
        target_onehot = self._one_hot_encoder(target)

        assert inputs.size() == target_onehot.size(), \
            f"Shape Mismatch: {inputs.size()} vs {target_onehot.size()}"

        smooth = 1e-5
        input_flat = inputs.view(inputs.size(0), inputs.size(1), -1)
        target_flat = target_onehot.view(target_onehot.size(0), target_onehot.size(1), -1)

        intersection = torch.sum(input_flat * target_flat, dim=2)
        union = torch.sum(input_flat, dim=2) + torch.sum(target_flat, dim=2)

        dice_score = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice_score.mean()