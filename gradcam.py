"""
Grad-CAM explainability for the RGB stream: highlights which regions of the
face the model relied on to decide real vs fake. Useful for your report/demo
to show the model isn't a total black box.
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, rgb_tensor, freq_tensor, class_idx=None):
        """
        rgb_tensor, freq_tensor: single-sample batches, shape (1, C, H, W), on the correct device.
        Returns a (H, W) heatmap normalized to [0, 1].
        """
        self.model.eval()
        output = self.model(rgb_tensor, freq_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        gradients = self.gradients[0]        # (C, h, w)
        activations = self.activations[0]    # (C, h, w)

        weights = gradients.mean(dim=(1, 2))  # (C,)
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (rgb_tensor.shape[3], rgb_tensor.shape[2]))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, class_idx


def overlay_heatmap(original_rgb_uint8: np.ndarray, cam: np.ndarray, alpha=0.45):
    """original_rgb_uint8: (H, W, 3) uint8 RGB image. cam: (H, W) float in [0,1]."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * original_rgb_uint8).astype(np.uint8)
    return overlay
