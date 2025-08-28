import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def smooth_curve(values, window_size=21):
    """
    使用滑动窗口对曲线进行简单平滑。
    window_size: 平滑窗口大小，必须为奇数。
    """
    if window_size < 3:
        return np.array(values)
    # 边界填充
    pad = window_size // 2
    padded = np.pad(values, (pad, pad), mode='edge')
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed

def plot_losses_from_pth(pth_path, save_dir=None, plot_together=False, smooth=True, window_size=21):
    """
    Read loss lists from a .pth file and plot loss curves, with optional smoothing.
    Supported loss names: true_losses, fake_losses, disc_losses, gen_losses, focal_losses, bce_losses, mask_losses, feature_losses

    Args:
        pth_path (str): Path to the .pth file.
        save_dir (str): Optional. Directory to save the images. If None, only display without saving.
        plot_together (bool): Whether to plot all losses in one figure. Default False (plot separately).
        smooth (bool): Whether to plot smoothed curves. Default True.
        window_size (int): Smoothing window size, default 21.
    """

    # Set English font to Times New Roman (workspace requirement)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = True

    # Assign colors and display names based on step_train checkpoints
    color_map = {
        'patch_true_losses': 'tab:blue',
        'patch_fake_losses': 'tab:orange',
        'patch_disc_losses': 'tab:green',
        'global_true_losses': 'tab:cyan',
        'global_fake_losses': 'tab:pink',
        'global_disc_losses': 'tab:olive',
        'gen_losses': 'tab:red',
        'focal_losses': 'tab:purple',
        'bce_patch_losses': 'tab:brown',
        'bce_global_losses': 'tab:gray',
        'mask_losses': 'tab:gray',
        'feature_losses': 'tab:blue'
    }
    loss_keys = [
        'patch_true_losses', 'patch_fake_losses', 'patch_disc_losses',
        'global_true_losses', 'global_fake_losses', 'global_disc_losses',
        'gen_losses', 'focal_losses', 'bce_patch_losses', 'bce_global_losses',
        'mask_losses', 'feature_losses'
    ]
    loss_display_names = {
        'patch_true_losses': 'Patch True loss',
        'patch_fake_losses': 'Patch Fake loss',
        'patch_disc_losses': 'Patch Discriminator loss',
        'global_true_losses': 'Global True loss',
        'global_fake_losses': 'Global Fake loss',
        'global_disc_losses': 'Global Discriminator loss',
        'gen_losses': 'Generator loss',
        'focal_losses': 'Focal loss',
        'bce_patch_losses': 'Patch BCE loss',
        'bce_global_losses': 'Global BCE loss',
        'mask_losses': 'Mask loss',
        'feature_losses': 'Feature loss'
    }
    # Load .pth file
    checkpoint = torch.load(pth_path, map_location='cpu')
    # Check which losses are available
    available_losses = [k for k in loss_keys if k in checkpoint]
    if not available_losses:
        print("No loss lists found in the .pth file.")
        return

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if plot_together:
        plt.figure(figsize=(10, 7))
        for loss_key in available_losses:
            losses = checkpoint[loss_key]
            x = np.arange(len(losses))
            plt.plot(x, losses, label=f"{loss_display_names[loss_key]} (raw)", color=color_map.get(loss_key, None), alpha=0.4)
            if smooth and len(losses) > window_size:
                smooth_losses = smooth_curve(losses, window_size)
                plt.plot(x, smooth_losses, label=f"{loss_display_names[loss_key]} (smoothed)", color=color_map.get(loss_key, None))
        plt.xlabel("Iteration")
        plt.ylabel("Loss value")
        plt.title("All Loss Curves")
        plt.legend()
        plt.grid(True)
        if save_dir is not None:
            save_path = os.path.join(save_dir, "all_losses.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    else:
        for loss_key in available_losses:
            losses = checkpoint[loss_key]
            x = np.arange(len(losses))
            plt.figure()
            plt.plot(x, losses, label=f"{loss_display_names[loss_key]} (raw)", color=color_map.get(loss_key, None), alpha=0.4)
            if smooth and len(losses) > window_size:
                smooth_losses = smooth_curve(losses, window_size)
                plt.plot(x, smooth_losses, label=f"{loss_display_names[loss_key]} (smoothed)", color=color_map.get(loss_key, None))
            plt.xlabel("Iteration")
            plt.ylabel("Loss value")
            plt.title(f"{loss_display_names[loss_key]} Curve")
            plt.legend(loc='upper right')
            plt.grid(True)
            if save_dir is not None:
                save_path = os.path.join(save_dir, f"{loss_key}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved: {save_path}")
            else:
                plt.show()
            plt.close()

if __name__ == "__main__":
    plot_losses_from_pth("checkpoints/stgan_20250813/iter1500.pth", save_dir="visual/loss_visual", plot_together=False)