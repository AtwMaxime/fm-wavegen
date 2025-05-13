import torch
import os
from pytorch_fid import fid_score


def calculate_fid_between_folders(path_real, path_fake, output_txt="outputs/fid_results_torchfid.txt"):
    """
    Computes FID between two folders using pytorch-fid.

    Args:
        path_real (str): Path to the folder containing real images.
        path_fake (str): Path to the folder containing generated images.
        output_txt (str): Path to save FID result.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 50
    dims = 2048

    print(f"📊 Calculating FID between:\n → {path_fake}\n → {path_real}")
    fid_value = fid_score.calculate_fid_given_paths(
        [path_fake, path_real],
        batch_size=batch_size,
        device=device,
        dims=dims
    )

    result_str = f"FID ({os.path.basename(path_fake)} vs {os.path.basename(path_real)}): {fid_value:.4f}\n"
    print("✅", result_str)

    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "a") as f:
        f.write(result_str)

    print(f"💾 Saved result to: {output_txt}")

from PIL import Image
import os

def remove_corrupt_images(folder):
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            with Image.open(path) as img:
                img.verify()  # Just checks header
        except Exception:
            print(f"❌ Removing broken image: {filename}")
            os.remove(path)



if __name__ == "__main__":
    remove_corrupt_images("data/processed/resized")
    remove_corrupt_images("data/processed/reconstruct")
    calculate_fid_between_folders(
        path_real="data/processed/resized",
        path_fake="data/processed/reconstruct"
    )