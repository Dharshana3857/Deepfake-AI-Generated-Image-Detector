"""
Downloads a REAL deepfake-detection dataset via the Kaggle API and arranges
it into data/raw/real and data/raw/fake.

This requires your own free Kaggle account + API token (kaggle.json), since
Kaggle datasets are gated behind Kaggle's own terms of use and can't be
redistributed inside this project.

SETUP (one time):
  1. Create a free account at https://www.kaggle.com
  2. Go to Account -> Settings -> API -> "Create New Token"
     This downloads a file called kaggle.json
  3. Place it at:
        Linux/Mac: ~/.kaggle/kaggle.json
        Windows:   C:\\Users\\<you>\\.kaggle\\kaggle.json
  4. pip install kaggle   (already in requirements.txt)

USAGE:
    python download_dataset.py --dataset 140k        # ~ 500MB, 140k real/fake faces (fast to train)
    python download_dataset.py --dataset ffpp_info    # prints instructions (FaceForensics++ requires signing a form)
"""
import os
import argparse
import shutil
import subprocess
import sys

DATASETS = {
    "140k": {
        "kaggle_id": "xhlulu/140k-real-and-fake-faces",
        "description": "140,000 real (Flickr-Faces-HQ) and fake (StyleGAN-generated) face images. "
                        "Good balanced dataset, ~500MB, easiest to get running quickly.",
    },
    "deepfake_detection_challenge_sample": {
        "kaggle_id": "unitednations/deepfake-detection-challenge",
        "description": "A sample slice of Meta/AWS's DFDC (Deepfake Detection Challenge). "
                        "Larger and harder (video-based); check the dataset page for size.",
    },
}


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()) + ["ffpp_info"], default="140k")
    parser.add_argument("--out_dir", default=os.path.join("data", "raw"))
    args = parser.parse_args()

    if args.dataset == "ffpp_info":
        print("FaceForensics++ requires filling out an access-request form with the authors")
        print("(academic dataset, not distributed via Kaggle/pip). See:")
        print("  https://github.com/ondyari/FaceForensics")
        return

    info = DATASETS[args.dataset]
    print(f"Dataset: {info['kaggle_id']}")
    print(info["description"])

    try:
        import kaggle  # noqa: F401
    except Exception:
        print("\nThe 'kaggle' package isn't configured yet. Run:")
        print("  pip install kaggle")
        print("and place your kaggle.json as described in this file's docstring, then re-run.")
        sys.exit(1)

    download_dir = "kaggle_download_tmp"
    os.makedirs(download_dir, exist_ok=True)
    run([sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", info["kaggle_id"], "-p", download_dir, "--unzip"])

    print("\nDownload complete.")
    print(f"Raw files are in '{download_dir}'.")
    print("NOTE: folder layout varies per dataset -- open it and move/rename subfolders so you end up with:")
    print(f"  {args.out_dir}/real/*.jpg")
    print(f"  {args.out_dir}/fake/*.jpg")
    print("For the '140k' dataset specifically, the extracted folders are usually named")
    print("'real' and 'fake' already inside a 'real_vs_fake' directory -- just copy them across, e.g.:")
    print(f"  cp -r {download_dir}/real_vs_fake/real-vs-fake/train/real/* {args.out_dir}/real/")
    print(f"  cp -r {download_dir}/real_vs_fake/real-vs-fake/train/fake/* {args.out_dir}/fake/")


if __name__ == "__main__":
    main()
