import os
import requests
import zipfile


def get_data_home_dir():
    """Returns the current script directory as the dataset root."""
    return os.path.dirname(os.path.abspath(__file__))


def download_and_unzip(url, extract_to):
    """Downloads a ZIP file from the given URL and extracts it to the specified directory."""
    os.makedirs(extract_to, exist_ok=True)
    zip_path = os.path.join(extract_to, "temp.zip")

    print(f"Downloading {url} to {zip_path}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")

    print(f"Unzipping {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Check if the zip contains a top-level folder (e.g., polyp/polyp/...)
        top_level_names = set(p.split("/")[0] for p in zip_ref.namelist() if "/" in p)
        if len(top_level_names) == 1 and list(top_level_names)[0].lower() == os.path.basename(url).replace(".zip", "").lower():
            # Extract without nesting (e.g., avoid polyp/polyp)
            for member in zip_ref.namelist():
                member_path = os.path.relpath(member, start=list(top_level_names)[0])
                target_path = os.path.join(extract_to, member_path)
                if not member.endswith('/'):
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with open(target_path, 'wb') as outfile, zip_ref.open(member) as infile:
                        outfile.write(infile.read())
        else:
            # Extract as-is
            zip_ref.extractall(extract_to)
    print("Unzip complete.")

    os.remove(zip_path)


if __name__ == "__main__":
    base_dir = get_data_home_dir()
    datasets = [
        "polyp",
        "leaf_disease_segmentation",
        "camo_sem_seg",
        "isic2017",
        "road_segmentation",
        "SBU-shadow"
    ]

    for name in datasets:
        url = f"https://automl-mm-bench.s3.amazonaws.com/semantic_segmentation/{name}.zip"
        dataset_dir = os.path.join(base_dir, name)  # Final path: ./polyp, ./isic2017, etc.
        if not os.path.exists(dataset_dir):
            download_and_unzip(url, dataset_dir)
        else:
            print(f"Dataset '{name}' already exists. Skipping download.")
