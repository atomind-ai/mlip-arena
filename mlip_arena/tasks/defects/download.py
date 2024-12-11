from tqdm.auto import tqdm
import requests

record_id = "10579527"
base_url = f"https://zenodo.org/record/{record_id}/files/"

# bulk_primitive_folders.tar.gz

files = ["bulk_primitive_folders.tar.gz", "defect_relaxations.tar.gz"]


for file in files:
    response = requests.get(base_url + file + "?download=1", stream=True)
    response.raise_for_status()  # Check if the request was successful

    total = int(response.headers.get("content-length", 0))

    # Save the file locally
    with open(file, "wb") as f, tqdm(
        desc=file, total=total, unit="iB", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

    print(f"{file} downloaded successfully.")
