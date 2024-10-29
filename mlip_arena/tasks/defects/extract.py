import tarfile
import glob
from tqdm.auto import tqdm

files = sorted(glob.glob("*.tar.gz"))

for file in files:

    # Check if the file is a tar archive and extract it
    if file.endswith('.tar.gz') or file.endswith('.tar'):
        with tarfile.open(file, 'r:gz' if file.endswith('.gz') else 'r:') as tar_ref:
            members = tar_ref.getmembers()

            # Initialize progress bar
            with tqdm(total=len(members), desc=file, unit="file") as pbar:
                for member in members:
                    tar_ref.extract(member, file.split('.')[0])
                    pbar.update()