import os
import warnings

# Locate the LICENSE file
license_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "LICENSE")

if os.path.exists(license_path):
    try:
        with open(license_path, "r") as license_file:
            license_content = license_file.read()
            warnings.warn(
                f"LICENSE content:\n{license_content}",
                category=UserWarning
            )
    except Exception as e:
        pass