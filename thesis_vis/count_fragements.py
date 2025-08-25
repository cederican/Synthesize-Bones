import os
import numpy as np

def analyze_obj_counts(base_folder="healthy"):
    results = {}

    # Iterate over type ("simple", "advanced")
    for prefix in ["simple", "advanced"]:
        counts = []

        # Walk through numbered folders inside "healthy"
        for number_folder in os.listdir(base_folder):
            number_path = os.path.join(base_folder, number_folder)
            if not os.path.isdir(number_path):
                continue

            # Look for subfolders starting with prefix
            for subfolder in os.listdir(number_path):
                if subfolder.startswith(prefix):
                    subfolder_path = os.path.join(number_path, subfolder)

                    # Inside this prefix-folder, iterate its subfolders
                    for inner in os.listdir(subfolder_path):
                        inner_path = os.path.join(subfolder_path, inner)
                        if os.path.isdir(inner_path):
                            # Count .obj files
                            obj_files = [
                                f for f in os.listdir(inner_path)
                                if f.endswith(".obj")
                            ]
                            counts.append(len(obj_files))

        if counts:
            results[prefix] = {
                "max": int(np.max(counts)),
                "mean": float(np.mean(counts))
            }
        else:
            results[prefix] = {"max": 0, "mean": 0.0}

    return results


# Example usage
if __name__ == "__main__":
    stats = analyze_obj_counts("/Users/cedimac/fracture-modes/data/healthy")
    print(stats)
