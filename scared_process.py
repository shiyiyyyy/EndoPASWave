import os

root = "../../../../dataset/scared/dpt/Test_scared_toolkit_gt"

train_sets = ["dataset_8", "dataset_9"]

output_txt = "scared_test.txt"

with open(output_txt, "w") as f:

    for ds in train_sets:

        ds_path = os.path.join(root, ds)

        if not os.path.isdir(ds_path):
            print(f"⚠ Missing dataset: {ds_path}")
            continue

        keyframes = sorted([
            kf for kf in os.listdir(ds_path)
            if kf.startswith("keyframe_")
        ])

        for kf in keyframes:

            rgb_dir = os.path.join(ds_path, kf, "data", "left_undistorted")
            depth_dir = os.path.join(ds_path, kf, "data", "depthmap_undistorted")

            if not os.path.isdir(rgb_dir):
                continue

            rgb_files = sorted([
                fn for fn in os.listdir(rgb_dir)
                if fn.endswith(".png")
            ])

            for rgb_name in rgb_files:

                rgb_path = os.path.join(rgb_dir, rgb_name)
                depth_path = os.path.join(depth_dir, rgb_name)

                if os.path.exists(depth_path):
                    f.write(f"{rgb_path} {depth_path}\n")

print(f"✅ Train pairs saved to {output_txt}")
