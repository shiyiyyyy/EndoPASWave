import os
root = "../../../../dataset/c3vd"
sequences = [
    "cecum_t4_b","sigmoid_t3_b","trans_t3_b","desc_t4_a"
]

output_txt = "test513.txt"

with open(output_txt, "w") as f:
    for seq in sequences:
        color_dir = os.path.join(root, seq, "color")
        depth_dir = os.path.join(root, seq, "depth")

        if not os.path.isdir(color_dir):
            print(f"⚠ Missing color dir: {color_dir}")
            continue

        color_files = sorted([
            fn for fn in os.listdir(color_dir)
            if fn.endswith("_color.png")
        ])
 #       color_files = color_files[:147]  #1222
        for color_name in color_files:
            frame_id = color_name.replace("_color.png", "")
            depth_name = frame_id + "_depth.tiff"

            color_path = os.path.join(color_dir, color_name)
            depth_path = os.path.join(depth_dir, depth_name)

            if not os.path.exists(depth_path):
                print(f"⚠ Missing depth: {depth_path}")
                continue

            f.write(f"{color_path} {depth_path}\n")

print(f"✅ Saved pairs to {output_txt}")

# "cecum_t1_a", "cecum_t2_b", "cecum_t2_c", "cecum_t3_a", "sigmoid_t1_a", "sigmoid_t2_a", "trans_t1_a", "trans_t1_b", "trans_t2_a", "trans_t2_b", "trans_t4_a",
# "trans_t4_b"

# "cecum_t1_a", "cecum_t1_b", "cecum_t2_a", "cecum_t2_b", "cecum_t2_c", "cecum_t3_a",
# "sigmoid_t1_a", "sigmoid_t2_a", "trans_t1_a", "trans_t1_b", "trans_t2_a", "trans_t2_b", "trans_t2_c", "trans_t3_a", "trans_t3_b"
# "cecum_t1_a", "cecum_t1_b", "cecum_t2_a", "cecum_t2_b", "cecum_t2_c", "cecum_t3_a",
# "sigmoid_t1_a", "sigmoid_t2_a", "trans_t1_a", "trans_t1_b", "trans_t2_a", "trans_t2_b", "trans_t2_c", "trans_t4_a", "trans_t4_b"