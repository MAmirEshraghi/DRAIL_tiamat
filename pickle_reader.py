import pickle
import open3d as o3d

def main():
    # load buffer from the pickle file
    with open("obs_buffer.pkl", "rb") as f:
        buffer = pickle.load(f)
    
    print(buffer.get_buffer_status())
    print("static transforms: ", buffer.entries.keys())

     # === Reconstruct point clouds from all entries ===
    merged_pcd = o3d.geometry.PointCloud()

    for stamp, entry in buffer.entries.items():
        try:
            if buffer.is_tf_static_ready() and entry.is_frame_full():
                pcds = entry.get_pointcloud(buffer.static_transforms)
                for pc in pcds:
                    merged_pcd += pc
        except Exception as e:
            print(f"Error processing {stamp}: {e}")
            continue

    print(f"Merged point cloud has {len(merged_pcd.points)} points")

    # draw the merged point cloud after downsampling
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.05)
    o3d.visualization.draw_geometries([merged_pcd])

if __name__ == "__main__":
    main()