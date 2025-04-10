import numpy as np
import cv2
import open3d as o3d
import os
from pathlib import Path

class DepthVisualizer:
    def __init__(self):
        self.original_depth_dir = "./depth-anything-large-hf"
        self.processed_depth_dir = "./processed_depth"
        self.mask_dir = "./mask"

    def read_depth(self, path):
        """Read depth image and convert disparity to depth."""
        print(f"Reading depth from: {path}")
        disparity = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if disparity is None:
            raise FileNotFoundError(f"Could not read: {path}")
        
        # Convert disparity to depth
        disparity = disparity.astype(np.float32)
        depth = 1.0 / (disparity + 1e-6)
        
        # Normalize depth
        depth_min = np.percentile(depth, 1)
        depth_max = np.percentile(depth, 99)
        depth_norm = np.clip((depth - depth_min) / (depth_max - depth_min), 0, 1)
        
        return depth_norm

    def read_mask(self, path, depth_shape):
        """Read and binarize mask, ensuring it matches depth image size."""
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read: {path}")
        
        if mask.shape != depth_shape:
            mask = cv2.resize(
                mask,
                (depth_shape[1], depth_shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        return mask > 128

    def create_point_cloud(self, depth, mask=None, color=[0, 0, 1]):
        """Create point cloud from depth map."""
        height, width = depth.shape
        
        # Create coordinate grid
        ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Normalize coordinates to [-1, 1]
        xs = (xs - width/2) / max(width, height)
        ys = (ys - height/2) / max(width, height)
        zs = depth * 2.0 - 1.0
        
        # Stack coordinates
        points = np.stack([xs, ys, zs], axis=-1)
        
        # Apply mask if provided
        if mask is not None:
            points = points[mask]
        else:
            points = points.reshape(-1, 3)
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))
        
        # Downsample
        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        
        return pcd

    def create_grid(self, size=2, steps=10):
        """Create a reference grid."""
        grid_points = []
        grid_colors = []
        step_size = size / steps
        
        for i in range(steps + 1):
            x = -size/2 + i * step_size
            grid_points.extend([[x, -size/2, -1], [x, size/2, -1]])
            grid_points.extend([[-size/2, x, -1], [size/2, x, -1]])
            grid_colors.extend([[0.7, 0.7, 0.7]] * 4)
        
        grid = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.array(grid_points)),
            lines=o3d.utility.Vector2iVector(np.array([[i, i+1] for i in range(0, len(grid_points), 2)]))
        )
        grid.colors = o3d.utility.Vector3dVector(np.array(grid_colors))
        
        return grid

    def visualize(self, image_id):
        """Interactive visualization function."""
        print(f"Processing image: {image_id}")
        
        # Read images
        orig_depth = self.read_depth(os.path.join(self.original_depth_dir, f"{image_id}.png"))
        proc_depth = self.read_depth(os.path.join(self.processed_depth_dir, f"{image_id}.png"))
        mask = self.read_mask(os.path.join(self.mask_dir, f"{image_id}.png"), orig_depth.shape)
        
        # Create point clouds
        orig_non_glass = self.create_point_cloud(orig_depth, ~mask, [0, 0, 1])  # Blue
        orig_glass = self.create_point_cloud(orig_depth, mask, [1, 0, 0])      # Red
        
        proc_non_glass = self.create_point_cloud(proc_depth, ~mask, [0, 0, 1])
        proc_glass = self.create_point_cloud(proc_depth, mask, [1, 0, 0])
        
        # Create reference grids
        grid1 = self.create_grid()
        grid2 = self.create_grid()
        
        # Offset processed point clouds and grid
        T = np.eye(4)
        T[0, 3] = 3
        proc_non_glass.transform(T)
        proc_glass.transform(T)
        grid2.transform(T)
        
        # Create coordinate frames
        frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        frame2.transform(T)

        # Create visualizer and set properties
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Depth Comparison - Image {image_id}",
            width=1280,
            height=720
        )
        
        # Add geometries
        geometries = [
            orig_non_glass, orig_glass,
            proc_non_glass, proc_glass,
            grid1, grid2,
            frame1, frame2
        ]
        for geom in geometries:
            vis.add_geometry(geom)

        # Set render options
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.9, 0.9, 0.9])
        opt.point_size = 2.0

        # Set default viewpoint that matches original image perspective
        ctr = vis.get_view_control()
        ctr.set_zoom(0.7)
        ctr.set_front([-0.3, -0.2, -0.5])  # Camera direction
        ctr.set_lookat([1.5, 0, 0])        # Look-at point (center between original and processed)
        ctr.set_up([0.0, -1.0, 0.0])       # Camera up direction

        # Print instructions
        print("\n操作指南：")
        print("- 左键点击并拖动：旋转视角")
        print("- 右键点击并拖动：平移视角")
        print("- 滚动鼠标滚轮：缩放")
        print("- 按Q或Esc：退出")

        # Run visualization
        vis.run()
        vis.destroy_window()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Interactive depth visualization')
    parser.add_argument('image_id', type=str, help='Image ID (e.g., "1")')
    args = parser.parse_args()
    
    visualizer = DepthVisualizer()
    visualizer.visualize(args.image_id)

if __name__ == "__main__":
    main()

