# 🐔 Chickenvision-Server

This repository contains the source code for the server or cloud service focusing on utilizing Open3D for 3D data processing and visualization. Open3D is a powerful library that streamlines mesh manipulation by enabling translation and rotation of a 3D model in Cartesian coordinates.

## Installation

Ensure you have Python 3.8.5 or a compatible version installed. Then, run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

Or, you can manually install Open3D with your preferred package manager:

```bash
pip install open3d==0.11.2
```

## Why Open3D?

1. **Ease of integration with Python**: Open3D is an accessible library for developers with varying levels of expertise in computer vision and 3D processing. Its Pythonic syntax and extensive documentation make it easy to start working with and integrate into your project.

2. **Feature-rich**: Open3D offers a rich set of features for 3D data processing and visualization. It includes robust algorithms for point cloud registration, 3D reconstruction, mesh processing, and more, making it much easier than implementing manual matrix transforms.

3. **Interactive visualizer**: The library's interactive visualizer allows developers to preview and tweak 3D models, meshes, and point clouds in real-time, with options to adjust lighting, materials, and textures.

4. **Open-source**: Open3D is open-source, promoting transparency and collaboration.

## Usage

The `Render()` function saves a single view of your chosen mesh:

```python
Render(savePath, objPath, angles, xyz, view_w, view_h)
```

- `savePath`: Path to save an image (output must be a .png file).
- `objPath`: Path to the source .obj file.
- `angles`: Tuple in the form (0, 0, 0), specifying Cartesian coordinate angles in degrees for rotation.
- `xyz`: Cartesian translation, as a list in the form [0, 0, 0].
- `view_w`, `view_h`: Image width and height.

## To-Do & Issues

- **Viewport settings**: Getting the right viewport settings can be challenging.
- **Colors**: Working on improving the colors.
- **Multiple mesh instances**: Loading copies of the mesh into one visualizer is a trivial task that can be implemented easily.

## 📚 License

Please ensure you follow the licensing requirements of the original repositories and give proper credit as mentioned in the previous answers.
