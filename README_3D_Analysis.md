# 3D Ventricle Analysis Feature

## Overview
This update adds a new "Ventricle Analysis" page to the AVA-TAR web application that provides 3D reconstruction and visualization of segmented brain ventricles.

## New Features

### 1. Ventricle Analysis Button
- Added a "View Ventricle Analysis" button on the results page
- Button appears after segmentation is complete
- Styled to match the existing design theme

### 2. 3D Reconstruction Pipeline
- Uses marching cubes algorithm to extract surface mesh from binary segmentation
- Applies Laplacian smoothing for better visual quality
- Preserves original voxel spacing and orientation
- Handles various NIfTI file formats

### 3. Interactive 3D Viewer
- Built with Three.js for smooth 3D rendering
- Interactive controls: rotate, zoom, pan
- Wireframe toggle for detailed inspection
- Screenshot functionality
- Real-time mesh statistics display

## Technical Implementation

### New Routes
- `/ventricle_analysis/<filename>` - Main analysis page
- `/generate_3d_reconstruction/<filename>` - API endpoint for mesh generation
- `/ventricle_3d_viewer` - 3D viewer page
- `/test_3d_viewer` - Test route for viewer

### New Templates
- `templates/ventricle_analysis.html` - Main analysis page
- `templates/3d_viewer.html` - 3D visualization component

### Dependencies Added
- `scikit-image` - For marching cubes algorithm
- `pyvista` - For mesh processing and smoothing
- `matplotlib` - For additional visualization support

## Usage

1. Upload and process a NIfTI brain MRI file
2. On the results page, click "View Ventricle Analysis"
3. Wait for 3D reconstruction to generate
4. Interact with the 3D viewer:
   - Mouse drag to rotate
   - Scroll to zoom
   - Right-click and drag to pan
   - Use control buttons for additional features

## File Structure
```
AVA-TAR/
├── app.py                          # Updated with new routes
├── templates/
│   ├── result.html                 # Updated with analysis button
│   ├── ventricle_analysis.html     # New analysis page
│   └── 3d_viewer.html             # New 3D viewer
├── requirements.txt                # Updated with new dependencies
└── README_3D_Analysis.md          # This documentation
```

## Installation
Install the new dependencies:
```bash
pip install -r requirements.txt
```

## Notes
- The 3D reconstruction process may take several seconds depending on the size of the segmentation
- Mesh data is passed via URL parameters (base64 encoded) for simplicity
- The viewer includes fallback to a placeholder sphere if mesh data is unavailable
- All mesh processing is done server-side for better performance
