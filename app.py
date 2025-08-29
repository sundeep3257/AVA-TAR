import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash, send_file, jsonify
from werkzeug.utils import secure_filename
from model_processing import load_model, run_segmentation_pipeline
import threading
import uuid
import time
from flask import jsonify
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import io
from skimage import measure
import pyvista as pv
import tempfile
import base64

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'nii', 'nii.gz'}
MODEL_WEIGHTS_PATH = os.path.join('model', 'EFB1_e50_best.pth') # Path to your model

# --- APP SETUP ---
app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload size
app.secret_key = 'super-secret-key' # Change this for production

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# --- LOAD THE MODEL (once, at startup) ---
# This is critical for performance and stability!
try:
    load_model(MODEL_WEIGHTS_PATH)
except RuntimeError as e:
    # If the model fails to load, print the error and stop the app.
    print(str(e))
    exit() # This stops the script from continuing.

# --- PROGRESS TRACKING ---
progress_dict = {}

# --- BACKGROUND SEGMENTATION TASK ---
def background_segmentation(task_id, input_filepath, output_filepath):
    try:
        progress_dict[task_id] = 10  # Start at 10%
        time.sleep(0.5)  # Simulate initial delay
        # Step 1: Resizing
        progress_dict[task_id] = 30
        # Actually run the pipeline, but update progress at key steps
        success, message = run_segmentation_pipeline(input_filepath, output_filepath, progress_callback=lambda p: progress_dict.update({task_id: p}))
        if success:
            progress_dict[task_id] = 100
        else:
            progress_dict[task_id] = -1  # Error
    except Exception as e:
        progress_dict[task_id] = -1

@app.route('/progress/<task_id>')
def progress(task_id):
    prog = progress_dict.get(task_id, 0)
    return jsonify({'progress': prog})


def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS)

@app.route('/', methods=['GET', 'POST'])
def upload_and_process():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part in the request.')
            return redirect(request.url)
        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            flash('No file selected.')
            return redirect(request.url)

        # If the file is valid, process it
        if file and allowed_file(file.filename):
            # Secure the filename to prevent directory traversal attacks
            filename = secure_filename(file.filename)
            input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_filename = f"mask_{filename}"
            output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

            # Save the uploaded file
            file.save(input_filepath)

            # Generate a unique task_id
            task_id = str(uuid.uuid4())
            # Start background thread
            thread = threading.Thread(target=background_segmentation, args=(task_id, input_filepath, output_filepath))
            thread.start()
            # Show progress page
            return render_template('progress.html', task_id=task_id, output_filename=output_filename)

        else:
            flash(f'Invalid file type. Allowed types are: {", ".join(ALLOWED_EXTENSIONS)}')
            return redirect(request.url)

    # For a GET request, just show the upload page
    return render_template('index.html')

@app.route('/result/<filename>')
def show_result(filename):
    """Displays the page with the download link."""
    return render_template('result.html', filename=filename)


@app.route('/download/<filename>')
def download_file(filename):
    """Serves the generated segmentation file for download."""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

@app.route('/preview_info/<filename>')
def preview_info(filename):
    # Returns the number of slices in the NIfTI file
    nii_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    nii = nib.load(nii_path)
    data = nii.get_fdata()
    num_slices = data.shape[2]
    return jsonify({'num_slices': num_slices})

@app.route('/preview_slice/<filename>/<int:slice_idx>')
def preview_slice(filename, slice_idx):
    # Returns a PNG image of the specified slice
    nii_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    nii = nib.load(nii_path)
    data = nii.get_fdata()
    if slice_idx < 0 or slice_idx >= data.shape[2]:
        return '', 404
    slice_img = data[:, :, slice_idx]
    # Fix orientation: rotate 90 degrees to the right and flip horizontally
    slice_img = np.rot90(slice_img, k=-1)
    slice_img = np.fliplr(slice_img)
    # Normalize to 0-1 for display
    if np.max(slice_img) > 0:
        slice_img = slice_img / np.max(slice_img)
    # Render as PNG
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=100)
    ax.imshow(slice_img, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/ventricle_analysis/<filename>')
def ventricle_analysis(filename):
    """Displays the ventricle analysis page with 3D reconstruction."""
    return render_template('ventricle_analysis.html', filename=filename)

@app.route('/generate_3d_reconstruction/<filename>')
def generate_3d_reconstruction(filename):
    """Generates a 3D reconstruction of the segmented ventricles."""
    try:
        # Get parameters from request
        slice_thickness = float(request.args.get('slice_thickness', 4.0))
        smoothing_iterations = int(request.args.get('smoothing_iterations', 50))
        smoothing_factor = float(request.args.get('smoothing_factor', 0.1))
        mesh_color = request.args.get('mesh_color', '#0085be')
        
        # Validate parameters
        if slice_thickness < 0.1 or slice_thickness > 10.0:
            return jsonify({'success': False, 'error': 'Invalid slice thickness parameter'})
        if smoothing_iterations < 0 or smoothing_iterations > 200:
            return jsonify({'success': False, 'error': 'Invalid smoothing iterations parameter'})
        if smoothing_factor < 0.001 or smoothing_factor > 1.0:
            return jsonify({'success': False, 'error': 'Invalid smoothing factor parameter'})
        # Validate color format (simple hex validation)
        if not mesh_color.startswith('#') or len(mesh_color) != 7:
            return jsonify({'success': False, 'error': 'Invalid color format'})
        
        # Path to the segmentation NIfTI file
        nii_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        if not os.path.exists(nii_path):
            return jsonify({'success': False, 'error': 'Segmentation file not found'})
        
        # Load the NIfTI file
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        # Make sure data is binary (ventricle = 1, background = 0)
        binary_data = (data > 0).astype(np.uint8)
        
        # Get voxel spacing from NIfTI header
        original_voxel_spacing = list(img.header.get_zooms())
        voxel_spacing = list(original_voxel_spacing)
        voxel_spacing[2] *= slice_thickness  # Apply user-specified slice thickness multiplier
        
        # Calculate volume using original voxel spacing (for accurate volume measurement)
        voxel_volume_mm3 = np.prod(original_voxel_spacing)
        ventricle_volume_mm3 = np.sum(binary_data) * voxel_volume_mm3
        
        # Calculate additional morphological metrics
        from scipy import ndimage
        from scipy.spatial import ConvexHull
        from scipy.spatial.distance import cdist
        
        # Get ventricle coordinates
        ventricle_coords = np.where(binary_data > 0)
        ventricle_coords = np.column_stack(ventricle_coords)
        
        # Calculate convexity (volume ratio of convex hull to actual volume)
        convexity = 1.0
        if len(ventricle_coords) > 4:  # Need at least 4 points for 3D convex hull
            try:
                # Scale coordinates by voxel spacing for accurate measurements
                scaled_coords = ventricle_coords * np.array(original_voxel_spacing)
                hull = ConvexHull(scaled_coords)
                convex_hull_volume = hull.volume
                convexity = ventricle_volume_mm3 / convex_hull_volume if convex_hull_volume > 0 else 1.0
            except Exception as e:
                print(f"Warning: Convexity calculation failed: {str(e)}")
                convexity = 1.0
        
        # Calculate symmetry (using center of mass and moment of inertia)
        symmetry_score = 1.0
        if len(ventricle_coords) > 0:
            try:
                # Calculate center of mass
                center_of_mass = np.mean(ventricle_coords, axis=0)
                
                # Calculate moment of inertia tensor
                centered_coords = ventricle_coords - center_of_mass
                inertia_tensor = np.zeros((3, 3))
                
                for coord in centered_coords:
                    x, y, z = coord
                    inertia_tensor[0, 0] += y*y + z*z
                    inertia_tensor[1, 1] += x*x + z*z
                    inertia_tensor[2, 2] += x*x + y*y
                    inertia_tensor[0, 1] -= x*y
                    inertia_tensor[0, 2] -= x*z
                    inertia_tensor[1, 2] -= y*z
                
                # Make tensor symmetric
                inertia_tensor[1, 0] = inertia_tensor[0, 1]
                inertia_tensor[2, 0] = inertia_tensor[0, 2]
                inertia_tensor[2, 1] = inertia_tensor[1, 2]
                
                # Calculate eigenvalues (principal moments of inertia)
                eigenvalues = np.linalg.eigvals(inertia_tensor)
                eigenvalues = np.real(eigenvalues)  # Remove imaginary parts
                eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
                
                # Calculate symmetry score based on eigenvalue ratios
                if eigenvalues[0] > 0:
                    # Normalize by the largest eigenvalue
                    normalized_eigenvalues = eigenvalues / eigenvalues[0]
                    # Symmetry score: how close the eigenvalues are to each other
                    # Perfect symmetry would have all eigenvalues equal
                    symmetry_score = 1.0 - np.std(normalized_eigenvalues)
                    symmetry_score = max(0.0, min(1.0, symmetry_score))  # Clamp to [0, 1]
                else:
                    symmetry_score = 1.0
                    
            except Exception as e:
                print(f"Warning: Symmetry calculation failed: {str(e)}")
                symmetry_score = 1.0
        
        # Check if there are any non-zero voxels
        if np.sum(binary_data) == 0:
            return jsonify({'success': False, 'error': 'No ventricle voxels found in segmentation'})
        
        # Apply marching cubes to extract the surface mesh
        try:
            verts, faces, normals, values = measure.marching_cubes(binary_data, level=0.5, spacing=voxel_spacing)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Marching cubes failed: {str(e)}'})
        
        # Check if mesh was generated
        if len(verts) == 0 or len(faces) == 0:
            return jsonify({'success': False, 'error': 'No mesh generated from segmentation'})
        
        # Calculate surface area from marching cubes mesh
        surface_area_mm2 = 0
        if len(verts) > 0 and len(faces) > 0:
            # Calculate area of each triangle face
            for face in faces:
                v1, v2, v3 = verts[face]
                # Calculate triangle area using cross product
                edge1 = v2 - v1
                edge2 = v3 - v1
                cross_product = np.cross(edge1, edge2)
                triangle_area = 0.5 * np.linalg.norm(cross_product)
                surface_area_mm2 += triangle_area
        
        # Convert faces to the format PyVista expects
        # PyVista expects faces as a 1D array where each face starts with the number of vertices
        faces_pv = np.column_stack((np.full(faces.shape[0], 3), faces)).astype(np.int32)
        faces_pv = faces_pv.flatten()  # Flatten to 1D array
        
        # Create a PyVista mesh
        try:
            mesh = pv.PolyData(verts, faces_pv)
        except Exception as e:
            return jsonify({'success': False, 'error': f'PyVista mesh creation failed: {str(e)}'})
        
        # Apply smoothing (Laplacian smoothing)
        try:
            if smoothing_iterations > 0:
                smoothed_mesh = mesh.smooth(n_iter=smoothing_iterations, relaxation_factor=smoothing_factor, feature_smoothing=False)
            else:
                smoothed_mesh = mesh  # No smoothing
        except Exception as e:
            print(f"Warning: Smoothing failed, using original mesh: {str(e)}")
            smoothed_mesh = mesh
        
        # Debug info
        print(f"Mesh created successfully with parameters:")
        print(f"  - Slice thickness multiplier: {slice_thickness}")
        print(f"  - Smoothing iterations: {smoothing_iterations}")
        print(f"  - Smoothing factor: {smoothing_factor}")
        print(f"  - Vertices: {len(verts)}")
        print(f"  - Original faces: {len(faces)}")
        print(f"  - Smoothed vertices: {len(smoothed_mesh.points)}")
        print(f"  - Smoothed faces array length: {len(smoothed_mesh.faces)}")
        
        # Get the smoothed vertices and faces
        smoothed_verts = smoothed_mesh.points
        # Extract faces from PyVista mesh - faces are stored as [n_verts, v1, v2, v3, ...]
        faces_array = smoothed_mesh.faces
        # Reshape to get individual faces (each face starts with number of vertices)
        faces_list = []
        i = 0
        while i < len(faces_array):
            n_verts = faces_array[i]
            if i + 1 + n_verts > len(faces_array):
                print(f"Warning: Invalid face data at index {i}")
                break
            face_vertices = faces_array[i+1:i+1+n_verts]
            faces_list.append(face_vertices.tolist())
            i += 1 + n_verts
        
        print(f"  - Processed faces: {len(faces_list)}")
        if len(faces_list) == 0:
            return jsonify({'success': False, 'error': 'No valid faces found in mesh'})
        
        # Save mesh data to a temporary file
        import json
        
        # Convert NumPy arrays to regular Python types for JSON serialization
        vertices_list = []
        for vertex in smoothed_verts:
            vertices_list.append([float(v) for v in vertex])
        
        mesh_data = {
            'vertices': vertices_list,
            'faces': faces_list,
            'color': mesh_color
        }
        
        # Create a unique filename for this mesh data
        mesh_filename = f"mesh_{filename.replace('.nii', '').replace('.gz', '')}_{uuid.uuid4().hex[:8]}.json"
        mesh_filepath = os.path.join(app.config['OUTPUT_FOLDER'], mesh_filename)
        
        # Save mesh data to file
        try:
            with open(mesh_filepath, 'w') as f:
                json.dump(mesh_data, f, default=lambda x: float(x) if hasattr(x, 'dtype') else x)
        except Exception as e:
            print(f"Error saving mesh data: {str(e)}")
            return jsonify({'success': False, 'error': f'Failed to save mesh data: {str(e)}'})
        
        # Create viewer URL with mesh filename
        viewer_url = url_for('ventricle_3d_viewer', mesh_file=mesh_filename)
        
        return jsonify({
            'success': True,
            'viewer_url': viewer_url,
            'mesh_info': {
                'vertices': len(smoothed_verts),
                'faces': len(faces_list),
                'spacing': [float(v) for v in voxel_spacing],
                'parameters': {
                    'slice_thickness': slice_thickness,
                    'smoothing_iterations': smoothing_iterations,
                    'smoothing_factor': smoothing_factor,
                    'mesh_color': mesh_color
                }
            },
            'volume_info': {
                'volume_mm3': float(ventricle_volume_mm3),
                'volume_ml': float(ventricle_volume_mm3 / 1000.0),  # Convert to mL
                'voxel_count': int(np.sum(binary_data)),
                'voxel_spacing': [float(v) for v in original_voxel_spacing]
            },
            'morphology_info': {
                'surface_area_mm2': float(surface_area_mm2),
                'convexity': float(convexity),
                'symmetry_score': float(symmetry_score)
            }
        })
        
    except Exception as e:
        print(f"Error in generate_3d_reconstruction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/ventricle_3d_viewer')
def ventricle_3d_viewer():
    """Serves the 3D viewer page."""
    return render_template('3d_viewer.html')

@app.route('/get_mesh_data/<mesh_filename>')
def get_mesh_data(mesh_filename):
    """Serves the mesh data JSON file."""
    try:
        mesh_filepath = os.path.join(app.config['OUTPUT_FOLDER'], mesh_filename)
        if os.path.exists(mesh_filepath):
            return send_from_directory(app.config['OUTPUT_FOLDER'], mesh_filename, mimetype='application/json')
        else:
            return jsonify({'error': 'Mesh file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test_3d_viewer')
def test_3d_viewer():
    """Test route for the 3D viewer without mesh data."""
    return render_template('3d_viewer.html')

@app.route('/test_3d_reconstruction/<filename>')
def test_3d_reconstruction(filename):
    """Test route to check if the basic 3D reconstruction works."""
    try:
        # Path to the segmentation NIfTI file
        nii_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        if not os.path.exists(nii_path):
            return jsonify({'success': False, 'error': 'Segmentation file not found'})
        
        # Load the NIfTI file
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        # Basic info about the data
        info = {
            'shape': data.shape,
            'min_value': float(np.min(data)),
            'max_value': float(np.max(data)),
            'mean_value': float(np.mean(data)),
            'non_zero_voxels': int(np.sum(data > 0))
        }
        
        return jsonify({
            'success': True,
            'info': info,
            'message': 'Basic file loading successful'
        })
        
    except Exception as e:
        print(f"Error in test_3d_reconstruction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_volume_info/<filename>')
def get_volume_info(filename):
    """Get volume information for a segmentation file."""
    try:
        # Path to the segmentation NIfTI file
        nii_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        if not os.path.exists(nii_path):
            return jsonify({'success': False, 'error': 'Segmentation file not found'})
        
        # Load the NIfTI file
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        # Make sure data is binary (ventricle = 1, background = 0)
        binary_data = (data > 0).astype(np.uint8)
        
        # Get voxel spacing from NIfTI header
        voxel_spacing = list(img.header.get_zooms())
        
        # Calculate volume
        voxel_volume_mm3 = np.prod(voxel_spacing)
        ventricle_volume_mm3 = np.sum(binary_data) * voxel_volume_mm3
        
        # Calculate additional morphological metrics
        from scipy import ndimage
        from scipy.spatial import ConvexHull
        from scipy.spatial.distance import cdist
        
        # Get ventricle coordinates
        ventricle_coords = np.where(binary_data > 0)
        ventricle_coords = np.column_stack(ventricle_coords)
        
        # Calculate surface area (approximate using marching cubes)
        surface_area_mm2 = 0
        if len(ventricle_coords) > 0:
            try:
                # Use marching cubes to get surface mesh for area calculation
                verts, faces, normals, values = measure.marching_cubes(binary_data, level=0.5, spacing=voxel_spacing)
                if len(verts) > 0 and len(faces) > 0:
                    # Calculate area of each triangle face
                    for face in faces:
                        v1, v2, v3 = verts[face]
                        # Calculate triangle area using cross product
                        edge1 = v2 - v1
                        edge2 = v3 - v1
                        cross_product = np.cross(edge1, edge2)
                        triangle_area = 0.5 * np.linalg.norm(cross_product)
                        surface_area_mm2 += triangle_area
            except Exception as e:
                print(f"Warning: Surface area calculation failed: {str(e)}")
                surface_area_mm2 = 0
        
        # Calculate convexity (volume ratio of convex hull to actual volume)
        convexity = 1.0
        if len(ventricle_coords) > 4:  # Need at least 4 points for 3D convex hull
            try:
                # Scale coordinates by voxel spacing for accurate measurements
                scaled_coords = ventricle_coords * np.array(voxel_spacing)
                hull = ConvexHull(scaled_coords)
                convex_hull_volume = hull.volume
                convexity = ventricle_volume_mm3 / convex_hull_volume if convex_hull_volume > 0 else 1.0
            except Exception as e:
                print(f"Warning: Convexity calculation failed: {str(e)}")
                convexity = 1.0
        
        # Calculate symmetry (using center of mass and moment of inertia)
        symmetry_score = 1.0
        if len(ventricle_coords) > 0:
            try:
                # Calculate center of mass
                center_of_mass = np.mean(ventricle_coords, axis=0)
                
                # Calculate moment of inertia tensor
                centered_coords = ventricle_coords - center_of_mass
                inertia_tensor = np.zeros((3, 3))
                
                for coord in centered_coords:
                    x, y, z = coord
                    inertia_tensor[0, 0] += y*y + z*z
                    inertia_tensor[1, 1] += x*x + z*z
                    inertia_tensor[2, 2] += x*x + y*y
                    inertia_tensor[0, 1] -= x*y
                    inertia_tensor[0, 2] -= x*z
                    inertia_tensor[1, 2] -= y*z
                
                # Make tensor symmetric
                inertia_tensor[1, 0] = inertia_tensor[0, 1]
                inertia_tensor[2, 0] = inertia_tensor[0, 2]
                inertia_tensor[2, 1] = inertia_tensor[1, 2]
                
                # Calculate eigenvalues (principal moments of inertia)
                eigenvalues = np.linalg.eigvals(inertia_tensor)
                eigenvalues = np.real(eigenvalues)  # Remove imaginary parts
                eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
                
                # Calculate symmetry score based on eigenvalue ratios
                if eigenvalues[0] > 0:
                    # Normalize by the largest eigenvalue
                    normalized_eigenvalues = eigenvalues / eigenvalues[0]
                    # Symmetry score: how close the eigenvalues are to each other
                    # Perfect symmetry would have all eigenvalues equal
                    symmetry_score = 1.0 - np.std(normalized_eigenvalues)
                    symmetry_score = max(0.0, min(1.0, symmetry_score))  # Clamp to [0, 1]
                else:
                    symmetry_score = 1.0
                    
            except Exception as e:
                print(f"Warning: Symmetry calculation failed: {str(e)}")
                symmetry_score = 1.0
        
        return jsonify({
            'success': True,
            'volume_info': {
                'volume_mm3': float(ventricle_volume_mm3),
                'volume_ml': float(ventricle_volume_mm3 / 1000.0),  # Convert to mL
                'voxel_count': int(np.sum(binary_data)),
                'voxel_spacing': [float(v) for v in voxel_spacing],
                'data_shape': data.shape
            },
            'morphology_info': {
                'surface_area_mm2': float(surface_area_mm2),
                'convexity': float(convexity),
                'symmetry_score': float(symmetry_score)
            }
        })
        
    except Exception as e:
        print(f"Error in get_volume_info: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':

    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
