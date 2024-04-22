from flask import Flask, render_template, request, send_from_directory
import os
import random
import compressor
from PIL import Image
import constants

app = Flask(__name__)

# Configuration for file upload and storage directories
app.config['UPLOAD_FOLDER'] = constants.UPLOAD_FOLDER
app.config['COMPRESSED_FOLDER'] = constants.COMPRESSED_FOLDER
app.config['PLOT_FOLDER'] = constants.PLOT_FOLDER

# Predefined number of clusters and maximum iterations for K-Means
K = 16
MAX_ITERS = 10

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    Handle file uploads and initiate image compression and plotting.

    This route accepts file uploads via a POST request. It saves the uploaded file, validates it, and processes
    it using the image compressor. Generated plots and compressed images are then made accessible to the user
    through rendering an HTML template.

    Returns:
        A rendered HTML template either for uploading a file ('upload.html') or showing generated plots
        ('show_plots.html') with the paths to the compressed image and plots.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '' or not file.filename.endswith('.png'):
            return 'No selected file or not a PNG'
        
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Check image size constraints
        # with Image.open(filename) as img:
        #     width, height = img.size
        #     if width * height > constants.MAX_PIXELS_IMAGE:
        #         remove = True
        #     else:
        #         remove = False
        
        # Remove oversized images and return error message
        # if remove:
        #     os.remove(filename) 
        #     return 'Image is too large. Please upload an image smaller than 1Mpx.'
        
        if not os.path.exists(constants.COMPRESSED_FOLDER):
            os.makedirs(constants.COMPRESSED_FOLDER)
        
        original_img, X_img, centroids, idx, compressed_filename, X_recovered = compressor.compress_image(filename, constants.K, constants.MAX_ITERS)
        os.remove(filename)
        
        plot_paths = compressor.generate_plots(X_img, centroids, idx, constants.K, original_img, compressed_filename, X_recovered)
        plot_filenames = [os.path.basename(path) for path in plot_paths]
        
        return render_template('show_plots.html', plot_filenames=plot_filenames, compressed_img_filename=compressed_filename)
    return render_template('upload.html')

@app.route('/load_sample', methods=['GET'])
def load_sample():
    """
    Handles the loading and display of a random sample image from a predefined directory. The image is processed
    in the same way as uploaded images.

    Returns:
        Response: Renders a template to display the processed sample image and its generated plots.
    """
    if not os.path.exists(constants.SAMPLE_IMAGE_PATH) or not os.listdir(constants.SAMPLE_IMAGE_PATH):
        return 'Sample image not found', 404

    files = [f for f in os.listdir(constants.SAMPLE_IMAGE_PATH) if os.path.isfile(os.path.join(constants.SAMPLE_IMAGE_PATH, f))]
    if not files:
        return 'No PNG images found in examples folder', 404

    # Randomly select a sample image
    sample_image_name = random.choice(files)
    filename = os.path.join(constants.SAMPLE_IMAGE_PATH, sample_image_name)
    
    original_img, X_img, centroids, idx, compressed_filename, X_recovered = compressor.compress_image(filename, constants.K, constants.MAX_ITERS)
    plot_paths = compressor.generate_plots(X_img, centroids, idx, constants.K, original_img, compressed_filename, X_recovered)
    plot_filenames = [os.path.basename(path) for path in plot_paths]
    
    return render_template('show_plots.html', plot_filenames=plot_filenames, compressed_img_filename=compressed_filename)

@app.route('/data/<filename>')
def static_file(filename):
    """
    Serves static files from the 'data' directory.

    Parameters:
        filename (str): The name of the file to be served.

    Returns:
        Response: Sends the requested file from the 'data' directory.
    """
    return send_from_directory('data', filename)

@app.route('/plots/<compression_folder>/<filename>')
def serve_plot(compression_folder, filename):
    """
    Serves generated plot images from their respective subdirectories within the plots directory.

    Parameters:
        compression_folder (str): Subdirectory name where the requested plot is stored.
        filename (str): The name of the file to be served.

    Returns:
        Response: Sends the requested plot image.
    """
    return send_from_directory(os.path.join(app.config['PLOT_FOLDER'], compression_folder), filename)

@app.route('/plots/compressed_image/<filename>')
def serve_compressed_image(filename):
    """
    Serves compressed image files, allowing them to be downloaded by the user.

    Parameters:
        filename (str): The name of the compressed image file to be served.

    Returns:
        Response: Sends the requested compressed image file as an attachment.
    """
    image_path = os.path.join(app.config['COMPRESSED_FOLDER'], filename)
    return send_from_directory(app.config['COMPRESSED_FOLDER'], filename, as_attachment=True, mimetype='image/png')

if __name__ == '__main__':
    # Ensure necessary folders exist before running the application
    for folder in [constants.UPLOAD_FOLDER, constants.PLOT_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    app.run(debug=True)
