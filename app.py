from flask import Flask, render_template, request, send_from_directory
import os
import random
import compressor
from PIL import Image
import constants

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = constants.UPLOAD_FOLDER
app.config['COMPRESSED_FOLDER'] = constants.COMPRESSED_FOLDER
app.config['PLOT_FOLDER'] = constants.PLOT_FOLDER

K = 16
MAX_ITERS = 10

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '' or not file.filename.endswith('.png'):
            return 'No selected file or not a PNG'
        
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        with Image.open(filename) as img:
            width, height = img.size
            if width * height > constants.MAX_PIXELS_IMAGE:
                remove = True
            else:
                remove = False
        
        if remove:
            os.remove(filename) 
            return 'Image is too large. Please upload an image smaller than 1Mpx.'
        
        if not os.path.exists(constants.COMPRESSED_FOLDER):
            os.makedirs(constants.COMPRESSED_FOLDER)
        
        # Call the image compressor
        original_img, X_img, centroids, idx, compressed_filename, X_recovered = compressor.compress_image(filename, K, MAX_ITERS)
        os.remove(filename)
        
        # Generate plots and get their paths
        plot_paths = compressor.generate_plots(X_img, centroids, idx, K, original_img, compressed_filename, X_recovered)
        
        # Extract filenames from paths for web serving
        plot_filenames = [os.path.basename(path) for path in plot_paths]
        
        return render_template('show_plots.html', plot_filenames=plot_filenames, compressed_img_filename=compressed_filename)
    return render_template('upload.html')

@app.route('/load_sample', methods=['GET'])
def load_sample():
    if not os.path.exists(constants.SAMPLE_IMAGE_PATH) or not os.listdir(constants.SAMPLE_IMAGE_PATH):
        return 'Sample image not found', 404

    # Obtain file list in directory
    files = [f for f in os.listdir(constants.SAMPLE_IMAGE_PATH) if os.path.isfile(os.path.join(constants.SAMPLE_IMAGE_PATH, f))]
    if not files:
        return 'No PNG images found in examples folder', 404

    # Get random example
    sample_image_name = random.choice(files)
    filename = os.path.join(constants.SAMPLE_IMAGE_PATH, sample_image_name)
    
    original_img, X_img, centroids, idx, compressed_filename, X_recovered = compressor.compress_image(filename, K, MAX_ITERS)
    plot_paths = compressor.generate_plots(X_img, centroids, idx, K, original_img, compressed_filename, X_recovered)
    plot_filenames = [os.path.basename(path) for path in plot_paths]
    
    return render_template('show_plots.html', plot_filenames=plot_filenames, compressed_img_filename=compressed_filename)

@app.route('/data/<filename>')
def static_file(filename):
    return send_from_directory('data', filename)

@app.route('/plots/<compression_folder>/<filename>')
def serve_plot(compression_folder, filename):
    return send_from_directory(os.path.join(app.config['PLOT_FOLDER'], compression_folder), filename)

@app.route('/plots/compressed_image/<filename>')
def serve_compressed_image(filename):
    image_path = os.path.join(app.config['COMPRESSED_FOLDER'], filename)
    return send_from_directory(app.config['COMPRESSED_FOLDER'], filename, as_attachment=True, mimetype='image/png')

if __name__ == '__main__':
    for folder in [constants.UPLOAD_FOLDER, constants.PLOT_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    app.run(debug=True)
