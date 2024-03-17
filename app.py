from flask import Flask, render_template, request, send_from_directory
import os
import compressor # Make sure to implement this module

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static/plots'
SAMPLE_IMAGE_PATH = 'data/Cat.png' 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLOT_FOLDER'] = PLOT_FOLDER

K = 16
MAX_ITERS = 25

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
        
        # Call the image compressor
        original_img, X_img, centroids, idx = compressor.compress_image(filename, K, MAX_ITERS)
        
        # Generate plots and get their paths
        plot_paths = compressor.generate_plots(X_img, centroids, idx, K, original_img)
        
        # Extract filenames from paths for web serving
        plot_filenames = [os.path.basename(path) for path in plot_paths]
        
        return render_template('show_plots.html', plot_filenames=plot_filenames)
    return render_template('upload.html')

@app.route('/load_sample', methods=['GET'])
def load_sample():
    if not os.path.exists(SAMPLE_IMAGE_PATH):
        return 'Sample image not found', 404
    
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sample.png')
    # Assuming the sample image doesn't need to be saved again if it's already there
    if not os.path.exists(filename):
        os.rename(SAMPLE_IMAGE_PATH, filename)
    
    original_img, X_img, centroids, idx = compressor.compress_image(filename, K, MAX_ITERS)
    plot_paths = compressor.generate_plots(X_img, centroids, idx, K, original_img)
    plot_filenames = [os.path.basename(path) for path in plot_paths]
    
    return render_template('show_plots.html', plot_filenames=plot_filenames)

@app.route('/data/<filename>')
def static_file(filename):
    return send_from_directory('data', filename)

@app.route('/plots/<filename>')
def serve_plot(filename):
    return send_from_directory(app.config['PLOT_FOLDER'], filename)

if __name__ == '__main__':
    for folder in [UPLOAD_FOLDER, PLOT_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    app.run(debug=True)
