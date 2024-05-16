from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import tensorflow as tf
import cv2
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def edge_detection(img):
    # Perform edge detection on the image
    edges = cv2.Canny(img, 100, 200)
    return edges

def segmentation(img):
    # Apply watershed segmentation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    # Convert markers to uint8 for visualization
    markers = markers.astype(np.uint8)
    # Apply colormap
    segmented_img = cv2.applyColorMap(markers, cv2.COLORMAP_VIRIDIS)
    return segmented_img

@app.route('/')
def upload_form():
    return render_template('upload_form.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        action = request.form['action']
        if action == 'predict':
            return redirect(url_for('predict', filename=filename))
        elif action == 'edge_detection':
            return redirect(url_for('edge_detect', filename=filename))
        elif action == 'segmentation':
            return redirect(url_for('segment_image', filename=filename))

@app.route('/predict/<filename>')
def predict(filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception("Error reading the image. Please check if the file exists and is in a supported format.")
    
    img_resized = cv2.resize(img, (28, 28))
    img_resized = img_resized / 255.0
    flattened_img = img_resized.flatten()
    flattened_img = 1 - flattened_img
    input_data = flattened_img.reshape(1, 28, 28, 1)
    cv2.imwrite(os.path.join('static', 'product.jpg'), img)
    model = tf.keras.models.load_model('image_classification_model.h5')
    class_probabilities = model.predict(input_data)
    predicted_class_index = np.argmax(class_probabilities)
    products = [
        "T-shirt/top", "Pants", "Coat", "Dress", "Pullover",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    product = products[predicted_class_index]
    
    return render_template('prediction_result.html', product=product)

@app.route('/edge_detection/<filename>')
def edge_detect(filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception("Error reading the image. Please check if the file exists and is in a supported format.")

    edges = edge_detection(img)

    # Save the edge-detected image
    cv2.imwrite(os.path.join('static', 'edges.jpg'), edges)

    return render_template('edge_detection_result.html')

@app.route('/segmentation/<filename>')
def segment_image(filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(img_path)
    if img is None:
        raise Exception("Error reading the image. Please check if the file exists and is in a supported format.")

    segmented_img = segmentation(img)

    # Save the segmented image
    cv2.imwrite(os.path.join('static', 'segmented.jpg'), segmented_img)

    return render_template('segmentation_result.html')

if __name__ == '__main__':
    app.run(debug=True)
