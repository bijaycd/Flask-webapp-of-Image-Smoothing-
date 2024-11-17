import os
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from knn_smooth import KNNSmooth  # Assume this is implemented as per your Tkinter code
from avg_smooth import SimpleAveragingSmooth  # Assume this is implemented as per your Tkinter code

app = Flask(__name__)
app.secret_key = "your_secret_key"  # For flash messages

# Folder to save uploaded and processed images
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route for homepage
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get file and inputs from form
        file = request.files.get("image")
        window_size = request.form.get("window_size")
        k_value = request.form.get("k_value")
        
        # Validate inputs
        if not file or file.filename == "":
            flash("No file selected.")
            return redirect(request.url)
        
        if not window_size.isdigit() or not k_value.isdigit():
            flash("Please enter valid numeric values for window size and k value.")
            return redirect(request.url)

        # Process inputs
        window_size = int(window_size)
        k_value = int(k_value)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Open image, convert to grayscale, and apply smoothing
        original_image = Image.open(file_path).convert("L")
        im_arr = np.array(original_image)

        # KNN Smoothing
        knn_smooth = KNNSmooth()
        knn_img_arr = knn_smooth.smooth(im_arr, window=window_size, k=k_value)
        knn_img_arr = np.clip(knn_img_arr, 0, 255).astype(np.uint8)
        knn_img = Image.fromarray(knn_img_arr)
        knn_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"knn_{filename}")
        knn_img.save(knn_img_path)

        # Simple Averaging Smoothing
        avg_smooth = SimpleAveragingSmooth()
        avg_img_arr = avg_smooth.smooth(im_arr, window=window_size)
        avg_img_arr = np.clip(avg_img_arr, 0, 255).astype(np.uint8)
        avg_img = Image.fromarray(avg_img_arr)
        avg_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"avg_{filename}")
        avg_img.save(avg_img_path)

        # Difference Image
        diff_img_arr = np.abs(knn_img_arr - avg_img_arr)
        diff_img = Image.fromarray(diff_img_arr)
        diff_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"diff_{filename}")
        diff_img.save(diff_img_path)

        # Pass results to template
        return render_template(
            "index.html",
            original_image=url_for("static", filename=f"images/{filename}"),
            knn_image=url_for("static", filename=f"images/knn_{filename}"),
            avg_image=url_for("static", filename=f"images/avg_{filename}"),
            diff_image=url_for("static", filename=f"images/diff_{filename}"),
            window_size=window_size,
            k_value=k_value
        )

    return render_template("index.html")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)