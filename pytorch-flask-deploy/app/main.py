from flask import Flask, request, jsonify
from torch_utils import transform_image, get_prediction


app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            imag_bytes = file.read()
            tensor = transform_image(imag_bytes)
            prediction = get_prediction(tensor)
            data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})

    # load image
    # image -> tensor
    # prediction
    # return json
    return jsonify({'result': 1})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)