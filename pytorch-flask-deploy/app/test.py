import requests


resp = requests.post("http://localhost:5000/predict", files={'file': open('C:\\Sowmya\\Personal\\PYTORCH\\pytorch-examples\\pytorch-flask-deploy\\app\\two.png', 'rb')})

print(resp.text)