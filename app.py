from flask import Flask, request, render_template
import torch
from torchvision import models, transforms
from PIL import Image
import requests
import os

app = Flask(__name__)

# Load model and labels once
model = models.resnet50(pretrained=True)
model.eval()
labels = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text.splitlines()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        if 'image' not in request.files:
            result = 'No file part'
        else:
            file = request.files['image']
            if file.filename == '':
                result = 'No selected file'
            else:
                img = Image.open(file.stream)
                input_tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.nn.functional.softmax(output[0], dim=0)
                top5 = torch.topk(probs, 5)
                result = [(labels[top5.indices[i]], float(top5.values[i])) for i in range(5)]
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
