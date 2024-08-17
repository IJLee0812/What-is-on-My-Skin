from flask import Flask, request, render_template
from PIL import Image
import torch
from torchvision import models, transforms
import io
import os

app = Flask(__name__)

# load a model
model_path = os.path.join('models', 'best_model_fold_1.pth')
model = models.densenet121(pretrained=False)
model.classifier = torch.nn.Linear(model.classifier.in_features, 4) # 4 : acne, acne_scar, hyperpigmentation, normal
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# preprocessing image from users
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# load a class names for classification
class_names_path = os.path.join('models', 'class_names.pth')
class_names = torch.load(class_names_path)

# mapping predicted labels to korean symptoms
english_to_korean = {
    'acne': '여드름',
    'acne_scar': '여드름 흉터',
    'hyperpigmentation': '색소침착',
    'normal': '정상 피부'
}

# solutions for each symptoms (matched with korean explanation)
solutions = {
    '여드름': '1. 얼굴을 깨끗이 씻으세요. 2. 벤조일 퍼옥사이드나 살리실산이 함유된 제품을 사용하세요. 3. 오일 프리 제품을 사용하세요. 4. 피부과 전문의와 상담하세요.',
    '여드름 흉터': '1. 비타민 C 세럼을 사용하세요. 2. 레티놀 제품을 사용하세요. 3. 화학적 필링을 고려해보세요. 4. 피부과에서 레이저 치료를 받아보세요.',
    '색소침착': '1. 자외선 차단제를 반드시 사용하세요. 2. 비타민 C 세럼을 사용하세요. 3. 각질 제거를 정기적으로 하세요. 4. 하이드로퀴논 제품 사용을 고려해보세요.',
    '정상 피부': '현재 피부 상태가 양호합니다. 꾸준한 관리로 건강한 피부를 유지하세요.'
}

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if'file' not in request.files:
            return render_template('index.html', error='파일이 선택되지 않았습니다.')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='파일이 선택되지 않았습니다.')
        
        if file:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            tensor = preprocess(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(tensor)
                _, predicted = torch.max(outputs, 1)
                label = english_to_korean[class_names[predicted.item()]]
                solution = solutions[label]
            
            return render_template('result.html', label=label, solution=solution)
    
    return render_template('index.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback = request.form['feedback']
    # (add feedback processing logics later)
    return "피드백이 제출되었습니다. 감사합니다!"

if __name__ == '__main__':
    app.run(debug=True)
