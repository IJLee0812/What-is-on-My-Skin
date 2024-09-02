# -*- coding: utf-8 -*-
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
    '여드름': 
        "1. 얼굴 청결 유지: 하루에 두 번, 미지근한 물과 여드름에 적합한 클렌저로 얼굴을 부드럽게 세안하세요. 너무 자주 세안하거나 얼굴을 과도하게 문지르면 피부 자극이 발생할 수 있습니다. <br><br> 2. 벤조일 퍼옥사이드(Benzoyl Peroxide) 또는 살리실산(Salicylic Acid) 사용: 이 성분들은 여드름을 유발하는 박테리아를 제거하고 모공을 깨끗하게 유지하는 데 도움을 줍니다. 제품을 사용할 때는 피부에 적절히 테스트한 후 사용하는 것이 좋습니다. <br><br> 3. 오일 프리 제품 사용: 여드름이 있는 피부에는 오일이 없는 비화학성 제품을 사용하는 것이 중요합니다. 이는 모공이 막히는 것을 방지해 줍니다. <br><br> 4. 피부과 전문의 상담: 지속적인 여드름 문제는 전문의와 상담하여 레티노이드, 항생제, 또는 기타 처방약을 고려하는 것이 좋습니다. 경우에 따라서는 레이저 치료나 경구약이 필요할 수도 있습니다.",
    '여드름 흉터':
        "1. 비타민 C 세럼 사용: 비타민 C는 피부를 밝게 하고 콜라겐 생성을 촉진하여 흉터의 개선에 도움을 줄 수 있습니다. 아침에 세럼을 바르고 자외선 차단제를 함께 사용하세요. <br><br> 2. 레티놀(Retinol) 사용: 레티놀은 피부 재생을 촉진하고 흉터를 옅게 만드는 데 효과적입니다. 저용량으로 시작해 피부가 적응할 수 있도록 하세요. <br><br> 3. 화학적 필링: 글리콜릭산(Glycolic Acid)이나 젖산(Lactic Acid) 필링은 피부의 표면층을 제거해 흉터를 완화하는 데 도움을 줄 수 있습니다. 전문의의 지도 하에 진행하는 것이 안전합니다. <br><br> 4. 레이저 치료: 피부과에서 제공하는 레이저 치료는 심한 흉터 개선에 매우 효과적일 수 있습니다. 흉터의 종류와 깊이에 따라 다양한 레이저 치료가 가능하므로, 전문의와 상담 후 적합한 치료법을 선택하세요.",
    '색소침착': 
        "1. 자외선 차단제 사용: 자외선은 색소침착을 악화시키기 때문에, SPF 30 이상의 광범위 자외선 차단제를 매일 사용하는 것이 중요합니다. 야외 활동 시 자외선 차단제를 자주 덧바르세요. <br><br> 2. 비타민 C 세럼 사용: 비타민 C는 멜라닌 생성 억제에 도움을 주어 색소침착을 완화할 수 있습니다. 아침에 사용하고, 자외선 차단제를 함께 사용하세요. <br><br> 3. 정기적인 각질 제거: 화학적 또는 물리적 각질 제거는 피부 표면의 색소침착을 줄이는 데 도움을 줍니다. AHA나 BHA 성분이 함유된 제품을 고려해 보세요. <br><br> 4. 하이드로퀴논(Hydroquinone) 사용: 하이드로퀴논은 색소침착을 억제하는 강력한 미백 성분입니다. 단, 이 성분은 자극적일 수 있으므로, 사용 전 전문의와 상담하고 필요한 경우 단계적으로 사용하세요.",
    '정상 피부':
        "1. 꾸준한 관리: 정상 피부라도 자외선 차단제를 매일 사용하고, 수분 공급을 위한 보습제를 잊지 마세요. 또한, 균형 잡힌 식단과 충분한 수면을 유지하여 피부 건강을 지속적으로 관리하세요. <br><br> 2. 정기적인 피부 체크: 피부 상태는 시간이 지남에 따라 변할 수 있으므로, 주기적으로 피부 상태를 확인하고 필요한 경우 스킨케어 루틴을 조정하는 것이 중요합니다."
}


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/disease')
def disease():
    return render_template('disease.html')
@app.route('/mypage')
def mypage():
    return render_template('mypage.html')


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
