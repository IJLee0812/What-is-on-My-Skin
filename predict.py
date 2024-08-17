import torch
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from PIL import Image
from model import initialize_model

base_dir = "./results"
data_transforms = torch.load(os.path.join(base_dir, "data_transforms.pth"))
class_names = torch.load(os.path.join(base_dir, "class_names.pth"))


# define the prediction function
def predict_image(image_path, model, class_names):
    model.eval()
    image = Image.open(image_path)
    image = data_transforms(image).unsqueeze(0) # Transform image, add batch dimension
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image) # Use the image as model input
        _, preds = torch.max(outputs, 1)

    plt.imshow(Image.open(image_path))
    plt.title(f'Predicted : {class_names[preds[0]]}')
    plt.show()
    return class_names[preds[0]]


# define the main function
def main():
    # device & model setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = initialize_model(len(class_names))

    model_path = os.path.join(base_dir, f"best_model_fold_1.pth")

    # load the model weights
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    # predict the class of the input image
    image_path = 'user_image.jpg'
    if not os.path.exists(image_path):
        print(f"Image at path {image_path} does not exist.")
        return

    predicted_class = predict_image(image_path, model, class_names)
    print()
    print(f'The predicted class (Your symptom predicted by the model) is: {predicted_class}')


if __name__ == "__main__":
    main()