# 모델 준비하기
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

# 이미지 열고 전처리
from PIL import Image
from torchvision import transforms
input_image = Image.open('dog.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# 가속 환경 설정
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
# 현재 해당 모델은 mps 에서 제대로 지원되지 않음
# elif torch.backends.mps.is_available():
    # device = 'mps'

input_batch.to(device)
model.to(device)

with torch.no_grad():
    output = model(input_batch)
# label 1000 개에 대한 confidence score
# print(output[0])
# softmax를 취해 확률 값으로 변환
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# 사람이 읽을 수 있는 이미지 레이블로 변환
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())