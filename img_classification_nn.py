import torch
from torchvision.models import get_weight
import requests
from PIL import Image
from io import BytesIO
import torch.nn.functional as F

# ── Device setup ──────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_image_from_url(url, headers=None):
    if headers is None:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def predict(path_or_url, model, transforms_fn, categories, topk=1, headers=None):
    if path_or_url.startswith('http'):
        img = get_image_from_url(path_or_url, headers=headers)
    else:
        img = Image.open(path_or_url)

    preproc_img = transforms_fn(img)

    if len(preproc_img.shape) == 3:
        preproc_img = preproc_img.unsqueeze(0)

    # Move input tensor to the same device as the model
    device = next(model.parameters()).device
    preproc_img = preproc_img.to(device)

    model.eval()

    # No gradient computation needed during inference
    with torch.no_grad():
        pred = model(preproc_img)

    probabilities = F.softmax(pred[0], dim=0)
    values, indices = torch.topk(probabilities, topk)

    return [{'label': categories[i], 'value': v.item()} for i, v in zip(indices, values)]


# ── Model setup ───────────────────────────────────────────────
weights = get_weight('MobileNet_V3_Small_Weights.DEFAULT')
model = torch.hub.load('pytorch/vision', 'mobilenet_v3_small', weights=weights)

model = model.to(device)  # Move model to GPU
model.eval()

categories = weights.meta['categories']
transforms_fn = weights.transforms()

# ── Inference ─────────────────────────────────────────────────
url = 'https://upload.wikimedia.org/wikipedia/commons/c/ce/Daisy_G%C3%A4nsebl%C3%BCmchen_Bellis_perennis_01.jpg'
headers = {'User-Agent': 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'}

print(predict(url, model, transforms_fn, categories, headers=headers))