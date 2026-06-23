import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as torch_models
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.measure import regionprops
import tifffile as tif
from sklearn.metrics import accuracy_score
from PIL import Image
from torchvision import transforms
from cellpose_omni import models as omni_models
from omnipose.utils import normalize99
import cv2

seg_model = omni_models.CellposeModel(gpu=True, model_type='bact_phase_omni')
params = {
    'channels': [0, 0],
    'rescale': 1.5,
    'mask_threshold': 0,
    'flow_threshold': 0,
    'transparency': True,
    'omni': True,
    'cluster': True,
    'resample': True,
    'verbose': False,
    'tile': False,
    'niter': None,
    'augment': False,
    'affinity_seg': False,
}

def prepare_bacteria_image(cell_img, target_size=30, padding_mode='reflect', output_dtype='float32'):
    if cell_img.ndim != 2:
        raise ValueError("Input must be 2D grayscale image")
    img_float = cell_img.astype(np.float32)
    h, w = cell_img.shape
    if target_size > max(h, w):
        scale = 1.0
    else:
        scale = target_size / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img_float, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    padding_dict = {
        'reflect': cv2.BORDER_REFLECT_101,
        'constant': cv2.BORDER_CONSTANT,
        'replicate': cv2.BORDER_REPLICATE
    }
    border_type = padding_dict.get(padding_mode, cv2.BORDER_REFLECT_101)
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, borderType=border_type)
    return padded.astype(np.float32)

def initialize_model(num_classes=6, use_pretrained=False):
    model = torch_models.resnet50(pretrained=use_pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def load_model_weights(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded base weight file from {checkpoint_path}")
    return model

def img_process(cell_img):
    cell_img = (cell_img - cell_img.min()) / (cell_img.max() - cell_img.min())
    cell_img = (cell_img * 255).astype(np.uint8)
    img_pil = Image.fromarray(cell_img, mode='L')
    img_pil = img_pil.convert('RGB')
    return img_pil

def normalize_channel(img, lower_percent=1, upper_percent=99):
    lower = np.percentile(img, lower_percent)
    upper = np.percentile(img, upper_percent)
    img_normalized = np.clip((img - lower) / (upper - lower + 1e-8), 0, 1)
    return img_normalized

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

FILE_PATH = "path/to/your/image.tif"
SAVE_DIR = "./results"
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PATH = "ESKAPe_Resnet.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Efm", "Sau", "Kpn", "Aba", "Pae", "Eco"]
MODEL_TO_DISPLAY = [5, 1, 2, 3, 4, 0]
COLORS = ['#00CCCC', '#00CC00', '#0000CC', '#CCCC00', '#CC00CC', '#CC0000']

# Modify type_A and type_B according to the fluorescent channel labels of your test image
type_A = 3 # Aba
type_B = 4 # Pae

model = initialize_model(num_classes=6, use_pretrained=False)
model = nn.DataParallel(model).to(device)
model = load_model_weights(model, MODEL_PATH)
model.eval()

img = tif.imread(FILE_PATH)
bright, flu_A, flu_B = img[0], img[1], img[2]

masks, _, _ = seg_model.eval(normalize99(bright), **params)
props = regionprops(masks)
total = len(props)

bg = masks == 0
th_A = flu_A[bg].mean() + 3 * flu_A[bg].std()
th_B = flu_B[bg].mean() + 3 * flu_B[bg].std()

cell_info, preds, trues = [], [], []
for prop in props:
    y1, x1, y2, x2 = prop.bbox
    mean_A = flu_A[y1:y2, x1:x2].mean()
    mean_B = flu_B[y1:y2, x1:x2].mean()
    
    is_A, is_B = mean_A > th_A, mean_B > th_B
    true_label = type_A if is_A and not is_B else type_B if is_B and not is_A else None
    
    pad = img_process(prepare_bacteria_image(bright[y1:y2, x1:x2]))
    with torch.no_grad():
        pred = model(test_transform(pad).unsqueeze(0).to(device)).argmax(1).item()
    
    cell_info.append({'bbox': (y1, x1, y2, x2), 'pred': pred, 'true': true_label})
    if true_label is not None:
        preds.append(pred)
        trues.append(true_label)

valid = sum(1 for c in cell_info if c['true'] is not None)
counts = [sum(1 for c in cell_info if MODEL_TO_DISPLAY[c['pred']] == i) for i in range(6)]

print(f"Total: {total}, Valid: {valid} ({valid/total*100:.1f}%)")

fig, axes = plt.subplots(2, 3, figsize=(22, 13))

for i, (ax, ch, title) in enumerate(zip(axes.flat, 
    [bright, flu_A, flu_B, bright, flu_A, flu_B],
    ["Brightfield", f"Flu A: {CLASS_NAMES[MODEL_TO_DISPLAY[type_A]]}", f"Flu B: {CLASS_NAMES[MODEL_TO_DISPLAY[type_B]]}"] * 2)):
    
    ax.imshow(normalize_channel(ch), cmap='gray' if i % 3 == 0 else 'viridis')
    ax.set_title(title)
    ax.axis('off')
    
    if i >= 3:
        for c in cell_info:
            y1, x1, y2, x2 = c['bbox']
            show = (i == 3) or (i == 4 and c['pred'] == type_A) or (i == 5 and c['pred'] == type_B)
            if show:
                color = COLORS[MODEL_TO_DISPLAY[c['pred']]]
                ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none'))

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "vis.png"), dpi=150, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(6), counts, color=COLORS, edgecolor='black')
ax.set_xticks(range(6))
ax.set_xticklabels(CLASS_NAMES, rotation=45)
ax.set_ylabel("Counts")
for i, c in enumerate(counts):
    ax.text(i, c + 0.05, str(c), ha='center')
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "bar.png"), dpi=150, bbox_inches='tight')
plt.show()
