import os
import io
import re
import random
import string
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as torch_models
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
from skimage.measure import regionprops
import tifffile as tif
from collections import Counter
from pathlib import Path
import cv2
from torchvision import transforms
from cellpose import models as omni_models
from omnipose.utils import normalize99

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

def generate_random_string():
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(3))

class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

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
    return model

def img_process(cell_img):
    cell_img = (cell_img - cell_img.min()) / (cell_img.max() - cell_img.min())
    cell_img = (cell_img * 255).astype(np.uint8)
    img_pil = Image.fromarray(cell_img, mode='L')
    img_pil = img_pil.convert('RGB')
    return img_pil

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

def setup_environment(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def well_pattern_extract(s):
    pattern = r'Well-([A-Z]\d{2})_'
    match = re.search(pattern, s)
    return match.group(1) if match else None

def add0(s):
    return s[0] + '0' + s[1:]

def find_file_by_index(root, include_str, n):
    return list(Path(root).rglob(f'*{include_str}*'))[n]

class Patchifier:
    def __init__(self, img_shape=(512, 512), patch_size=256, pad=32):
        self._shape = img_shape
        self.size = patch_size
        self.pad = pad
        self.shape = (max(self._shape[0], self.size), max(self._shape[1], self.size))
        self.pad_h = max(0, self.size - self._shape[0])
        self.pad_w = max(0, self.size - self._shape[1])
        self.ref_coords = self.generate_patch_coords()

    def generate_patch_coords(self):
        h, w = self.shape
        xs = list(np.arange(0, h - self.size, self.size - 2 * self.pad)) + [h - self.size]
        if len(xs) > 1:
            if xs[-1] == xs[-2]:
                xs = xs[:-1]
        ys = list(np.arange(0, w - self.size, self.size - 2 * self.pad)) + [w - self.size]
        if len(ys) > 1:
            if ys[-1] == ys[-2]:
                ys = ys[:-1]
        ref_coords = np.array([[x, y, np.random.randint(2)] for x in xs for y in ys])
        return ref_coords

    def patchify(self, img, random_rotate=False):
        if self.shape != img.shape[:2]:
            self.__init__(img.shape[:2])
        pad_config = np.zeros((len(img.shape), 2))
        pad_config[0][1] = self.pad_h
        pad_config[1][1] = self.pad_w
        pad_config = pad_config.astype(int)
        if self.pad_h > 0 or self.pad_w > 0:
            padded_img = np.pad(img.copy(), pad_config, mode='constant')
        else:
            padded_img = img.copy()
        patches = []
        for x, y, t in self.ref_coords:
            p = padded_img[x:x + self.size, y:y + self.size]
            if random_rotate and t:
                p = p.T
            patches.append(p)
        return np.array(patches)

    def unpatchify(self, patches, n_channel):
        canvas = np.zeros(list(self.shape) + [n_channel])
        canvas_counter = np.zeros(self.shape)
        for i, p in enumerate(patches):
            x, y = self.ref_coords[i][0], self.ref_coords[i][1]
            canvas[x:x + self.size, y:y + self.size] += p
            canvas_counter[x:x + self.size, y:y + self.size] += 1
        mean_canvas = canvas / canvas_counter[:, :, np.newaxis]
        return mean_canvas[:self._shape[0], :self._shape[1]]

    def unpatchify_max(self, patches, n_channel):
        canvas = np.zeros(list(self.shape) + [n_channel])
        for i, p in enumerate(patches):
            x, y = self.ref_coords[i][0], self.ref_coords[i][1]
            p0 = canvas[x:x + self.size, y:y + self.size].copy()
            max_p = np.array([p, p0]).max(axis=0)
            canvas[x:x + self.size, y:y + self.size] = max_p
        return canvas[:self._shape[0], :self._shape[1]]

def score_z(stack, func):
    return np.array([func(x) for x in stack])

def eol_operator(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplace = cv2.Laplacian(image, cv2.CV_64F)
    f_eol = np.sum(laplace[1:-1, 1:-1] ** 2)
    return f_eol

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

setup_environment()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILE_PATH = "path/to/sputum_or_balf.tif"
OUTPUT_DIR = "path/to/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = "path/to/model.pth"
HIGH_PROB_THRESHOLD = 0.998
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GROUND_TRUTH_LABEL = 3

CLASS_NAMES = ["Eco", "Sau", "Kpn", "Aba", "Pae", "Efm"]
CLASS_COLORS = ['#CC0000', '#00CC00', '#0000CC', '#CCCC00', '#CC00CC', '#00CCCC']

model = initialize_model(num_classes=6, use_pretrained=False)
model = nn.DataParallel(model).to(DEVICE)
model = load_model_weights(model, MODEL_PATH)
model.eval()

img = tif.imread(FILE_PATH)
bright_field = np.moveaxis(img[:, 0, :, :], 0, -1)
pat = Patchifier(img_shape=bright_field.shape[:2], pad=0, patch_size=256)
zpat = pat.patchify(bright_field)
zpat = np.moveaxis(zpat, -1, 1)
sharpness_records = np.array([score_z(x, eol_operator) for x in zpat])
max_index = np.argmax(sharpness_records.mean(axis=0))
final_bright_field = bright_field[:, :, max_index]

masks, _, _ = seg_model.eval(normalize99(final_bright_field), **params)
props = regionprops(masks)
total = len(props)

cell_info, pred_counts, high_conf_counts = [], Counter(), Counter()

for prop in props:
    y1, x1, y2, x2 = prop.bbox
    cell_img = final_bright_field[max(0, y1):min(final_bright_field.shape[0], y2), max(0, x1):min(final_bright_field.shape[1], x2)]
    try:
        pad = img_process(prepare_bacteria_image(cell_img))
    except:
        continue
    
    tensor = test_transform(pad).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
        max_prob, pred = torch.max(probs, dim=0)
        predicted_class = int(pred.item())
    
    is_high = max_prob.item() >= HIGH_PROB_THRESHOLD
    cell_info.append({'bbox': (y1, x1, y2, x2), 'pred': predicted_class, 'high': is_high})
    pred_counts[predicted_class] += 1
    if is_high:
        high_conf_counts[predicted_class] += 1

high_conf_cells = [c for c in cell_info if c['high']]
total_cells = len(cell_info)
high_conf_total = len(high_conf_cells)
pred_vec_filtered = [high_conf_counts.get(i, 0) for i in range(6)]

acc = sum(1 for c in high_conf_cells if c['pred'] == GROUND_TRUTH_LABEL) / high_conf_total if high_conf_total > 0 else 0

fig1, axes = plt.subplots(1, 3, figsize=(24, 10))

def norm(img):
    img = img.astype(np.float32)
    return img / img.max() if img.max() > 0 else img

for i, ax in enumerate(axes):
    ax.imshow(norm(final_bright_field), cmap='gray')
    ax.set_title(["Brightfield", "All Predictions", "Filtered Predictions"][i])
    ax.axis('off')
    if i == 0:
        continue
    for c in (cell_info if i == 1 else high_conf_cells):
        y1, x1, y2, x2 = c['bbox']
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2.0, edgecolor=CLASS_COLORS[c['pred']], facecolor='none'))

plt.tight_layout()
buf1 = io.BytesIO()
fig1.savefig(buf1, format='png', dpi=150, bbox_inches='tight', facecolor='white')
buf1.seek(0)
img_top = Image.open(buf1)
plt.close(fig1)

fig2, axes2 = plt.subplots(2, 2, figsize=(24, 10), gridspec_kw={'height_ratios': [3, 1], 'width_ratios': [4, 6]})

ax_bar = axes2[0, 0]
bars = ax_bar.bar(range(6), pred_vec_filtered, color=CLASS_COLORS, edgecolor='black')
ax_bar.set_xlabel("Cell Class")
ax_bar.set_ylabel("Counts")
ax_bar.set_xticks(range(6))
ax_bar.set_xticklabels(CLASS_NAMES, rotation=45)
ax_bar.set_ylim(0, max(pred_vec_filtered) * 1.25 if pred_vec_filtered else 1)
for bar, c in zip(bars, pred_vec_filtered):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, str(c), ha='center')

ax_table = axes2[0, 1]
ax_table.axis('off')
table_data = [
    ['Total', str(total_cells), 'High Conf', f"{high_conf_total} ({high_conf_total/total_cells*100:.1f}%)"],
    ['Acc.', f"{acc*100:.2f}%" if high_conf_total else "N/A", 'True Label', CLASS_NAMES[GROUND_TRUTH_LABEL] if GROUND_TRUTH_LABEL is not None else 'N/A'],
]
for i in range(0, 6, 2):
    c1, c2 = pred_vec_filtered[i], pred_vec_filtered[i+1] if i+1 < 6 else 0
    p1 = c1 / high_conf_total * 100 if high_conf_total > 0 else 0
    p2 = c2 / high_conf_total * 100 if high_conf_total > 0 else 0
    table_data.append([CLASS_NAMES[i], f"{c1} ({p1:.1f}%)", CLASS_NAMES[i+1] if i+1 < 6 else '', f"{c2} ({p2:.1f}%)" if i+1 < 6 else ''])
table = ax_table.table(cellText=table_data, loc='center', cellLoc='left', colWidths=[0.22, 0.28, 0.22, 0.28], bbox=[0.05, 0.05, 0.9, 0.9])
table.auto_set_font_size(False)
table.scale(1, 2.2)
for j in range(4):
    table[(0, j)].set_facecolor('#40466e')
    table[(0, j)].set_text_props(weight='bold', color='white')
ax_table.set_title('Statistics')

ax_legend = axes2[1, :]
ax_legend.axis('off')
handles = [Rectangle((0, 0), 1, 1, color=c, ec='k') for c in CLASS_COLORS]
ax_legend.legend(handles, CLASS_NAMES, loc='center', ncol=6, title="Cell Classes")

plt.tight_layout()
buf2 = io.BytesIO()
fig2.savefig(buf2, format='png', dpi=150, bbox_inches='tight', facecolor='white')
buf2.seek(0)
img_bottom = Image.open(buf2)
plt.close(fig2)

target_width = img_top.width
img_bottom_resized = img_bottom.resize((target_width, int(img_bottom.height * target_width / img_bottom.width)), Image.Resampling.LANCZOS)
img_top_resized = img_top.resize((target_width, int(img_top.height * target_width / img_top.width)), Image.Resampling.LANCZOS)

combined = Image.new('RGB', (target_width, img_top_resized.height + img_bottom_resized.height), (255, 255, 255))
combined.paste(img_top_resized, (0, 0))
combined.paste(img_bottom_resized, (0, img_top_resized.height))

plt.figure(figsize=(24, 20))
plt.imshow(combined)
plt.axis('off')
plt.tight_layout()
plt.show()

combined.save(os.path.join(OUTPUT_DIR, "output.png"))