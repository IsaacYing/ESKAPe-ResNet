import os
import io
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as torch_models
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
from skimage.measure import regionprops
import tifffile as tif
from collections import Counter
import cv2
from torchvision import transforms
from cellpose_omni import models as omni_models
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILE_PATH = "path/to/your/image.tif"
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = "ESKAPe_Resnet.pth"
HIGH_PROB_THRESHOLD = 0.998
GROUND_TRUTH_LABEL = 3 # Aba

CLASS_NAMES = ["Efm", "Sau", "Kpn", "Aba", "Pae", "Eco"]
MODEL_TO_DISPLAY = [5, 1, 2, 3, 4, 0]
DISPLAY_TO_MODEL = [5, 1, 2, 3, 4, 0]
COLORS = ['#00CCCC', '#00CC00', '#0000CC', '#CCCC00', '#CC00CC', '#CC0000']

model = initialize_model(num_classes=6, use_pretrained=False)
model = nn.DataParallel(model).to(device)
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
    
    tensor = test_transform(pad).unsqueeze(0).to(device)
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

pred_vec_filtered = [high_conf_counts.get(DISPLAY_TO_MODEL[i], 0) for i in range(6)]

fig1 = plt.figure(figsize=(22, 8))
gs1 = fig1.add_gridspec(1, 3, wspace=0.08)

def norm(img):
    img = img.astype(np.float32)
    return img / img.max() if img.max() > 0 else img

channels = [norm(final_bright_field)] * 3
titles = ["Brightfield", "All Predictions", "Filtered Predictions"]
axes = [fig1.add_subplot(gs1[0, i]) for i in range(3)]

for i, ax in enumerate(axes):
    ax.imshow(channels[i], cmap='gray')
    ax.set_title(titles[i], fontweight='bold', pad=20)
    ax.axis('off')
    if i == 0:
        continue
    for c in (cell_info if i == 1 else high_conf_cells):
        y1, x1, y2, x2 = c['bbox']
        display_idx = MODEL_TO_DISPLAY[c['pred']]
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2.0, edgecolor=COLORS[display_idx], facecolor='none'))

plt.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.02, wspace=0.08)

buf1 = io.BytesIO()
fig1.savefig(buf1, format='png', dpi=150, bbox_inches='tight', facecolor='white')
buf1.seek(0)
img_top = Image.open(buf1)
plt.close(fig1)

fig2 = plt.figure(figsize=(24, 9))
gs2 = fig2.add_gridspec(2, 2, height_ratios=[2.8, 1.2], width_ratios=[3, 7], hspace=0.5, wspace=0.25)

ax_bar = fig2.add_subplot(gs2[0, 0])
bars = ax_bar.bar(range(len(CLASS_NAMES)), pred_vec_filtered,
                  color=[COLORS[i] for i in range(len(CLASS_NAMES))],
                  edgecolor='black')

ax_bar.set_xlabel("Cell Class")
ax_bar.set_ylabel("Counts")
ax_bar.set_xticks(range(len(CLASS_NAMES)))
ax_bar.set_xticklabels(CLASS_NAMES, rotation=30, ha='center')

max_count = max(pred_vec_filtered) if pred_vec_filtered else 0
ax_bar.set_ylim(0, max_count * 1.25)

for bar, c in zip(bars, pred_vec_filtered):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(c), ha='center', va='bottom')

ax_table = fig2.add_subplot(gs2[0, 1])
ax_table.axis('off')

table_data = [
    ['Total', str(total_cells), 'High Conf.', str(high_conf_total)],
    ['Label', CLASS_NAMES[MODEL_TO_DISPLAY[GROUND_TRUTH_LABEL]] if GROUND_TRUTH_LABEL is not None else 'N/A', '', ''],
]

for i in range(0, 6, 2):
    name1 = CLASS_NAMES[i]
    model_idx1 = DISPLAY_TO_MODEL[i]
    count1 = high_conf_counts.get(model_idx1, 0)
    pct1 = count1 / high_conf_total * 100 if high_conf_total > 0 else 0

    if i + 1 < 6:
        name2 = CLASS_NAMES[i + 1]
        model_idx2 = DISPLAY_TO_MODEL[i + 1]
        count2 = high_conf_counts.get(model_idx2, 0)
        pct2 = count2 / high_conf_total * 100 if high_conf_total > 0 else 0
        table_data.append([
            name1, f"{count1} ({pct1:.1f}%)",
            name2, f"{count2} ({pct2:.1f}%)"
        ])
    else:
        table_data.append([
            name1, f"{count1} ({pct1:.1f}%)", '', ''
        ])

table = ax_table.table(cellText=table_data, loc='center', cellLoc='left',
                       colWidths=[0.18, 0.22, 0.35, 0.25], 
                       bbox=[0.0, 0.02, 1.0, 0.96])
table.auto_set_font_size(False)
table.scale(1, 2.5)

for j in range(4):
    table[(0, j)].set_facecolor('#40466e')
    table[(0, j)].set_text_props(weight='bold', color='white')

# Green highlight for true label row
if GROUND_TRUTH_LABEL is not None:
    true_display_idx = MODEL_TO_DISPLAY[GROUND_TRUTH_LABEL]
    true_name = CLASS_NAMES[true_display_idx]
    for i in range(2, len(table_data)):
        for j in range(0, 4, 2):
            cell_text = table_data[i][j]
            if cell_text == true_name:
                table[(i, j)].set_facecolor('#ccffcc')
                table[(i, j + 1)].set_facecolor('#ccffcc')

ax_table.set_title('Statistics', fontweight='bold', pad=15)

ax_legend = fig2.add_subplot(gs2[1, :])
ax_legend.axis('off')
handles = [Rectangle((0, 0), 1, 1, color=c, ec='k') for c in COLORS]
ax_legend.legend(handles, CLASS_NAMES, loc='lower center', ncol=len(CLASS_NAMES))

plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.08, hspace=0.5, wspace=0.25)

buf2 = io.BytesIO()
fig2.savefig(buf2, format='png', dpi=150, bbox_inches='tight', facecolor='white')
buf2.seek(0)
img_bottom = Image.open(buf2)
plt.close(fig2)

target_width = img_top.width

ratio_b = target_width / img_bottom.width
new_height_b = int(img_bottom.height * ratio_b)
img_bottom_resized = img_bottom.resize((target_width, new_height_b), Image.Resampling.LANCZOS)

ratio_t = target_width / img_top.width
new_height_t = int(img_top.height * ratio_t)
img_top_resized = img_top.resize((target_width, new_height_t), Image.Resampling.LANCZOS)

total_height = img_top_resized.height + img_bottom_resized.height
combined_img = Image.new('RGB', (target_width, total_height), (255, 255, 255))
combined_img.paste(img_top_resized, (0, 0))
combined_img.paste(img_bottom_resized, (0, img_top_resized.height))

plt.figure(figsize=(18, 16))
plt.imshow(combined_img)
plt.axis('off')
plt.tight_layout()
plt.show()

combined_img.save(os.path.join(OUTPUT_DIR, "output.png"))
