# main.py
# One-file training script for tobacco-leaf grading (4 classes)
# Key features:
# - No TTA (evaluation is single-view)
# - EMA fixed (updates parameters + BN buffers), optional delayed start
# - LR scheduler & early-stop driven by ONLINE model; EMA used for checkpoint/test
# - Optional Mask 4th channel with normalization
# - Gentler augmentation (no vertical flip; ±max_rotate)
# - WeightedRandomSampler with optional mid-training switch to natural sampling
# - CE + CORAL heads with configurable fusion weights
# - Optional channel/CBAM attention

import os
import time
import csv
import argparse
import numpy as np
import random
import cv2
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights, ResNet18_Weights

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ----------------- Utils -----------------
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For reproducibility without triggering non-deterministic op errors:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if getattr(m, 'bias', None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

# ----------------- Attention -----------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
    def forward(self, x):
        avg = self.avg_pool(x)
        mx = torch.amax(x, dim=(2,3), keepdim=True)
        out = self.fc(avg) + self.fc(mx)
        return torch.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx  = torch.amax(x, dim=1, keepdim=True)
        out = torch.cat([avg, mx], dim=1)
        out = self.conv(out)
        return torch.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ECAAttention(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
    def forward(self, x):
        y = self.avg_pool(x)                    # (B,C,1,1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)) # (B,1,C)
        y = torch.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)

# ----------------- Encoder -----------------
class ResNetEncoder(nn.Module):
    def __init__(self, model_name="resnet18", use_attention=None, freeze_layers=0, in_ch=3):
        super().__init__()
        assert model_name in ["resnet18","resnet50"]
        if model_name == "resnet50":
            backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            out_dim = 2048
        else:
            backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            out_dim = 512

        if in_ch != 3:
            old = backbone.conv1
            backbone.conv1 = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                                       stride=old.stride, padding=old.padding, bias=False)
            with torch.no_grad():
                backbone.conv1.weight[:, :3] = old.weight
                if in_ch > 3:
                    mean_w = old.weight.mean(dim=1, keepdim=True)
                    backbone.conv1.weight[:, 3:in_ch] = mean_w.repeat(1, in_ch-3, 1, 1)

        if freeze_layers > 0:
            frozen = [backbone.conv1, backbone.bn1]
            if freeze_layers >= 1: frozen.append(backbone.layer1)
            if freeze_layers >= 2: frozen.append(backbone.layer2)
            if freeze_layers >= 3: frozen.append(backbone.layer3)
            for m in frozen:
                for p in m.parameters(): p.requires_grad=False
            print(f"[Info] Froze backbone up to layer {min(freeze_layers,3)}")

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.out_dim = out_dim

        self.attention = nn.Identity()
        if use_attention == "eca":
            print("[Info] ECA enabled.")
            self.attention = ECAAttention(out_dim)
            self.attention.apply(initialize_weights)
        elif use_attention == "cbam":
            print("[Info] CBAM enabled.")
            self.attention = CBAM(out_dim)
            self.attention.apply(initialize_weights)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.attention(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x

# ----------------- Heads -----------------
class CEHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes); self.apply(initialize_weights)
    def forward(self, x): return self.fc(x)

class CORALHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes-1); self.apply(initialize_weights)
    def forward(self, x): return self.fc(x)  # (B,K-1)

def coral_targets(y, num_classes):
    B, K = y.size(0), num_classes
    tgt = torch.zeros(B, K-1, device=y.device)
    for k in range(K-1):
        tgt[:, k] = (y > k).float()
    return tgt

def coral_probs_from_logits_monotonic(logits):
    # enforce s1>=s2>=... by cumulative min to stabilize probs
    s = torch.sigmoid(logits)  # (B,K-1)
    s_rev = torch.flip(s, dims=[1])
    s_mon = torch.flip(torch.cummin(s_rev, dim=1).values, dims=[1])
    B, K_1 = s_mon.shape; K = K_1 + 1
    p = [1 - s_mon[:, 0]]
    for k in range(K-2): p.append(s_mon[:, k] - s_mon[:, k+1])
    p.append(s_mon[:, -1])
    probs = torch.stack(p, dim=1).clamp_(1e-8, 1-1e-8)
    return probs

class HybridClassifier(nn.Module):
    def __init__(self, encoder, emb_dim, num_classes, use_ce=True, use_coral=False):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(nn.Linear(encoder.out_dim, emb_dim), nn.LeakyReLU(inplace=True))
        self.ce_head = CEHead(emb_dim, num_classes) if use_ce else None
        self.coral_head = CORALHead(emb_dim, num_classes) if use_coral else None
        self.num_classes = num_classes
    def forward(self, x):
        feat = self.encoder(x)
        emb = self.projector(feat)
        logits_ce = self.ce_head(emb) if self.ce_head is not None else None
        logits_coral = self.coral_head(emb) if self.coral_head is not None else None
        return emb, logits_ce, logits_coral

# ----------------- Mask & Transforms -----------------
def simple_leaf_mask(pil_img):
    img = np.array(pil_img)[:, :, ::-1]            # RGB->BGR
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    _, th = cv2.threshold(A, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(th)
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(mask, [c], -1, 255, -1)
    return (mask > 0).astype(np.float32)

class ToTensorWithMask:
    def __init__(self, size=(256,256), add_mask_channel=False):
        self.size = size
        self.add_mask_channel = add_mask_channel
        self.rgb_tf = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    def __call__(self, pil_img):
        rgb = self.rgb_tf(pil_img)
        if not self.add_mask_channel:
            return rgb
        mask = simple_leaf_mask(pil_img)  # 0/1
        mask = cv2.resize(mask, self.size[::-1], interpolation=cv2.INTER_NEAREST)
        mask = (mask - 0.5) / 0.5         # 标准化到 ~[-1,1]
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return torch.cat([rgb, mask], dim=0)

# ----------------- Dataset & Split -----------------
class EvaluationDataset(Dataset):
    def __init__(self, data, transform): self.data, self.transform = data, transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img, label = self.data[idx]
        return self.transform(img), label

def stratified_split(dataset, val_ratio=0.1, test_ratio=0.1, random_state=42):
    targets = np.array(dataset.targets)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio+test_ratio, random_state=random_state)
    train_idx, temp_idx = next(sss1.split(np.zeros(len(targets)), targets))
    temp_targets = targets[temp_idx]
    val_ratio_adj = val_ratio / (val_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_ratio_adj, random_state=random_state)
    val_idx, test_idx = next(sss2.split(np.zeros(len(temp_targets)), temp_targets))
    val_idx = temp_idx[val_idx]; test_idx = temp_idx[test_idx]
    return (torch.utils.data.Subset(dataset, train_idx),
            torch.utils.data.Subset(dataset, val_idx),
            torch.utils.data.Subset(dataset, test_idx))

# ----------------- EMA (fixed) -----------------
class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.device = next(model.parameters()).device
        self.ema.to(self.device, non_blocking=True)
    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k, v in esd.items():
            if k not in msd: continue
            src = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src.to(v.dtype), alpha=1.0 - d)
            else:
                v.copy_(src)

# ----------------- Metrics & Viz -----------------
def quadratic_weighted_kappa(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    O = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).astype(np.float64)
    N = O.sum()
    if N == 0: return 0.0
    w = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            w[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist) / N
    num = (w * O).sum()
    den = (w * E).sum() + 1e-12
    return 1.0 - num / den

def visualize_embeddings(embeddings, labels, writer, epoch, tag="embedding"):
    emb = embeddings.numpy(); lbs = labels.numpy()
    pca = PCA(n_components=2, random_state=42)
    p = pca.fit_transform(emb)
    plt.figure(figsize=(6,5))
    for lb in np.unique(lbs):
        idx = lbs == lb; plt.scatter(p[idx,0], p[idx,1], s=8, alpha=0.6, label=str(lb))
    plt.legend(); plt.title('PCA'); writer.add_figure(f'{tag}/PCA', plt.gcf(), epoch); plt.close()
    tsne = TSNE(n_components=2, perplexity=30, init='pca', max_iter=300, random_state=42)
    t = tsne.fit_transform(emb)
    plt.figure(figsize=(6,5))
    for lb in np.unique(lbs):
        idx = lbs == lb; plt.scatter(t[idx,0], t[idx,1], s=8, alpha=0.6, label=str(lb))
    plt.legend(); plt.title('t-SNE'); writer.add_figure(f'{tag}/tSNE', plt.gcf(), epoch); plt.close()

# ----------------- Evaluation (No-TTA) -----------------
def evaluate(model, loader, device, num_classes, ce_w=1.0, coral_w=1.0):
    model.eval()
    y_true, y_pred = [], []
    all_probs, all_labels, all_emb = [], [], []

    def _forward(x):
        emb, logits_ce, logits_coral = model(x)
        probs, sum_w = None, 0.0
        if logits_ce is not None and ce_w > 0:
            p_ce = F.softmax(logits_ce, dim=1)
            probs = p_ce * ce_w if probs is None else probs + p_ce * ce_w
            sum_w += ce_w
        if logits_coral is not None and coral_w > 0:
            p_coral = coral_probs_from_logits_monotonic(logits_coral)
            probs = p_coral * coral_w if probs is None else probs + p_coral * coral_w
            sum_w += coral_w
        if sum_w == 0:
            raise ValueError("At least one of the heads must be active for inference.")
        probs = probs / sum_w
        pred = probs.argmax(dim=1)
        return emb, pred, probs

    with torch.no_grad():
        for x, y in tqdm(loader, desc='[Eval]', ncols=100):
            x = x.to(device); y = y.to(device)
            emb, pred, probs = _forward(x)
            y_true.extend(y.cpu().tolist()); y_pred.extend(pred.cpu().tolist())
            all_probs.append(probs.cpu()); all_labels.append(y.cpu()); all_emb.append(emb.cpu())

    acc = np.mean(np.array(y_pred) == np.array(y_true))
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    probs_tensor = torch.cat(all_probs, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    emb_tensor = torch.cat(all_emb, dim=0)
    y_true_bin = np.zeros((len(y_true), num_classes))
    y_true_bin[np.arange(len(y_true)), labels_tensor.numpy()] = 1
    mAP = average_precision_score(y_true_bin, probs_tensor.numpy(), average='macro')
    qwk = quadratic_weighted_kappa(y_true, y_pred, num_classes)
    return {
        'accuracy': acc, 'f1': f1, 'mAP': mAP, 'qwk': qwk,
        'cm': cm, 'emb': emb_tensor, 'labels': labels_tensor,
        'y_true': np.array(y_true), 'y_pred': np.array(y_pred)
    }

# ----------------- Main -----------------
def main():
    from torch.utils.tensorboard import SummaryWriter

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='resnet50', choices=['resnet18','resnet50'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--attention', type=str, default='none', choices=['none','eca','cbam'])
    parser.add_argument('--freeze_layers', type=int, default=0)
    parser.add_argument('--log_root', type=str, default='run')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # augmentation knobs
    parser.add_argument('--max_rotate', type=int, default=10, help='max rotation degrees for RandomRotation(+/-)')
    parser.add_argument('--use_vertical_flip', action='store_true', help='enable vertical flip (disabled by default)')

    # modules
    parser.add_argument('--use_ce', action='store_true')
    parser.add_argument('--use_coral', action='store_true')
    parser.add_argument('--ce_weight', type=float, default=0.5)
    parser.add_argument('--coral_weight', type=float, default=0.5)
    parser.add_argument('--add_mask_channel', action='store_true')
    parser.add_argument('--use_weighted_sampler', action='store_true')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0.995)
    parser.add_argument('--ema_start_epoch', type=int, default=3, help='start EMA update after this epoch (1-indexed)')
    parser.add_argument('--switch_off_sampler_epoch', type=int, default=-1,
                        help='epoch index (1-based) to switch to natural sampling; -1 means keep WRS')
    parser.add_argument('--infer_ce_weight', type=float, default=None,
                    help='CE prob weight at inference; default = ce_weight')
    parser.add_argument('--infer_coral_weight', type=float, default=None,
                    help='CORAL prob weight at inference; default = coral_weight')
    # early stop
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--delta', type=float, default=1e-4)

    # testing
    parser.add_argument('--test_use_online', action='store_true', help='use online model for test instead of EMA')

    args = parser.parse_args()
    set_seed(args.seed)

    if args.infer_ce_weight is None:    args.infer_ce_weight = args.ce_weight
    if args.infer_coral_weight is None: args.infer_coral_weight = args.coral_weight
    # default to CE only if none chosen
    if not args.use_ce and not args.use_coral:
        args.use_ce, args.ce_weight = True, 1.0
        args.coral_weight = 0.0

    attn = None if args.attention == 'none' else args.attention

    # build log dir suffix
    suffix = []
    if attn: suffix.append(attn.upper())
    if args.freeze_layers>0: suffix.append(f"Freeze{args.freeze_layers}")
    if args.add_mask_channel: suffix.append("Mask4Ch")
    if args.use_weighted_sampler: suffix.append("WRS")
    if args.use_ema: suffix.append("EMA")
    suffix += ['CE' if args.use_ce else '', 'CORAL' if args.use_coral else '']
    suffix = '_'.join([s for s in suffix if s])

    dataset = ImageFolder(root=args.data)
    num_classes = len(dataset.classes)
    data_name = os.path.basename(args.data.rstrip('/'))
    log_dir = os.path.join(args.log_root, f"{data_name}_{args.model_name}_{suffix if suffix else 'BASE'}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"[Config] log_dir={log_dir}")

    # transforms
    train_aug_list = [
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(args.max_rotate),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ]
    if args.use_vertical_flip:
        train_aug_list.insert(2, transforms.RandomVerticalFlip(p=0.5))
    train_aug = transforms.Compose(train_aug_list)

    def build_train_tf(add_mask):
        return transforms.Compose([train_aug, ToTensorWithMask(size=(256,256), add_mask_channel=add_mask)])
    def build_eval_tf(add_mask):
        return ToTensorWithMask(size=(256,256), add_mask_channel=add_mask)

    train_subset, val_subset, test_subset = stratified_split(dataset, val_ratio=0.1, test_ratio=0.1, random_state=args.seed)

    train_set = EvaluationDataset(train_subset, build_train_tf(args.add_mask_channel))
    val_set   = EvaluationDataset(val_subset,   build_eval_tf(args.add_mask_channel))
    test_set  = EvaluationDataset(test_subset,  build_eval_tf(args.add_mask_channel))

    # seeding for workers
    def seed_worker(worker_id):
        wseed = (args.seed + worker_id) % (2**32 - 1)
        random.seed(wseed); np.random.seed(wseed); torch.manual_seed(wseed)
    g = torch.Generator(); g.manual_seed(args.seed)
    pw = args.num_workers > 0

    def make_train_loader(weighted=True):
        if weighted:
            labels = np.array([train_subset.dataset.targets[i] for i in train_subset.indices])
            class_cnt = np.bincount(labels, minlength=num_classes)
            sample_weights = 1.0 / np.maximum(class_cnt[labels], 1)
            sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            return DataLoader(train_set, batchsize := args.batchsize, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=pw, worker_init_fn=seed_worker, generator=g)
        else:
            return DataLoader(train_set, batch_size=args.batchsize, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=pw, worker_init_fn=seed_worker, generator=g)

    train_loader = make_train_loader(weighted=args.use_weighted_sampler)
    val_loader   = DataLoader(val_set, batch_size=args.batchsize, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=pw, worker_init_fn=seed_worker, generator=g)
    test_loader  = DataLoader(test_set, batch_size=args.batchsize, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=pw, worker_init_fn=seed_worker, generator=g)

    # model
    in_ch = 4 if args.add_mask_channel else 3
    encoder = ResNetEncoder(args.model_name, use_attention=attn, freeze_layers=args.freeze_layers, in_ch=in_ch)
    model = HybridClassifier(encoder, emb_dim=512, num_classes=num_classes,
                             use_ce=args.use_ce, use_coral=args.use_coral).to(args.device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                           patience=3, verbose=True, min_lr=1e-6)
    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None

    best_ema_val_acc = 0.0
    best_online_val_acc = 0.0
    patience_counter = 0
    global_step = 0

    use_amp = (args.device.startswith('cuda') and torch.cuda.is_available())
    scaler = torch.amp.GradScaler('cuda') if use_amp else torch.amp.GradScaler(enabled=False)

    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        # Optional switch from WRS to natural sampling
        if args.use_weighted_sampler and args.switch_off_sampler_epoch > 0 and (epoch+1) == args.switch_off_sampler_epoch:
            print(f"[Info] Switch train loader to natural sampling at epoch {epoch+1}")
            train_loader = make_train_loader(weighted=False)

        pbar = tqdm(train_loader, ncols=100, desc=f'[Train] {epoch+1}/{args.epochs}')
        for x, y in pbar:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                emb, logits_ce, logits_coral = model(x)
                loss = 0.0
                if args.use_ce:
                    loss_ce = F.cross_entropy(logits_ce, y)
                    loss = loss + args.ce_weight * loss_ce
                if args.use_coral:
                    tg = coral_targets(y, num_classes)
                    loss_coral = F.binary_cross_entropy_with_logits(logits_coral, tg)
                    loss = loss + args.coral_weight * loss_coral
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer); scaler.update()

            # EMA update (delayed start)
            if ema is not None and (epoch + 1) >= args.ema_start_epoch:
                ema.update(model)

            running += float(loss)
            writer.add_scalar('train/loss', float(loss), global_step)
            global_step += 1
            pbar.set_postfix(loss=float(loss))
        writer.add_scalar('train/epoch_avg_loss', running / max(1, len(train_loader)), epoch)

        # -------- Validation: ONLINE drives scheduler/early-stop; EMA for checkpoint ----------
        val_on  = evaluate(model,   val_loader, args.device, num_classes,
                   ce_w=args.infer_ce_weight, coral_w=args.infer_coral_weight)
        val_ema = evaluate(ema.ema, val_loader, args.device, num_classes,
                   ce_w=args.infer_ce_weight, coral_w=args.infer_coral_weight) if ema is not None else val_on

        writer.add_scalar('val_online/acc', val_on['accuracy'], epoch)
        writer.add_scalar('val_online/f1',  val_on['f1'], epoch)
        writer.add_scalar('val_ema/acc',    val_ema['accuracy'], epoch)
        writer.add_scalar('val_ema/f1',     val_ema['f1'], epoch)

        # LR schedule uses ONLINE acc
        scheduler.step(val_on['accuracy'])

        print(f"[Epoch {epoch+1}] Online Acc={val_on['accuracy']:.4f} F1={val_on['f1']:.4f} "
              f"| EMA Acc={val_ema['accuracy']:.4f} F1={val_ema['f1']:.4f} "
              f"QWK(EMA)={val_ema['qwk']:.4f} mAP(EMA)={val_ema['mAP']:.4f}")

        best_online_val_acc = max(best_online_val_acc, val_on['accuracy'])

        metric_for_ckpt = val_ema['accuracy']  # choose EMA as selection metric
        if metric_for_ckpt > best_ema_val_acc + args.delta:
            best_ema_val_acc = metric_for_ckpt
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pt'))             # online
            if ema is not None:
                torch.save(ema.ema.state_dict(), os.path.join(log_dir, 'best_model_ema.pt'))   # ema
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[Info] Early stopping at epoch {epoch+1}")
                break

    # -------- Test (prefer EMA) ----------
    test_model = model if (args.test_use_online or ema is None) else ema.ema
    test_res = evaluate(test_model, test_loader, args.device, num_classes,
                    ce_w=args.infer_ce_weight, coral_w=args.infer_coral_weight)
    print("========== Test Report ==========")
    print(f"Test Acc={test_res['accuracy']:.4f} F1={test_res['f1']:.4f} "
          f"QWK={test_res['qwk']:.4f} mAP={test_res['mAP']:.4f}")
    print(classification_report(
        test_res['y_true'], test_res['y_pred'],
        target_names=dataset.classes, digits=4, zero_division=0
    ))

    # Confusion matrix
    try:
        fig = plt.figure(figsize=(6,5))
        sns.heatmap(test_res['cm'], annot=True, fmt='d', cmap='Blues')
        plt.title('Test Confusion Matrix'); plt.tight_layout()
        fig.savefig(os.path.join(log_dir, 'test_confusion_matrix.png')); plt.close(fig)
    except Exception:
        pass

    # Embedding viz
    try:
        visualize_embeddings(test_res['emb'], test_res['labels'], writer, 0, tag='test')
    except Exception:
        pass

    # -------- CSV (ablation log) ----------
    elapsed_min = (time.time() - start_time) / 60.0
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    ablation_row = {
        'exp_dir': log_dir,
        'model': args.model_name,
        'attention': args.attention,
        'mask4ch': int(args.add_mask_channel),
        'sampler': int(args.use_weighted_sampler),
        'ema': int(args.use_ema),
        'tta': 0,  # always 0 in V7 (No-TTA)
        'ce': int(args.use_ce), 'coral': int(args.use_coral),
        'epochs': args.epochs, 'bs': args.batchsize, 'lr': args.lr,
        'val_online_best_acc': round(best_online_val_acc, 4),
        'val_ema_best_acc': round(best_ema_val_acc, 4),
        'test_acc': round(test_res['accuracy'], 4),
        'test_f1': round(test_res['f1'], 4),
        'test_qwk': round(test_res['qwk'], 4),
        'test_mAP': round(test_res['mAP'], 4),
        'ema_decay': args.ema_decay,
        'ema_start_epoch': args.ema_start_epoch,
        'switch_off_sampler_epoch': args.switch_off_sampler_epoch,
        'mins': round(elapsed_min, 2),
        'params_M': round(total_params, 2),
        'seed': args.seed,
        'max_rotate': args.max_rotate,
        'use_vertical_flip': int(args.use_vertical_flip),
        'ce_weight': args.ce_weight,
        'coral_weight': args.coral_weight
    }
    csv_path = os.path.join(args.log_root, 'ablation_results.csv')
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(ablation_row.keys()))
        if not file_exists:
            writer_csv.writeheader()
        writer_csv.writerow(ablation_row)
    print(f"[Ablation] appended results to {csv_path}\nRow: {ablation_row}")

if __name__ == '__main__':
    main()
