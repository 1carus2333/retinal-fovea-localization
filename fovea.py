"""Fovea localization: training and inference are explicit CLI commands only."""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_index(folder):
    result = {}
    for p in sorted(Path(folder).rglob('*')):
        if p.suffix.lower() in EXTENSIONS:
            keys = [p.name.lower(), p.stem.lower()]
            if p.stem.isdecimal():
                keys.append(str(int(p.stem)))
            for key in keys:
                if key in result and result[key] != p:
                    raise ValueError(f'Ambiguous image name: {key}')
                result[key] = p
    if not result:
        raise ValueError(f'No images found in {folder}')
    return result


def resolve(index, name):
    name = str(name).replace('\\', '/').split('/')[-1].lower()
    if name not in index:
        raise ValueError(f'Image missing: {name}')
    return index[name]


def labels(root):
    df = pd.read_csv(root / 'fovea_localization_train_GT.csv')
    required = ['data', 'Fovea_X', 'Fovea_Y']
    if not set(required).issubset(df.columns):
        raise ValueError(f'Expected columns: {required}')
    if df['data'].duplicated().any():
        raise ValueError('Duplicate training image names')
    index = image_index(root / 'train')
    rows = []
    for row in df[required].itertuples(index=False, name=None):
        path = resolve(index, row[0])
        x, y = float(row[1]), float(row[2])
        with Image.open(path) as im:
            w, h = im.size
        if not np.isfinite([x, y]).all() or x < 0 or y < 0 or x >= w or y >= h:
            raise ValueError(f'Invalid coordinates for {path}: {(x, y)}')
        rows.append((str(path), x, y, w, h))
    return rows


class FundusDataset(Dataset):
    def __init__(self, rows, size, augment=False):
        self.rows, self.size, self.augment = rows, size, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, x, y, w, h = self.rows[i]
        visible = float(x != 0 or y != 0)
        xy = np.array([x / w, y / h], dtype=np.float32)
        with Image.open(path) as source:
            im = source.convert('RGB').resize((self.size, self.size), Image.Resampling.BILINEAR)
        if self.augment:
            if random.random() < .5:
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if visible:
                    xy[0] = (w - 1 - x) / w
            im = ImageEnhance.Brightness(im).enhance(random.uniform(.8, 1.2))
            im = ImageEnhance.Contrast(im).enhance(random.uniform(.8, 1.2))
        a = np.asarray(im, dtype=np.float32).copy() / 255
        tensor = torch.from_numpy(a).permute(2, 0, 1)
        tensor = (tensor - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
        return tensor, torch.from_numpy(xy), torch.tensor(visible), torch.tensor([w, h], dtype=torch.float32)


class Locator(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.heatmap = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 1, 1))
        self.visibility = nn.Linear(512, 1)

    def forward(self, x):
        feature = self.encoder(x)
        logits = self.heatmap(F.interpolate(feature, scale_factor=4, mode='bilinear', align_corners=False))[:, 0]
        b, h, w = logits.shape
        prob = logits.flatten(1).softmax(1).view(b, h, w)
        ys = (torch.arange(h, device=x.device) + .5) / h
        xs = (torch.arange(w, device=x.device) + .5) / w
        xy = torch.stack([(prob.sum(1) * xs).sum(1), (prob.sum(2) * ys).sum(1)], 1)
        return xy, self.visibility(feature.mean((2, 3))).squeeze(1), logits


def loss_fn(xy, vis, heat, target, visible):
    mask = visible.bool()
    loss = F.binary_cross_entropy_with_logits(vis, visible)
    if mask.any():
        h, w = heat.shape[-2:]
        yy, xx = torch.meshgrid((torch.arange(h, device=heat.device) + .5) / h,
                                (torch.arange(w, device=heat.device) + .5) / w, indexing='ij')
        delta = (xx[None] - target[:, 0, None, None]) ** 2 + (yy[None] - target[:, 1, None, None]) ** 2
        gaussian = torch.exp(-delta / (2 * .025 ** 2)).flatten(1)
        gaussian = gaussian / gaussian.sum(1, keepdim=True).clamp_min(1e-8)
        loss = loss + 20 * F.mse_loss(xy[mask], target[mask])
        loss = loss - (gaussian[mask] * heat[mask].flatten(1).log_softmax(1)).sum(1).mean()
    return loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    coords, probs, targets, dims = [], [], [], []
    for image, target, visible, wh in loader:
        xy, vis, _ = model(image.to(device))
        coords.append(xy.cpu()); probs.append(vis.sigmoid().cpu())
        targets.append(target); dims.append(wh)
    return [torch.cat(a).numpy() for a in (coords, probs, targets, dims)]


def decode(xy, prob, wh, threshold):
    pixels = np.clip(xy * wh, 0, wh - 1)
    pixels[prob < threshold] = 0
    return pixels


def train(args):
    rows = labels(args.data)
    visible = np.array([r[1] != 0 or r[2] != 0 for r in rows], dtype=int)
    counts = np.bincount(visible, minlength=2)
    if len(rows) < args.folds:
        raise ValueError('Number of images must be >= folds')
    stratified = counts.min() >= args.folds
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed) if stratified else KFold(args.folds, shuffle=True, random_state=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob('fold_*.pt')):
        raise ValueError('Output already has checkpoints; choose a new --output directory')
    device = torch.device(args.device)
    oof = np.zeros((len(rows), 2)); truth = np.array([[r[1], r[2]] for r in rows])
    for fold, (tr, va) in enumerate(splitter.split(rows, visible)):
        seed_all(args.seed + fold)
        train_loader = DataLoader(FundusDataset([rows[i] for i in tr], args.size, True), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        valid_loader = DataLoader(FundusDataset([rows[i] for i in va], args.size), batch_size=args.batch_size, num_workers=args.workers)
        model = Locator(not args.no_pretrained).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        best, best_pred, history = float('inf'), None, []
        for epoch in range(args.epochs):
            model.train()
            # Small datasets/batches: keep pretrained BatchNorm statistics stable.
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
            total = 0.
            for image, target, vis, _ in tqdm(train_loader, desc=f'Fold {fold} epoch {epoch + 1}'):
                optimizer.zero_grad(set_to_none=True)
                xy, v, heat = model(image.to(device))
                loss = loss_fn(xy, v, heat, target.to(device), vis.to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                total += loss.item() * len(image)
            scheduler.step()
            xy, prob, target, wh = evaluate(model, valid_loader, device)
            pred = decode(xy, prob, wh, args.threshold)
            rmse = float(np.sqrt(np.mean((pred - target * wh) ** 2)))
            history.append({'epoch': epoch + 1, 'loss': total / len(tr), 'val_rmse_pixels': rmse})
            print(f'fold={fold} epoch={epoch + 1} RMSE={rmse:.4f}', flush=True)
            if rmse < best:
                best, best_pred = rmse, pred.copy()
                torch.save({'state': model.state_dict(), 'size': args.size, 'threshold': args.threshold, 'fold': fold, 'rmse': best}, args.output / f'fold_{fold}.pt')
            (args.output / f'history_{fold}.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        oof[va] = best_pred
    pd.DataFrame({'data': [Path(r[0]).name for r in rows], 'true_x': truth[:, 0], 'true_y': truth[:, 1], 'pred_x': oof[:, 0], 'pred_y': oof[:, 1]}).to_csv(args.output / 'oof.csv', index=False)
    metrics = {'oof_rmse_pixels': float(np.sqrt(np.mean((oof - truth) ** 2))), 'seed': args.seed, 'folds': args.folds}
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(metrics)


def predict(args):
    sample = pd.read_csv(args.data / 'sample_submission.csv', dtype={'ImageID': str})
    if list(sample.columns) != ['ImageID', 'value'] or sample['ImageID'].duplicated().any():
        raise ValueError('Expected unique ImageID and value columns')
    index = image_index(args.data / 'test')
    parsed, paths, axes = [], [], {}
    for key in sample.ImageID:
        match = re.fullmatch(r'(.+?)[_\s.-]*Fovea_([XY])', key, re.IGNORECASE)
        if not match:
            raise ValueError(f'Unsupported ImageID: {key}; expected image name + Fovea_X/Y')
        path, axis = resolve(index, match[1]), match[2].upper()
        if axis in axes.setdefault(path, set()):
            raise ValueError(f'Duplicate coordinate axis for {path}')
        axes[path].add(axis)
        if path not in paths:
            paths.append(path)
        parsed.append((path, 0 if axis == 'X' else 1))
    if any(a != {'X', 'Y'} for a in axes.values()) or set(paths) != set(index.values()):
        raise ValueError('Template must contain exactly X and Y for every test image')
    rows = []
    for path in paths:
        with Image.open(path) as im:
            w, h = im.size
        rows.append((str(path), 0, 0, w, h))
    checkpoints = sorted(args.output.glob('fold_*.pt'))
    if not checkpoints:
        raise FileNotFoundError('No trained fold checkpoints found')
    coords, probs, thresholds = [], [], []
    for path in checkpoints:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        model = Locator().to(args.device)
        model.load_state_dict(ckpt['state'])
        loader = DataLoader(FundusDataset(rows, ckpt['size']), batch_size=args.batch_size, num_workers=args.workers)
        xy, prob, _, wh = evaluate(model, loader, args.device)
        coords.append(xy); probs.append(prob); thresholds.append(ckpt['threshold'])
    result = decode(np.mean(coords, axis=0), np.mean(probs, axis=0), wh, float(np.mean(thresholds)))
    lookup = dict(zip(paths, result))
    sample['value'] = [float(lookup[p][axis]) for p, axis in parsed]
    if not np.isfinite(sample.value).all():
        raise ValueError('Non-finite predictions')
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.submission, index=False)
    print(f'Saved {len(sample)} rows to {args.submission}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'predict'])
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--submission', type=Path, default=Path('outputs/submission.csv'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--no-pretrained', action='store_true')
    args = parser.parse_args()
    if args.size < 64 or args.size % 32 or args.batch_size < 1 or args.epochs < 1 or args.folds < 2 or not 0 < args.threshold < 1:
        parser.error('size must be >=64 and divisible by 32; batch/epochs positive; folds>=2; threshold in (0,1)')
    if args.command == 'train':
        train(args)
    else:
        predict(args)


if __name__ == '__main__':
    main()
