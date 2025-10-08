# test_model.py
import argparse, json, glob, os
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# import your existing builders
from mlp_model_classification import build_mlp
from cnn_model_classification import build_cnn

# ---------- Dataset ----------
class AnimationJsonDataset(Dataset):
    """
    Expects each JSON file like:
    {
      "label": "walking",             # <-- added in AnimationDataRecorder
      "frames": [{"time":..., "rotations":[...]} , ...]
    }
    For CNN: returns (C,T) tensor (rotations stacked as channels across time)
    For MLP: returns a single feature vector by temporal aggregation.
    """
    def __init__(self, paths, label2idx=None, model_type="cnn",
                 aggregate="mean"):  # aggregate: mean|flatten
        self.paths = paths
        self.model_type = model_type
        self.aggregate = aggregate
        self.samples = []

        # read all files, build label map if needed
        labels = []
        for p in self.paths:
            with open(p, "r") as f:
                js = json.load(f)
            lab = js.get("label", None)
            labels.append(lab)
            # collect rotations: shape (T, F)
            R = torch.tensor([fr["rotations"] for fr in js["frames"]], dtype=torch.float32)
            self.samples.append((R, lab, os.path.basename(p)))

        if label2idx is None:
            # produce deterministic label order (strings) ignoring None
            uniq = sorted(set([l for l in labels if l is not None]))
            self.label2idx = {l:i for i,l in enumerate(uniq)}
        else:
            self.label2idx = label2idx
        self.idx2label = {v:k for k,v in self.label2idx.items()}

        # infer feature dim
        if len(self.samples):
            self.feature_dim = self.samples[0][0].shape[1]
        else:
            self.feature_dim = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        R, lab, fname = self.samples[i]  # R: (T, F)
        # Prepare x depending on model type
        if self.model_type == "cnn":
            # Return as (C,F_time) = (F, T)
            x = R.transpose(0,1)  # (F, T)
        else:
            # MLP: aggregate over time
            if self.aggregate == "mean":
                x = R.mean(dim=0)          # (F,)
            elif self.aggregate == "flatten":
                x = R.reshape(-1)           # (T*F,)
            else:
                raise ValueError("aggregate must be mean|flatten")

        # Target (may be None)
        if lab is None:
            y = None
        else:
            y = torch.tensor(self.label2idx[lab], dtype=torch.long)
        return x, y, fname

# ---------- Evaluation ----------
@torch.no_grad()
def evaluate(model, loader, criterion=None, binary=False):
    model.eval()
    device = next(model.parameters()).device
    tot_loss, n = 0.0, 0
    correct = 0
    per_file = []
    for xb, yb, names in loader:
        xb = xb.to(device)
        logits = model(xb if xb.ndim > 2 else xb)  # supports both
        if yb[0] is not None:
            y = yb.to(device)
        else:
            y = None

        if criterion is not None and y is not None:
            loss = criterion(logits if not binary else logits.squeeze(1), y if not binary else y.float())
            bs = xb.size(0)
            tot_loss += loss.item() * bs
            n += bs

        if binary:
            probs = torch.sigmoid(logits.squeeze(1))
            preds = (probs >= 0.5).long()
            for i in range(xb.size(0)):
                per_file.append((names[i], float(probs[i].cpu()), int(preds[i].cpu())))
            if y is not None:
                correct += (preds == y).sum().item()
        else:
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            for i in range(xb.size(0)):
                per_file.append((names[i], probs[i].cpu().tolist(), int(preds[i].cpu())))
            if y is not None:
                correct += (preds == y).sum().item()

    metrics = {}
    if n > 0:
        metrics["loss"] = tot_loss / n
    if len(loader.dataset) > 0 and loader.dataset.samples[0][1] is not None:
        metrics["acc"] = correct / len(loader.dataset)
    return metrics, per_file

def parse_hidden(s: str):
    return [int(x) for x in s.split(",")] if s else []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp","cnn"], required=True)
    ap.add_argument("--hidden", type=str, default='256,128', help="The hidden dimensions of the model.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data_glob", default="*.json")
    ap.add_argument("--binary", action="store_true",
                    help="Binary classification (1 logit with BCE).")
    ap.add_argument("--num_classes", type=int, default=None,
                    help="Required for multi-class if label set > 2 can’t be inferred.")
    ap.add_argument("--aggregate", choices=["mean","flatten"], default="mean",
                    help="How to aggregate time for MLP.")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.data_glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.data_glob}")

    # Peek first file to infer dims
    js0 = json.load(open(paths[0],"r"))
    T0 = len(js0["frames"])
    F0 = len(js0["frames"][0]["rotations"])

    # Build dataset to get label map
    ds = AnimationJsonDataset(paths, model_type=args.model, aggregate=args.aggregate)
    label2idx = ds.label2idx
    num_labels = len(label2idx)

    if args.binary:
        out_dim = 1
    else:
        out_dim = args.num_classes or (num_labels if num_labels >= 2 else 2)

    # Build model  (mirror training setup)
    if args.model == "cnn":
        in_channels = F0
        model = build_cnn(input_channels=in_channels, output_dim=out_dim)
    else:
        if args.aggregate == "flatten":
            input_dim = T0 * F0
        else:
            input_dim = F0
        model = build_mlp(input_dim=input_dim, hidden_layers=parse_hidden(args.hidden), output_dim=out_dim, binary=args.binary, classification=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Choose criterion only if labels present (for loss)
    has_labels = ds.samples[0][1] is not None
    if has_labels:
        if args.binary:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = None

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    metrics, per_file = evaluate(model, loader, criterion, binary=args.binary)

    # Print summary
    if metrics:
        print({k: round(v,4) for k,v in metrics.items()})
    print("label_map:", label2idx)
    for row in per_file:
        print("pred:", row)  # (filename, prob / prob_list, pred_idx)

    # Optional: confusion matrix if labels exist
    if has_labels:
        y_true, y_pred = [], []
        for (x, y, n) in ds:
            if y is None: continue
            if args.model == "cnn":
                x = x.unsqueeze(0).to(device)  # (1, C, T)
            else:
                x = x.unsqueeze(0).to(device)  # (1, F) or (1, T*F)
            logits = model(x)
            if args.binary:
                pred = int((torch.sigmoid(logits.squeeze(1)) >= 0.5).item())
            else:
                pred = int(torch.argmax(logits, dim=1).item())
            y_true.append(int(y))
            y_pred.append(pred)
        from collections import defaultdict
        cm = defaultdict(lambda: defaultdict(int))
        for t,p in zip(y_true,y_pred):
            cm[t][p]+=1
        print("confusion_matrix:", dict((int(k), dict(v)) for k,v in cm.items()))

if __name__ == "__main__":
    main()
