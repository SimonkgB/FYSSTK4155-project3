# LLM-assisted (Claude, Anthropic): network classes, make_activation, n_params and the examples.
from typing import cast

import torch
from torch import nn

from config import N_CLASSES
from transforms import Standardize, AvgPoolFlatten








ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def make_activation(name: str, **kwargs): # feed as many at a time as you want, i hope :|
    return ACTIVATIONS[name](**kwargs)

import numpy as np
class LogisticRegressionNet(nn.Module):
    def __init__(self, n_features: int, n_classes: int=N_CLASSES):
        super().__init__()
        self.flatten = nn.Flatten()
        # z = x W^T + b with W (n_classes, n_features), b (n_classes,)
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print(self.flatten(x).shape)
        return self.linear(self.flatten(x))   # (B, n_classes) logits


class FFNN(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...],
                 activation: str, n_classes: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(make_activation(activation))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))     # output logits no activation
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.flatten(x))





    # Next: make the CNN and the 2d conv and activation block




def skibedi():
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import balanced_accuracy_score
    from config import Config, set_seed
    from GravitySpyLoader import GravitySpyData

    set_seed(2026)
    data = GravitySpyData(Config(views=("1.0",)))
    data.load()
    X_tr, y_tr = data.split("train")
    X_va, y_va = data.split("validation")

    # Standardise with training statistics
    # mean = X_tr.mean()
    # std = X_tr.std()
    # x_tr = torch.from_numpy((X_tr - mean) / std).flatten(1)
    # x_va = torch.from_numpy((X_va - mean) / std).flatten(1)
    X_tr_t = torch.from_numpy(X_tr)
    X_va_t = torch.from_numpy(X_va)
    standard = Standardize().fit(X_tr_t)   # statistics from TRAIN only
    pool_flat = AvgPoolFlatten(4)
    x_tr = pool_flat(standard(X_tr_t))
    x_va = pool_flat(standard(X_va_t))  # with k=4 torch.Size([5457, 1470]),,, ((35*42))
    print(x_tr.shape)

    t_tr = torch.from_numpy(y_tr)
    #print(y_tr)

    # class inballance
    counts = np.bincount(y_tr)
    # print(counts)
    # print(len(counts))
    # print(len(y_tr))
    w = torch.tensor(len(y_tr) / (N_CLASSES * counts), dtype=torch.float32) # the initial weights should allow for each to be eaqually probable and we must thus amplify the least numerous and downscale the most numeros  labels
    # must give dtype, otherwise it  loss_fn reads double and gets sad
    # print(w)

    model = LogisticRegressionNet(n_features=x_tr.shape[1])
    loss_fn = nn.CrossEntropyLoss(weight=w) # softmax plus nll, the loss is multipoed by the weight of each class so rare counts more.
    # ridge penelisation
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(x_tr, t_tr), batch_size=64, shuffle=True)
    #print(loader)
    #print(model.parameters())
    for epoch in range(0, 31):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad() # reset the gradients, as torch axxumulates them
            loss = loss_fn(model(xb), yb)
            loss.backward() # run the backwards propegation and fill .grad for all params
            opt.step() # updates the params using the last lines gradients
            total += loss.item() * len(yb)
        if epoch % 5 == 0:
            model.eval() # must run this before validating
            with torch.no_grad():   # skipp tracking nthe gradients when evaluate, which uses less memory
                pred = model(x_va).argmax(dim=1).numpy()
            acc = (pred == y_va).mean()
            bacc = balanced_accuracy_score(y_va, pred)  # the mean of the accuracies so certain classes (liek blimp) dont dominate
            print(f"epoch {epoch}: train loss {total / len(y_tr)}, val acc {acc} val balanced acc {bacc}\n")

    param_sum = 0
    for p in model.parameters():
        param_sum+= p.numel()

    print(f"parameters: ",param_sum)
    return model





# simple test to see if all activations are created and applied correclty, should print (input(ie. 8), 22)
def shapes():
    x_flat = torch.randn(8, 1470)
    tests = [
        ("LR", LogisticRegressionNet(1470), x_flat),
        ("FFNN", FFNN(1470, (25,100, 50), "relu", N_CLASSES), x_flat),
    ]
    for name, model, x in tests:
        out = model(x)
        if out.requires_grad:
            sum_aaa = 0
            for p in model.parameters():
                sum_aaa += p.numel()
        print(f"{name}:  out {tuple(out.shape)}  params {sum_aaa}")


if __name__ == "__main__":
    # skibedi()
    shapes()



"""
❯ python3 Networks.py
epoch 0: train loss 2.060758862108544, val acc 0.7668085106382979 val balanced acc 0.7480057302089729

epoch 5: train loss 0.12814375396598787, val acc 0.865531914893617 val balanced acc 0.7900512063810351

epoch 10: train loss 0.047377082051267956, val acc 0.8663829787234043 val balanced acc 0.7655380468935511

epoch 15: train loss 0.23039501637078047, val acc 0.854468085106383 val balanced acc 0.7645372049345472

epoch 20: train loss 0.007342166340134991, val acc 0.8757446808510638 val balanced acc 0.7701645228140733

epoch 25: train loss 0.6487761540781275, val acc 0.8348936170212766 val balanced acc 0.7799289111790827

epoch 30: train loss 0.06734093562357575, val acc 0.8663829787234043 val balanced acc 0.7725811053399734

parameters:  523622
"""