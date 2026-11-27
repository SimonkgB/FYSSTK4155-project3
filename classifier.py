import copy

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import N_CLASSES, Config
from GravitySpyLoader import GlitchDataset




class Trainer:
    def __init__(self, config: Config, class_weights: torch.Tensor | None = None, optimizer: str = "adam"):
        # tried to rewrite the example func in Networks.py to a class for propper testing, no longer a pofc
        self.config = config
        self.class_weights = class_weights
        self.optimizer_name = optimizer
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
        self.best_epoch = 0
        self.loss_fn: nn.Module = nn.CrossEntropyLoss()
        self.optimizer: torch.optim.Optimizer | None = None

    def make_optimizer(self, model: nn.Module):
        p = model.parameters()
        lr = self.config.lr
        wd = self.config.weight_decay
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(p, lr=lr, weight_decay=wd)
        if self.optimizer_name == "adam":
            return torch.optim.Adam(p, lr=lr, weight_decay=wd)

    def train_epoch(self, model: nn.Module, loader) -> float:
        """One run over the training data"""
        model.train()
        total = 0.0
        n = 0
        for xb, yb in loader:
            self.optimizer.zero_grad()
            loss = self.loss_fn(model(xb), yb)
            loss.backward()
            self.optimizer.step()
            total += loss.item() * len(yb)
            n += len(yb)
        return total / n

    def evaluate(self, model: nn.Module, loader) -> tuple[float, float]:
        """(loss, accuracy) not using gradient updates."""
        model.eval()
        total = 0.0
        correct = 0
        n = 0
        with torch.no_grad():
            for xb, yb in loader:
                logits = model(xb)
                total += self.loss_fn(logits, yb).item() * len(yb)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                n += len(yb)
        return total / n, correct / n

    def fit(self, model: nn.Module, train: GlitchDataset, val: GlitchDataset | None = None) -> nn.Module:
        """this is actual full loop with val"""
        w = self.class_weights
        self.loss_fn = nn.CrossEntropyLoss(weight=w)
        self.optimizer = self.make_optimizer(model)
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

        train_loader = DataLoader(train, batch_size=self.config.batch_size, shuffle=True)

        val_loader = DataLoader(val, batch_size=256)

        best_loss, best_state, wait = float("inf"), None, 0
        for epoch in range(1, self.config.epochs + 1):
            tr_loss = self.train_epoch(model, train_loader)
            self.history["train_loss"].append(tr_loss)
            if False:
                print(f"epoch {epoch}: train loss {tr_loss:.3}")


            va_loss, va_acc = self.evaluate(model, val_loader)
            self.history["val_loss"].append(va_loss)
            self.history["val_acc"].append(va_acc)
            if True:
                print(f"epoch {epoch}: train loss {tr_loss:.3}, val loss {va_loss:.3}, val acc {va_acc:.3}")

            if va_loss < best_loss:
                best_loss = va_loss
                wait = 0
                self.best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            else:
                wait += 1

        if best_state is not None:
            model.load_state_dict(best_state)
        return model



# WHAST NEXT
# need classifiers for sklearn and torch
# sklearn for teh grad boost and/or forrest (dunno what i wanna do yet). Torch for what we already have,
# do  torch first then

# i dont know if we are going ot need to do large amount of parallelisation, we need to see how long propper runs
# takes when we get to that point




if __name__ == "__main__":
    from config import set_seed
    from GravitySpyLoader import GravitySpyData, build_datasets, class_weights
    from Networks import LogisticRegressionNet

    config = Config(views=("1.0",), pool=4, epochs=30, lr=1e-3, weight_decay=1e-4)
    set_seed(config.seed)
    data = GravitySpyData(config)
    data.load()
    train, val, test = build_datasets(data, flat=True, pool=config.pool)
    w = class_weights(train.y.numpy())

    model = LogisticRegressionNet(n_features=train.x.shape[1])
    trainer = Trainer(config, class_weights=w, optimizer="adam") # optimizer="sgd" doesnt work yet see:: AttributeError: 'NoneType' object has no attribute'zero_grad'
    trainer.fit(model, train, val)

    print(f"best epoch: {trainer.best_epoch}")
    print(f"val loss per epoch: {trainer.history["val_loss"]}")
    print(f"val acc per epoch: {trainer.history["val_acc"]}")




"""

❯ python3 classifier.py
epoch 1: train loss 1.6963558262884781, val loss 1.0864953021800265, val acc 0.8417021276595744
epoch 2: train loss 0.8612558974305723, val loss 0.8365305653024228, val acc 0.8238297872340425
epoch 3: train loss 0.601496859885232, val loss 0.7313522249587039, val acc 0.8834042553191489
epoch 4: train loss 0.4894095392826387, val loss 0.689999872775788, val acc 0.8782978723404256
epoch 5: train loss 0.39647464701297935, val loss 0.65026376957589, val acc 0.8774468085106383
epoch 6: train loss 0.3430459645018869, val loss 0.6317625338980493, val acc 0.8910638297872341
epoch 7: train loss 0.2963876670831635, val loss 0.617967342011472, val acc 0.8817021276595745
epoch 8: train loss 0.2582656529789904, val loss 0.6045345772073624, val acc 0.9012765957446809
epoch 9: train loss 0.23193355760164525, val loss 0.621425742494299, val acc 0.8995744680851064
epoch 10: train loss 0.20944686782858146, val loss0.592406180665848, val acc 0.8987234042553192
epoch 11: train loss 0.18932951896814762, val loss0.606090325294657, val acc 0.9046808510638298
epoch 12: train loss 0.17154807178516931, val loss0.6190989518672861, val acc 0.8868085106382979
epoch 13: train loss 0.15966302032749885, val loss0.6000808022884613, val acc 0.9055319148936171
epoch 14: train loss 0.15488623965730242, val loss0.6186495438027889, val acc 0.9055319148936171
epoch 15: train loss 0.13843691678892153, val loss0.6048692340039192, val acc 0.9097872340425532
epoch 16: train loss 0.13675973349620685, val loss0.6461512894326068, val acc 0.902127659574468
epoch 17: train loss 0.12499677734126947, val loss0.6146417926220183, val acc 0.9072340425531915
epoch 18: train loss 0.11615779655269591, val loss0.6095401427086363, val acc 0.9097872340425532
epoch 19: train loss 0.10648051558469643, val loss0.6314582701946827, val acc 0.9072340425531915
epoch 20: train loss 0.10260090311903514, val loss0.6308781453396411, val acc 0.9029787234042553
epoch 21: train loss 0.09588615807373707, val loss0.6382288018693315, val acc 0.9080851063829787
epoch 22: train loss 0.09300226031019068, val loss0.6586964703620748, val acc 0.9072340425531915
epoch 23: train loss 0.0885848555367782, val loss 0.6555174348709432, val acc 0.9063829787234042
epoch 24: train loss 0.09808917804140736, val loss0.645235625530811, val acc 0.9063829787234042
epoch 25: train loss 0.08129526394650965, val loss0.6526250736764136, val acc 0.9097872340425532
epoch 26: train loss 0.07744422470251348, val loss0.6446199419143352, val acc 0.9080851063829787
epoch 27: train loss 0.07018313069624933, val loss0.6622765631371356, val acc 0.9038297872340425
epoch 28: train loss 0.0689163707157118, val loss 0.6645976699666774, val acc 0.9097872340425532
epoch 29: train loss 0.06850641866197336, val loss0.6769564401342514, val acc 0.9063829787234042
epoch 30: train loss 0.0626435424215697, val loss 0.6733562572966231, val acc 0.902127659574468
# these results look comparable to the test func in networks, id say its valid then
best epoch: 10
val loss per epoch: [1.0864953021800265, 0.8365305653024228, 0.7313522249587039, 0.689999872775788, 0.65026376957589, 0.6317625338980493, 0.617967342011472, 0.6045345772073624, 0.621425742494299, 0.592406180665848, 0.606090325294657, 0.6190989518672861, 0.6000808022884613, 0.6186495438027889, 0.6048692340039192, 0.6461512894326068, 0.6146417926220183, 0.6095401427086363, 0.6314582701946827, 0.6308781453396411, 0.6382288018693315, 0.6586964703620748, 0.6555174348709432, 0.645235625530811, 0.6526250736764136, 0.6446199419143352, 0.6622765631371356, 0.6645976699666774, 0.6769564401342514, 0.6733562572966231]
val acc per epoch: [0.8417021276595744, 0.8238297872340425, 0.8834042553191489, 0.8782978723404256, 0.8774468085106383, 0.8910638297872341, 0.8817021276595745, 0.9012765957446809, 0.8995744680851064, 0.8987234042553192, 0.9046808510638298, 0.8868085106382979, 0.9055319148936171, 0.9055319148936171, 0.9097872340425532, 0.902127659574468, 0.9072340425531915, 0.9097872340425532, 0.9072340425531915, 0.9029787234042553, 0.9080851063829787, 0.9072340425531915,0.9063829787234042, 0.9063829787234042, 0.9097872340425532, 0.9080851063829787, 0.9038297872340425, 0.9097872340425532, 0.9063829787234042, 0.902127659574468]


"""