import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from config import Config, DURATIONS, BLANK_STD, N_CLASSES
from transforms import Standardize, AvgPoolFlatten

# we notice that some of the classified samples contains at least one view
# where every pixel sits at the image minimum  and are completly empty.
# I have not gotten into why this is atm.
class GravitySpyData:
    def __init__(self, config: Config):
        self.config = config
        self.classes: list[str] = []
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}  # given split gives cuts: (images, labels, ids)
        self.removed: list[tuple[str, str, str]] = []  # (class, split, id) of dropped blank samples

    @staticmethod
    def is_blank(views: np.ndarray, threshold: float = BLANK_STD) -> np.ndarray:
        """If an image is uniform or close to, the SD of the values
        in all pixels will be low, and will be cut out
        returns (4,N) """
        return views.std(axis=(-2, -1)) < threshold

    def _keep(self, all_views: np.ndarray) -> bool:
        """using any or all on the (4,N) we can cut samples that contain one uniform
        image or all four (over the tiem windows)"""
        blank = self.is_blank(all_views)
        #print(blank)
        if self.config.blank_rule == "any":
            return not blank.any()
        if self.config.blank_rule == "all":
            return not blank.all()

    def load(self) -> None:
        """Read all splits into memory as (N, 4, 140, 170).
        Blank checks always use all 4 views, so every experiment keeps the same samples.

        Independant on the set self.view we load all 4 windows to have code omly read that
        file ones and then do as we please for any other combiantion of windows."""
        idx = []
        for v in self.config.views:
            idx.append(DURATIONS.index(v))
        #print(idx)
        self.data = {}
        self.removed = []
        with h5py.File(self.config.h5_path, "r") as f:
            self.classes = sorted(f.keys())
            #print(self.classes)
            for split in ("train", "validation", "test"):
                images = []
                labels = []
                ids = []
                for y, cls in enumerate(self.classes): # ["Extremely_Loud",])
                    g = f[cls][split]
                    for eid in sorted(g):
                        all_views = np.stack([g[eid][f"{d}.png"][0] for d in DURATIONS])  # (4, 140, 170)
                        if not self._keep(all_views):
                            self.removed.append((cls, split, eid))
                            continue
                        images.append(all_views[idx])
                        labels.append(y)
                        ids.append(eid)
                self.data[split] = (np.stack(images).astype(np.float32),
                                    np.array(labels,dtype=np.int64),
                                    np.array(ids))


    def split(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        images, labels, _ = self.data[name]
        return images, labels

    def blank_report(self):
        """Number of removed blank samples per class and split (for the report)"""
        import pandas as pd
        df = pd.DataFrame(self.removed, columns=["label", "split", "id"])
        return df.groupby(["label", "split"]).size().unstack(fill_value=0)




class GlitchDataset(Dataset):
    def __init__(self, x: torch.Tensor, y, transform=None):
        self.x = x
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x[i]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[i]

    def numpy(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x.flatten(1).numpy(), self.y.numpy()


def class_weights(y: np.ndarray, n_classes: int = N_CLASSES) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes)
    return torch.tensor(len(y) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)



def build_datasets(data, flat: bool, pool: int = 4, train_transform=None):
    """Only need to read h5 file once, contruct
    Train/validation/test GlitchDatasets from a loaded GravitySpyData."""
    splits = ("train", "validation", "test")
    raw = {s: data.split(s) for s in splits}
    scaler = Standardize().fit(torch.from_numpy(raw["train"][0]))
    pool_flat = AvgPoolFlatten(pool)
    out = []
    for s in splits:
        x = scaler(torch.from_numpy(raw[s][0]))
        if flat:
            x = pool_flat(x)

        if s=="train":
            innp = train_transform
        else:
            innp = None
        out.append(GlitchDataset(x, raw[s][1], innp))
    return tuple(out)






if __name__=="__main__":
    conf = Config(views=("0.5", "1.0", "4.0"))
    load_data = GravitySpyData(conf)
    load_data.load()
    images, labels = load_data.split("train")
    print(load_data.blank_report())
    print(images.shape, images.dtype)   # (5457, 3, 140, 170) float32
    print(labels.shape, labels.dtype)   # (5457,) int64
    import matplotlib.pyplot as plt

    for i in range(images.shape[0]):
        plt.figure()
        plt.imshow(images[i,2,:,:])
        plt.colorbar()
        plt.savefig(f"{i}_3.png")

