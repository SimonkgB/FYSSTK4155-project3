import numpy as np
from config import Config,DURATIONS, BLANK_STD
import h5py

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
            print(v)
        print(idx)
        self.data = {}
        self.removed = []
        with h5py.File(self.config.h5_path, "r") as f:
            self.classes = sorted(f.keys())
            for split in ("train", "validation", "test"):
                images = []
                labels = []
                ids = []
                for y, cls in enumerate(self.classes):
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


# TODO:
# Consider savin the cuts to npy format for faster loading.




if __name__=="__main__":
    conf = Config(views=("0.5", "1.0", "4.0"))
    load_data = GravitySpyData(conf)
    load_data.load()
    images, labels = load_data.split("train")
    print(images.shape, images.dtype)   # (5457, 3, 140, 170) float32
    print(labels.shape, labels.dtype)   # (5457,) int64