import numpy as np
import h5py
import matplotlib.pyplot as plt

def load_split(path, split, durations=("0.5", "1.0", "2.0", "4.0")):
    X, y = [], []
    with h5py.File(path, "r") as f:
        classes = sorted(f.keys())
        for ci, cls in enumerate(classes):
            for sid in f[cls][split]:
                g = f[cls][split][sid]
                imgs = []
                for d in durations:
                    imgs.append(g[f"{d}.png"][0])
                X.append(np.stack(imgs))  # (n_dur, 140, 170)
                y.append(ci)
    return np.array(X), np.array(y), classes


path = "trainingsetv1d1.h5"
Xtr, ytr, classes = load_split(path, "train")
Xva, yva, _ = load_split(path, "validation")
Xte, yte, _ = load_split(path, "test")
print(Xtr.shape, np.bincount(ytr))  # (5587, 4, 140, 170) [ 229   59   41 1274   41
# 316  195  498  361  437  315  107   57   19 314  185  308  237   73  284   29  208]

plt.imshow(Xtr[0, 1], aspect="auto", origin="lower")
plt.title(classes[ytr[0]])
plt.show()

#   idx	class	            n
#   0	1080Lines	        229
#   1	1400Ripples	        59
#   2	Air_Compressor	    41
#   3	Blip	            1274
#   4	Chirp	            41
#   5	Extremely_Loud	    316
#   6	Helix	            195
#   7	Koi_Fish	        498
#   8	Light_Modulation	361
#   9	Low_Frequency_Burst	437
#   10	Low_Frequency_Lines	315
#   11	No_Glitch	        107
#   12	None_of_the_Above	57
#   13	Paired_Doves	    19
#   14	Power_Line	        314
#   15	Repeating_Blips	    185
#   16	Scattered_Light	    308
#   17	Scratchy	        237
#   18	Tomte	            73
#   19	Violin_Mode	        284
#   20	Wandering_Line	    29
#   21	Whistle	            208