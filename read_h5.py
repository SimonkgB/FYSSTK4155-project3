# pyright: reportIndexIssue=false, reportGeneralTypeIssues=false, reportArgumentType=false
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt



DURATIONS = ("0.5", "1.0", "2.0", "4.0")
F_MIN, F_MAX = 10.0, 2048.0
N_F, N_T = 140, 170
path_spy = "./trainingsetv1d1.h5"


def freq_axis(nf):
    return np.geomspace(F_MAX, F_MIN, nf)


def time_axis(duration):
    d = float(duration)
    return np.linspace(-d / 2, d / 2, N_T)




def get_event(path, meta, event_id=None, cls=None, split="train", index=0):
    # dunno if i wanna do event id or index, so remove either when that time comes
    if event_id is not None:
        cls, split = meta.loc[event_id, ["label", "sample_type"]]
    with h5py.File(path, "r") as f:
        if event_id is None:
            event_id = sorted(f[cls][split])[index]
        g = f[cls][split][event_id]
        images = {}
        for d in DURATIONS:
            images[d] = g[f"{d}.png"][0]
    time_ax = {}
    for d in DURATIONS:
        time_ax[d] = time_axis(d)
    return {
        "id": event_id,
        "class": cls,
        "split": split,
        "images": images,  # {duration: (140, 170) array}
        "freq": freq_axis(N_F),  # (140,) Hz
        "time": time_ax,  # {duration: (170,) s}
        "meta": meta.loc[event_id].to_dict(),
    }


def plot_event(ev):
    fig, axes = plt.subplots(1, len(DURATIONS), figsize=(16, 4.5), sharey=True, layout="constrained")
    f_edges = np.geomspace(F_MIN, F_MAX, N_F + 1)
    m = ev["meta"]
    to_max = []
    for img in ev["images"].values():
        to_max.append(img.max())
    vmax = max(to_max)
    mesh = None
    for ax, d in zip(axes, DURATIONS):
        t_edges = np.linspace(-float(d) / 2, float(d) / 2, N_T + 1)
        img = ev["images"][d][::-1]
        mesh = ax.pcolormesh(t_edges, f_edges, img, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_yscale("log")
        ax.set_title(f"{d} s window")
        ax.axhline(m["peak_frequency"], color="red", ls="--", lw=1.5)
    axes[0].set_ylabel("Frequency [Hz]")
    fig.supxlabel("Time relative to event [s]")
    fig.colorbar(mesh, ax=axes, label="Normalized energy [a.u.]")
    title = f"{ev['class']} ({ev['split']}) - {ev['id']}"
    print(f"{m['ifo']}, SNR {m['snr']:.1f}, peak {m['peak_frequency']:.0f} Hz, (red dashed), duration {m['duration']:.2f} s")
    fig.suptitle(title)
    return fig


def print_event_id(path):
    with h5py.File(path, "r") as f:
        for cls in sorted(f.keys()):
            event_id = sorted(f[cls]["train"])[0]
            print(f"{cls}: {event_id}")
print_event_id(path_spy)


meta = pd.read_csv("trainingset_v1d1_metadata.csv", index_col="gravityspy_id")
ev1 = get_event(path_spy, meta, event_id="08UZkPPyFQ")
ev2 = get_event(path_spy, meta, event_id=None, cls="Power_Line", split="train", index=10)
ev3 = get_event(path_spy, meta, event_id=None, cls="Extremely_Loud", split="train", index=10)
# ev = get_event(path_spy, meta, cls="Power_Line", split="train", index=0)

plot_event(ev1)
plot_event(ev2)
plot_event(ev3)
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