import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pymatgen.core import Element

from mlip_arena.models import REGISTRY

DATA_DIR = Path(__file__).parent

mlip_models = ["MACE-MP(M)", "MatterSim", "ORBv2", "M3GNet", "CHGNet", "SevenNet"]

fcc_pbe = pd.read_csv(DATA_DIR / "Table-A1-fcc.csv")
hcp_pbe = pd.read_csv(DATA_DIR / "Table-A2-hcp.csv")

# fcc

# Initialize an empty DataFrame
results_df = pd.DataFrame(columns=["symbol", "model", "fit_path", "fit_energies"])

for model in mlip_models:
    out_dir = Path(REGISTRY[model]["family"])

    for index, row in fcc_pbe.iterrows():
        symbol = row["symbol"]

        if Element(symbol.split("_")[0]).is_noble_gas:
            continue

        files = glob.glob(str(out_dir / f"{model}-fcc-{symbol.split('_')[0]}108.pkl"))
        if len(files) == 0:
            print("skip", model, symbol)
            # Add missing data to the DataFrame
            # if symbol not in results_df['symbol'].values:
            # Create a new row if the symbol is not yet in the DataFrame
            new_row = {
                "symbol": symbol,
                "model": model,
                "pbe_e_vacmig": row["e_vacmig"],
                "fit_path": [],
                "fit_energies": [],
            }
            results_df = pd.concat(
                [results_df, pd.DataFrame([new_row])], ignore_index=True
            )
            continue
        file = files[0]
        with open(file, "rb") as f:
            result = pickle.load(f)

        # Add data to the DataFrame
        # if symbol not in results_df['symbol'].values:
        # Create a new row if the symbol is not yet in the DataFrame
        forcefit = result["neb"]["forcefit"]
        new_row = {
            "symbol": symbol,
            "model": model,
            "pbe_e_vacmig": row["e_vacmig"],
            "fit_path": forcefit.fit_path,
            "fit_energies": forcefit.fit_energies,
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)


nrows = 2
ncols = len(mlip_models) // nrows

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(6, 4),
    sharex=True,
    sharey=True,
    constrained_layout=True,
    dpi=300,
)

for i, (ax, model) in enumerate(zip(axes.ravel(), mlip_models, strict=False)):
    filtered_df = results_df[results_df["model"] == model]

    asymmetries = []
    middle_deviations = []

    for index, row in filtered_df.iterrows():
        if len(row["fit_path"]) == 0 or pd.isna(row["pbe_e_vacmig"]):
            continue

        x = row["fit_path"] / max(row["fit_path"])
        y = row["fit_energies"] / row["pbe_e_vacmig"]

        # middle_idx = np.argmin(np.abs(x - 0.5))

        left_side = y[x <= 0.5]
        right_side = y[x >= 0.5][::-1]
        min_len = min(len(left_side), len(right_side))
        left_side = left_side[:min_len]
        right_side = right_side[:min_len]

        asymmetry = np.abs(left_side - right_side).mean()
        # middle = (left_side[-1] + right_side[-1]) / 2
        middle = max(y)

        if np.abs(np.array(y)).max() > 10:
            continue

        asymmetries.append(asymmetry)
        middle_deviations.append(middle - 1)

        ax.plot(
            x,
            y,
            alpha=0.5,
            color=method_color_mapping[model],
            label=model,
        )

    asymmetries = np.array(asymmetries)
    middle_deviations = np.array(middle_deviations)

    ax.text(
        0.05,
        0.95,
        "\n".join(
            [
                f"Miss: {len(filtered_df) - len(asymmetries) - filtered_df['pbe_e_vacmig'].isna().sum()}",
                f"Asym: {asymmetries.mean():.3f}",
                f"MAPE@max: {np.abs(middle_deviations).mean() * 100:.1f}",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize="small",
        # fontsize=6,
    )

    ax.set(
        title=model,
        xlabel="Normalized path" if i >= len(models) - ncols else None,
        ylabel="Normalized energy" if i % ncols == 0 else None,
        ylim=(-0.1, 2),
    )

with open(DATA_DIR / "fcc.pkl", "wb") as f:
    pickle.dump(fig, f)

# hcp

# Initialize an empty DataFrame
results_df = pd.DataFrame(columns=["symbol", "model", "fit_path", "fit_energies"])

for model in mlip_models:
    out_dir = Path(REGISTRY[model]["family"])

    for index, row in hcp_pbe.iterrows():
        symbol = row["symbol"]

        if Element(symbol.split("_")[0]).is_noble_gas:
            continue

        files = glob.glob(str(out_dir / f"{model}-hcp-{symbol.split('_')[0]}36.pkl"))
        if len(files) == 0:
            print("skip", model, symbol)
            # Add missing data to the DataFrame
            # if symbol not in results_df['symbol'].values:
            # Create a new row if the symbol is not yet in the DataFrame
            new_row = {
                "symbol": symbol,
                "model": model,
                "pbe_e_vacmig": row["e_vacmig"],
                "fit_path": [],
                "fit_energies": [],
            }
            results_df = pd.concat(
                [results_df, pd.DataFrame([new_row])], ignore_index=True
            )
            # else:
            #     # Update the existing row with the model's prediction
            #     results_df.loc[results_df['symbol'] == symbol, model] = pd.NA
            continue
        file = files[0]
        with open(file, "rb") as f:
            result = pickle.load(f)

        # Add data to the DataFrame
        # if symbol not in results_df['symbol'].values:
        # Create a new row if the symbol is not yet in the DataFrame
        forcefit = result["neb"]["forcefit"]
        new_row = {
            "symbol": symbol,
            "model": model,
            "pbe_e_vacmig": row["e_vacmig"],
            "fit_path": forcefit.fit_path,
            "fit_energies": forcefit.fit_energies,
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)



nrows = 2
ncols = len(mlip_models) // nrows

threshold = 0.10

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(6, 4),
    sharex=True,
    sharey=True,
    constrained_layout=True,
    dpi=300,
)

for i, (ax, model) in enumerate(zip(axes.ravel(), mlip_models, strict=False)):
    filtered_df = results_df[results_df["model"] == model]

    asymmetries = []
    middle_deviations = []

    for index, row in filtered_df.iterrows():
        if len(row["fit_path"]) == 0 or pd.isna(row["pbe_e_vacmig"]):
            continue

        x = row["fit_path"] / max(row["fit_path"])
        y = row["fit_energies"] / row["pbe_e_vacmig"]

        # middle_idx = np.argmin(np.abs(x - 0.5))

        left_side = y[x <= 0.5]
        right_side = y[x >= 0.5][::-1]
        min_len = min(len(left_side), len(right_side))
        left_side = left_side[:min_len]
        right_side = right_side[:min_len]

        asymmetry = np.abs(left_side - right_side).mean()
        # middle = (left_side[-1] + right_side[-1]) / 2
        middle = max(y)

        if np.abs(np.array(y)).max() > 10:
            continue

        asymmetries.append(asymmetry)
        middle_deviations.append(middle - 1)

        ax.plot(
            x,
            y,
            alpha=0.5,
            color=method_color_mapping[model],
            label=model,
        )

    asymmetries = np.array(asymmetries)
    middle_deviations = np.array(middle_deviations)

    ax.text(
        0.05,
        0.95,
        "\n".join(
            [
                f"Miss: {len(filtered_df) - len(asymmetries) - filtered_df['pbe_e_vacmig'].isna().sum()}",
                f"Asym: {asymmetries.mean():.3f}",
                f"MAPE@max: {np.abs(middle_deviations).mean() * 100:.1f}",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize="small",
    )

    ax.set(
        title=model,
        xlabel="Normalized path" if i >= len(mlip_models) - ncols else None,
        ylabel="Normalized energy" if i % ncols == 0 else None,
        ylim=(-0.1, 2),
    )

with open(DATA_DIR / "hcp.pkl", "wb") as f:
    pickle.dump(fig, f)
