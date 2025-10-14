from math import exp, log  # noqa: D100, F401

import matplotlib.pyplot as plt
import numpy as np
import z_add_path  # noqa: D100, F401

from cobrak.constants import LNCONC_VAR_PREFIX
from cobrak.dataclasses import Model
from cobrak.io import ensure_folder_existence, json_load  # noqa: F401, F811

model = json_load(
    "examples/iCH360/prepared_external_resources/iCH360_cobrak.json", Model
)

max_round = 100
targets = [
    "Biomass_fw",
    "EX_glc__D_e_bw",
    "EX_ac_e_fw",
    f"{LNCONC_VAR_PREFIX}glu__L_c",
]
original_data = json_load(
    "examples/iCH360/RESULTS_GLCUPTAKE/final_best_result__2_maxglc1000.json"
)
jsonprefix = (
    "examples/iCH360/RESULTS_VARIATION_ALL_KCAT_KM_VARIED_50_PCT/final_best_result__"
)
jsonsuffix = "_maxglc1000.0.json"
target_to_data = {
    target: [
        exp(original_data[target])
        if target.startswith(LNCONC_VAR_PREFIX)
        else original_data[target]
    ]
    for target in targets
}
found_files = 0
for round_num in range(max_round):
    try:
        data = json_load(f"{jsonprefix}{round_num}{jsonsuffix}")
    except FileNotFoundError:
        continue
    found_files += 1
    for target in targets:
        if target.startswith(LNCONC_VAR_PREFIX):
            target_to_data[target].append(exp(data[target]))
        else:
            target_to_data[target].append(data[target])

# Create a figure and a set of subplots
fig, axs = plt.subplot_mosaic(
    [
        ["A", "B"],
        ["C", "D"],
    ],
    figsize=(7.25, 5.0),
)

# Iterate over the target_to_data dictionary
i_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
for i, (label, values) in enumerate(target_to_data.items()):
    # Separate the original value from the sensitivity analysis values
    original_value = values[0]
    sensitivity_values = values[1:]

    # Plot the histogram of sensitivity analysis values
    edges = np.linspace(np.min(sensitivity_values), np.max(sensitivity_values), 15)
    n, bins, patches = axs[i_to_letter[i]].hist(
        sensitivity_values,
        bins=edges,
        alpha=0.75,
        color="blue",
    )
    # Set y-axis to show integer counts
    axs[i_to_letter[i]].yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Calculate mean and standard deviation of sensitivity values
    mean_value = np.mean(sensitivity_values)
    median_value = np.median(sensitivity_values)
    std_deviation = np.std(sensitivity_values)

    # Mark the original value on the histogram
    axs[i_to_letter[i]].axvline(
        x=original_value,
        color="r",
        linestyle="dashed",
        linewidth=2,
        label=f"Original Value: {original_value:.2f}",
    )

    # Mark the mean value on the histogram
    axs[i_to_letter[i]].axvline(
        x=mean_value,
        color="g",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {mean_value:.2f}",
    )

    # Mark the median value on the histogram
    # axs[i_to_letter[i]].axvline(x=median_value, color="b", linestyle=":", linewidth=1, label=f"Median: {median_value:.2f}")

    # Mark one standard deviation above and below the mean
    axs[i_to_letter[i]].axvspan(
        mean_value - std_deviation,
        mean_value + std_deviation,
        color="yellow",
        alpha=0.3,
        label=f"Std. Dev.: ±{round(std_deviation, 2)}",
    )

    # Set the title of the subplot to the label

    # Set y-axis label
    axs[i_to_letter[i]].set_ylabel("# occurrences", fontsize=8)

    match i_to_letter[i]:
        case "A":
            title = "A"
            xlabel = "Growth rate [h⁻¹]"
        case "B":
            title = "B"
            xlabel = "Glucose uptake [mmol⋅gDW⋅h⁻¹]"
        case "C":
            title = "C"
            xlabel = "Acetate secretion [mmol⋅gDW⋅h⁻¹]"
        case "D":
            title = "D"
            xlabel = "Glutamate concentration [M]"
    axs[i_to_letter[i]].tick_params(axis="x", labelsize=8)
    axs[i_to_letter[i]].tick_params(axis="y", labelsize=8)
    axs[i_to_letter[i]].set_xlabel(xlabel, fontsize=8)
    axs[i_to_letter[i]].set_title(title, loc="left", fontweight="bold", fontsize=9)

    # Add legend to the subplot
    axs[i_to_letter[i]].legend(fontsize=7.5)

    axs[i_to_letter[i]].set_xlim(0.0, max(sensitivity_values) * 1.05)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the figure
print("n:", found_files)
# plt.show()
ensure_folder_existence("examples/iCH360/variation_plot")
plt.savefig(
    "./examples/iCH360/variation_plot/Z_"
    + jsonprefix.split("/")[2].replace("RESULTS_GLCUPTAKE_", "")
    + ".jpeg",
    dpi=300,
)
plt.savefig(
    "./examples/iCH360/variation_plot/Figure_6.pdf",
)
