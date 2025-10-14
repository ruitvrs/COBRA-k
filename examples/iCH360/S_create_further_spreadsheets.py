import z_add_path  # noqa: D100, F401

from cobrak.dataclasses import Model
from cobrak.io import ensure_folder_existence, json_load, standardize_folder
from cobrak.spreadsheet_functionality import (
    OptimizationDataset,
    create_cobrak_spreadsheet,
)

datafolder = standardize_folder("examples/iCH360")
sheetfolder = standardize_folder(datafolder + "extra_spreadsheets")
ensure_folder_existence(sheetfolder)

# Mtot variations spreadsheet #
basefolder = standardize_folder(datafolder + "RESULTS_GLCUPTAKE_DIFFERENT_MAXCONCSUMS")
create_cobrak_spreadsheet(
    path=sheetfolder + "different_concsums.xlsx",
    cobrak_model=json_load(
        f"{basefolder}/used_cobrak_model__5_maxconcsuminf.json", Model
    ),
    variability_datasets={},
    optimization_datasets={
        "100 mM Mtot": OptimizationDataset(
            data=json_load(f"{basefolder}/final_best_result__1_maxconcsum0.1.json"),
            with_df=True,
        ),
        "200 mM Mtot": OptimizationDataset(
            data=json_load(f"{basefolder}/final_best_result__1_maxconcsum0.2.json"),
            with_df=True,
        ),
        "300 mM Mtot": OptimizationDataset(
            data=json_load(f"{basefolder}/final_best_result__1_maxconcsum0.3.json"),
            with_df=True,
        ),
        "400 mM Mtot": OptimizationDataset(
            data=json_load(f"{basefolder}/final_best_result__1_maxconcsum0.4.json"),
            with_df=False,
        ),
        "500 mM Mtot": OptimizationDataset(
            data=json_load(f"{basefolder}/final_best_result__1_maxconcsum0.5.json"),
            with_df=False,
        ),
        "600 mM Mtot": OptimizationDataset(
            data=json_load(f"{basefolder}/final_best_result__1_maxconcsum0.6.json"),
            with_df=False,
        ),
        "700 mM Mtot": OptimizationDataset(
            data=json_load(f"{basefolder}/final_best_result__1_maxconcsum0.7.json"),
            with_df=False,
        ),
        "800 mM Mtot": OptimizationDataset(
            data=json_load(f"{basefolder}/final_best_result__1_maxconcsum0.8.json"),
            with_df=False,
        ),
        "inf Mtot": OptimizationDataset(
            data=json_load(f"{basefolder}/final_best_result__1_maxconcsuminf.json")
        ),
    },
    show_regulation_coefficients=False,
)

# With and without manual changes spreadsheet #
create_cobrak_spreadsheet(
    path=sheetfolder + "with_and_wo_manual_kcat_changes.xlsx",
    cobrak_model=json_load(
        f"{basefolder}/used_cobrak_model__5_maxconcsuminf.json", Model
    ),
    variability_datasets={},
    optimization_datasets={
        "9.65 with manual changes": OptimizationDataset(
            data=json_load(
                "examples/iCH360/RESULTS_GLCUPTAKE/final_best_result__1_maxglc9.65.json"
            )
        ),
        "9.65 w/o manual changes": OptimizationDataset(
            data=json_load(
                "examples/iCH360/RESULTS_GLCUPTAKE_NO_MANUAL_KCAT_CHANGES_DONEPROTADJ/final_best_result__1_maxglc9.65.json"
            )
        ),
        "1000 with manual changes": OptimizationDataset(
            data=json_load(
                "examples/iCH360/RESULTS_GLCUPTAKE/final_best_result__1_maxglc1000.json"
            )
        ),
        "1000 w/o manual changes": OptimizationDataset(
            data=json_load(
                "examples/iCH360/RESULTS_GLCUPTAKE_NO_MANUAL_KCAT_CHANGES_DONEPROTADJ/final_best_result__1_maxglc1000.0.json"
            )
        ),
    },
    show_regulation_coefficients=False,
)


# Acetate comparison #
create_cobrak_spreadsheet(
    path=sheetfolder + "acetate_comparison.xlsx",
    cobrak_model=json_load(
        "examples/iCH360/RESULTS_UPTAKE_ACETATE/used_cobrak_model__2_maxglc1000.json",
        Model,
    ),
    variability_datasets={},
    optimization_datasets={
        "protcorrected max(µ) w/ ac": OptimizationDataset(
            data=json_load(
                "examples/iCH360/RESULTS_UPTAKE_ACETATE/final_best_result__5_maxglc1000.json"
            ),
            with_gamma=True,
            with_vplus=True,
            with_kappa=True,
        ),
    },
    show_regulation_coefficients=False,
)
