from os.path import exists  # noqa: D100

import z_add_path  # noqa: F401

from cobrak.dataclasses import Model
from cobrak.io import ensure_folder_existence, json_load, json_write
from cobrak.lps import perform_lp_optimization, perform_lp_variability_analysis
from cobrak.spreadsheet_functionality import (
    OptimizationDataset,
    VariabilityDataset,
    create_cobrak_spreadsheet,
)
from cobrak.standard_solvers import CPLEX, CPLEX_FOR_VARIABILITY_ANALYSIS
from cobrak.constants import OBJECTIVE_VAR_NAME, PROT_POOL_REAC_NAME

biomass_reac_id = "Biomass_fw"
glcuptake_reac_id = "EX_glc__D_e_bw"
cobrak_model: Model = json_load(
    "examples/iCH360/prepared_external_resources/iCH360_cobrak.json",
    Model,
)

cobrak_model.max_prot_pool = 0.75  # just a high value for our following protein pool optimization
with cobrak_model as growth_0_7_cobrak_model:
    growth_0_7_cobrak_model.reactions["Biomass_fw"].min_flux = 0.7
    opt_result_protpool = perform_lp_optimization(
        growth_0_7_cobrak_model,
        objective_target=PROT_POOL_REAC_NAME,
        objective_sense=-1,
        with_enzyme_constraints=True,
        with_thermodynamic_constraints=True,
        with_loop_constraints=True,
        variability_dict={},
        solver=CPLEX,
    )
    print("Protein pool to reach growth 0.7 h⁻¹:", round(opt_result_protpool[PROT_POOL_REAC_NAME], 6), "g⋅gDW⁻¹")
cobrak_model.max_prot_pool = opt_result_protpool[PROT_POOL_REAC_NAME]

results_folder = "examples/iCH360/ecTFBA_and_ecTFVA_results/"
ensure_folder_existence(results_folder)
opt_datasets: dict[str, OptimizationDataset] = {}
for glc_uptake in (
    9.65,
    1000.0,
):
    cobrak_model.reactions[glcuptake_reac_id].max_flux = glc_uptake
    opt_result_maxflux = perform_lp_optimization(
        cobrak_model,
        objective_target=biomass_reac_id,
        objective_sense=+1,
        with_enzyme_constraints=True,
        with_thermodynamic_constraints=True,
        with_loop_constraints=True,
        variability_dict={},
        solver=CPLEX,
    )
    json_write(f"{results_folder}ectfba_glc{glc_uptake}.json", opt_result_maxflux)
    opt_datasets[str(glc_uptake)] = OptimizationDataset(
        data=opt_result_maxflux,
        with_df=True,
        with_vplus=True,
    )

    min_biomass_flux = opt_result_maxflux[biomass_reac_id] - 1e-6
    cobrak_model.reactions[biomass_reac_id].min_flux = min_biomass_flux

"""
# START OF PROLINETEST
cobrak_model.max_prot_pool = .224
opt_result_noprolineflux = perform_lp_optimization(
    cobrak_model,
    objective_target=biomass_reac_id,
    objective_sense=+1,
    with_enzyme_constraints=True,
    with_thermodynamic_constraints=True,
    with_loop_constraints=True,
    variability_dict={},
    solver=CPLEX,
)

other_model: Model = json_load(
    "examples/iCH360/prepared_external_resources/proline_intake_reaction.json",
    Model,
)
for reac_id, reaction in other_model.reactions.items():
    if reac_id not in cobrak_model.reactions:
        cobrak_model.reactions[reac_id] = other_model.reactions[reac_id]
for met_id, metabolite in other_model.metabolites.items():
    if reac_id not in cobrak_model.metabolites:
        cobrak_model.metabolites[met_id] = other_model.metabolites[met_id]
for enzyme_id, enzyme in other_model.enzymes.items():
    if reac_id not in cobrak_model.enzymes:
        cobrak_model.reactions[met_id] = other_model.reactions[met_id]
opt_result_prolineflux = perform_lp_optimization(
    cobrak_model,
    objective_target=biomass_reac_id,
    objective_sense=+1,
    with_enzyme_constraints=True,
    with_thermodynamic_constraints=True,
    with_loop_constraints=True,
    variability_dict={},
    solver=CPLEX,
)
print("µ with proline:", opt_result_prolineflux[OBJECTIVE_VAR_NAME], "uptake is", opt_result_prolineflux["EX_pro__L_c_IN"])
print("µ w/o proline:", opt_result_noprolineflux[OBJECTIVE_VAR_NAME])
print(opt_result_prolineflux[OBJECTIVE_VAR_NAME] - opt_result_noprolineflux[OBJECTIVE_VAR_NAME])
# END OF PROLINETEST
"""

reference_result = json_load(f"{results_folder}ectfba_glc9.65.json")
cobrak_model.reactions[biomass_reac_id].min_flux = reference_result[biomass_reac_id]
cobrak_model.reactions[biomass_reac_id].max_flux = reference_result[biomass_reac_id]
cobrak_model.reactions[glcuptake_reac_id].min_flux = reference_result[glcuptake_reac_id]
cobrak_model.reactions[glcuptake_reac_id].max_flux = reference_result[glcuptake_reac_id]

varpath = (
    f"{results_folder}ectfva_glc{glc_uptake}_mu{reference_result[biomass_reac_id]}.json"
)
if not exists(varpath):
    var_result = perform_lp_variability_analysis(
        cobrak_model=cobrak_model,
        with_enzyme_constraints=True,
        with_thermodynamic_constraints=True,
        calculate_reacs=True,
        calculate_concs=True,
        calculate_rest=True,
        solver=CPLEX_FOR_VARIABILITY_ANALYSIS,
        min_enzyme_cutoff=1e-8,
        max_active_enzyme_cutoff=1e-8,
    )
    json_write(varpath, var_result)
else:
    var_result = json_load(varpath)

var_datasets = {
    "ecTFVA_at_fixed_glcuptake_and_growth": VariabilityDataset(
        data=var_result, with_df=True
    )
}
create_cobrak_spreadsheet(
    "examples/iCH360/ecTFBA_and_ecTFVA_results/ectfba_ectfva_spreadsheet.xlsx",
    cobrak_model=cobrak_model,
    variability_datasets=var_datasets,
    optimization_datasets=opt_datasets,
)
