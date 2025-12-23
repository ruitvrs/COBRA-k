"""Functionality for using molmass"""

from molmass import Formula, FormulaError
from pydantic import validate_call

from .dataclasses import Model


@validate_call(validate_return=True)
def add_molar_masses_to_model_metabolites(
    model: Model,
    verbose: bool = False,
) -> Model:
    """Calculates and assigns molar masses to metabolites in a Model instance.

    This function iterates through all metabolites in the provided model, parses
    their chemical formulas using the `molmass` library, and updates the
    `molar_mass` attribute for each metabolite.

    Args:
        model: A COBRA-k Model instance.
        verbose: If True, prints status messages regarding missing formulas,
            successful calculations, or parsing errors.

    Returns:
        Model: The modified model object with updated 'molar_mass' attributes.

    Raises:
        Note: molmass's FormulaError (e.g. a wrong formula with unknown symbols) is handled internally
    """
    for met_id, metabolite in model.metabolites.items():
        if not metabolite.formula:
            if verbose:
                print(f"No formula given for {met_id} - no molar mass computable!")
            continue
        try:
            molmass_formula = Formula(metabolite.formula)
            metabolite.molar_mass = molmass_formula.mass
            if verbose:
                print(f"Average Mass (g⋅mol⁻¹) of {met_id}: {molmass_formula.mass:.4f}")
        except FormulaError:
            if verbose:
                print(
                    f"FormulaError with {met_id} with formula {metabolite.formula} - no molar mass computable!"
                )
    return model
