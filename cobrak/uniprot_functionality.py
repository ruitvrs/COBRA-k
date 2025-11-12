"""get_protein_mass_mapping.py

Functions for the generation of a model's mapping of its proteins and their masses.
"""

# IMPORTS SECTION #
import time
from copy import deepcopy

import cobra
import requests
from pydantic import ConfigDict, validate_call

from .io import ensure_folder_existence, json_load, standardize_folder


# FUNCTIONS SECTION #
@validate_call(config=ConfigDict(arbitrary_types_allowed=True), validate_return=True)
def uniprot_get_enzyme_molecular_weights_for_sbml(
    sbml_path: str,
    cache_basepath: str,
    base_species: str,
    multiplication_factor: float = 1 / 1000,
) -> dict[str, float]:
    """Returns a JSON with a mapping of protein IDs as keys, and as values the protein mass in kDa.

    The protein masses are taken  from UniProt (retrieved using
    UniProt's REST API).

    Arguments
    ----------
    * sbml_path: str ~ The SBML's file path

    Output
    ----------
    A JSON file with the path project_folder+project_name+'_protein_id_mass_mapping.json'
    and the following structure:
    <pre>
    {
        "$PROTEIN_ID": $PROTEIN_MASS_IN_KDA,
        (...),
    }
    </pre>
    """
    model = cobra.io.read_sbml_model(sbml_path)
    # GET UNIPROT ID - PROTEIN MAPPING
    uniprot_id_protein_id_mapping: dict[str, list[str]] = {}
    for gene in model.genes:
        # Without a UniProt ID, no mass mapping can be found
        if "uniprot" not in gene.annotation:
            uniprot_id_protein_id_mapping[gene.id] = [gene.id]
            continue
        uniprot_id = gene.annotation["uniprot"]
        if uniprot_id in uniprot_id_protein_id_mapping:
            uniprot_id_protein_id_mapping[uniprot_id].extend([gene.id, uniprot_id])
        else:
            uniprot_id_protein_id_mapping[uniprot_id] = [gene.id, uniprot_id]

    # GET UNIPROT ID<->PROTEIN MASS MAPPING
    uniprot_id_protein_mass_mapping: dict[str, float] = {}
    # The cache stored UniProt masses for already searched
    # UniProt IDs (each file in the cache folder has the name
    # of the corresponding UniProt ID). This prevents searching
    # UniProt for already found protein masses. :-)
    cache_basepath = standardize_folder(cache_basepath)
    ensure_folder_existence(cache_basepath)
    cache_filepath = f"{cache_basepath}_cache_uniprot_molecular_weights.json"
    try:
        cache_json: dict[str, float] = json_load(cache_filepath, dict[str, float])
    except Exception:
        cache_json: dict[str, float] = {}
    original_cache_json_keys = deepcopy(list(cache_json.keys()))
    # Go through each batch of UniProt IDs (multiple UniProt IDs
    # are searched at once in order to save an amount of UniProt API calls)
    # and retrieve the amino acid sequences and using these sequences, their
    # masses.
    print("Starting UniProt ID<->Protein mass search using UniProt API...")
    uniprot_ids = list(uniprot_id_protein_id_mapping.keys())

    batch_size = 12
    batch_start = 0
    while batch_start < len(uniprot_ids):
        # Create the batch with all UniProt IDs
        prebatch = uniprot_ids[batch_start : batch_start + batch_size]
        batch = []
        # Remove all IDs which are present in the cache (i.e.,
        # which were searched for already).
        # The cache consists of pickled protein mass floats, each
        # onein a file with the name of the associated protein.
        for uniprot_id in prebatch:
            if uniprot_id not in cache_json:
                batch.append(uniprot_id)
            else:
                uniprot_id_protein_mass_mapping[uniprot_id] = cache_json[uniprot_id]

        # If all IDs could be found in the cache, continue with the next batch.
        if len(batch) == 0:
            batch_start += batch_size
            continue

        # Create the UniProt query for the batch
        # With 'OR', all given IDs are searched, and subsequently in this script,
        # the right associated masses are being picked.
        query = " OR ".join(batch)
        uniprot_query_url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&format=tsv&fields=accession,id,mass,gene_names,gene_orf,gene_oln,organism_name"
        print(f"UniProt batch search for: {query}")

        # Call UniProt's API :-)
        uniprot_data: list[str] = requests.get(
            uniprot_query_url, timeout=1e6
        ).text.split("\n")
        # Wait in order to cool down their server :-)
        time.sleep(1.0)

        # Read out the API-returned lines
        found_ids = []
        for line in uniprot_data[1:]:
            if not line:
                continue
            accession_id = line.split("\t")[0].lstrip().rstrip()
            entry_id = line.split("\t")[1].lstrip().rstrip()
            mass_string = line.split("\t")[2].lstrip().rstrip()
            gene_names = line.split("\t")[3].lstrip().rstrip().split(" ")
            gene_names_orf = line.split("\t")[4].lstrip().rstrip().split(" ")
            gene_names_ordered_locus = line.split("\t")[5].lstrip().rstrip().split(" ")
            organism_name = line.split("\t")[6].lstrip().rstrip()
            if base_species.lower() not in organism_name.lower():
                continue
            try:
                # Note that the mass entry from UniProt uses a comma as a thousand separator, so it has to be removed before parsing
                mass = float(mass_string.replace(",", ""))
            except ValueError:  # We may also risk the entry is missing
                continue
            uniprot_id_protein_mass_mapping[accession_id] = float(mass)
            uniprot_id_protein_mass_mapping[entry_id] = float(mass)
            for extraname in [
                extraname
                for extraname in gene_names + gene_names_orf + gene_names_ordered_locus
                if len(extraname) > 0
            ]:
                uniprot_id_protein_mass_mapping[extraname] = float(mass)
                found_ids.append(extraname)
            found_ids.extend((accession_id, entry_id))

        # Continue with the next batch :D
        batch_start += batch_size

    # Create the final protein ID <-> mass mapping
    protein_id_mass_mapping: dict[str, float] = {}
    not_found_ids = set(uniprot_ids) - set(cache_json.keys())
    if len(not_found_ids):
        print(
            f"INFO: Molecular weights not found for the following IDs: {'; '.join(not_found_ids)}"
        )
        print(
            "You may try to re-run the Uniprot MW search, this helps sometimes to find missing MWs."
        )
    for uniprot_id in list(uniprot_id_protein_mass_mapping.keys()):
        try:
            protein_ids = uniprot_id_protein_id_mapping[uniprot_id]
        except KeyError:
            continue
        for protein_id in protein_ids:
            if protein_id not in original_cache_json_keys:
                protein_id_mass_mapping[protein_id] = uniprot_id_protein_mass_mapping[
                    uniprot_id
                ] * (
                    multiplication_factor
                    if protein_id not in original_cache_json_keys
                    else 1.0
                )

    # Return protein mass list JSON :D
    return protein_id_mass_mapping | cache_json
