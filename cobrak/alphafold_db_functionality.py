"""Functionality to retrieve PDB files from the AlphaFold Protein Structure Database

Link to database (as of December 2, 2025): https://alphafold.ebi.ac.uk/
"""

import os
from time import sleep

import requests

from .dataclasses import Model
from .io import ensure_folder_existence, get_files, gzip_write_file, standardize_folder


def download_alphafold_pdb(
    uniprot_id: str, output_dir: str = ".", as_gzip: bool = False
) -> None:
    """Downloads the predicted Protein Data Bank (PDB) file for a given UniProt ID
    from the AlphaFold Protein Structure Database (AlphaFold DB).

    This function queries the AlphaFold API, identifies the most complete/latest PDB
    URL (specifically by finding the highest numbered entry in fragmented predictions),
    and downloads the file to the specified directory.

    Args:
        uniprot_id: The UniProt accession ID (e.g., "P00520") for the target protein.
        output_dir: The directory path where the PDB file should be saved.
                    Defaults to the current directory (``"."``).
        as_gzip: Whether or not the file shall be compressed through gzip (.gz is added to the file name).
            Defaults to False.

    Returns:
        None: The function prints status messages and saves the file to disk.

    Raises:
        requests.exceptions.HTTPError: If the initial API request fails due to an
                                       HTTP error (e.g., 404, 500).

    Notes:
        * The download file is named using the format:
          ``<uniprot_id>__<original_filename>`` (e.g., ``P00520__AF-P00520-F1-model_v4.pdb``).
        * The AlphaFold DB API is used to handle multi-fragment predictions (for
          proteins over ~2700 residues) by attempting to select the final fragment
          or the entry with the highest numerical identifier in the URL path.
    """
    output_dir = standardize_folder(output_dir)
    ensure_folder_existence(output_dir)

    output_dir_files = get_files(output_dir)
    for file in output_dir_files:
        if file.startswith(uniprot_id + "__"):
            print(f"File for {uniprot_id} already exists as {file} in {output_dir}")
        return

    # 1. API Endpoint for the specific UniProt ID
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        # 2. Iterate through predictions (essential for proteins > 2700 residues)
        if not data:
            print(f"No AlphaFold prediction found for {uniprot_id}")
            return

        last_pdb_url = ""
        last_pdb_number = -1
        for entry in data:
            # Extract the PDB URL from the JSON response
            pdb_url = entry.get("pdbUrl")
            url_split = pdb_url.split("-")[2]
            try:
                int(url_split)
                int_url_split_possible = True
            except ValueError:
                int_url_split_possible = False
            if pdb_url and (
                (not int_url_split_possible)
                or (int_url_split_possible and int(url_split) > last_pdb_number)
            ):
                last_pdb_url = pdb_url
                if int_url_split_possible:
                    last_pdb_number = int(url_split)

        if last_pdb_url:
            # 3. Download the file
            file_name = uniprot_id + "__" + os.path.basename(last_pdb_url)
            save_path = os.path.join(output_dir, file_name)

            print(f"Downloading {file_name}...")
            pdb_response = requests.get(last_pdb_url)
            if as_gzip:
                gzip_write_file(f"{save_path}.gz", [pdb_response.text])
            else:
                with open(save_path, "wb") as f:
                    f.write(pdb_response.content)

        if not last_pdb_url:
            print(f"PDB file not found in metadata for {uniprot_id}")

    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error for {uniprot_id}: {err}")


def download_alphafold_pdb_for_all_enzymes(
    cobrak_model: Model,
    uniprot_annotation_id: str = "uniprot",
    output_dir: str = ".",
    sleep_time: float = 1.0,
    as_gzip: bool = False,
) -> None:
    """Downloads AlphaFold PDB files for all enzymes in a COBRA-k Model
    that have a given UniProt ID annotation.

    It iterates through the reactions in the COBRA-k Model, extracts the UniProt ID
    from the reaction's annotation, and calls `download_alphafold_pdb` for each.
    A delay is introduced between downloads to respect API rate limits.

    Args:
        cobrak_model: A COBRA-k Model instance.
        uniprot_annotation_id: The key used in the reaction's `annotation`
                               dictionary to store the UniProt ID (defaults to "uniprot").
        output_dir: The directory path where the PDB files should be saved.
                    Defaults to the current directory (``"."``).
        sleep_time: The time in seconds to pause between consecutive downloads
                    to cool down the AlphaFold API server. Defaults to 1.0 second.
        as_gzip: Whether or not the file shall be compressed through gzip (.gz is added to the file name).
            Defaults to False.

    Returns:
        None: The function manages file downloads and prints status messages.
    """
    for enzyme_data in cobrak_model.enzymes.values():
        if uniprot_annotation_id not in enzyme_data.annotation:
            continue
        uniprot_id = enzyme_data.annotation[uniprot_annotation_id]
        download_alphafold_pdb(
            uniprot_id=uniprot_id, output_dir=output_dir, as_gzip=as_gzip
        )
        sleep(sleep_time)
