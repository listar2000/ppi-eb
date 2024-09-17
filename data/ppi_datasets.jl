# collection of utils to load datasets

using GoogleDrive: google_download
using NPZ: npzread

GDRIVE_IDS = Dict(
    "alphafold" => "1lOhdSJEcFbZmcIoqmlLxo3LgLG1KqPho",
    "ballots" => "1DJvTWvPM6zQD0V4yGH1O7DL3kfnTE06u",
    "census_education" => "15iq7nLjwogb46v3stknMmx7kMuK9cnje",
    "census_income" => "15dZeWw-RTw17-MieG4y1ILTZlreJOmBS",
    "census_healthcare" => "1RjWsnq-gMngRFRj22DvezcdCVl2MxAIX",
    "forest" => "1Vqi1wSmVnWh_2lLQuDwrhkGcipvoWBc0",
    "galaxies" => "1pDLQesPhbH5fSZW1m4aWC-wnJWnp1rGV",
    "gene_expression" => "17PwlvAAKeBYGLXPz9L2LVnNJ66XjuyZd",
    "plankton" => "1KEk0ZFZ6KiB7_2tdPc5fyBDFNhhJUS_W",
)

function load_gdrive_dataset(dataset_folder::String, dataset_name::String)
    # check if the folder exists. Create it if it doesn't
    if !isdir(dataset_folder)
        mkdir(dataset_folder)
    end
    # throw error if the dataset_name is not in the GDRIVE_IDS
    if !haskey(GDRIVE_IDS, dataset_name)
        error("Dataset [$dataset_name] not found. Please check the dataset name.")
    end
    # construct the data path
    data_path = joinpath(dataset_folder, "$dataset_name.npz")
    if !isfile(data_path)
        println("Dataset [$dataset_name] not found. Downloading from Google Drive...")
        # download the dataset from Google Drive
        google_url = "https://drive.google.com/file/d/" * GDRIVE_IDS[dataset_name]
        google_download(google_url, data_path)
    end
    # load the dataset
    npzread(data_path)
end

