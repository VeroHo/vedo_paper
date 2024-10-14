exec(open("../ML_Response_prediction_HelperScript.py").read())
import sys
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

sys.stdout = Logger('../ML_outputLog.txt')

# Where are modality information stored?
clinicaliloc = range(12, 14)
cytof_freiloc = range(22, 38)
cytof_totaliloc = range(38, 54)
cytof_totalAbiloc = range(55, 70)
cytof_Abiloc = range(71, 87)
facsiloc = range(163, 193)
olinikiloc = range(87, 161)

# Read in metadata file
metadata = pd.read_csv("../Metadata_file.csv", sep=';', index_col=0, decimal=',')
# Select data frame for response prediction
only_timepoint_one = metadata[metadata["timepoint"] == 1]
# Select data frame for treatment prediction
only_treated = metadata[metadata["treatment"] == "VDZ"]

# Run only single modalities
run_modality(dataframe = only_timepoint_one,input_features =olinikiloc , data_modality_name = "onlyOLINK")

contrast_data = only_timepoint_one.iloc[:, np.r_[clinicaliloc]]
contrast_data['Response'] = only_timepoint_one['response']
X, y = Run_analysis(data_frame=contrast_data)

print(str("onlyClinical") + ":\n================================================")
features = scan_over_models(X=X, y=y, data_type=str("onlyClinical"))

run_modality(dataframe = only_timepoint_one,input_features =cytof_freiloc , data_modality_name = "onlyCyTOF")
run_modality(dataframe = only_timepoint_one,input_features =cytof_Abiloc , data_modality_name = "onlyCyTOFab")
run_modality(dataframe = only_timepoint_one,input_features =facsiloc , data_modality_name = "onlyFACS")

# Run all duo modalities
run_modality(dataframe = only_timepoint_one,input_features =(olinikiloc,clinicaliloc) ,data_modality_name = "OLINK_Clinical")
run_modality(dataframe = only_timepoint_one,input_features =(olinikiloc,facsiloc) ,data_modality_name = "OLINK_FACS")
run_modality(dataframe = only_timepoint_one,input_features =(olinikiloc,cytof_freiloc) ,data_modality_name = "OLINK_CyTOF")
run_modality(dataframe = only_timepoint_one,input_features =(olinikiloc,cytof_Abiloc) ,data_modality_name = "OLINK_CyTOFab")

run_modality(dataframe = only_timepoint_one,input_features =(facsiloc,cytof_freiloc) ,data_modality_name = "FACS_CyTOF")
run_modality(dataframe = only_timepoint_one,input_features =(facsiloc,cytof_Abiloc) ,data_modality_name = "FACS_CyTOFab")
run_modality(dataframe = only_timepoint_one,input_features =(facsiloc,clinicaliloc) ,data_modality_name = "FACS_Clinical")

run_modality(dataframe = only_timepoint_one,input_features =(clinicaliloc,cytof_freiloc) ,data_modality_name = "Clinical_CyTOF")
run_modality(dataframe = only_timepoint_one,input_features =(clinicaliloc,cytof_Abiloc) ,data_modality_name = "Clinical_CyTOFab")

# Run all trio modalities
run_modality(dataframe = only_timepoint_one,input_features =(facsiloc,clinicaliloc,olinikiloc ) ,data_modality_name = "FACS_Clinical_OLINK")
run_modality(dataframe = only_timepoint_one,input_features =(facsiloc,cytof_freiloc,olinikiloc ) ,data_modality_name = "FACS_CyTOF_OLINK")
run_modality(dataframe = only_timepoint_one,input_features =(facsiloc,cytof_Abiloc,olinikiloc ) ,data_modality_name = "FACS_CyTOFab_OLINK")

run_modality(dataframe = only_timepoint_one,input_features =(facsiloc,cytof_Abiloc,cytof_freiloc ) ,data_modality_name = "FACS_CyTOF_Clinical")
run_modality(dataframe = only_timepoint_one,input_features =(facsiloc,cytof_Abiloc,clinicaliloc ) ,data_modality_name = "FACS_CyTOFab_Clinical")

run_modality(dataframe = only_timepoint_one,input_features =(olinikiloc,cytof_freiloc,clinicaliloc ) ,data_modality_name = "OLINK_CyTOF_Clinical")
run_modality(dataframe = only_timepoint_one,input_features =(olinikiloc,cytof_Abiloc,clinicaliloc ) ,data_modality_name = "OLINK_CyTOFab_Clinical")


# Run all fully combined modalities
run_modality(dataframe = only_timepoint_one,input_features =(olinikiloc,cytof_freiloc,clinicaliloc, facsiloc ) ,data_modality_name = "all_CyTOF")
run_modality(dataframe = only_timepoint_one,input_features =(olinikiloc,cytof_Abiloc,clinicaliloc, facsiloc ) ,data_modality_name = "all_CyTOFab")

sys.stdout.close()