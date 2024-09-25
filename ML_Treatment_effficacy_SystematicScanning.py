exec(open("../ML_Response_prediction_HelperScript.py").read())
import sys
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

sys.stdout = Logger('../ML_outputLog_treatment_efficacy.txt')

# Where are modality information stored?
clinicaliloc = range(12, 14)
cytof_freiloc = range(22, 38)
cytof_totaliloc = range(38, 54)
cytof_totalAbiloc = range(55, 70)
cytof_Abiloc = range(71, 87)
facsiloc = range(163, 193)
olinikiloc = range(87, 161)

metadata = pd.read_csv("../Metadata_file.csv", sep='\t', index_col=0, decimal=',')
only_treated = metadata[metadata["treatment"] == "VDZ"]

# single
run_modality(dataframe = only_treated,input_features =olinikiloc , data_modality_name = "onlyOLINK", treatment_or_response = "treatment")

contrast_data = only_treated.iloc[:, np.r_[clinicaliloc]]
contrast_data['Response'] = only_treated['timepoint']
X, y = Prepare_input(data_frame=contrast_data, treatment_or_response = "treatment")
print(str("onlyClinical") + ":\n================================================")
features = scan_over_models(X=X, y=y, data_type=str("onlyClinical"))

run_modality(dataframe = only_treated,input_features =cytof_freiloc , data_modality_name = "onlyCyTOF", treatment_or_response = "treatment")
run_modality(dataframe = only_treated,input_features =cytof_Abiloc , data_modality_name = "onlyCyTOFab", treatment_or_response = "treatment")
run_modality(dataframe = only_treated,input_features =facsiloc , data_modality_name = "onlyFACS", treatment_or_response = "treatment")

#duos
run_modality(dataframe = only_treated,input_features =(olinikiloc,clinicaliloc) ,data_modality_name = "OLINK_Clinical", treatment_or_response = "treatment")
run_modality(dataframe = only_treated,input_features =(olinikiloc,facsiloc) ,data_modality_name = "OLINK_FACS", treatment_or_response = "treatment")
run_modality(dataframe = only_treated,input_features =(olinikiloc,cytof_freiloc) ,data_modality_name = "OLINK_CyTOF", treatment_or_response = "treatment")
run_modality(dataframe = only_treated,input_features =(olinikiloc,cytof_Abiloc) ,data_modality_name = "OLINK_CyTOFab", treatment_or_response = "treatment")

run_modality(dataframe = only_treated,input_features =(facsiloc,cytof_freiloc) ,data_modality_name = "FACS_CyTOF", treatment_or_response = "treatment")
run_modality(dataframe = only_treated,input_features =(facsiloc,cytof_Abiloc) ,data_modality_name = "FACS_CyTOFab", treatment_or_response = "treatment")
run_modality(dataframe = only_treated,input_features =(facsiloc,clinicaliloc) ,data_modality_name = "FACS_Clinical", treatment_or_response = "treatment")

run_modality(dataframe = only_treated,input_features =(clinicaliloc,cytof_freiloc) ,data_modality_name = "Clinical_CyTOF", treatment_or_response = "treatment")
run_modality(dataframe = only_treated,input_features =(clinicaliloc,cytof_Abiloc) ,data_modality_name = "Clinical_CyTOFab", treatment_or_response = "treatment")

# trios
run_modality(dataframe = only_treated,input_features =(facsiloc,clinicaliloc,olinikiloc ) ,data_modality_name = "FACS_Clinical_OLINK", treatment_or_response = "treatment")
run_modality(dataframe = only_treated,input_features =(facsiloc,cytof_freiloc,olinikiloc ) ,data_modality_name = "FACS_CyTOF_OLINK", treatment_or_response = "treatment")
run_modality(dataframe = only_treated,input_features =(facsiloc,cytof_Abiloc,olinikiloc ) ,data_modality_name = "FACS_CyTOFab_OLINK", treatment_or_response = "treatment")

run_modality(dataframe = only_treated,input_features =(facsiloc,cytof_Abiloc,cytof_freiloc ) ,data_modality_name = "FACS_CyTOF_Clinical", treatment_or_response = "treatment")
run_modality(dataframe = only_treated,input_features =(facsiloc,cytof_Abiloc,clinicaliloc ) ,data_modality_name = "FACS_CyTOFab_Clinical", treatment_or_response = "treatment")

run_modality(dataframe = only_treated,input_features =(olinikiloc,cytof_freiloc,clinicaliloc ) ,data_modality_name = "OLINK_CyTOF_Clinical", treatment_or_response = "treatment")
run_modality(dataframe = only_treated,input_features =(olinikiloc,cytof_Abiloc,clinicaliloc ) ,data_modality_name = "OLINK_CyTOFab_Clinical", treatment_or_response = "treatment")


# four
run_modality(dataframe = only_treated,input_features =(olinikiloc,cytof_freiloc,clinicaliloc, facsiloc ) ,data_modality_name = "all_CyTOF", treatment_or_response = "treatment")
run_modality(dataframe = only_treated,input_features =(olinikiloc,cytof_Abiloc,clinicaliloc, facsiloc ) ,data_modality_name = "all_CyTOFab", treatment_or_response = "treatment")

sys.stdout.close()