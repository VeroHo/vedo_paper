# ML Vedolizumab response prediction
These scripts were used to generate the data presented in Horn et al. <br>
More specifically the illustration depicted in Figure 4, Figure 5, and supplementary Figure 18+19 and one technical Supplement. <br>

The structure is as follows: <br>
* ML_Response_prediction_HelperScript.py contains all helper function with were used for systematically running the classification task on the IBD data as well as AUC visualizations.
* ML_Treatment_efficacy_SystematicScanning.py contains the script with runs over all single modalities and every data combination aiming to classify treatment efficacy, providing AUC scores and feature coefficients.
* ML_Treatment_efficacy_visualization.rmd contains the visualizations scripts used to generate Figure 4.
* ML_Response_prediction_SystematicScanning.py contains the script with runs over all single modalities and every data combination aiming to classify treatment response, providing AUC scores and feature coefficients.
* ML_Response_prediction_visualization.rmd contains the visualizations scripts used to generate Figures 5 as well as Supplementary Figures 18.
* ML_TreatmentPrediction_ResponsePrediction_AUC_Visualization.py contains the script used to generate displayed ROC plots in Figure 4, Figure 5 (not external validation) and Supplementary Figures 18.
* ML_Iterative_kfold_model_validation_Run.py contains the run commands for the iterative k-fold cv runs to validate the 4 best performing models. 
* ML_Iteratuve_model_valdiation_StabililtySelection.rmd contains the R-Scripts which evaluation the stability of marker panels from the itaterive CV approach. It also generated the technical supplementary figure and Supplementary Figure 19.
* ML_Iterative_HelperFunctionsUpdated.py contains updated helper function used for the itertative CV approach.
* ML_ModelValidation_AUC_Visualizations.ipynb contains the AUC plos to produce Figure 5, and Supplementary Figure 19. Here the most stable iterations were tested and the predictive capacity of different marker combinations were tested alongside the external validation cohort.
