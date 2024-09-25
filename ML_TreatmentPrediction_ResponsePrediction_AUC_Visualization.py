exec(open("../ML_Response_prediction_HelperScript.py").read())
rc_ticks = {
    "figure.figsize": (1.5, 1.2),
    "text.usetex": False,
    "font.size": 8,
    "legend.fontsize":4,
    "legend.title_fontsize":4,
    "xtick.labelsize":6,
    "ytick.labelsize": 6,
    "axes.labelsize":8,
    "axes.titlesize":8,
    "lines.linewidth": 0.5,
    "axes.linewidth": 0.4,
    "lines.markersize": 3,
    "xtick.major.size": 2.5,
    "xtick.major.width": 0.5,
    "xtick.major.pad": 1,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.5,
    "xtick.minor.pad": 1,
    "ytick.major.size": 2.5,
    "ytick.major.width": 0.5,
    "ytick.major.pad": 1,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    "ytick.minor.pad": 1,
    "axes.labelpad": 1,
    "axes.titlepad": 1
}
sns.set_theme(context="talk", style="ticks", rc=rc_ticks)

clinicaliloc = range(12, 14)
cytof_freiloc = range(22, 38)
cytof_totaliloc = range(38, 54)
cytof_totalAbiloc = range(55, 70)
cytof_Abiloc = range(71, 87)
facsiloc = range(163, 193)
olinikiloc = range(87, 161)

metadata = pd.read_csv("../Metadata_file.csv", sep=';', index_col=0, decimal=',')

# Code Figure 4B
# ---------------------------------------------------------------------------------------------------
selected_features = ['IL8', 'VEGFA', 'CD8A', 'MCP-3', 'GDNF', 'CDCP1', 'CD244', 'IL7', 'OPG',
       'LAP TGF-beta-1', 'uPA', 'IL6', 'IL-17C', 'MCP-1', 'IL-17A', 'CXCL11',
       'AXIN1', 'TRAIL', 'CXCL9', 'CST5', 'OSM', 'CXCL1', 'CCL4', 'CD6', 'SCF',
       'IL18', 'SLAMF1', 'TGF-alpha', 'MCP-4', 'CCL11', 'TNFSF14', 'IL-10RA',
       'MMP-1', 'LIF-R', 'FGF-21', 'CCL19', 'IL-15RA', 'IL-10RB', 'IL-18R1',
       'PD-L1', 'CXCL5', 'TRANCE', 'HGF', 'IL-12B', 'MMP-10', 'IL10', 'TNF',
       'CCL23', 'CD5', 'CCL3', 'Flt3L', 'CXCL6', 'CXCL10', '4E-BP1', 'SIRT2',
       'CCL28', 'DNER', 'EN-RAGE', 'CD40', 'IFN-gamma', 'FGF-19', 'MCP-2',
       'CASP-8', 'CCL25', 'CX3CL1', 'TNFRSF9', 'NT-3', 'TWEAK', 'CCL20',
       'ST1A1', 'STAMBP', 'ADA', 'TNFB', 'CSF-1']
mean_fpr_olink, mean_tpr_olink, mean_auc_olink, std_auc_olink = get_input_data(data=only_treated,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="treatment")

selected_features = ['Leukocytes', 'CRP' ]
mean_fpr_clinic, mean_tpr_clinic, mean_auc_clinic, std_auc_clinic = get_input_data(data=only_treated,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="treatment")

selected_features = ['RorgT_Tbet_FACS', 'EOMES_FACS', 'GATA3_FACS',
       'RorgT_FACS', 'Tbet_FACS', 'Treg_FACS', 'IFNg_FACS', 'IL10_FACS',
       'IL17A_FACS', 'IL22_FACS', 'IL4_FACS', 'TNFa_FACS', 'IL10_Treg_FACS',
       'A4_FACs', 'B1_FACS', 'B7_FACS', 'CCR6_FACS', 'CCR7_FACS', 'CCR9_FACS',
       'CD25_FACS', 'CD38_FACS', 'CD103_FACS', 'CD127_FACS', 'CD161_FACS',
       'CTLA4_FACS', 'CXCR3_FACS', 'GPR15_FACS', 'HLADR_FACS',
       'Ki67_FACS', 'PD1_FACS']
mean_fpr_facs, mean_tpr_facs, mean_auc_facs, std_auc_facs = get_input_data(data=only_treated,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="treatment")

selected_features = ['Basophils', 'cDCs', 'cMonocytes', 'Eosinophils', 'gd T cells',
       'Memory B cells', 'Naive B cells', 'MAIT cells', 'Memory CD4',
       'Memory CD8', 'Naive CD4', 'Naive CD8', 'ncMonocytes', 'NK cells',
       'pDCs', 'Plasmablasts']
mean_fpr_cytof, mean_tpr_cytof, mean_auc_cytof, std_auc_cytof = get_input_data(data=only_treated,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="treatment")

selected_features = ['Basophils_A4B7', 'cDCs_A4B7', 'cMonocytes_A4B7', 'Eosinophils_A4B7',
       'gd T cells_A4B7', 'IgD- B cells_A4B7', 'IgD+ B cells_A4B7',
       'MAIT cells_A4B7', 'Memory CD4_A4B7', 'Memory CD8_A4B7',
       'Naive CD4_A4B7', 'Naive CD8_A4B7', 'ncMonocytes_A4B7', 'NK cells_A4B7',
       'pDCs_A4B7', 'Plasmablasts_A4B7']
mean_fpr_cytofab, mean_tpr_cytofab, mean_auc_cytofab, std_auc_cytofab = get_input_data(data=only_treated,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="treatment")

plt.figure()
lw = 1.25
plt.plot(mean_fpr_olink, mean_tpr_olink,
         label=r"OLINK, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_olink, std_auc_olink),
         color="#d8b365", lw=lw)
plt.plot(mean_fpr_clinic, mean_tpr_clinic,
         label=r"Clinic, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_clinic, std_auc_clinic),
         color="#8F8F8F", lw=lw)
plt.plot(mean_fpr_facs, mean_tpr_facs,
         label=r"FACS, AUC %0.2f $\pm$ %0.2f" % (mean_auc_facs, std_auc_facs),
         color="#8c510a", lw=lw)
plt.plot(mean_fpr_cytof, mean_tpr_cytof,
         label=r"CyTOF, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_cytof, std_auc_cytof),
         color="#01665e", lw=lw)
plt.plot(mean_fpr_cytofab, mean_tpr_cytofab,
         label=r"CyTOF A4B7, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_cytofab, std_auc_cytofab),
         color="#5ab4ac", lw=lw)
plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Single Modalities")
plt.legend(loc="lower right")
plt.savefig('F_4B.pdf', dpi=500)

# Code Figure 5B(upper left)
# ---------------------------------------------------------------------------------------------------
only_timepoint_one = metadata[metadata["timepoint"] == 1]
only_treated = metadata[metadata["treatment"] == "VDZ"]

selected_features = ['IL8', 'VEGFA', 'CD8A', 'MCP-3', 'GDNF', 'CDCP1', 'CD244', 'IL7', 'OPG',
       'LAP TGF-beta-1', 'uPA', 'IL6', 'IL-17C', 'MCP-1', 'IL-17A', 'CXCL11',
       'AXIN1', 'TRAIL', 'CXCL9', 'CST5', 'OSM', 'CXCL1', 'CCL4', 'CD6', 'SCF',
       'IL18', 'SLAMF1', 'TGF-alpha', 'MCP-4', 'CCL11', 'TNFSF14', 'IL-10RA',
       'MMP-1', 'LIF-R', 'FGF-21', 'CCL19', 'IL-15RA', 'IL-10RB', 'IL-18R1',
       'PD-L1', 'CXCL5', 'TRANCE', 'HGF', 'IL-12B', 'MMP-10', 'IL10', 'TNF',
       'CCL23', 'CD5', 'CCL3', 'Flt3L', 'CXCL6', 'CXCL10', '4E-BP1', 'SIRT2',
       'CCL28', 'DNER', 'EN-RAGE', 'CD40', 'IFN-gamma', 'FGF-19', 'MCP-2',
       'CASP-8', 'CCL25', 'CX3CL1', 'TNFRSF9', 'NT-3', 'TWEAK', 'CCL20',
       'ST1A1', 'STAMBP', 'ADA', 'TNFB', 'CSF-1']
mean_fpr_olink, mean_tpr_olink, mean_auc_olink, std_auc_olink = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")


selected_features = ['Leukocytes', 'CRP' ]
mean_fpr_clinic, mean_tpr_clinic, mean_auc_clinic, std_auc_clinic = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['RorgT_Tbet_FACS', 'EOMES_FACS', 'GATA3_FACS',
       'RorgT_FACS', 'Tbet_FACS', 'Treg_FACS', 'IFNg_FACS', 'IL10_FACS',
       'IL17A_FACS', 'IL22_FACS', 'IL4_FACS', 'TNFa_FACS', 'IL10_Treg_FACS',
       'A4_FACs', 'B1_FACS', 'B7_FACS', 'CCR6_FACS', 'CCR7_FACS', 'CCR9_FACS',
       'CD25_FACS', 'CD38_FACS', 'CD103_FACS', 'CD127_FACS', 'CD161_FACS',
       'CTLA4_FACS', 'CXCR3_FACS', 'GPR15_FACS', 'HLADR_FACS',
       'Ki67_FACS', 'PD1_FACS']
mean_fpr_facs, mean_tpr_facs, mean_auc_facs, std_auc_facs = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['Basophils', 'cDCs', 'cMonocytes', 'Eosinophils', 'gd T cells',
       'Memory B cells', 'Naive B cells', 'MAIT cells', 'Memory CD4',
       'Memory CD8', 'Naive CD4', 'Naive CD8', 'ncMonocytes', 'NK cells',
       'pDCs', 'Plasmablasts']
mean_fpr_cytof, mean_tpr_cytof, mean_auc_cytof, std_auc_cytof = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['Basophils_A4B7', 'cDCs_A4B7', 'cMonocytes_A4B7', 'Eosinophils_A4B7',
       'gd T cells_A4B7', 'IgD- B cells_A4B7', 'IgD+ B cells_A4B7',
       'MAIT cells_A4B7', 'Memory CD4_A4B7', 'Memory CD8_A4B7',
       'Naive CD4_A4B7', 'Naive CD8_A4B7', 'ncMonocytes_A4B7', 'NK cells_A4B7',
       'pDCs_A4B7', 'Plasmablasts_A4B7']
mean_fpr_cytofab, mean_tpr_cytofab, mean_auc_cytofab, std_auc_cytofab = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

plt.figure()
lw = 1.25
plt.plot(mean_fpr_olink, mean_tpr_olink,
         label=r"OLINK, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_olink, std_auc_olink),
         color="#d8b365", lw=lw)
plt.plot(mean_fpr_clinic, mean_tpr_clinic,
         label=r"Clinic, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_clinic, std_auc_clinic),
         color="#8F8F8F", lw=lw)
plt.plot(mean_fpr_facs, mean_tpr_facs,
         label=r"FACS, AUC %0.2f $\pm$ %0.2f" % (mean_auc_facs, std_auc_facs),
         color="#8c510a", lw=lw)
plt.plot(mean_fpr_cytof, mean_tpr_cytof,
         label=r"CyTOF, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_cytof, std_auc_cytof),
         color="#01665e", lw=lw)
plt.plot(mean_fpr_cytofab, mean_tpr_cytofab,
         label=r"CyTOF A4B7, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_cytofab, std_auc_cytofab),
         color="#5ab4ac", lw=lw)
plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Single Modalities")
plt.plot(legend=None)
plt.savefig('F_5B_upper_left.pdf', dpi=500)


# Code Figure 5B (lower left)
# ---------------------------------------------------------------------------------------------------
selected_features = ['TNF', 'TNFB', 'IL-10RA', 'MCP-3', 'IL18', 'MMP-10', 'CASP-8']
mean_fpr_olink, mean_tpr_olink, mean_auc_olink, std_auc_olink = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['CRP']
mean_fpr_clinic, mean_tpr_clinic, mean_auc_clinic, std_auc_clinic = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['Ki67_FACS', 'HLADR_FACS', 'GPR15_FACS', 'CCR7_FACS', 'CD127_FACS',
       'IL17A_FACS']
mean_fpr_facs, mean_tpr_facs, mean_auc_facs, std_auc_facs = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['Plasmablasts', 'Naive CD8', 'Memory B cells']
mean_fpr_cytof, mean_tpr_cytof, mean_auc_cytof, std_auc_cytof = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['Basophils_A4B7', 'ncMonocytes_A4B7', 'MAIT cells_A4B7']
mean_fpr_cytofab, mean_tpr_cytofab, mean_auc_cytofab, std_auc_cytofab = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

plt.figure()
lw = 1.25
plt.plot(mean_fpr_olink, mean_tpr_olink,
         label=r"OLINK, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_olink, std_auc_olink),
         color="#d8b365", lw=lw)
plt.plot(mean_fpr_clinic, mean_tpr_clinic,
         label=r"Clinic, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_clinic, std_auc_clinic),
         color="#8F8F8F", lw=lw)
plt.plot(mean_fpr_facs, mean_tpr_facs,
         label=r"FACS, AUC %0.2f $\pm$ %0.2f" % (mean_auc_facs, std_auc_facs),
         color="#8c510a", lw=lw)
plt.plot(mean_fpr_cytof, mean_tpr_cytof,
         label=r"CyTOF, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_cytof, std_auc_cytof),
         color="#01665e", lw=lw)
plt.plot(mean_fpr_cytofab, mean_tpr_cytofab,
         label=r"CyTOF A4B7, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_cytofab, std_auc_cytofab),
         color="#5ab4ac", lw=lw)
plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Single Modalities")
plt.plot(legend=None)
plt.savefig('F_5B_lowerleft.pdf', dpi=500)

# Code Figure 5B(upper right)
# ---------------------------------------------------------------------------------------------------
selected_features =  ['IL8', 'VEGFA', 'CD8A', 'MCP-3', 'GDNF', 'CDCP1', 'CD244', 'IL7', 'OPG',
       'LAP TGF-beta-1', 'uPA', 'IL6', 'IL-17C', 'MCP-1', 'IL-17A', 'CXCL11',
       'AXIN1', 'TRAIL', 'CXCL9', 'CST5', 'OSM', 'CXCL1', 'CCL4', 'CD6', 'SCF',
       'IL18', 'SLAMF1', 'TGF-alpha', 'MCP-4', 'CCL11', 'TNFSF14', 'IL-10RA',
       'MMP-1', 'LIF-R', 'FGF-21', 'CCL19', 'IL-15RA', 'IL-10RB', 'IL-18R1',
       'PD-L1', 'CXCL5', 'TRANCE', 'HGF', 'IL-12B', 'MMP-10', 'IL10', 'TNF',
       'CCL23', 'CD5', 'CCL3', 'Flt3L', 'CXCL6', 'CXCL10', '4E-BP1', 'SIRT2',
       'CCL28', 'DNER', 'EN-RAGE', 'CD40', 'IFN-gamma', 'FGF-19', 'MCP-2',
       'CASP-8', 'CCL25', 'CX3CL1', 'TNFRSF9', 'NT-3', 'TWEAK', 'CCL20',
       'ST1A1', 'STAMBP', 'ADA', 'TNFB', 'CSF-1','RorgT_Tbet_FACS', 'EOMES_FACS', 'GATA3_FACS',
       'RorgT_FACS', 'Tbet_FACS', 'Treg_FACS', 'IFNg_FACS', 'IL10_FACS',
       'IL17A_FACS', 'IL22_FACS', 'IL4_FACS', 'TNFa_FACS', 'IL10_Treg_FACS',
       'A4_FACs', 'B1_FACS', 'B7_FACS', 'CCR6_FACS', 'CCR7_FACS', 'CCR9_FACS',
       'CD25_FACS', 'CD38_FACS', 'CD103_FACS', 'CD127_FACS', 'CD161_FACS',
       'CTLA4_FACS', 'CXCR3_FACS', 'GPR15_FACS', 'HLADR_FACS',
       'Ki67_FACS', 'PD1_FACS' ]
mean_fpr_olink_facs, mean_tpr_olink_facs, mean_auc_olink_facs, std_auc_olink_facs = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features =  ['IL8', 'VEGFA', 'CD8A', 'MCP-3', 'GDNF', 'CDCP1', 'CD244', 'IL7', 'OPG',
       'LAP TGF-beta-1', 'uPA', 'IL6', 'IL-17C', 'MCP-1', 'IL-17A', 'CXCL11',
       'AXIN1', 'TRAIL', 'CXCL9', 'CST5', 'OSM', 'CXCL1', 'CCL4', 'CD6', 'SCF',
       'IL18', 'SLAMF1', 'TGF-alpha', 'MCP-4', 'CCL11', 'TNFSF14', 'IL-10RA',
       'MMP-1', 'LIF-R', 'FGF-21', 'CCL19', 'IL-15RA', 'IL-10RB', 'IL-18R1',
       'PD-L1', 'CXCL5', 'TRANCE', 'HGF', 'IL-12B', 'MMP-10', 'IL10', 'TNF',
       'CCL23', 'CD5', 'CCL3', 'Flt3L', 'CXCL6', 'CXCL10', '4E-BP1', 'SIRT2',
       'CCL28', 'DNER', 'EN-RAGE', 'CD40', 'IFN-gamma', 'FGF-19', 'MCP-2',
       'CASP-8', 'CCL25', 'CX3CL1', 'TNFRSF9', 'NT-3', 'TWEAK', 'CCL20',
       'ST1A1', 'STAMBP', 'ADA', 'TNFB', 'CSF-1','RorgT_Tbet_FACS', 'EOMES_FACS', 'GATA3_FACS',
       'RorgT_FACS', 'Tbet_FACS', 'Treg_FACS', 'IFNg_FACS', 'IL10_FACS',
       'IL17A_FACS', 'IL22_FACS', 'IL4_FACS', 'TNFa_FACS', 'IL10_Treg_FACS',
       'A4_FACs', 'B1_FACS', 'B7_FACS', 'CCR6_FACS', 'CCR7_FACS', 'CCR9_FACS',
       'CD25_FACS', 'CD38_FACS', 'CD103_FACS', 'CD127_FACS', 'CD161_FACS',
       'CTLA4_FACS', 'CXCR3_FACS', 'GPR15_FACS', 'HLADR_FACS',
       'Ki67_FACS', 'PD1_FACS', 'Leukocytes', 'CRP']
mean_fpr_facs_clinical_olink, mean_tpr_facs_clinical_olink, mean_auc_facs_clinical_olink, std_auc_facs_clinical_olink = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['RorgT_Tbet_FACS', 'EOMES_FACS', 'GATA3_FACS',
       'RorgT_FACS', 'Tbet_FACS', 'Treg_FACS', 'IFNg_FACS', 'IL10_FACS',
       'IL17A_FACS', 'IL22_FACS', 'IL4_FACS', 'TNFa_FACS', 'IL10_Treg_FACS',
       'A4_FACs', 'B1_FACS', 'B7_FACS', 'CCR6_FACS', 'CCR7_FACS', 'CCR9_FACS',
       'CD25_FACS', 'CD38_FACS', 'CD103_FACS', 'CD127_FACS', 'CD161_FACS',
       'CTLA4_FACS', 'CXCR3_FACS', 'GPR15_FACS', 'HLADR_FACS',
       'Ki67_FACS', 'PD1_FACS', 'Basophils', 'cDCs', 'cMonocytes', 'Eosinophils', 'gd T cells',
       'Memory B cells', 'Naive B cells', 'MAIT cells', 'Memory CD4',
       'Memory CD8', 'Naive CD4', 'Naive CD8', 'ncMonocytes', 'NK cells',
       'pDCs', 'Plasmablasts','IL8', 'VEGFA', 'CD8A', 'MCP-3', 'GDNF', 'CDCP1', 'CD244', 'IL7', 'OPG',
       'LAP TGF-beta-1', 'uPA', 'IL6', 'IL-17C', 'MCP-1', 'IL-17A', 'CXCL11',
       'AXIN1', 'TRAIL', 'CXCL9', 'CST5', 'OSM', 'CXCL1', 'CCL4', 'CD6', 'SCF',
       'IL18', 'SLAMF1', 'TGF-alpha', 'MCP-4', 'CCL11', 'TNFSF14', 'IL-10RA',
       'MMP-1', 'LIF-R', 'FGF-21', 'CCL19', 'IL-15RA', 'IL-10RB', 'IL-18R1',
       'PD-L1', 'CXCL5', 'TRANCE', 'HGF', 'IL-12B', 'MMP-10', 'IL10', 'TNF',
       'CCL23', 'CD5', 'CCL3', 'Flt3L', 'CXCL6', 'CXCL10', '4E-BP1', 'SIRT2',
       'CCL28', 'DNER', 'EN-RAGE', 'CD40', 'IFN-gamma', 'FGF-19', 'MCP-2',
       'CASP-8', 'CCL25', 'CX3CL1', 'TNFRSF9', 'NT-3', 'TWEAK', 'CCL20',
       'ST1A1', 'STAMBP', 'ADA', 'TNFB', 'CSF-1']
mean_fpr_facs_cytof_clinical, mean_tpr_facs_cytof_clinical, mean_auc_facs_cytof_clinical, std_auc_facs_cytof_clinical = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['RorgT_Tbet_FACS', 'EOMES_FACS', 'GATA3_FACS',
       'RorgT_FACS', 'Tbet_FACS', 'Treg_FACS', 'IFNg_FACS', 'IL10_FACS',
       'IL17A_FACS', 'IL22_FACS', 'IL4_FACS', 'TNFa_FACS', 'IL10_Treg_FACS',
       'A4_FACs', 'B1_FACS', 'B7_FACS', 'CCR6_FACS', 'CCR7_FACS', 'CCR9_FACS',
       'CD25_FACS', 'CD38_FACS', 'CD103_FACS', 'CD127_FACS', 'CD161_FACS',
       'CTLA4_FACS', 'CXCR3_FACS', 'GPR15_FACS', 'HLADR_FACS',
       'Ki67_FACS', 'PD1_FACS', 'Basophils', 'cDCs', 'cMonocytes', 'Eosinophils', 'gd T cells',
       'Memory B cells', 'Naive B cells', 'MAIT cells', 'Memory CD4',
       'Memory CD8', 'Naive CD4', 'Naive CD8', 'ncMonocytes', 'NK cells','IL8', 'VEGFA', 'CD8A', 'MCP-3', 'GDNF', 'CDCP1', 'CD244', 'IL7', 'OPG',
       'LAP TGF-beta-1', 'uPA', 'IL6', 'IL-17C', 'MCP-1', 'IL-17A', 'CXCL11',
       'AXIN1', 'TRAIL', 'CXCL9', 'CST5', 'OSM', 'CXCL1', 'CCL4', 'CD6', 'SCF',
       'IL18', 'SLAMF1', 'TGF-alpha', 'MCP-4', 'CCL11', 'TNFSF14', 'IL-10RA',
       'MMP-1', 'LIF-R', 'FGF-21', 'CCL19', 'IL-15RA', 'IL-10RB', 'IL-18R1',
       'PD-L1', 'CXCL5', 'TRANCE', 'HGF', 'IL-12B', 'MMP-10', 'IL10', 'TNF',
       'CCL23', 'CD5', 'CCL3', 'Flt3L', 'CXCL6', 'CXCL10', '4E-BP1', 'SIRT2',
       'CCL28', 'DNER', 'EN-RAGE', 'CD40', 'IFN-gamma', 'FGF-19', 'MCP-2',
       'CASP-8', 'CCL25', 'CX3CL1', 'TNFRSF9', 'NT-3', 'TWEAK', 'CCL20',
       'ST1A1', 'STAMBP', 'ADA', 'TNFB', 'CSF-1','pDCs', 'Plasmablasts','Leukocytes', 'CRP' ]
mean_fpr_allcytof, mean_tpr_allcytof, mean_auc_allcytof, std_auc_allcytof = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

fig, ax = plt.subplots()
plt.figure()
lw = 1.25
plt.plot(mean_fpr_olink_facs, mean_tpr_olink_facs,
         label=r"FACS & OLINK, = %0.2f $\pm$ %0.2f" % (mean_auc_olink_facs, std_auc_olink_facs),
         color="#bdd7e7", lw=lw)
plt.plot(mean_fpr_facs_clinical_olink, mean_tpr_facs_clinical_olink,
         label=r"FACS & OLINK & Clinical,= %0.2f $\pm$ %0.2f" % (mean_auc_facs_clinical_olink, std_auc_facs_clinical_olink),
         color="#6baed6", lw=lw)
plt.plot(mean_fpr_facs_cytof_clinical, mean_tpr_facs_cytof_clinical,
         label=r"FACS & CyTOF & OLINK, = %0.2f $\pm$ %0.2f" % (mean_auc_facs_cytof_clinical, std_auc_facs_cytof_clinical),
         color="#2171b5", lw=lw)
plt.plot(mean_fpr_allcytof, mean_tpr_allcytof,
         label=r"all CyTOF, = %0.2f $\pm$ %0.2f" % (mean_auc_allcytof, std_auc_allcytof),
         color="#000000", lw=lw)
plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Modality Combination")
plt.legend(loc="lower right")
plt.savefig('F_5B_upperright.pdf', dpi=500)

# Code Figure 5B(lower right)
# ---------------------------------------------------------------------------------------------------
selected_features =  ['Ki67_FACS', 'TNF', 'IL4_FACS', 'CXCR3_FACS', 'GPR15_FACS', 'B7_FACS',
       'IL-10RA', 'IL17A_FACS', 'NT-3', 'HLADR_FACS']
mean_fpr_olink_facs, mean_tpr_olink_facs, mean_auc_olink_facs, std_auc_olink_facs = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features =  ['Ki67_FACS', 'TNF', 'IL-10RA', 'NT-3', 'GPR15_FACS', 'CXCR3_FACS',
       'B7_FACS', 'CCR7_FACS', 'CASP-8', 'MCP-3', 'AXIN1']
mean_fpr_facs_clinical_olink, mean_tpr_facs_clinical_olink, mean_auc_facs_clinical_olink, std_auc_facs_clinical_olink = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['Ki67_FACS', 'NK cells', 'CXCR3_FACS', 'Treg_FACS', 'IL4_FACS',
       'GPR15_FACS', 'IFNg_FACS', 'IL-10RA', 'MCP-3', 'TNFB', 'CCR7_FACS',
       'CCL25']
mean_fpr_facs_cytof_clinical, mean_tpr_facs_cytof_clinical, mean_auc_facs_cytof_clinical, std_auc_facs_cytof_clinical = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")
selected_features = ['Ki67_FACS', 'NK cells', 'CXCR3_FACS', 'IL4_FACS', 'Treg_FACS',
       'GPR15_FACS', 'IL-10RA', 'IFNg_FACS', 'MCP-3', 'TNFB', 'CCL25',
       'CCR7_FACS']
mean_fpr_allcytof, mean_tpr_allcytof, mean_auc_allcytof, std_auc_allcytof = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

fig, ax = plt.subplots()
plt.figure()
lw = 1.25
plt.plot(mean_fpr_olink_facs, mean_tpr_olink_facs,
         label=r"FACS & OLINK, = %0.2f $\pm$ %0.2f" % (mean_auc_olink_facs, std_auc_olink_facs),
         color="#bdd7e7", lw=lw)
plt.plot(mean_fpr_facs_clinical_olink, mean_tpr_facs_clinical_olink,
         label=r"FACS & OLINK & Clinical,= %0.2f $\pm$ %0.2f" % (mean_auc_facs_clinical_olink, std_auc_facs_clinical_olink),
         color="#6baed6", lw=lw)
plt.plot(mean_fpr_facs_cytof_clinical, mean_tpr_facs_cytof_clinical,
         label=r"FACS & CyTOF & OLINK, = %0.2f $\pm$ %0.2f" % (mean_auc_facs_cytof_clinical, std_auc_facs_cytof_clinical),
         color="#2171b5", lw=lw)
plt.plot(mean_fpr_allcytof, mean_tpr_allcytof,
         label=r"all CyTOF, = %0.2f $\pm$ %0.2f" % (mean_auc_allcytof, std_auc_allcytof),
         color="#000000", lw=lw)
plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Modality Combination")
plt.legend(loc="lower right")
plt.savefig('../F_5B_lowerright.pdf', dpi=500)

# Code Figure 5E(upper)
# ---------------------------------------------------------------------------------------------------
selected_features =  ['Ki67_FACS']
mean_fpr_facs, mean_tpr_facs, mean_auc_facs, std_auc_facs = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features =  ['Ki67_FACS', 'GPR15_FACS', 'IFNg_FACS']
mean_fpr_facs3, mean_tpr_facs3, mean_auc_facs3, std_auc_facs3 = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features =  ['Ki67_FACS', 'GPR15_FACS', 'IFNg_FACS', "Treg_FACS", 'IL4_FACS']
mean_fpr_facs5, mean_tpr_facs5, mean_auc_facs5, std_auc_facs5 = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features =  ['Ki67_FACS', 'GPR15_FACS', 'IFNg_FACS', "Treg_FACS",'IL4_FACS', "CXCR3_FACS", "B7_FACS", "HLADR_FACS", "IL17A_FACS"]
mean_fpr_facs9, mean_tpr_facs9, mean_auc_facs9, std_auc_facs9 = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features =  ['Ki67_FACS', 'GPR15_FACS', 'TNF']
mean_fpr_facs2_olink, mean_tpr_facs2_olink, mean_auc_facs2_olink, std_auc_facs2_olink = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features =  ['Ki67_FACS', 'GPR15_FACS', 'TNF', 'IFNg_FACS', 'CASP-8']
mean_fpr_facs5_olink, mean_tpr_facs5_olink, mean_auc_facs5_olink, std_auc_facs5_olink = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features =  ['Ki67_FACS', 'GPR15_FACS', 'TNF', 'IFNg_FACS', 'CASP-8', "Treg_FACS", "IL4_FACS", "CCL25", "IL-10RA", "CXCR3_FACS"]
mean_fpr_facs10_olink, mean_tpr_facs10_olink, mean_auc_facs10_olink, std_auc_facs10_olink = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

fig, ax = plt.subplots()
plt.figure()
lw = 1.25
plt.plot(mean_fpr_facs, mean_tpr_facs,
         label=r"only Ki67 = %0.2f $\pm$ %0.2f" % (mean_auc_facs, std_auc_facs),
         color="#FF9000", lw=lw)
plt.plot(mean_fpr_facs3, mean_tpr_facs3,
         label=r"Top3 FACS= %0.2f $\pm$ %0.2f" % (mean_auc_facs3, std_auc_facs3),
         color="#A55D00", lw=lw)
plt.plot(mean_fpr_facs5, mean_tpr_facs5,
         label=r"Top5 FACS= %0.2f $\pm$ %0.2f" % (mean_auc_facs5, std_auc_facs5),
         color="#6E3E00", lw=lw)
plt.plot(mean_fpr_facs9, mean_tpr_facs9,
         label=r"Top9 FACS= %0.2f $\pm$ %0.2f" % (mean_auc_facs9, std_auc_facs9),
         color="#392000", lw=lw)
plt.plot(mean_fpr_facs2_olink, mean_tpr_facs2_olink,
         label=r"Top3 FACS+OLINK= %0.2f $\pm$ %0.2f" % (mean_auc_facs2_olink, std_auc_facs2_olink),
         color="#BDD7E7", lw=lw)
plt.plot(mean_fpr_facs5_olink, mean_tpr_facs5_olink,
         label=r"Top5 FACS+OLINK= %0.2f $\pm$ %0.2f" % (mean_auc_facs5_olink, std_auc_facs5_olink),
         color="#98AEBC", lw=lw)
plt.plot(mean_fpr_facs5_olink, mean_tpr_facs5_olink,
         label=r"Top10 FACS+OLINK= %0.2f $\pm$ %0.2f" % (mean_auc_facs5_olink, std_auc_facs5_olink),
         color="#73838E", lw=lw)

plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Modality Combination")
plt.legend(loc="lower right")
plt.savefig('../F_5E.pdf', dpi=500)

# Code Figure 5F
# ---------------------------------------------------------------------------------------------------
# these are the patients were OLINK and FACS were measured
selected_patients = ['P1.1', 'P100.1', 'P101.1', 'P140.1', 'P141.1', 'P143.1', 'P144.1',
       'P147.1', 'P148.1', 'P15.1', 'P22.1', 'P27.1', 'P28.1', 'P37.1', 'P4.1',
       'P46.1', 'P5.1', 'P7.1', 'P8.1', 'P90.1', 'P92.1', 'P94.1']
random.seed(123)
selected_patients_red = random.sample(selected_patients, 15)

only_timepoint_one = metadata[metadata["timepoint"] == 1]
only_timepoint_one = only_timepoint_one.loc[selected_patients_red]
only_treated =metadata[metadata["treatment"] == "VDZ"]

selected_features =  ['Ki67_FACS']
mean_fpr_facs, mean_tpr_facs, mean_auc_facs, std_auc_facs = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features =  ['Ki67_FACS', 'GPR15_FACS', 'IFNg_FACS']
mean_fpr_facs3, mean_tpr_facs3, mean_auc_facs3, std_auc_facs3 = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features =  ['Ki67_FACS', 'GPR15_FACS', 'IFNg_FACS', "Treg_FACS", 'IL4_FACS']
mean_fpr_facs5, mean_tpr_facs5, mean_auc_facs5, std_auc_facs5 = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features =  ['Ki67_FACS', 'GPR15_FACS', 'IFNg_FACS', "Treg_FACS",'IL4_FACS', "CXCR3_FACS", "B7_FACS", "HLADR_FACS", "IL17A_FACS"]
mean_fpr_facs9, mean_tpr_facs9, mean_auc_facs9, std_auc_facs9 = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features =  ['Ki67_FACS', 'GPR15_FACS', 'TNF']
mean_fpr_facs2_olink, mean_tpr_facs2_olink, mean_auc_facs2_olink, std_auc_facs2_olink = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features =  ['Ki67_FACS', 'GPR15_FACS', 'TNF', 'IFNg_FACS', 'CASP-8']
mean_fpr_facs5_olink, mean_tpr_facs5_olink, mean_auc_facs5_olink, std_auc_facs5_olink = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features =  ['Ki67_FACS', 'GPR15_FACS', 'TNF', 'IFNg_FACS', 'CASP-8', "Treg_FACS", "IL4_FACS", "CCL25", "IL-10RA", "CXCR3_FACS"]
mean_fpr_facs10_olink, mean_tpr_facs10_olink, mean_auc_facs10_olink, std_auc_facs10_olink = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")
fig, ax = plt.subplots()
plt.figure()
lw = 1.25
plt.plot(mean_fpr_facs, mean_tpr_facs,
         label=r"only Ki67 = %0.2f $\pm$ %0.2f" % (mean_auc_facs, std_auc_facs),
         color="#252525", lw=lw)
plt.plot(mean_fpr_facs2_olink, mean_tpr_facs2_olink,
         label=r"Top3 FACS+OLINK= %0.2f $\pm$ %0.2f" % (mean_auc_facs2_olink, std_auc_facs2_olink),
         color="#636363", lw=lw)
plt.plot(mean_fpr_facs5_olink, mean_tpr_facs5_olink,
         label=r"Top5 FACS+OLINK= %0.2f $\pm$ %0.2f" % (mean_auc_facs5_olink, std_auc_facs5_olink),
         color="#969696", lw=lw)
plt.plot(mean_fpr_facs3, mean_tpr_facs3,
         label=r"Top3 FACS= %0.2f $\pm$ %0.2f" % (mean_auc_facs3, std_auc_facs3),
         color="#cccccc", lw=lw)

plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Modality Combination")
plt.legend(loc="lower right")
plt.savefig('../F_5F.pdf', dpi=500)

# Code for supplementary Figure 18 C
# ---------------------------------------------------------------------------------------------------
selected_patients = ['P1.1', 'P105.1', 'P141.1', 'P15.1', 'P17.1', 'P22.1', 'P27.1', 'P28.1', 'P37.1', 'P4.1', 'P5.1', 'P7.1', 'P8.1', 'P90.1', 'P92.1']

only_timepoint_one = metadata[metadata["timepoint"] == 1]
only_timepoint_one = only_timepoint_one.loc[selected_patients]
only_treated =metadata[metadata["treatment"] == "VDZ"]

selected_features = ['IL8', 'VEGFA', 'CD8A', 'MCP-3', 'GDNF', 'CDCP1', 'CD244', 'IL7', 'OPG',
       'LAP TGF-beta-1', 'uPA', 'IL6', 'IL-17C', 'MCP-1', 'IL-17A', 'CXCL11',
       'AXIN1', 'TRAIL', 'CXCL9', 'CST5', 'OSM', 'CXCL1', 'CCL4', 'CD6', 'SCF',
       'IL18', 'SLAMF1', 'TGF-alpha', 'MCP-4', 'CCL11', 'TNFSF14', 'IL-10RA',
       'MMP-1', 'LIF-R', 'FGF-21', 'CCL19', 'IL-15RA', 'IL-10RB', 'IL-18R1',
       'PD-L1', 'CXCL5', 'TRANCE', 'HGF', 'IL-12B', 'MMP-10', 'IL10', 'TNF',
       'CCL23', 'CD5', 'CCL3', 'Flt3L', 'CXCL6', 'CXCL10', '4E-BP1', 'SIRT2',
       'CCL28', 'DNER', 'EN-RAGE', 'CD40', 'IFN-gamma', 'FGF-19', 'MCP-2',
       'CASP-8', 'CCL25', 'CX3CL1', 'TNFRSF9', 'NT-3', 'TWEAK', 'CCL20',
       'ST1A1', 'STAMBP', 'ADA', 'TNFB', 'CSF-1']
mean_fpr_olink, mean_tpr_olink, mean_auc_olink, std_auc_olink = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['Leukocytes', 'CRP' ]
mean_fpr_clinic, mean_tpr_clinic, mean_auc_clinic, std_auc_clinic = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['RorgT_Tbet_FACS', 'EOMES_FACS', 'GATA3_FACS',
       'RorgT_FACS', 'Tbet_FACS', 'Treg_FACS', 'IFNg_FACS', 'IL10_FACS',
       'IL17A_FACS', 'IL22_FACS', 'IL4_FACS', 'TNFa_FACS', 'IL10_Treg_FACS',
       'A4_FACs', 'B1_FACS', 'B7_FACS', 'CCR6_FACS', 'CCR7_FACS', 'CCR9_FACS',
       'CD25_FACS', 'CD38_FACS', 'CD103_FACS', 'CD127_FACS', 'CD161_FACS',
       'CTLA4_FACS', 'CXCR3_FACS', 'GPR15_FACS', 'HLADR_FACS',
       'Ki67_FACS', 'PD1_FACS']
mean_fpr_facs, mean_tpr_facs, mean_auc_facs, std_auc_facs = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['Basophils', 'cDCs', 'cMonocytes', 'Eosinophils', 'gd T cells',
       'Memory B cells', 'Naive B cells', 'MAIT cells', 'Memory CD4',
       'Memory CD8', 'Naive CD4', 'Naive CD8', 'ncMonocytes', 'NK cells',
       'pDCs', 'Plasmablasts']
mean_fpr_cytof, mean_tpr_cytof, mean_auc_cytof, std_auc_cytof = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['Basophils_A4B7', 'cDCs_A4B7', 'cMonocytes_A4B7', 'Eosinophils_A4B7',
       'gd T cells_A4B7', 'IgD- B cells_A4B7', 'IgD+ B cells_A4B7',
       'MAIT cells_A4B7', 'Memory CD4_A4B7', 'Memory CD8_A4B7',
       'Naive CD4_A4B7', 'Naive CD8_A4B7', 'ncMonocytes_A4B7', 'NK cells_A4B7',
       'pDCs_A4B7', 'Plasmablasts_A4B7']
mean_fpr_cytofab, mean_tpr_cytofab, mean_auc_cytofab, std_auc_cytofab = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

plt.figure()
lw = 1.25
plt.plot(mean_fpr_olink, mean_tpr_olink,
         label=r"OLINK, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_olink, std_auc_olink),
         color="#d8b365", lw=lw)
plt.plot(mean_fpr_clinic, mean_tpr_clinic,
         label=r"Clinic, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_clinic, std_auc_clinic),
         color="#8F8F8F", lw=lw)
plt.plot(mean_fpr_facs, mean_tpr_facs,
         label=r"FACS, AUC %0.2f $\pm$ %0.2f" % (mean_auc_facs, std_auc_facs),
         color="#8c510a", lw=lw)
plt.plot(mean_fpr_cytof, mean_tpr_cytof,
         label=r"CyTOF, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_cytof, std_auc_cytof),
         color="#01665e", lw=lw)
plt.plot(mean_fpr_cytofab, mean_tpr_cytofab,
         label=r"CyTOF A4B7, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_cytofab, std_auc_cytofab),
         color="#5ab4ac", lw=lw)
plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Single Modalities")
plt.plot(legend=None)
plt.savefig('../S_18C_upper_left.pdf', dpi=500)

selected_features = ['TNF', 'TNFB', 'IL-10RA', 'MCP-3', 'IL18', 'MMP-10', 'CASP-8']
mean_fpr_olink, mean_tpr_olink, mean_auc_olink, std_auc_olink = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['CRP']
mean_fpr_clinic, mean_tpr_clinic, mean_auc_clinic, std_auc_clinic = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['Ki67_FACS', 'HLADR_FACS', 'GPR15_FACS', 'CCR7_FACS', 'CD127_FACS',
       'IL17A_FACS']
mean_fpr_facs, mean_tpr_facs, mean_auc_facs, std_auc_facs = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['Plasmablasts', 'Naive CD8', 'Memory B cells']
mean_fpr_cytof, mean_tpr_cytof, mean_auc_cytof, std_auc_cytof = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['Basophils_A4B7', 'ncMonocytes_A4B7', 'MAIT cells_A4B7']
mean_fpr_cytofab, mean_tpr_cytofab, mean_auc_cytofab, std_auc_cytofab = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

plt.figure()
lw = 1.25
plt.plot(mean_fpr_olink, mean_tpr_olink,
         label=r"OLINK, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_olink, std_auc_olink),
         color="#d8b365", lw=lw)
plt.plot(mean_fpr_clinic, mean_tpr_clinic,
         label=r"Clinic, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_clinic, std_auc_clinic),
         color="#8F8F8F", lw=lw)
plt.plot(mean_fpr_facs, mean_tpr_facs,
         label=r"FACS, AUC %0.2f $\pm$ %0.2f" % (mean_auc_facs, std_auc_facs),
         color="#8c510a", lw=lw)
plt.plot(mean_fpr_cytof, mean_tpr_cytof,
         label=r"CyTOF, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_cytof, std_auc_cytof),
         color="#01665e", lw=lw)
plt.plot(mean_fpr_cytofab, mean_tpr_cytofab,
         label=r"CyTOF A4B7, AUC = %0.2f $\pm$ %0.2f" % (mean_auc_cytofab, std_auc_cytofab),
         color="#5ab4ac", lw=lw)
plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Single Modalities")
plt.legend(loc="lower right")
plt.savefig('../S_18C_upper_right.pdf', dpi=500)

selected_features =  ['IL8', 'VEGFA', 'CD8A', 'MCP-3', 'GDNF', 'CDCP1', 'CD244', 'IL7', 'OPG',
       'LAP TGF-beta-1', 'uPA', 'IL6', 'IL-17C', 'MCP-1', 'IL-17A', 'CXCL11',
       'AXIN1', 'TRAIL', 'CXCL9', 'CST5', 'OSM', 'CXCL1', 'CCL4', 'CD6', 'SCF',
       'IL18', 'SLAMF1', 'TGF-alpha', 'MCP-4', 'CCL11', 'TNFSF14', 'IL-10RA',
       'MMP-1', 'LIF-R', 'FGF-21', 'CCL19', 'IL-15RA', 'IL-10RB', 'IL-18R1',
       'PD-L1', 'CXCL5', 'TRANCE', 'HGF', 'IL-12B', 'MMP-10', 'IL10', 'TNF',
       'CCL23', 'CD5', 'CCL3', 'Flt3L', 'CXCL6', 'CXCL10', '4E-BP1', 'SIRT2',
       'CCL28', 'DNER', 'EN-RAGE', 'CD40', 'IFN-gamma', 'FGF-19', 'MCP-2',
       'CASP-8', 'CCL25', 'CX3CL1', 'TNFRSF9', 'NT-3', 'TWEAK', 'CCL20',
       'ST1A1', 'STAMBP', 'ADA', 'TNFB', 'CSF-1','RorgT_Tbet_FACS', 'EOMES_FACS', 'GATA3_FACS',
       'RorgT_FACS', 'Tbet_FACS', 'Treg_FACS', 'IFNg_FACS', 'IL10_FACS',
       'IL17A_FACS', 'IL22_FACS', 'IL4_FACS', 'TNFa_FACS', 'IL10_Treg_FACS',
       'A4_FACs', 'B1_FACS', 'B7_FACS', 'CCR6_FACS', 'CCR7_FACS', 'CCR9_FACS',
       'CD25_FACS', 'CD38_FACS', 'CD103_FACS', 'CD127_FACS', 'CD161_FACS',
       'CTLA4_FACS', 'CXCR3_FACS', 'GPR15_FACS', 'HLADR_FACS',
       'Ki67_FACS', 'PD1_FACS' ]
mean_fpr_olink_facs, mean_tpr_olink_facs, mean_auc_olink_facs, std_auc_olink_facs = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features =  ['IL8', 'VEGFA', 'CD8A', 'MCP-3', 'GDNF', 'CDCP1', 'CD244', 'IL7', 'OPG',
       'LAP TGF-beta-1', 'uPA', 'IL6', 'IL-17C', 'MCP-1', 'IL-17A', 'CXCL11',
       'AXIN1', 'TRAIL', 'CXCL9', 'CST5', 'OSM', 'CXCL1', 'CCL4', 'CD6', 'SCF',
       'IL18', 'SLAMF1', 'TGF-alpha', 'MCP-4', 'CCL11', 'TNFSF14', 'IL-10RA',
       'MMP-1', 'LIF-R', 'FGF-21', 'CCL19', 'IL-15RA', 'IL-10RB', 'IL-18R1',
       'PD-L1', 'CXCL5', 'TRANCE', 'HGF', 'IL-12B', 'MMP-10', 'IL10', 'TNF',
       'CCL23', 'CD5', 'CCL3', 'Flt3L', 'CXCL6', 'CXCL10', '4E-BP1', 'SIRT2',
       'CCL28', 'DNER', 'EN-RAGE', 'CD40', 'IFN-gamma', 'FGF-19', 'MCP-2',
       'CASP-8', 'CCL25', 'CX3CL1', 'TNFRSF9', 'NT-3', 'TWEAK', 'CCL20',
       'ST1A1', 'STAMBP', 'ADA', 'TNFB', 'CSF-1','RorgT_Tbet_FACS', 'EOMES_FACS', 'GATA3_FACS',
       'RorgT_FACS', 'Tbet_FACS', 'Treg_FACS', 'IFNg_FACS', 'IL10_FACS',
       'IL17A_FACS', 'IL22_FACS', 'IL4_FACS', 'TNFa_FACS', 'IL10_Treg_FACS',
       'A4_FACs', 'B1_FACS', 'B7_FACS', 'CCR6_FACS', 'CCR7_FACS', 'CCR9_FACS',
       'CD25_FACS', 'CD38_FACS', 'CD103_FACS', 'CD127_FACS', 'CD161_FACS',
       'CTLA4_FACS', 'CXCR3_FACS', 'GPR15_FACS', 'HLADR_FACS',
       'Ki67_FACS', 'PD1_FACS', 'Leukocytes', 'CRP']
mean_fpr_facs_clinical_olink, mean_tpr_facs_clinical_olink, mean_auc_facs_clinical_olink, std_auc_facs_clinical_olink = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['RorgT_Tbet_FACS', 'EOMES_FACS', 'GATA3_FACS',
       'RorgT_FACS', 'Tbet_FACS', 'Treg_FACS', 'IFNg_FACS', 'IL10_FACS',
       'IL17A_FACS', 'IL22_FACS', 'IL4_FACS', 'TNFa_FACS', 'IL10_Treg_FACS',
       'A4_FACs', 'B1_FACS', 'B7_FACS', 'CCR6_FACS', 'CCR7_FACS', 'CCR9_FACS',
       'CD25_FACS', 'CD38_FACS', 'CD103_FACS', 'CD127_FACS', 'CD161_FACS',
       'CTLA4_FACS', 'CXCR3_FACS', 'GPR15_FACS', 'HLADR_FACS',
       'Ki67_FACS', 'PD1_FACS', 'Basophils', 'cDCs', 'cMonocytes', 'Eosinophils', 'gd T cells',
       'Memory B cells', 'Naive B cells', 'MAIT cells', 'Memory CD4',
       'Memory CD8', 'Naive CD4', 'Naive CD8', 'ncMonocytes', 'NK cells',
       'pDCs', 'Plasmablasts','IL8', 'VEGFA', 'CD8A', 'MCP-3', 'GDNF', 'CDCP1', 'CD244', 'IL7', 'OPG',
       'LAP TGF-beta-1', 'uPA', 'IL6', 'IL-17C', 'MCP-1', 'IL-17A', 'CXCL11',
       'AXIN1', 'TRAIL', 'CXCL9', 'CST5', 'OSM', 'CXCL1', 'CCL4', 'CD6', 'SCF',
       'IL18', 'SLAMF1', 'TGF-alpha', 'MCP-4', 'CCL11', 'TNFSF14', 'IL-10RA',
       'MMP-1', 'LIF-R', 'FGF-21', 'CCL19', 'IL-15RA', 'IL-10RB', 'IL-18R1',
       'PD-L1', 'CXCL5', 'TRANCE', 'HGF', 'IL-12B', 'MMP-10', 'IL10', 'TNF',
       'CCL23', 'CD5', 'CCL3', 'Flt3L', 'CXCL6', 'CXCL10', '4E-BP1', 'SIRT2',
       'CCL28', 'DNER', 'EN-RAGE', 'CD40', 'IFN-gamma', 'FGF-19', 'MCP-2',
       'CASP-8', 'CCL25', 'CX3CL1', 'TNFRSF9', 'NT-3', 'TWEAK', 'CCL20',
       'ST1A1', 'STAMBP', 'ADA', 'TNFB', 'CSF-1']
mean_fpr_facs_cytof_clinical, mean_tpr_facs_cytof_clinical, mean_auc_facs_cytof_clinical, std_auc_facs_cytof_clinical = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['RorgT_Tbet_FACS', 'EOMES_FACS', 'GATA3_FACS',
       'RorgT_FACS', 'Tbet_FACS', 'Treg_FACS', 'IFNg_FACS', 'IL10_FACS',
       'IL17A_FACS', 'IL22_FACS', 'IL4_FACS', 'TNFa_FACS', 'IL10_Treg_FACS',
       'A4_FACs', 'B1_FACS', 'B7_FACS', 'CCR6_FACS', 'CCR7_FACS', 'CCR9_FACS',
       'CD25_FACS', 'CD38_FACS', 'CD103_FACS', 'CD127_FACS', 'CD161_FACS',
       'CTLA4_FACS', 'CXCR3_FACS', 'GPR15_FACS', 'HLADR_FACS',
       'Ki67_FACS', 'PD1_FACS', 'Basophils', 'cDCs', 'cMonocytes', 'Eosinophils', 'gd T cells',
       'Memory B cells', 'Naive B cells', 'MAIT cells', 'Memory CD4',
       'Memory CD8', 'Naive CD4', 'Naive CD8', 'ncMonocytes', 'NK cells','IL8', 'VEGFA', 'CD8A', 'MCP-3', 'GDNF', 'CDCP1', 'CD244', 'IL7', 'OPG',
       'LAP TGF-beta-1', 'uPA', 'IL6', 'IL-17C', 'MCP-1', 'IL-17A', 'CXCL11',
       'AXIN1', 'TRAIL', 'CXCL9', 'CST5', 'OSM', 'CXCL1', 'CCL4', 'CD6', 'SCF',
       'IL18', 'SLAMF1', 'TGF-alpha', 'MCP-4', 'CCL11', 'TNFSF14', 'IL-10RA',
       'MMP-1', 'LIF-R', 'FGF-21', 'CCL19', 'IL-15RA', 'IL-10RB', 'IL-18R1',
       'PD-L1', 'CXCL5', 'TRANCE', 'HGF', 'IL-12B', 'MMP-10', 'IL10', 'TNF',
       'CCL23', 'CD5', 'CCL3', 'Flt3L', 'CXCL6', 'CXCL10', '4E-BP1', 'SIRT2',
       'CCL28', 'DNER', 'EN-RAGE', 'CD40', 'IFN-gamma', 'FGF-19', 'MCP-2',
       'CASP-8', 'CCL25', 'CX3CL1', 'TNFRSF9', 'NT-3', 'TWEAK', 'CCL20',
       'ST1A1', 'STAMBP', 'ADA', 'TNFB', 'CSF-1','pDCs', 'Plasmablasts','Leukocytes', 'CRP' ]
mean_fpr_allcytof, mean_tpr_allcytof, mean_auc_allcytof, std_auc_allcytof = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

fig, ax = plt.subplots()
plt.figure()
lw = 1.25
plt.plot(mean_fpr_olink_facs, mean_tpr_olink_facs,
         label=r"FACS & OLINK, = %0.2f $\pm$ %0.2f" % (mean_auc_olink_facs, std_auc_olink_facs),
         color="#bdd7e7", lw=lw)
plt.plot(mean_fpr_facs_clinical_olink, mean_tpr_facs_clinical_olink,
         label=r"FACS & OLINK & Clinical,= %0.2f $\pm$ %0.2f" % (mean_auc_facs_clinical_olink, std_auc_facs_clinical_olink),
         color="#6baed6", lw=lw)
plt.plot(mean_fpr_facs_cytof_clinical, mean_tpr_facs_cytof_clinical,
         label=r"FACS & CyTOF & OLINK, = %0.2f $\pm$ %0.2f" % (mean_auc_facs_cytof_clinical, std_auc_facs_cytof_clinical),
         color="#2171b5", lw=lw)
plt.plot(mean_fpr_allcytof, mean_tpr_allcytof,
         label=r"all CyTOF, = %0.2f $\pm$ %0.2f" % (mean_auc_allcytof, std_auc_allcytof),
         color="#000000", lw=lw)
plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Modality Combination")
plt.legend(loc="lower right")
plt.savefig('../S_18C_lower_left.pdf', dpi=500)

selected_features =  ['Ki67_FACS', 'TNF', 'IL4_FACS', 'CXCR3_FACS', 'GPR15_FACS', 'B7_FACS',
       'IL-10RA', 'IL17A_FACS', 'NT-3', 'HLADR_FACS']
mean_fpr_olink_facs, mean_tpr_olink_facs, mean_auc_olink_facs, std_auc_olink_facs = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features =  ['Ki67_FACS', 'TNF', 'IL-10RA', 'NT-3', 'GPR15_FACS', 'CXCR3_FACS',
       'B7_FACS', 'CCR7_FACS', 'CASP-8', 'MCP-3', 'AXIN1']
mean_fpr_facs_clinical_olink, mean_tpr_facs_clinical_olink, mean_auc_facs_clinical_olink, std_auc_facs_clinical_olink = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

selected_features = ['Ki67_FACS', 'NK cells', 'CXCR3_FACS', 'Treg_FACS', 'IL4_FACS',
       'GPR15_FACS', 'IFNg_FACS', 'IL-10RA', 'MCP-3', 'TNFB', 'CCR7_FACS',
       'CCL25']
mean_fpr_facs_cytof_clinical, mean_tpr_facs_cytof_clinical, mean_auc_facs_cytof_clinical, std_auc_facs_cytof_clinical = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")
selected_features = ['Ki67_FACS', 'NK cells', 'CXCR3_FACS', 'IL4_FACS', 'Treg_FACS',
       'GPR15_FACS', 'IL-10RA', 'IFNg_FACS', 'MCP-3', 'TNFB', 'CCL25',
       'CCR7_FACS']
mean_fpr_allcytof, mean_tpr_allcytof, mean_auc_allcytof, std_auc_allcytof = get_input_data(data=only_timepoint_one,
                                                                               what2choose=selected_features,
                                                                               treatment_or_response="response")

fig, ax = plt.subplots()
plt.figure()
lw = 1.25
plt.plot(mean_fpr_olink_facs, mean_tpr_olink_facs,
         label=r"FACS & OLINK, = %0.2f $\pm$ %0.2f" % (mean_auc_olink_facs, std_auc_olink_facs),
         color="#bdd7e7", lw=lw)
plt.plot(mean_fpr_facs_clinical_olink, mean_tpr_facs_clinical_olink,
         label=r"FACS & OLINK & Clinical,= %0.2f $\pm$ %0.2f" % (mean_auc_facs_clinical_olink, std_auc_facs_clinical_olink),
         color="#6baed6", lw=lw)
plt.plot(mean_fpr_facs_cytof_clinical, mean_tpr_facs_cytof_clinical,
         label=r"FACS & CyTOF & OLINK, = %0.2f $\pm$ %0.2f" % (mean_auc_facs_cytof_clinical, std_auc_facs_cytof_clinical),
         color="#2171b5", lw=lw)
plt.plot(mean_fpr_allcytof, mean_tpr_allcytof,
         label=r"all CyTOF, = %0.2f $\pm$ %0.2f" % (mean_auc_allcytof, std_auc_allcytof),
         color="#000000", lw=lw)
plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Modality Combination")
plt.legend(loc="lower right")
plt.savefig('../S_18C_lower_right.pdf', dpi=500)