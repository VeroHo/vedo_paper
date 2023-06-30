library(Seurat)
library(dplyr)

path <- getwd()
pbmc <- readRDS(file.path(path,'seurat','PBMC_combined.rds'))

study <- read.csv(file.path(path,'sample_sheet','s_hegazy_IBD.txt'),
                 sep='\t') %>%
  dplyr::mutate(Sample.Name=gsub('p','P',Sample.Name))
assay <- read.csv(file.path(path,'sample_sheet','a_hegazy_IBD_totalseq_nucleotide_sequencing.txt'),
                 sep='\t') %>%
  dplyr::mutate(Sample.Name=gsub('p','P',Sample.Name),
                Library.Name=gsub('p','P',Library.Name))

CD4.list <- lapply(pbmc, function(x) subset(x, predicted.celltype.l2 %in% c('CD4 CTL','CD4 Proliferating','CD4 TCM','CD4 TEM','Treg')))
for (nn in names(CD4.list)) {
  DefaultAssay(CD4.list[[nn]]) <- 'SCT'
}
CD4.list <- lapply(CD4.list, RunPCA, verbose=FALSE)

features <- SelectIntegrationFeatures(object.list = CD4.list, nfeatures = 2000, verbose=FALSE)
features <- features[!grepl('MT-|TR[AB][VDJ][0-9]|^IG[HLK][VDJ][0-9]',features)]

CD4 <- merge(CD4.list[[1]],CD4.list[2:length(CD4.list)],add.cell.ids=names(CD4.list)) %>%
  RunPCA(features=features, verbose = FALSE) %>%
  RunUMAP(reduction = "pca", dims = 1:20, verbose=FALSE) %>%
  FindNeighbors(dims = 1:20, verbose = FALSE) %>%
  FindClusters(verbose = FALSE)

CD4@meta.data <- CD4@meta.data %>%
  tibble::rownames_to_column('cell') %>%
  dplyr::left_join(study %>% 
                     dplyr::mutate(Patient.ID=gsub('[P]*([0-9]*)[\\/_][12]','P\\1',Characteristics.Patient.ID.),
                                   Responder=Characteristics.Responder.,
                                   Diagnosis=Characteristics.Diagnosis.,
                                   Batch=gsub(' ','',gsub('_scRNAseq','',Parameter.Value.Method.)),
                                   Timepoint=ifelse(grepl('_1_',Sample.Name),'Vedo_0d',
                                                    ifelse(grepl('_2_',Sample.Name),'Vedo_6wk','ctrl')),
                                   Material=ifelse(Characteristics.Material.=='PBMC','PBMC','CD4')) %>%
                     dplyr::select(Patient.ID,Responder,Diagnosis,Sample.Name,Material,Timepoint,Batch),
                   by=c('sample'='Sample.Name')) %>%
  tibble::column_to_rownames('cell')

# remove batch effect

CD4.list <- SplitObject(CD4, split.by='sample')
CD4.list <- lapply(CD4.list, SCTransform, verbose=FALSE)
features <- SelectIntegrationFeatures(object.list = CD4.list, nfeatures = 2000, verbose=FALSE)
features <- features[!grepl('MT-|TR[AB][VDJ][0-9]|^IG[HLK][VDJ][0-9]',features)]
refs <- which(grepl('CD4',names(CD4.list)))
CD4.list <- PrepSCTIntegration(object.list = CD4.list, anchor.features = features, verbose=FALSE)
anchors <- FindIntegrationAnchors(object.list = CD4.list, normalization.method = "SCT",
                                  anchor.features = features, reduction='rpca', verbose=FALSE,
                                  reference=refs)
CD4.integrated <- IntegrateData(anchorset = anchors, normalization.method = "SCT", verbose=FALSE) %>%
  ScaleData(verbose=FALSE) %>%
  RunPCA(verbose = FALSE) %>%
  RunUMAP(reduction = "pca", dims = 1:20, verbose=FALSE) %>%
  FindNeighbors(dims = 1:20, verbose = FALSE) %>%
  FindClusters(resolution=.5,verbose = FALSE)

saveRDS(CD4.integrated,file.path(path,'seurat','CD4_reclustered.rds'))
