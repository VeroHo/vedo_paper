# scRNAseq analysis for Horn et al.

## data access

- processed data will be available from NCBI GEO under accession [GSE261334](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE261334)
- `pbmc_multimodal.h5seurat` is available [here](https://atlas.fredhutch.org/data/nygc/multimodal/pbmc_multimodal.h5seurat) 

input data structure should looks like so (after stripping `GSMXXX_` prefixes from files):

```
data
├── cellranger
│   ├── HC-1_PBMC_filtered_feature_bc_matrix.h5
│   ├── HC-1_PBMC_vdj_t_filtered_contig_annotations.csv
│   ├── HC-1_PBMC_vdj_b_filtered_contig_annotations.csv
│   ├── IBD-1_1_PBMC_filtered_feature_bc_matrix.h5
│   ├── IBD-1_1_PBMC_vdj_t_filtered_contig_annotations.csv
│   ├── IBD-1_1_PBMC_vdj_b_filtered_contig_annotations.csv
│   ├── ...
├── seurat
│   └── pbmc_multimodal.h5seurat
└── vireo
    ├── IBD-2_IBD-6_HC-4_CD4_donor_ids.tsv
    ├── ...
```

## Seurat processing

use `processing.Rmd`

## CD4 reclustering

use `recluster_CD4.R`

## paper figures

all code for paper figures is in `paper_figures.Rmd`

## session info

**R version 4.0.3 (2020-10-10)**

**Platform:** x86_64-pc-linux-gnu (64-bit) 

**locale:**
_LC_CTYPE=en_US.UTF-8_, _LC_NUMERIC=C_, _LC_TIME=en_US.UTF-8_, _LC_COLLATE=en_US.UTF-8_, _LC_MONETARY=en_US.UTF-8_, _LC_MESSAGES=en_US.UTF-8_, _LC_PAPER=en_US.UTF-8_, _LC_NAME=C_, _LC_ADDRESS=C_, _LC_TELEPHONE=C_, _LC_MEASUREMENT=en_US.UTF-8_ and _LC_IDENTIFICATION=C_

**attached base packages:** 
_grid_, _parallel_, _stats4_, _stats_, _graphics_, _grDevices_, _utils_, _datasets_, _methods_ and _base_

**other attached packages:** 
_pander(v.0.6.5)_, _gtools(v.3.9.4)_, _Binarize(v.1.3)_, _diptest(v.0.76-0)_, _ComplexHeatmap(v.2.6.2)_, _ggpubr(v.0.5.0)_, _pheatmap(v.1.0.12)_, _ggalluvial(v.0.12.3)_, _fitdistrplus(v.1.1-8)_, _survival(v.3.5-0)_, _cluster(v.2.1.4)_, _scRepertoire(v.1.7.4)_, _SeuratDisk(v.0.0.0.9020)_, _sctransform(v.0.3.5)_, _akima(v.0.6-3.4)_, _MASS(v.7.3-58.2)_, _ggh4x(v.0.2.3)_, _scatterpie(v.0.1.8)_, _ggfortify(v.0.4.15)_, _RColorBrewer(v.1.1-3)_, _ggrepel(v.0.9.2)_, _tmod(v.0.50.11)_, _dendextend(v.1.16.0)_, _DESeq2(v.1.30.1)_, _SummarizedExperiment(v.1.20.0)_, _Biobase(v.2.50.0)_, _MatrixGenerics(v.1.2.1)_, _matrixStats(v.0.63.0)_, _GenomicRanges(v.1.42.0)_, _GenomeInfoDb(v.1.26.7)_, _IRanges(v.2.24.1)_, _S4Vectors(v.0.28.1)_, _BiocGenerics(v.0.36.1)_, _plyr(v.1.8.8)_, _stringr(v.1.5.0)_, _tidyr(v.1.3.0)_, _Matrix(v.1.5-3)_, _igraph(v.1.3.5)_, _cowplot(v.1.1.1)_, _ggplot2(v.3.4.0)_, _dplyr(v.1.1.0)_, _SeuratObject(v.4.1.3)_, _Seurat(v.4.3.0)_ and _reticulate(v.1.27)_

**loaded via a namespace (and not attached):** 
_SparseM(v.1.81)_, _scattermore(v.0.8)_, _evmix(v.2.12)_, _bit64(v.4.0.5)_, _knitr(v.1.42)_, _irlba(v.2.3.5.1)_, _DelayedArray(v.0.16.3)_, _data.table(v.1.14.6)_, _RCurl(v.1.98-1.10)_, _doParallel(v.1.0.17)_, _generics(v.0.1.3)_, _RSQLite(v.2.2.20)_, _RANN(v.2.6.1)_, _VGAM(v.1.1-7)_, _future(v.1.31.0)_, _bit(v.4.0.5)_, _spatstat.data(v.3.0-0)_, _httpuv(v.1.6.8)_, _viridis(v.0.6.2)_, _xfun(v.0.36)_, _plotwidgets(v.0.5.1)_, _evaluate(v.0.20)_, _promises(v.1.2.0.1)_, _fansi(v.1.0.4)_, _caTools(v.1.18.2)_, _DBI(v.1.1.3)_, _geneplotter(v.1.68.0)_, _htmlwidgets(v.1.6.1)_, _powerTCR(v.1.16.0)_, _spatstat.geom(v.3.0-5)_, _stringdist(v.0.9.10)_, _purrr(v.1.0.1)_, _ellipsis(v.0.3.2)_, _backports(v.1.4.1)_, _permute(v.0.9-7)_, _annotate(v.1.68.0)_, _deldir(v.1.0-6)_, _vctrs(v.0.5.2)_, _SingleCellExperiment(v.1.12.0)_, _Cairo(v.1.6-0)_, _ROCR(v.1.0-11)_, _abind(v.1.4-5)_, _cachem(v.1.0.6)_, _withr(v.2.5.0)_, _ggforce(v.0.4.1)_, _progressr(v.0.13.0)_, _vegan(v.2.6-4)_, _goftest(v.1.2-3)_, _gsl(v.2.1-8)_, _lazyeval(v.0.2.2)_, _crayon(v.1.5.2)_, _genefilter(v.1.72.1)_, _hdf5r(v.1.3.8)_, _spatstat.explore(v.3.0-5)_, _pkgconfig(v.2.0.3)_, _labeling(v.0.4.2)_, _tweenr(v.2.0.2)_, _nlme(v.3.1-161)_, _rlang(v.1.0.6)_, _globals(v.0.16.2)_, _lifecycle(v.1.0.3)_, _miniUI(v.0.1.1.1)_, _polyclip(v.1.10-4)_, _lmtest(v.0.9-40)_, _carData(v.3.0-5)_, _zoo(v.1.8-11)_, _beeswarm(v.0.4.0)_, _ggridges(v.0.5.4)_, _GlobalOptions(v.0.1.2)_, _png(v.0.1-8)_, _viridisLite(v.0.4.1)_, _rjson(v.0.2.21)_, _bitops(v.1.0-7)_, _KernSmooth(v.2.23-20)_, _blob(v.1.2.3)_, _shape(v.1.4.6)_, _parallelly(v.1.34.0)_, _spatstat.random(v.3.1-3)_, _rstatix(v.0.7.1)_, _ggsignif(v.0.6.4)_, _scales(v.1.2.1)_, _memoise(v.2.0.1)_, _magrittr(v.2.0.3)_, _ica(v.1.0-3)_, _gplots(v.3.1.3)_, _zlibbioc(v.1.36.0)_, _compiler(v.4.0.3)_, _clue(v.0.3-63)_, _cli(v.3.6.0)_, _XVector(v.0.30.0)_, _listenv(v.0.9.0)_, _patchwork(v.1.1.2)_, _pbapply(v.1.7-0)_, _mgcv(v.1.8-41)_, _tidyselect(v.1.2.0)_, _stringi(v.1.7.12)_, _yaml(v.2.3.7)_, _locfit(v.1.5-9.6)_, _tools(v.4.0.3)_, _future.apply(v.1.10.0)_, _circlize(v.0.4.15)_, _rstudioapi(v.0.14)_, _foreach(v.1.5.2)_, _tagcloud(v.0.6)_, _gridExtra(v.2.3)_, _cubature(v.2.0.4.6)_, _farver(v.2.1.1)_, _Rtsne(v.0.16)_, _ggraph(v.2.1.0)_, _digest(v.0.6.31)_, _shiny(v.1.7.4)_, _Rcpp(v.1.0.10)_, _car(v.3.1-1)_, _broom(v.1.0.3)_, _later(v.1.3.0)_, _RcppAnnoy(v.0.0.20)_, _httr(v.1.4.4)_, _AnnotationDbi(v.1.52.0)_, _colorspace(v.2.1-0)_, _XML(v.3.99-0.13)_, _tensor(v.1.5)_, _splines(v.4.0.3)_, _uwot(v.0.1.14)_, _spatstat.utils(v.3.0-1)_, _graphlayouts(v.0.8.4)_, _sp(v.1.6-0)_, _plotly(v.4.10.1)_, _xtable(v.1.8-4)_, _jsonlite(v.1.8.4)_, _truncdist(v.1.0-2)_, _tidygraph(v.1.2.3)_, _ggfun(v.0.0.9)_, _R6(v.2.5.1)_, _pillar(v.1.8.1)_, _htmltools(v.0.5.4)_, _mime(v.0.12)_, _DT(v.0.27)_, _glue(v.1.6.2)_, _fastmap(v.1.1.0)_, _BiocParallel(v.1.24.1)_, _codetools(v.0.2-18)_, _utf8(v.1.2.3)_, _lattice(v.0.20-45)_, _spatstat.sparse(v.3.0-0)_, _tibble(v.3.1.8)_, _evd(v.2.3-6.1)_, _leiden(v.0.4.3)_, _rmarkdown(v.2.20)_, _munsell(v.0.5.0)_, _GetoptLong(v.1.0.5)_, _GenomeInfoDbData(v.1.2.4)_, _iterators(v.1.0.14)_, _reshape2(v.1.4.4)_ and _gtable(v.0.3.1)_

