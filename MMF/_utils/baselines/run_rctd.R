args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 3L)
input <- args[1]; output <- args[2]; lib <- args[3]
Sys.setenv(R_LIBS_USER=lib)
.libPaths(c(lib,.libPaths()))
suppressPackageStartupMessages(library(spacexr))
suppressPackageStartupMessages(library(Matrix))
dir.create(output, recursive=TRUE, showWarnings=FALSE)
started <- proc.time()[['elapsed']]
set.seed(240L)
genes <- readLines(file.path(input,'genes.txt'))
spatial <- as(readMM(file.path(input,'spatial_counts.mtx')),'dgCMatrix')
reference_counts <- as(readMM(file.path(input,'reference_counts.mtx')),'dgCMatrix')
rownames(spatial) <- rownames(reference_counts) <- genes
colnames(spatial) <- sprintf('spot%04d',seq_len(ncol(spatial)))
colnames(reference_counts) <- sprintf('ref%04d',seq_len(ncol(reference_counts)))
coords <- read.delim(file.path(input,'coords.tsv'),header=FALSE)
colnames(coords) <- c('x','y'); rownames(coords) <- colnames(spatial)
types <- factor(readLines(file.path(input,'reference_labels.txt')))
names(types) <- colnames(reference_counts)
ref <- Reference(reference_counts,types,colSums(reference_counts))
puck <- SpatialRNA(coords,spatial,colSums(spatial))
rctd <- create.RCTD(puck,ref,max_cores=4,test_mode=FALSE)
rctd <- run.RCTD(rctd,doublet_mode='full')
weights <- normalize_weights(rctd@results$weights)
weights <- weights[colnames(spatial),levels(types),drop=FALSE]
stopifnot(nrow(weights)==ncol(spatial),all(is.finite(as.matrix(weights))),
          all(weights>=0),max(abs(rowSums(weights)-1))<1e-8)
saveRDS(rctd,file.path(output,'fit.rds'))
write.csv(as.matrix(weights),file.path(output,'proportions.csv'))
runtime <- proc.time()[['elapsed']]-started
writeLines(c(sprintf('spacexr_version=%s',packageVersion('spacexr')),
             'mode=full','test_mode=FALSE','max_cores=4',
             sprintf('elapsed_seconds=%.6f',runtime),
             'evaluation_truth_read=FALSE','complete=TRUE'),file.path(output,'run_receipt.txt'))
capture.output(sessionInfo(),file=file.path(output,'sessionInfo.txt'))
cat('RCTD_COMPLETE elapsed_seconds=',runtime,'\n')
