library(reshape2)
library(ggplot2)
library(cluster)
library(pvclust)
library(som)
library(pvclust)
library(fpc)

raw_X <- read.table("/Users/Jason/Desktop/SETI_TimeSeries/DATA/Complete/Data.csv", header=TRUE, sep=",", row.names="id")
raw_X[,1:72] <- scale(raw_X[,1:72]) # Do not scale labels

# Scale up (by factor of sqrt(63)) non-DFT features 
raw_X[,1:9] <- raw_X[,1:9]*sqrt(63)

# Separate into unknown and squiggle
unknown <- raw_X[raw_X[,73] == 0,] 
squiggle <- raw_X[raw_X[,73] == 1,] 

# Remove the last column corresp. to labels
unknown <- unknown[,1:72]
squiggle <- squiggle[,1:72]

# Replace NA with 0
unknown[is.na(unknown)] = 0
squiggle[is.na(squiggle)] = 0

# Note that data is already centered and scaled
squiggle.pca <- prcomp(squiggle, center=FALSE, scale=FALSE)
plot(squiggle.pca, type='l')
summary(squiggle.pca)

# Get top 5 PCA components; plot all pairings
pca.5 = squiggle.pca$x[,1:5]
plot(pca.5[,1], pca.5[,2])
plot(pca.5[,1], pca.5[,3])
plot(pca.5[,1], pca.5[,4])
plot(pca.5[,1], pca.5[,5])

plot(pca.5[,2], pca.5[,3])
plot(pca.5[,2], pca.5[,4])
plot(pca.5[,2], pca.5[,5])

plot(pca.5[,3], pca.5[,4])
plot(pca.5[,3], pca.5[,5])

plot(pca.5[,4], pca.5[,5])

# Visualize using SOMs
# TODO: https://cran.r-project.org/web/packages/som/som.pdf

# Perform Hierarchical Clustering: Use different linkages

# Using full dataset
# Average Linkage
hc <- hclust(dist(squiggle, method='euclidean'), method = 'average')
plot(hc, hang = -1, cex = 0.01)
pruned <- cutree(hc, k=10)
rect.hclust(hc, k=10, cluster=pruned)

clusplot(squiggle, pruned, color=TRUE, shade=TRUE, labels=2, lines=0)
plotcluster(squiggle, pruned)

#fit <- pvclust(t(as.matrix(squiggle)), method.hclust="average",method.dist="euclidean") # Tranpose before using!
#plot(fit)

# Single
hc <- hclust(dist(squiggle, method='euclidean'), method = 'single')
plot(hc, hang = -1, cex = 0.01)
pruned <- cutree(hc, k=10)
rect.hclust(hc, k=10, cluster=pruned)

clusplot(squiggle, pruned, color=TRUE, shade=TRUE, labels=2, lines=0)
plotcluster(squiggle, pruned)

# Complete ****
hc <- hclust(dist(squiggle, method='euclidean'), method = 'complete')
plot(hc, hang = -1, cex = 0.01)
pruned <- cutree(hc, k=10)
rect.hclust(hc, k=10, cluster=pruned)

clusplot(squiggle, pruned, color=TRUE, shade=TRUE, labels=2, lines=0)
plotcluster(squiggle, pruned)

# Ward D ***
hc <- hclust(dist(squiggle, method='euclidean'), method = 'ward.D')
plot(hc, hang = -1, cex = 0.01)
pruned <- cutree(hc, k=10)
rect.hclust(hc, k=10, cluster=pruned)

clusplot(squiggle, pruned, color=TRUE, shade=TRUE, labels=2, lines=0)
plotcluster(squiggle, pruned)

# Ward D2 ***
hc <- hclust(dist(squiggle, method='euclidean'), method = 'ward.D2')
plot(hc, hang = -1, cex = 0.01)
pruned <- cutree(hc, k=10)
rect.hclust(hc, k=10, cluster=pruned)

clusplot(squiggle, pruned, color=TRUE, shade=TRUE, labels=2, lines=0)
plotcluster(squiggle, pruned)

# Mcquitty
hc <- hclust(dist(squiggle, method='euclidean'), method = 'mcquitty')
plot(hc, hang = -1, cex = 0.01)
pruned <- cutree(hc, k=10)
rect.hclust(hc, k=10, cluster=pruned)

clusplot(squiggle, pruned, color=TRUE, shade=TRUE, labels=2, lines=0)
plotcluster(squiggle, pruned)

# Median
hc <- hclust(dist(squiggle, method='euclidean'), method = 'median')
plot(hc, hang = -1, cex = 0.01)
pruned <- cutree(hc, k=10)
rect.hclust(hc, k=10, cluster=pruned)

clusplot(squiggle, pruned, color=TRUE, shade=TRUE, labels=2, lines=0)
plotcluster(squiggle, pruned)

# Centroid
hc <- hclust(dist(squiggle, method='euclidean'), method = 'centroid')
plot(hc, hang = -1, cex = 0.01)
pruned <- cutree(hc, k=10)
rect.hclust(hc, k=10, cluster=pruned)

clusplot(squiggle, pruned, color=TRUE, shade=TRUE, labels=2, lines=0)
plotcluster(squiggle, pruned)

#######################################
# Using top 2 PCA components

# Complete
pca.5 = squiggle.pca$x[,1:5]
hc <- hclust(dist(pca.5, method='euclidean'), method = 'complete')
plot(hc, hang = -1, cex = 0.01)
pruned <- cutree(hc, k=5)
rect.hclust(hc, k=5, cluster=pruned)

# Ward D
pca.5 = squiggle.pca$x[,1:5]
hc <- hclust(dist(pca.5, method='euclidean'), method = 'ward.D')
plot(hc, hang = -1, cex = 0.01)
pruned <- cutree(hc, k=5)
rect.hclust(hc, k=5, cluster=pruned)

# Ward D2
pca.5 = squiggle.pca$x[,1:5]
hc <- hclust(dist(pca.5, method='euclidean'), method = 'ward.D2')
plot(hc, hang = -1, cex = 0.01)
pruned <- cutree(hc, k=5)
rect.hclust(hc, k=5, cluster=pruned)

################# 
# DIVISIVE

# Complete dataset
div <- diana(squiggle, metric='euclidean')
pltree(div, cex = 0.01, hang = -1, main = "Dendrogram of diana")
div$dc





