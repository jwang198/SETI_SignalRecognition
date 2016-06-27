# Applications of various unsupervised machine learning methods to the problem of identifying squiggle subgroups
# Hierarchical, divisive, and k-means clustering using varying kernels + distance metrics

library(reshape2)
library(ggplot2)
library(cluster)
library(pvclust)
library(som)
library(pvclust)
library(fpc)
library(sets)
library(scatterplot3d)

raw_X <- read.table("./dataset/Full/DATA.csv", header = T, sep = ",", row.name = "id")
raw_X[,1:72] <- scale(raw_X[,1:72]) # Do not scale labels

# Scale up (by factor of sqrt(63)) non-DFT features 
raw_X[,1:9] <- raw_X[,1:9]*sqrt(63)

# Separate into nonsquiggle and squiggle
nonsquiggle <- raw_X[raw_X[,73] == 0,] 
squiggle <- raw_X[raw_X[,73] == 1,] 

# Remove the last column corresp. to labels
nonsquiggle <- nonsquiggle[,1:72]
squiggle <- squiggle[,1:72]

# Replace NA with 0
nonsquiggle[is.na(nonsquiggle)] = 0
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

#### CLUSTERING: POST-DIMENSIONALITY REDUCTION ####
squiggle_reduced = pca.5[,1:2]
k_chosen = 3
linkage_vector = c('average', 'single', 'complete', 'ward.D', 'ward.D2', 'mcquitty'
                   ,'median', 'centroid')
linkage <- 'ward.D2'

distance_vector = c('euclidean', 'manhattan', 'canberra')
distance <- 'manhattan'

# K-Means, Euclidean
fit <- kmeans(squiggle_reduced, centers=k_chosen, nstart=10)
plotcluster(squiggle_reduced, fit$cluster,cex=1, main='K-Means', xlab='Linear Discriminant Function 1', ylab='Linear Discriminant Function 2')
silhouette <- silhouette(fit$cluster, dist(squiggle_reduced, method='euclidean'))
summary(silhouette)

#silhouette <- abs(fit$withinss - fit$betweenss)/max(fit$withinss, fit$betweenss)

# Hierarchical, ? Linkage, Euclidean
hc <- hclust(dist(squiggle_reduced, method=distance), method = linkage)
plot(hc, hang = -1, cex = 0.01)
pruned <- cutree(hc, k=k_chosen)
rect.hclust(hc, k=k_chosen, cluster=pruned)

silhouette <- silhouette(pruned, dist(squiggle_reduced, method='euclidean'))
summary(silhouette)

plotcluster(squiggle_reduced, pruned, main='Hierarchical', xlab='Linear Discriminant Function 1', ylab='Linear Discriminant Function 2')

# Plot silhouette scores against number of clusters k = 1:10
silhouette_vector_kmeans <- rep(0,15)
for (k in 2:15) {
  fit <- kmeans(squiggle_reduced, centers=k, nstart=10)
  silhouette <- silhouette(fit$cluster, dist(squiggle_reduced, method='euclidean'))
  silhouette_vector_kmeans[k] <-  mean(summary(silhouette)$clus.avg.widths)
}

silhouette_vector_hc_wardD2_euc <- rep(0,15)
for (k_chosen in 2:15) {
  hc <- hclust(dist(squiggle_reduced, method='euclidean'), method = 'ward.D2')
  pruned <- cutree(hc, k=k_chosen)
  silhouette <- silhouette(pruned, dist(squiggle_reduced, method='euclidean'))
  silhouette_vector_hc_wardD2_euc[k_chosen] <-  mean(summary(silhouette)$clus.avg.widths)
}

silhouette_vector_hc_wardD1_euc <- rep(0,15)
for (k_chosen in 2:15) {
  hc <- hclust(dist(squiggle_reduced, method='manhattan'), method = 'ward.D')
  pruned <- cutree(hc, k=k_chosen)
  silhouette <- silhouette(pruned, dist(squiggle_reduced, method='manhattan'))
  silhouette_vector_hc_wardD1_euc[k_chosen] <-  mean(summary(silhouette)$clus.avg.widths)
}

silhouette_vector_hc_wardD2_man <- rep(0,15)
for (k_chosen in 2:15) {
  hc <- hclust(dist(squiggle_reduced, method='manhattan'), method = 'ward.D2')
  pruned <- cutree(hc, k=k_chosen)
  silhouette <- silhouette(pruned, dist(squiggle_reduced, method='manhattan'))
  silhouette_vector_hc_wardD2_man[k_chosen] <-  mean(summary(silhouette)$clus.avg.widths)
}

par(mar=c(4,4,2,1)) 
plot(2:15, silhouette_vector_kmeans[2:15], type='l', col='red', 
     main='Silhouette Score Varying K', ylab='Whole-Cluster Average Silhouette Score', 
     xlab='K (Number of Clusters)', ylim=c(0.30,0.60), lwd=3)
lines(2:15, silhouette_vector_hc_wardD2_euc[2:15], col='blue', lwd=3)
lines(2:15, silhouette_vector_hc_wardD2_man[2:15], col='green', lwd=3)
lines(2:15, silhouette_vector_hc_wardD1_euc[2:15], col='orange', lwd=3)

legend("topright", cex = 1,
       c("K-Means (Euclidean)","Hierarchical (Euclidean, Ward.D2)",
         "Hierarchical (Manhattan, Ward.D2)", 'Hierarchical (Euclidean, Ward.D1)'), 
       lty=c(1,1,1,1), # gives the legend appropriate symbols (lines)
       lwd=c(2.5,2.5,2.5,2.5),col=c("red","blue","green", 'orange'), pch=16)

# Purity Analysis
# Only on the three euclidean

hc1 <- hclust(dist(squiggle_reduced, method='euclidean'), method = 'ward.D2')
clusters1 <- cutree(hc1, k=4)
hc2 <- hclust(dist(squiggle_reduced, method='euclidean'), method = 'ward.D')
clusters2 <- cutree(hc2, k=4)
clusters3 <- kmeans(squiggle_reduced, centers=4, nstart=10)

#perm <- c(1,2,3,4)
#sims <- rep(0,1000)
#for (i in 1:1000) {
#  sims[(i-1)*4 + j] <- set_similarity(clusters1, clusters2, method = "Jaccard")
#}

# Plot 3-D clusters
clusters <- read.csv("/Users/Jason/Desktop/SETI_TimeSeries/clusts.csv", header=TRUE, sep=",", row.names="id")

# Visualize the data
pc1 <- pca.5[,1]
pc2 <- pca.5[,2]
pc3 <- pca.5[,3]

scatterplot3d(pc1, pc2, pc3, as.factor(clusters[,1]+1), xlab='Principal Component 1',
              ylab='Principal Component 2', zlab='Principal Component 3', main='Hierarchical, Ward D2')

legend("bottomright", cex = 1,
       c("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"),
       lty=c(1,1,1,1), # gives the legend appropriate symbols (lines)
       lwd=c(2.5,2.5,2.5,2.5),col=c("red","green", "blue", "cyan"), pch=16)

#### CLUSTERING USING FULL 72 FEATURES ####

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