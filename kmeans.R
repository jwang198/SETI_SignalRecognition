raw_X <- read.table("/Users/Jason/Desktop/SETI_TimeSeries/DATA/official_ts_dataset_dft.csv", header=TRUE, sep=",", row.names="id")
#print(X1)

# Normalize data
X <- scale(raw_X)
X[,1:5] <- X[,1:5]*sqrt(63)
dim(X)
colMeans(X)

# 68 FEATURES TOTAL

# K-Means Clustering
fit <- kmeans(X, centers=3, nstart=10)
aggregate(X,by=list(fit$cluster),FUN=mean)

# Cluster Plot against 1st 2 principal components

# vary parameters for most readable graph
library(cluster) 
clusplot(X, fit$cluster, color=TRUE, shade=TRUE, 
         labels=1, lines=0, cex=1)

# Centroid Plot against 1st 2 discriminant functions
library(fpc)
plotcluster(X, fit$cluster,cex=1)

##################################
# Ward Hierarchical Clustering
d <- dist(X, method = "euclidean") # distance matrix
fit <- hclust(d, method="ward") 
plot(fit, cex=0.3) # display dendogram
groups <- cutree(fit, k=3) # cut tree into 5 clusters
# draw dendogram with red borders around the 5 clusters 
rect.hclust(fit, k=3, border="red")

##################################
# Model Based Clustering
library(mclust)
fit <- Mclust(X)
plot(fit) # plot results 
summary(fit) # display the best model

##################################
X <- data.frame(X, fit$cluster)

# K-Means Clustering: Round 2 (Minus Cluster 1)
fit <- kmeans(X[X[,69] != 1,1:68], centers=3, nstart=10)
aggregate(X[X[,69] != 1,1:68],by=list(fit$cluster),FUN=mean)

# Cluster Plot against 1st 2 principal components

# vary parameters for most readable graph
library(cluster) 
clusplot(X[X[,69] != 1,1:68], fit$cluster, color=TRUE, shade=TRUE, 
         labels=1, lines=0, cex=1)

# Centroid Plot against 1st 2 discriminant functions
library(fpc)
plotcluster(X[X[,69] != 1,1:68], fit$cluster,cex=1)

##################################
X <- data.frame(X[X[,69] != 1,], fit$cluster)

# K-Means Clustering: Round 3 (Minus Cluster 1'')
fit <- kmeans(X[X[,70] != 1,1:68], centers=3, nstart=10)
aggregate(X[X[,70] != 1,1:68],by=list(fit$cluster),FUN=mean)

# Cluster Plot against 1st 2 principal components

# vary parameters for most readable graph
library(cluster) 
clusplot(X[X[,70] != 1,1:68], fit$cluster, color=TRUE, shade=TRUE, 
         labels=1, lines=0, cex=1)

# Centroid Plot against 1st 2 discriminant functions
library(fpc)
plotcluster(X[X[,70] != 1,1:68], fit$cluster,cex=1)

##################################
X <- data.frame(X[X[,70] != 1,1:68], fit$cluster)

# K-Means Clustering: Round 4 (Minus Cluster 1''')
fit <- kmeans(X[X[,71] != 1,1:68], centers=3, nstart=10)
aggregate(X[X[,71] != 1,1:68],by=list(fit$cluster),FUN=mean)

# Cluster Plot against 1st 2 principal components

# vary parameters for most readable graph
library(cluster) 
clusplot(X[X[,71] != 1,1:68], fit$cluster, color=TRUE, shade=TRUE, 
         labels=1, lines=0, cex=1)

# Centroid Plot against 1st 2 discriminant functions
library(fpc)
plotcluster(X[X[,71] != 1,1:68], fit$cluster,cex=1)

##################################
X <- data.frame(X[X[,71] != 1,1:68], fit$cluster)

# K-Means Clustering: Round 5 (Minus Cluster 1'''')
fit <- kmeans(X[X[,72] != 1,1:68], centers=3, nstart=10)
aggregate(X[X[,72] != 1,1:68],by=list(fit$cluster),FUN=mean)

# Cluster Plot against 1st 2 principal components

# vary parameters for most readable graph
library(cluster) 
clusplot(X[X[,72] != 1,1:68], fit$cluster, color=TRUE, shade=TRUE, 
         labels=1, lines=0, cex=1)

# Centroid Plot against 1st 2 discriminant functions
library(fpc)
plotcluster(X[X[,72] != 1,1:68], fit$cluster,cex=1)
