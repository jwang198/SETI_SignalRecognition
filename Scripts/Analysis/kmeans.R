# Iterative K-Means: 
# 1. Select k based on kink in silhouette score v. k graph
# 2. Remove cluster with best silhouette score
# 3. Iteratively, continue conducting k-means clustering (return to step 1)

set.seed(341)
raw_X <- read.table("./dataset/Full/DATA.csv", header = T, sep = ",", row.name = "id")

# Normalize data
X <- scale(raw_X)
X[,1:5] <- X[,1:5]*sqrt(63)
dim(X)
colMeans(X)

# 68 FEATURES TOTAL

# Iterative K-Means Clustering

### Choose Number of Clusters: 8
wss <- (nrow(X)-1)*sum(apply(X,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(X,centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

fit <- kmeans(X, centers=8, nstart=10)
aggregate(X,by=list(fit$cluster),FUN=mean)

# Cluster Plot against 1st 2 principal components

# vary parameters for most readable graph
library(cluster) 
clusplot(X, fit$cluster, color=TRUE, shade=TRUE, 
         labels=1, lines=0, cex=1)
#text(X, labels=fit$cluster, col=fit$cluster)

# Centroid Plot against 1st 2 discriminant functions
library(fpc)
plotcluster(X, fit$cluster,cex=1)

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
X <- data.frame(X[X[,70] != 1,], fit$cluster)

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
X <- data.frame(X[X[,71] != 1,], fit$cluster)

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

##################################
X <- data.frame(X[X[,72] != 1,], fit$cluster)

# K-Means Clustering: Round 6 (Minus Cluster 1''''')
fit <- kmeans(X[X[,73] != 1,1:68], centers=3, nstart=10)
aggregate(X[X[,73] != 1,1:68],by=list(fit$cluster),FUN=mean)

# Cluster Plot against 1st 2 principal components

# vary parameters for most readable graph
library(cluster) 
clusplot(X[X[,73] != 1,1:68], fit$cluster, color=TRUE, shade=TRUE, 
         labels=1, lines=0, cex=1)

# Centroid Plot against 1st 2 discriminant functions
library(fpc)
plotcluster(X[X[,73] != 1,1:68], fit$cluster,cex=1)

##################################
X <- data.frame(X[X[,73] != 1,], fit$cluster)

# K-Means Clustering: Round 7 (Minus Cluster 1'''''')
fit <- kmeans(X[X[,74] != 1,1:68], centers=3, nstart=10)
aggregate(X[X[,74] != 1,1:68],by=list(fit$cluster),FUN=mean)

# Cluster Plot against 1st 2 principal components

# vary parameters for most readable graph
library(cluster) 
clusplot(X[X[,74] != 1,1:68], fit$cluster, color=TRUE, shade=TRUE, 
         labels=1, lines=0, cex=1)

# Centroid Plot against 1st 2 discriminant functions
library(fpc)
plotcluster(X[X[,74] != 1,1:68], fit$cluster,cex=1)

#Cluster using only modulation and variance
####################################### 
X <- X[,4:5]

# K-Means Clustering
fit <- kmeans(X, centers=8, nstart=10)
aggregate(X,by=list(fit$cluster),FUN=mean)

# Cluster Plot against 1st 2 principal components

# vary parameters for most readable graph
library(cluster) 
clusplot(X, fit$cluster, color=TRUE, shade=TRUE, 
         labels=1, lines=0, cex=1)

# Centroid Plot against 1st 2 discriminant functions
library(fpc)
plotcluster(X, fit$cluster,cex=1)

wss <- (nrow(X)-1)*sum(apply(X,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(X,centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

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