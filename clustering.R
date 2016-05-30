setwd('/Users/travischen/Box Sync/Sophomore/spring/cs229/SETI_TimeSeries/DATA/Complete(7657+833)')

raw_X <- read.table("Data.csv", header=TRUE, sep=",", row.names="id")
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

# Scale up (by factor of sqrt(63)) non-DFT features 
unknown[,1:9] <- unknown[,1:9]*sqrt(63)
squiggle[,1:9] <- squiggle[,1:9]*sqrt(63)

squiggleCopy = squiggle

###CLUSTERING CODE###

clusterNum = 0

###Round 1###
# Determine number of clusters
clusterNum = clusterNum + 1
wss <- (nrow(squiggleCopy)-1)*sum(apply(squiggleCopy,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(squiggleCopy, 
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")
# K-Means Cluster Analysis
fit <- kmeans(squiggleCopy, 5) 
# get cluster means 
aggregate(squiggleCopy,by=list(fit$cluster),FUN=mean)
# append cluster assignment
squiggleCopy <- data.frame(squiggleCopy, fit$cluster)
# vary parameters for most readable graph
closestCluster = which.min(fit$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")
cluster1 = squiggleCopy[squiggleCopy$fit.cluster == toString(closestCluster),]
cluster1$fit.cluster <- clusterNum
squiggleCopy = squiggleCopy[squiggleCopy$fit.cluster != toString(closestCluster),]
squiggleCopy$fit.cluster <- NULL

###Round 2###
clusterNum = clusterNum + 1
wss <- (nrow(squiggleCopy)-1)*sum(apply(squiggleCopy,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(squiggleCopy, 
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")
# K-Means Cluster Analysis
fit <- kmeans(squiggleCopy, 4) 
# get cluster means 
aggregate(squiggleCopy,by=list(fit$cluster),FUN=mean)
# append cluster assignment
squiggleCopy <- data.frame(squiggleCopy, fit$cluster)
# vary parameters for most readable graph
closestCluster = which.min(fit$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")
cluster2 = squiggleCopy[squiggleCopy$fit.cluster == toString(closestCluster),]
cluster2$fit.cluster <- clusterNum
squiggleCopy = squiggleCopy[squiggleCopy$fit.cluster != toString(closestCluster),]
squiggleCopy$fit.cluster <- NULL

###Round 3###
clusterNum = clusterNum + 1
wss <- (nrow(squiggleCopy)-1)*sum(apply(squiggleCopy,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(squiggleCopy, 
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")
# K-Means Cluster Analysis
fit <- kmeans(squiggleCopy, 3) 
# get cluster means 
aggregate(squiggleCopy,by=list(fit$cluster),FUN=mean)
# append cluster assignment
squiggleCopy <- data.frame(squiggleCopy, fit$cluster)
# vary parameters for most readable graph
closestCluster = which.min(fit$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")
cluster3 = squiggleCopy[squiggleCopy$fit.cluster == toString(closestCluster),]
cluster3$fit.cluster <- clusterNum
squiggleCopy = squiggleCopy[squiggleCopy$fit.cluster != toString(closestCluster),]
squiggleCopy$fit.cluster <- NULL


###Round 4###
clusterNum = clusterNum + 1
wss <- (nrow(squiggleCopy)-1)*sum(apply(squiggleCopy,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(squiggleCopy, 
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")
# K-Means Cluster Analysis
fit <- kmeans(squiggleCopy, 2) 
# get cluster means 
aggregate(squiggleCopy,by=list(fit$cluster),FUN=mean)
# append cluster assignment
squiggleCopy <- data.frame(squiggleCopy, fit$cluster)
# vary parameters for most readable graph
closestCluster = which.min(fit$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")
cluster4 = squiggleCopy[squiggleCopy$fit.cluster == toString(closestCluster),]
cluster4$fit.cluster <- clusterNum
squiggleCopy = squiggleCopy[squiggleCopy$fit.cluster != toString(closestCluster),]
squiggleCopy$fit.cluster <- NULL

###Round 4###
clusterNum = clusterNum + 1
wss <- (nrow(squiggleCopy)-1)*sum(apply(squiggleCopy,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(squiggleCopy, 
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")
# K-Means Cluster Analysis
fit <- kmeans(squiggleCopy, 1) 
# get cluster means 
aggregate(squiggleCopy,by=list(fit$cluster),FUN=mean)
# append cluster assignment
squiggleCopy <- data.frame(squiggleCopy, fit$cluster)
# vary parameters for most readable graph
closestCluster = which.min(fit$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")
cluster5 = squiggleCopy[squiggleCopy$fit.cluster == toString(closestCluster),]
cluster5$fit.cluster <- clusterNum
squiggleCopy = squiggleCopy[squiggleCopy$fit.cluster != toString(closestCluster),]
squiggleCopy$fit.cluster <- NULL

allClusters = rbind(cluster1, cluster2, cluster3, cluster4, cluster5)
write.csv(allClusters, file = 'sketch_iterative_kmeans.csv')
