library(glmnet)
library(ROCR)

#read data
raw_squiggle <- read.table("/Users/Jason/Desktop/SETI_TimeSeries/DATA/official_ts_dataset_dft.csv", header=TRUE, sep=",", row.names="id")
raw_nonsquiggle <- read.table("/Users/Jason/Desktop/SETI_TimeSeries/DATA/official_ts_dataset_nonsquiggle_dft.csv", header=TRUE, sep=",", row.names="id")

raw_squiggle <- cbind(raw_squiggle, label = T)
raw_nonsquiggle <- cbind(raw_nonsquiggle, label = F)

#put all the data together
data <- rbind(raw_squiggle, raw_nonsquiggle)
#scale data
data[,1:(ncol(data)-1)] <- scale(data[,1:(ncol(data)-1)]) #SCALE AFTER CONCATENATING
data[,1:5] <- data[,1:5]*sqrt(63) 

#randomize the dataset
data.randomized <- data[sample(nrow(data)), ]

#training and test splits. Also split by independent variables / labels
split = round(nrow(data.randomized)*.9)
train <- data.randomized[1:split, ]
train.indep <- train[,1:(ncol(train)-1)]
train.labels <- train[,ncol(train)]
test <- data.randomized[(split+1):nrow(data.randomized), ]
test.indep <- test[,1:(ncol(test)-1)]
test.labels <- test[,ncol(test)]

par(mfrow=c(1, 1))
# Control randomness in Lasso CV fit
set.seed(229)
# Produce a Lasso CV fit
grid=10^seq(-2,-1.8, length=100)
#10-fold cross validation by default, homie
lasso.cv <- cv.glmnet(x=data.matrix(train.indep), y=train.labels, lambda = grid, family = "binomial")
plot(lasso.cv)
# Use the 1 standard-error rule to pick lambda
lambda_1se <- lasso.cv$lambda.1se
#Print min lambda
lambda_min <- lasso.cv$lambda.min

#training set performance: cross validation error
train.fit <- predict(lasso.cv,newx=data.matrix(train.indep),s="lambda.1se",type = "response")
train.pred <- prediction(train.fit, train.labels)
# Obtain performance statistics
# Plot ROC
train.roc <- performance(train.pred, measure = "tpr", x.measure = "fpr")
plot(train.roc,colorize=FALSE, col="black")
lines(c(0,1),c(0,1),col = "gray", lty = 4 )
performance(train.pred, measure = "auc")@y.values #AUC
max(performance(train.pred, measure="acc")@y.values[[1]]) #ACC
plot(performance(train.pred, measure="acc")@y.values[[1]]) # varying ACC

#test set performance: validation set error
test.fit <- predict(lasso.cv,newx=data.matrix(test.indep),s="lambda.1se",type = "response")
test.pred <- prediction(test.fit, test.labels)
# Obtain performance statistics
# Plot ROC
test.roc <- performance(test.pred, measure = "tpr", x.measure = "fpr")
plot(test.roc,colorize=FALSE, col="black")
lines(c(0,1),c(0,1),col = "gray", lty = 4 )
performance(test.pred, measure = "auc")@y.values
max(performance(test.pred, measure="acc")@y.values[[1]]) #ACC
plot(performance(test.pred, measure="acc")@y.values[[1]]) # varying ACC

#simple logistic: baseline?
glm.fit <- glm(as.factor(train.labels) ~ ., data = data.frame(train.indep), family = binomial())
glm.probs <- predict.glm(glm.fit, newdata = data.frame(test.indep), type = "response")
glm.test.pred = prediction(glm.probs, test.labels)
glm.test.roc <- performance(glm.test.pred, measure = "tpr", x.measure = "fpr")
plot(glm.test.roc,colorize=FALSE, col="black")
lines(c(0,1),c(0,1),col = "gray", lty = 4 )

performance(glm.test.pred, measure = "auc")@y.values #True positive v.s. False positive (AUC of TPR v.s. FPR plot)
plot(performance(glm.test.pred, measure = "acc")) #Varying by cutoff
max(performance(glm.test.pred, measure="acc")@y.values[[1]]) #ACC
plot(performance(glm.test.pred, measure="acc")@y.values[[1]]) # varying ACC


