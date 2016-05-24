library(glmnet)
library(ROCR)

#read data
raw_squiggle <- read.table("/Users/Jason/Desktop/SETI_TimeSeries/DATA/official_ts_dataset_dft.csv", header=TRUE, sep=",", row.names="id")
raw_nonsquiggle <- read.table("/Users/Jason/Desktop/SETI_TimeSeries/DATA/official_ts_dataset_nonsquiggle_dft.csv", header=TRUE, sep=",", row.names="id")
#add labels to the data
squiggle <- cbind(squiggle,T)
nonsquiggle <- cbind(nonsquiggle,F)
#put all the data together
data <- rbind(squiggle, nonsquiggle)
data <- scale(data)
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

#training set performance
train.fit <- predict(lasso.cv,newx=data.matrix(train.indep),s="lambda.1se",type = "response")
train.pred <- prediction(train.fit, train.labels)
# Obtain performance statistics
# Plot ROC
train.roc <- performance(train.pred, measure = "tpr", x.measure = "fpr")
plot(train.roc,colorize=FALSE, col="black")
lines(c(0,1),c(0,1),col = "gray", lty = 4 )

#test set performance
test.fit <- predict(lasso.cv,newx=data.matrix(test.indep),s="lambda.1se",type = "response")
test.pred <- prediction(test.fit, test.labels)
# Obtain performance statistics
# Plot ROC
test.roc <- performance(test.pred, measure = "tpr", x.measure = "fpr")
plot(test.roc,colorize=FALSE, col="black")
lines(c(0,1),c(0,1),col = "gray", lty = 4 )

#simple logistic
glm.fit <- glm(as.factor(train.labels) ~ ., data = data.frame(train.indep), family = binomial())
glm.probs <- predict.glm(glm.fit, newdata = data.frame(train.indep), type = "response")
glm.train.pred = prediction(glm.probs, train.labels)
glm.train.roc <- performance(glm.train.pred, measure = "tpr", x.measure = "fpr")
plot(glm.train.roc,colorize=FALSE, col="black")
lines(c(0,1),c(0,1),col = "gray", lty = 4 )
plot(performance(glm.train.pred, measure = "acc")) #Varying by cutoff

performance(glm.train.pred, measure = "auc")@y.values #True positive v.s. False positive (AUC of TPR v.s. FPR plot)


#logistic using only modulation
glm.fit <- glm(as.factor(train.labels) ~ ., data = data.frame(train.indep[,"modulation"]), family = binomial())
glm.probs <- predict.glm(glm.fit, newdata = data.frame(train.indep[,"modulation"]), type = "response")
glm.train.pred = prediction(glm.probs, train.labels)
glm.train.roc <- performance(glm.train.pred, measure = "tpr", x.measure = "fpr")
plot(glm.train.roc,colorize=FALSE, col="black")
lines(c(0,1),c(0,1),col = "gray", lty = 4 )
plot(performance(glm.train.pred, measure = "acc")) #Varying by cutoff

performance(glm.train.pred, measure = "auc")@y.values #True positive v.s. False positive (AUC of TPR v.s. FPR plot)
