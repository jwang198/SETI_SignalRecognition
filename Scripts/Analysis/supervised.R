# Applications of various supervised machine learning models to the squiggle v. nonsquiggle classification problem

library(glmnet)
library(ROCR)
library(e1071)
library(randomForest)
library(MASS)
library(adabag)
library(gbm)
library(class)

set.seed(216)

## LOAD + PRE-PROCESS DATA ##
raw <- read.table("./dataset/Full/DATA.csv", header = T, sep = ",", row.name = "id")
raw[,1:72] <- scale(raw[,1:72]) # Do not scale labels

# Scale up (by factor of sqrt(63)) non-DFT features 
#raw[,1:9] <- raw[,1:9]*sqrt(63) #Don't scale for supervised, scale for unsupervised
raw[is.na(raw)] = 0
raw.randomized <- raw[sample(nrow(raw)), ]

#training and test splits. Also split by independent variables / labels
split = round(nrow(raw.randomized)*.9)

train <- raw.randomized[1:split, ]
train.indep <- train[,1:(ncol(train)-1)]
train.labels <- train[,ncol(train)]

test <- raw.randomized[(split+1):nrow(raw.randomized), ]
test.indep <- test[,1:(ncol(test)-1)]
test.labels <- test[,ncol(test)]

# Control randomness in Lasso CV fit
set.seed(229)

## L-1 REGULARIZED LOGISTIC REGRESSION ##
## LASSO: CV TO CHOOSE OPTIMAL PARAMETERS ##

#grid=10^seq(-2,-1.8, length=100)
#10-fold cross validation by default
#lasso.cv <- cv.glmnet(x=data.matrix(train.indep), y=train.labels, lambda = grid, family = "binomial")
lasso.cv <- cv.glmnet(x=data.matrix(train.indep), y=train.labels, family = "binomial", alpha = 1)
plot(lasso.cv) # use default grid, instead of our own

# Use the 1 standard-error rule to pick lambda
lambda_1se <- lasso.cv$lambda.1se
#Print min lambda
lambda_min <- lasso.cv$lambda.min

#training set performance: cross validation error
train.fit <- predict(lasso.cv,newx=data.matrix(train.indep),s="lambda.1se",type = "response")
train.pred <- prediction(train.fit, train.labels)

# Obtain performance statistics
# Plot ROC
lasso.roc <- performance(train.pred, measure = "tpr", x.measure = "fpr")
plot(lasso.roc,colorize=FALSE, col="black")
lines(c(0,1),c(0,1),col = "gray", lty = 4 )
performance(train.pred, measure = "auc")@y.values #AUC
max(performance(train.pred, measure="acc")@y.values[[1]]) #ACC
plot(performance(train.pred, measure="acc")) # varying ACC

## LASSO: VALIDATION TEST SET ##
test.fit <- predict(lasso.cv,newx=data.matrix(test.indep),s="lambda.1se",type = "response")
test.pred <- prediction(test.fit, test.labels)

# Obtain performance statistics
# Plot ROC
lasso.roc <- performance(test.pred, measure = "tpr", x.measure = "fpr")
plot(lasso.roc,colorize=FALSE, col="black")
lines(c(0,1),c(0,1),col = "gray", lty = 4 )
performance(test.pred, measure = "auc")@y.values
max(performance(test.pred, measure="acc")@y.values[[1]]) #ACC
plot(performance(test.pred, measure="acc")) # varying ACC
dimnames(coef(lasso.cv))[[1]][which(coef(lasso.cv, s = "lambda.1se") != 0)]

## L-2 REGULARIZED LOGISTIC REGRESSION ##
## RIDGE: CV TO CHOOSE OPTIMAL PARAMETERS ##

#10-fold cross validation by default
ridge.cv <- cv.glmnet(x=data.matrix(train.indep), y=train.labels, family = "binomial", alpha = 0)
plot(ridge.cv) # use default grid, instead of our own

# Use the 1 standard-error rule to pick lambda
lambda_1se <- ridge.cv$lambda.1se
#Print min lambda
lambda_min <- ridge.cv$lambda.min

#training set performance: cross validation error
train.fit <- predict(ridge.cv,newx=data.matrix(train.indep),s="lambda.1se",type = "response")
train.pred <- prediction(train.fit, train.labels)

# Obtain performance statistics
# Plot ROC
train.roc <- performance(train.pred, measure = "tpr", x.measure = "fpr")
plot(train.roc,colorize=FALSE, col="black")
lines(c(0,1),c(0,1),col = "gray", lty = 4 )
performance(train.pred, measure = "auc")@y.values #AUC
max(performance(train.pred, measure="acc")@y.values[[1]]) #ACC
plot(performance(train.pred, measure="acc")) # varying ACC

## RIDGE: VALIDATION TEST SET ##
test.fit <- predict(ridge.cv,newx=data.matrix(test.indep),s="lambda.1se",type = "response")
test.pred <- prediction(test.fit, test.labels)
ridge.roc <- performance(test.pred, measure = "tpr", x.measure = "fpr")
performance(test.pred, measure = "auc")@y.values #AUC
max(performance(test.pred, measure="acc")@y.values[[1]]) #ACC
plot(performance(test.pred, measure="acc")) # varying ACC

w = which(test.fit < 0.5)
test.fit[w] = 0
w = which(test.fit != 0)
test.fit[w] = 1

tab <- table(test.fit, test.labels)

## UNREGULARIZED LOGISTIC REGRESSION ##
glm.fit <- glm(as.factor(train.labels) ~ ., data = data.frame(train.indep), family = binomial())
glm.probs <- predict.glm(glm.fit, newdata = data.frame(train.indep), type = "response")
glm.train.pred = prediction(glm.probs, train.labels)
performance(glm.train.pred, measure = "auc")@y.values #True positive v.s. False positive (AUC of TPR v.s. FPR plot)
max(performance(glm.train.pred, measure="acc")@y.values[[1]]) #ACC
plot(performance(glm.train.pred, measure="acc")) # varying ACC

glm.probs <- predict.glm(glm.fit, newdata = data.frame(test.indep), type = "response")
glm.test.pred = prediction(glm.probs, test.labels)
glm.test.roc <- performance(glm.test.pred, measure = "tpr", x.measure = "fpr")
plot(glm.test.roc,colorize=FALSE, col="black")
lines(c(0,1),c(0,1),col = "gray", lty = 4 )

performance(glm.test.pred, measure = "auc")@y.values #True positive v.s. False positive (AUC of TPR v.s. FPR plot)
max(performance(glm.test.pred, measure="acc")@y.values[[1]]) #ACC
plot(performance(glm.test.pred, measure="acc")) # varying ACC

w = which(glm.probs < 0.5)
glm.probs[w] = 0
w = which(glm.probs != 0)
glm.probs[w] = 1

tab <- table(glm.probs, test.labels)

plot(lasso.roc, col = "green", xlim=c(0,.5), main = "Logistic Regression Family ROC Curve", lwd=3)
plot(ridge.roc, add = TRUE, col = "red", xlim=c(0,.5), lwd = 3, lty=3)
plot(glm.test.roc, add = TRUE, col = "blue", xlim=c(0,.5), lwd = 3, lty=4)

legend("bottomright", cex = 1,
       c("Lasso","Ridge","Unregularized"),
       lty=c(1,1,1), # gives the legend appropriate symbols (lines)
       lwd=c(2.5,2.5,2.5),col=c("green","red","blue"), pch=16)

## SUPPORT VECTOR MACHINE: CHOOSE OPTIMAL PARAMETERS BASED ON CV ERROR ##

## SUPPORT VECTOR MACHINE: TRY DIFFERENT KERNELS? ## 

optimal.cost <- function(train.indep, train.labels, test.indep, test.labels){
  kernels <- c("linear", "radial", "polynomial", "sigmoid")
  best.cost <- c()
  grid = 10^seq(-3,2,length = 10)
  for (kern in kernels){
    auc = c()
    for(cost.check in grid){
      svmfit = svm(as.factor(train.labels)~.,data = data.frame(train.indep),kernel=kern,cost=cost.check,scale=FALSE, probability=TRUE)
      svm.pred <- predict(svmfit, newdata = data.frame(test.indep), probability = TRUE)
      svm.test.pred = prediction(attr(svm.pred, "probabilities")[,1], test.labels)
      svm.test.auc <- performance(svm.test.pred, measure = "auc")
      auc <- c(auc, svm.test.auc@y.values[[1]])
    } 
    best.cost <- c(best.cost, grid[which.max(auc)])
  }
  return(best.cost)
}

#linear: 0.59948425
#radial: 0.07742637
#polynomial: 35.93813664
#sigmoid: 0.00129155

best.costs <- optimal.cost(train.indep, train.labels, test.indep, test.labels)
best.costs <- c(0.59948425,0.07742637,35.93813664,0.00129155)

# LINEAR
svmfit = svm(as.factor(train.labels)~.,data = data.frame(train.indep),kernel="linear",cost=best.costs[1],scale=FALSE, probability=TRUE)
svm.pred <- predict(svmfit, newdata = data.frame(test.indep), probability = TRUE)
svm.test.pred = prediction(attr(svm.pred, "probabilities")[,1], test.labels)
linear.auc <- (performance(svm.test.pred, measure = "auc"))@y.values[[1]]
linear.roc <- performance(svm.test.pred, measure = "tpr", x.measure = "fpr")
max(performance(svm.test.pred, measure="acc")@y.values[[1]]) #ACC

w = which(svm.pred < 0.5)
svm.pred[w] = 0
w = which(svm.pred != 0)
svm.pred[w] = 1

tab <- table(svm.pred, test.labels)

svm.pred <- predict(svmfit, newdata = data.frame(train.indep), probability = TRUE)
svm.train.pred = prediction(attr(svm.pred, "probabilities")[,1], train.labels)
linear.auc <- (performance(svm.train.pred, measure = "auc"))@y.values[[1]]
max(performance(svm.train.pred, measure="acc")@y.values[[1]]) #ACCw = which(glm.probs < 0.5)

# RADIAL
svmfit = svm(as.factor(train.labels)~.,data = data.frame(train.indep),kernel="radial",cost=best.costs[2],scale=FALSE, probability=TRUE)
svm.pred <- predict(svmfit, newdata = data.frame(test.indep), probability = TRUE)
svm.test.pred = prediction(attr(svm.pred, "probabilities")[,1], test.labels)
radial.auc <- (performance(svm.test.pred, measure = "auc"))@y.values[[1]]
radial.roc <- performance(svm.test.pred, measure = "tpr", x.measure = "fpr")
max(performance(svm.test.pred, measure="acc")@y.values[[1]]) #ACC

w = which(svm.pred < 0.5)
svm.pred[w] = 0
w = which(svm.pred != 0)
svm.pred[w] = 1

tab <- table(svm.pred, test.labels)

svm.pred <- predict(svmfit, newdata = data.frame(train.indep), probability = TRUE)
svm.train.pred = prediction(attr(svm.pred, "probabilities")[,1], train.labels)
radial.auc <- (performance(svm.train.pred, measure = "auc"))@y.values[[1]]
max(performance(svm.train.pred, measure="acc")@y.values[[1]]) #ACC

# POLYNOMIAL
svmfit = svm(as.factor(train.labels)~.,data = data.frame(train.indep),kernel="polynomial",cost=best.costs[3],scale=FALSE, probability=TRUE)
svm.pred <- predict(svmfit, newdata = data.frame(test.indep), probability = TRUE)
svm.test.pred = prediction(attr(svm.pred, "probabilities")[,1], test.labels)
poly.auc <- (performance(svm.test.pred, measure = "auc"))@y.values[[1]]
poly.roc <- performance(svm.test.pred, measure = "tpr", x.measure = "fpr")
max(performance(svm.test.pred, measure="acc")@y.values[[1]]) #ACC

w = which(svm.pred < 0.5)
svm.pred[w] = 0
w = which(svm.pred != 0)
svm.pred[w] = 1

tab <- table(svm.pred, test.labels)

svm.pred <- predict(svmfit, newdata = data.frame(train.indep), probability = TRUE)
svm.train.pred = prediction(attr(svm.pred, "probabilities")[,1], train.labels)
poly.auc <- (performance(svm.train.pred, measure = "auc"))@y.values[[1]]
max(performance(svm.train.pred, measure="acc")@y.values[[1]]) #ACC

# SIGMOID
svmfit = svm(as.factor(train.labels)~.,data = data.frame(train.indep),kernel="sigmoid",cost=best.costs[4],scale=FALSE, probability=TRUE)
svm.pred <- predict(svmfit, newdata = data.frame(test.indep), probability = TRUE)
svm.test.pred = prediction(attr(svm.pred, "probabilities")[,1], test.labels)
sig.auc <- (performance(svm.test.pred, measure = "auc"))@y.values[[1]]
sig.roc <- performance(svm.test.pred, measure = "tpr", x.measure = "fpr")
max(performance(svm.test.pred, measure="acc")@y.values[[1]]) #ACC

w = which(svm.pred < 0.5)
svm.pred[w] = 0
w = which(svm.pred != 0)
svm.pred[w] = 1

tab <- table(svm.pred, test.labels)

svm.pred <- predict(svmfit, newdata = data.frame(train.indep), probability = TRUE)
svm.train.pred = prediction(attr(svm.pred, "probabilities")[,1], train.labels)
sig.auc <- (performance(svm.train.pred, measure = "auc"))@y.values[[1]]
max(performance(svm.train.pred, measure="acc")@y.values[[1]]) #ACC

plot(linear.roc, col = "black", xlim=c(0,.5), main='SVM Family ROC Curve', lwd=3)
plot(radial.roc, add = TRUE, col = "red", xlim=c(0,.5), lty = 2, lwd=3)
plot(poly.roc, add = TRUE, col = "blue", xlim=c(0,.5), lty = 3, lwd=3)
plot(sig.roc, add = TRUE, col = "green", xlim=c(0,.5), lty = 4, lwd=3)

legend("bottomright", cex = 1,
       c("linear","radial","polynomial", "sigmoid"),
       lty=c(1,1,1,1), # gives the legend appropriate symbols (lines)
       lwd=c(2.5,2.5,2.5,2.5),col=c("black","red","blue", "green"), pch=16)

linear.auc
radial.auc
poly.auc
sig.auc

## TREE-BASED METHOD (BOOSTING, RANDOM FORESTS, BAGGING/CART)
boost_vector = rep(0,50)
bag_vector = rep(0,50)
rf_vector = rep(0,50)

#to make boosting work
train.data <- data.frame(train)
train.data$X0 <- as.factor(train.data$X0)
for (i in 1:50) {
  
  #boosting: decision stumps
  boost_model = boosting(X0~., data = train.data, boos=TRUE,mfinal=i)
  yhat.boost = predict(boost_model ,newdata=data.frame(test.indep))
  tab <- table(yhat.boost$class, test.labels)
  boost_vector[i] = 1-sum(diag(tab))/sum(tab)
  
  #bagging: use all 72 predictors
  bag_model = randomForest(as.factor(train.labels)~.,data = data.frame(train.indep),mtry=72,importance =TRUE, ntree=i)
  yhat.bag = predict(bag_model ,newdata=data.frame(test.indep))
  tab <- table(yhat.bag, test.labels)
  bag_vector[i] = 1-sum(diag(tab))/sum(tab)
  
  #random forests: use default number of trees ~sqrt(p) where p = number of predictors
  rf_model = randomForest(as.factor(train.labels)~.,data.frame(train.indep),importance =TRUE, ntree=i)
  yhat.rf = predict(rf_model ,newdata=data.frame(test.indep))
  tab <- table(yhat.rf, test.labels)
  rf_vector[i] = 1-sum(diag(tab))/sum(tab)
}

#plot(1:50,bag_vector,col='red',pch=16,xlab='Number of Trees',ylab='Misclassification Error',main="Tree-Based Methods Varying Number of Trees")
plot(predict(rf_lo),col='blue',lwd=4, type='l',xlab='Number of Trees',ylab='Misclassification Error',main="Tree-Based Methods Varying Number of Trees")
points(1:50,rf_vector,col='blue',pch=16)
points(1:50,boost_vector,col='green',pch=16)
points(1:50,bag_vector,col='red',pch=16)

#rf_lo <- loess(rf_vector~c(1:50)) # Moving average
#bag_lo <- loess(bag_vector~c(1:50))
#boost_lo <- loess(boost_vector~c(1:50))

rf_lo <- smooth.spline(c(1:50),rf_vector,spar=0.35) # Spline fitting
bag_lo <- smooth.spline(c(1:50),bag_vector,spar=0.35) 
boost_lo <- smooth.spline(c(1:50),boost_vector,spar=0.35) 

lines(predict(rf_lo),col='blue',lwd=4)
lines(predict(bag_lo),col='red',lwd=4)
lines(predict(boost_lo),col='green',lwd=4)

legend('topright',legend=c('Bagging','Random Forest','Boosting'), pch=16, col=c('Red','Blue','Green'))

# Using the optimal values
# Boosting: optimal Number of Iterations: 19
boost_model = boosting(X0~., data = train.data, boos=TRUE,mfinal=19)
yhat.boost = predict(boost_model ,newdata=data.frame(test.indep))
tab <- table(yhat.boost$class, test.labels)
acc = 1-(1-sum(diag(tab))/sum(tab))

yhat.boost.pred = prediction(as.numeric(yhat.boost$class), test.labels)
(performance(yhat.boost.pred, measure = "auc"))@y.values[[1]]
boost.roc <- performance(yhat.boost.pred, measure = "tpr", x.measure = "fpr")

yhat.boost = predict(boost_model ,newdata=data.frame(train.indep))
tab <- table(yhat.boost$class, train.labels)
acc = 1-(1-sum(diag(tab))/sum(tab))

yhat.boost.pred = prediction(as.numeric(yhat.boost$class), train.labels)
(performance(yhat.boost.pred, measure = "auc"))@y.values[[1]]

# Optimal: 16 Trees
#bagging: use all 72 predictors
bag_model = randomForest(as.factor(train.labels)~.,data = data.frame(train.indep),mtry=72,importance =TRUE, ntree=16)
yhat.bag = predict(bag_model ,newdata=data.frame(test.indep))
tab <- table(yhat.bag, test.labels)
acc = 1-(1-sum(diag(tab))/sum(tab))

yhat.bag = as.numeric(yhat.bag)
w = which(yhat.bag==1)
yhat.bag[w] = 0
w = which(yhat.bag==2)
yhat.bag[w] = 1
yhat.bag.pred = prediction(yhat.bag, test.labels)
(performance(yhat.bag.pred, measure = "auc"))@y.values[[1]]
bag.roc <- performance(yhat.bag.pred, measure = "tpr", x.measure = "fpr")

yhat.bag = predict(bag_model ,newdata=data.frame(train.indep))
tab <- table(yhat.bag, train.labels)
acc = 1-(1-sum(diag(tab))/sum(tab))

yhat.bag = as.numeric(yhat.bag)
w = which(yhat.bag==1)
yhat.bag[w] = 0
w = which(yhat.bag==2)
yhat.bag[w] = 1
yhat.bag.pred = prediction(yhat.bag, train.labels)
(performance(yhat.bag.pred, measure = "auc"))@y.values[[1]]

# Optimal: 17 Trees
#random forests: use default number of trees ~sqrt(p) where p = number of predictors
rf_model = randomForest(as.factor(train.labels)~.,data.frame(train.indep),importance =TRUE, ntree=17)
yhat.rf = predict(rf_model ,newdata=data.frame(test.indep))
tab <- table(yhat.rf, test.labels)
acc = 1-(1-sum(diag(tab))/sum(tab))

yhat.rf = as.numeric(yhat.rf)
w = which(yhat.rf==1)
yhat.rf[w] = 0
w = which(yhat.rf==2)
yhat.rf[w] = 1
yhat.rf.pred = prediction(yhat.rf, test.labels)
(performance(yhat.rf.pred, measure = "auc"))@y.values[[1]]
rf.roc <- performance(yhat.rf.pred, measure = "tpr", x.measure = "fpr")

yhat.rf = predict(rf_model ,newdata=data.frame(train.indep))
tab <- table(yhat.rf, train.labels)
acc = 1-(1-sum(diag(tab))/sum(tab))

yhat.rf = as.numeric(yhat.rf)
w = which(yhat.rf==1)
yhat.rf[w] = 0
w = which(yhat.rf==2)
yhat.rf[w] = 1
yhat.rf.pred = prediction(yhat.rf, train.labels)
(performance(yhat.rf.pred, measure = "auc"))@y.values[[1]]

#Finding Most Important Variables
rf_model$importance[order(-rf_model$importance[,2]),]
bag_model$importance[order(-bag_model$importance[,2]),]
#boost_model$importance[order(-boost_model$importance[,2]),]

plot(boost.roc, col = "black", xlim=c(0,.5), main='Tree-Based Methods ROC Curve', lwd=3)
plot(bag.roc, add = TRUE, col = "red", xlim=c(0,.5), lty = 2, lwd=3)
plot(rf.roc, add = TRUE, col = "blue", xlim=c(0,.5), lty = 3, lwd=3)

legend("bottomright", cex = 1,
       c("Boosting","Bagging","Random Forests"),
       lty=c(1,1,1,1), # gives the legend appropriate symbols (lines)
       lwd=c(2.5,2.5,2.5,2.5),col=c("black","red","blue"), pch=16)
#############################
## K-NEAREST NEIGHBORS ##
knn_vector = rep(0,10)
for (i in 1:10) {
  knn.fit <- knn(train=train.indep, test=test.indep, cl=train.labels, k = i, prob=TRUE)
  tab <- table(knn.fit, test.labels)
  knn_vector[i] = 1-sum(diag(tab))/sum(tab)
}
plot(1:10,knn_vector,col='red',pch=16,xlab='K (Number of Neighbors)',ylab='Misclassification Error', main='KNN Varying K')
knn_lo <- loess(knn_vector~c(1:10))
lines(predict(knn_lo),col='black',lwd=4)

# K = 4 is optimal
knn.fit <- knn(train=train.indep, test=test.indep, cl=train.labels, k=4, prob=TRUE)
tab <- table(knn.fit, test.labels)
acc = 1-(1-sum(diag(tab))/sum(tab))

knn.fit <- knn(train=train.indep, test=train.indep, cl=train.labels, k=4, prob=TRUE)
tab <- table(knn.fit, train.labels)
acc = 1-(1-sum(diag(tab))/sum(tab))

prob <- attr(knn.fit, "prob")
prob <- 2*ifelse(knn.fit == "-1", 1-prob, prob) - 1
knn.pred <- prediction(prob, test.indep)
knn.pred <- performance(knn.pred, "tpr", "fpr")
plot(knn.pred,colorize=FALSE, col="black")

# Future Test: Single-Layer Hidden Neural Network