data <- read.table("~/Desktop/SETI_TimeSeries/DATA/nonsquiggle_temp.csv", header = T, sep = ",", row.name = "id")
#data <- scale(data)
#data <- cbind(1, data)

data.mat <- as.matrix(data)
data.mat <- t(data.mat)

data.mat <- data.mat[-1,]
dat.lm <- data.frame(data.mat)

dat.lm$xs <- 1:129
mses = c()
for (i in 1:ncol(dat.lm)){
  temp.lm <- lm(dat.lm[,i] ~ xs, data = dat.lm)
  mses <- c(mses, mean((temp.lm$residuals)^2))
}

write.csv(mses, "new_mes.csv")