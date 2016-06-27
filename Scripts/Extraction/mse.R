# Compute the mean-squared error of a linear regression line fitted to the raw time series; this is equivalent to modulation

# Extract raw time series; you can toggle squiggle/nonsquiggle directory
data <- read.table("./dataset/Squiggles(833)/COMPLETE_squiggle_raw.csv", header = T, sep = ",", row.name = "id")
data.mat <- as.matrix(data)

# Extract raw time series from full feature set
data.mat <- data.mat[,8:136]
data.mat <- t(data.mat)

# Keep only raw time series columns
dat.lm <- data.frame(data.mat)

dat.lm$xs <- 1:129
mses = c()
for (i in 1:ncol(dat.lm)){
  temp.lm <- lm(dat.lm[,i] ~ xs, data = dat.lm)
  mses <- c(mses, mean((temp.lm$residuals)^2))
}

write.csv(mses, "new_mes.csv")