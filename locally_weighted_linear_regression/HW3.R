#a
data1 <- read.csv("/Users/estelle/Desktop/MF850/data1.csv",header = FALSE)
data2 <- read.csv("/Users/estelle/Desktop/MF850/data2.csv",header = FALSE)

train1 <- data.frame(X = data1$V1,Y = data1$V2)
val1 <- data.frame(X = data1$V3,Y = data1$V4)

getp <- function(train,val){
  p_list <- 1:20
  mse_values <- numeric(length(p_list))
  for (p in p_list) {
    model <- lm(formula = Y ~ poly(X,p), data = train)
    pred <- predict(model, newdata = val)
    mse <- mean((val$Y - pred)^2)
    mse_values[p] <- mse
  }
  return(mse_values)
}
p1 <- which.min(getp(train1,val1))
#p = 12 the MSE minimized, and the value is 6.005382

#minimize the MSE on the training set
p2 <- which.min(getp(train1,train1))
#p = 20 the MSE minimized and the value is 2.555897
#it may get smaller if p value goes bigger
#compare with p1, this method appears some overfitting

#b
train2 <- data.frame(X = data2$V1,Y = data2$V2)
val2 <- data.frame(X = data2$V3,Y = data2$V4)
p3 <- which.min(getp(train2,val2))
#p = 6 the MSE minimized and the value is 19.26092
#compare to data1, p gets smaller and MSE gets bigger

#c
model1 <- lm(formula = Y ~ poly(X,p1), data = train1)
residuals <- resid(model1)

getweight <- function(residuals,t,val){
  weights <- numeric(length(residuals))
  for (i in 1:length(residuals)){
    weights[i] <- exp(-residuals[i]^2/(2*val[i]^t))
  }
  return(weights)
}

getmse <- function(residuals,p,train,val){
  t_list <- 1:5
  mse_values <- numeric(length(t_list))
  for (t in t_list) {
    w <- getweight(residuals,t,val$X)
    model <- lm(formula = Y ~ poly(X,p), data = train, weights = w)
    pred <- predict(model, newdata = val)
    mse <- mean((val$Y - pred)^2)
    mse_values[t] <- mse
  }
  return(mse_values)
}

#d
mse1 <- getmse(residuals,p1,train1,val1)
plot(mse1)
t1 <- which.min(mse1)
#t=4 is the validation error minimized and the validation MSE is 6.388083

#e
mse2 <- getmse(residuals,p3,train2,val2)
plot(mse2)
t2 <- which.min(mse2)
#when t=3 is the validation error minimized and the validation MSE is 24.36539



























