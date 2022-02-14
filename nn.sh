https://rviews.rstudio.com/2020/07/20/shallow-neural-net-from-scratch-using-r-part-1/
https://rviews.rstudio.com/2020/07/24/building-a-neural-net-from-scratch-using-r-part-2/
and refrence in the end

R
simple 3-layer neural net with 1 hidden layer
epoch, deep (>1 hidden layers),loss, gradient,
activation function: sigmoid, tanh, relu, leaky relu, maxout, elu
loss, gradients, learning rate, update weight

# dry run:
see nn-html folder for math


# classical get dataset
:
planar_dataset <- function(){
  set.seed(1)
  m <- 400
  N <- m/2
  D <- 2
  X <- matrix(0, nrow = m, ncol = D)
  Y <- matrix(0, nrow = m, ncol = 1)
  a <- 4
  
  for(j in 0:1){
    ix <- seq((N*j)+1, N*(j+1))
    t <- seq(j*3.12,(j+1)*3.12,length.out = N) + rnorm(N, sd = 0.2)
    r <- a*sin(4*t) + rnorm(N, sd = 0.2)
    X[ix,1] <- r*sin(t)
    X[ix,2] <- r*cos(t)
    Y[ix,] <- j
  }
  
  d <- as.data.frame(cbind(X, Y))
  names(d) <- c('X1','X2','Y')
  d
}

# try out
:
df <- planar_dataset()

library(ggplot2)
ggplot(df, aes(x = X1, y = X2, color = factor(Y))) + geom_point()

# ML
#df <- read.csv(file = "planar_flower.csv")
:
df <- planar_dataset()
df <- df[sample(nrow(df)), ]
set.seed(69)

# train and test
:
train_test_split_index <- 0.8 * nrow(df)
train <- df[1:train_test_split_index,]
head(train)
test <- df[(train_test_split_index+1): nrow(df),]
head(test)

ggplot(test, aes(Y)) + geom_bar()

#Preprocess, stardize
:
X_train <- scale(train[, c(1:2)])
y_train <- train$Y
dim(y_train) <- c(length(y_train), 1) # add extra dimension to vector

X_test <- scale(test[, c(1:2)])
y_test <- test$Y
dim(y_test) <- c(length(y_test), 1) # add extra dimension to vector

# matrix
:
X_train <- as.matrix(X_train, byrow=TRUE)
X_train <- t(X_train)
y_train <- as.matrix(y_train, byrow=TRUE)
y_train <- t(y_train)

X_test <- as.matrix(X_test, byrow=TRUE)
X_test <- t(X_test)
y_test <- as.matrix(y_test, byrow=TRUE)
y_test <- t(y_test)

#steps
Define the neural net architecture.
Initialize the model’s parameters from a random-uniform distribution.
Loop:
Implement forward propagation.
Compute loss.
Implement backward propagation to get the gradients.
Update parameters.
To generate matrices with random parameters, we need to first obtain the size (number of neurons) of all the layers in our neural-net. 
Let’s denote n_x, n_h, and n_y as the number of neurons in input layer, hidden layer, and output layer respectively.

#get layer size
:
getLayerSize <- function(X, y, hidden_neurons, train=TRUE) {
  n_x <- dim(X)[1]
  n_h <- hidden_neurons
  n_y <- dim(y)[1]   
  
  size <- list("n_x" = n_x,
               "n_h" = n_h,
               "n_y" = n_y)
  
  return(size)
}
layer_size <- getLayerSize(X_train, y_train, hidden_neurons = 4)
layer_size

# initialize the parameters based on random uniform distribution.
The function initializeParameters() takes as argument an input matrix and a list which contains the layer sizes i.e. number of neurons. 
The function returns the trainable parameters W1, b1, W2, b2.
The sizes of these weights matrices are 
W1 = (n_h, n_x)
b1 = (n_h, 1)
W2 = (n_y, n_h)
b2 = (n_y, 1)

# initialize
:
initializeParameters <- function(X, list_layer_size){

    m <- dim(data.matrix(X))[2]
    
    n_x <- list_layer_size$n_x
    n_h <- list_layer_size$n_h
    n_y <- list_layer_size$n_y
        
    W1 <- matrix(runif(n_h * n_x), nrow = n_h, ncol = n_x, byrow = TRUE) * 0.01
    b1 <- matrix(rep(0, n_h), nrow = n_h)
    W2 <- matrix(runif(n_y * n_h), nrow = n_y, ncol = n_h, byrow = TRUE) * 0.01
    b2 <- matrix(rep(0, n_y), nrow = n_y)
    
    params <- list("W1" = W1,
                   "b1" = b1, 
                   "W2" = W2,
                   "b2" = b2)
    
    return (params)
}

init_params <- initializeParameters(X_train, layer_size)
lapply(init_params, function(x) dim(x))


#Define the Activation Functions.
#An activation function only introduces non-linearity in a network.
tanh(x)=(exp(​x)-​exp(​‑x))/​(exp(​x)+​exp(​‑x))
sigmoid(x)=1/(1+exp(​-x))
ReLU(x)=max(0,x)
Leaky ReLU(x)=max(0.1x,x)
ELU(z)=x	x>=0
      =alpha(exp(​x)-1)	x<0
Maxout(x)=max(w1%*%x+b1,w2%*%x+b2)

#sigmoid
:
sigmoid <- function(x){
    return(1 / (1 + exp(-x)))
}
plot(-10:10, sigmoid(-10:10))


#Forward Propagation
#matrix dimentions
W1: (4, 2)
b1: (4, 1)
W2: (1, 4)
b2 : (1, 1)
X: (2,320)
Y(1,320)

Y=W2*A1+b2=W*(W1%*%X+b1)+b2

W1%*%X-> (4,320),   change/repeat b1->(4:320)
A1=W1%*%X+b1 -> (4:320)
W2*A1 -> (1,320),  change/repeat b2 -> (1,320), same as Y

We use the tanh() activation for the hidden layer and sigmoid() activation for the output layer.

#forwardPropagation
:
forwardPropagation <- function(X, params, list_layer_size){

    m <- dim(X)[2]
    n_h <- list_layer_size$n_h
    n_y <- list_layer_size$n_y

    W1 <- params$W1
    b1 <- params$b1
    W2 <- params$W2
    b2 <- params$b2

    b1_new <- matrix(rep(b1, m), nrow = n_h)
    b2_new <- matrix(rep(b2, m), nrow = n_y)

    Z1 <- W1 %*% X + b1_new
    #A1 <- sigmoid(Z1)  # backwardPropagation  dZ1 <- (t(W2) %*% dZ2) * A1 * (1-A1),  website mistake by usin tanh(Z1) backwardPropagation, so the spikes in the cost plot on the website
    A1 <- tanh(Z1)   # backwardPropagation  dZ1 <- (t(W2) %*% dZ2) * (1 - A1^2)
    Z2 <- W2 %*% A1 + b2_new
    A2 <- sigmoid(Z2)

    cache <- list("Z1" = Z1,
                  "A1" = A1, 
                  "Z2" = Z2,
                  "A2" = A2)

    return (cache)
}

fwd_prop <- forwardPropagation(X_train, init_params, layer_size)
lapply(fwd_prop, function(x) dim(x))


# backpropagation by writing functions to calculate gradients and update the weights
# compute cost
Binary Cross Entropy loss function (aka log loss). Here, y is the true label and ^y is the predicted output.
cost=−1/NN∑i=1yilog(^yi)+(1−yi)(log(1−^yi))

:
computeCost <- function(X, y, cache) {
    m <- dim(X)[2]
    A2 <- cache$A2
    logprobs <- (log(A2) * y) + (log(1-A2) * (1-y))
    cost <- -sum(logprobs/m)
    return (cost)
}
cost <- computeCost(X_train, y_train, fwd_prop)
cost


#Backpropagation
see html for Backpropagation with cache
list of gradient matrices,  note it is gradient, DL/dZ etc. not dZ itself
see https://towardsdatascience.com/shallow-neural-networks-23594aa97a5  for detailed derivation

:
backwardPropagation <- function(X, y, cache, params, list_layer_size){
    
    m <- dim(X)[2]
    
    n_x <- list_layer_size$n_x
    n_h <- list_layer_size$n_h
    n_y <- list_layer_size$n_y

    A2 <- cache$A2
    A1 <- cache$A1
    W2 <- params$W2

    dZ2 <- A2 - y  #  A2=sigma(Z2)
    dW2 <- 1/m * (dZ2 %*% t(A1)) 
    db2 <- matrix(1/m * sum(dZ2), nrow = n_y)
    db2_new <- matrix(rep(db2, m), nrow = n_y)
    
    #dZ1 <- (t(W2) %*% dZ2) * A1 * (1 - A1)  # forward A1 <- sigmoid(Z1)
    dZ1 <- (t(W2) %*% dZ2) * (1 - A1^2)  # forward A1 <- tanh(Z1)
    dW1 <- 1/m * (dZ1 %*% t(X))
    db1 <- matrix(1/m * sum(dZ1), nrow = n_h)
    db1_new <- matrix(rep(db1, m), nrow = n_h)
    
    grads <- list("dW1" = dW1, 
                  "db1" = db1,
                  "dW2" = dW2,
                  "db2" = db2)
    
    return(grads)
}

back_prop <- backwardPropagation(X_train, y_train, fwd_prop, init_params, layer_size)
lapply(back_prop, function(x) dim(x))


#Update Parameters
learning rate is a hyper-parameter that is set by us, the user, to control the impact of weight updates. The value of learning rate lies between 
0 and 1. This learning rate is multiplied with the gradients before being subtracted from the weights.
:
updateParameters <- function(grads, params, learning_rate){

    W1 <- params$W1
    b1 <- params$b1
    W2 <- params$W2
    b2 <- params$b2
    
    dW1 <- grads$dW1
    db1 <- grads$db1
    dW2 <- grads$dW2
    db2 <- grads$db2
    
    
    W1 <- W1 - learning_rate * dW1
    b1 <- b1 - learning_rate * db1
    W2 <- W2 - learning_rate * dW2
    b2 <- b2 - learning_rate * db2
    
    updated_params <- list("W1" = W1,
                           "b1" = b1,
                           "W2" = W2,
                           "b2" = b2)
    
    return (updated_params)
}

update_params <- updateParameters(back_prop, init_params, learning_rate = 0.01)
lapply(update_params, function(x) dim(x))


r#Train the Model
We will use all the functions we have written above in the following order.
Run forward propagation
Calculate loss
Calculate gradients
Update parameters
Repeat

Get the sizes for layers and initialize random parameters.
Initialize a vector called cost_history which we’ll use to store the calculated loss value per epoch.
Run a for-loop:
Run forward prop.
Calculate loss.
Update parameters.
Replace the current parameters with updated parameters.

:
trainModel <- function(X, y, num_iteration, hidden_neurons, lr){
    
    layer_size <- getLayerSize(X, y, hidden_neurons)
    init_params <- initializeParameters(X, layer_size)
    cost_history <- c()
    for (i in 1:num_iteration) {
        fwd_prop <- forwardPropagation(X, init_params, layer_size)
        cost <- computeCost(X, y, fwd_prop)
        back_prop <- backwardPropagation(X, y, fwd_prop, init_params, layer_size)
        update_params <- updateParameters(back_prop, init_params, learning_rate = lr)
        init_params <- update_params
        cost_history <- c(cost_history, cost)
        
        if (i %% 10000 == 0) cat("Iteration", i, " | Cost: ", cost, "\n")
    }
    
    model_out <- list("updated_params" = update_params,
                      "cost_hist" = cost_history)
    return (model_out)
}

EPOCHS = 60000
HIDDEN_NEURONS = 40
LEARNING_RATE = 0.9

train_model <- trainModel(X_train, y_train, hidden_neurons = HIDDEN_NEURONS, num_iteration = EPOCHS, lr = LEARNING_RATE)
write(train_model$cost_hist, ncol=1,file="test/ML/cost_hist.tanh.tanh")

HIDDEN_NEURONS = 60
train_model <- trainModel(X_train, y_train, hidden_neurons = HIDDEN_NEURONS, num_iteration = EPOCHS, lr = LEARNING_RATE)
write(train_model$cost_hist, ncol=1,file="test/ML/cost_hist.tanh.tanh.nns60")


# compare sigmoid, tanh activation, number of neurons
data=read.csv("test/ML/cost_hist.tanh.tanh")
plot(data[,1],type="b")
data=read.csv("test/ML/cost_hist.sigmoid.sigmoid")
lines(data[,1],type="b", col="red")
data=read.csv("test/ML/cost_hist.sigmoid.sigmoid.nn60")
lines(data[,1],type="b", col="blue")
data=read.csv("test/ML/cost_hist.tanh.tanh.nns60")
lines(data[,1],type="b", col="green")

tanh is better



#Logistic Regression comapre
We’ll use the glm() function in R to build this model.

:
lr_model <- glm(Y ~ X1 + X2, data = train)
lr_model
lr_pred <- round(as.vector(predict(lr_model, test[, 1:2])))
lr_pred


# Test the Model
We only perform forward propagation and return the final output from our neural network using the trained parameters here. 
:
makePrediction <- function(X, y, hidden_neurons){
    layer_size <- getLayerSize(X, y, hidden_neurons)
    params <- train_model$updated_params
    fwd_prop <- forwardPropagation(X, params, layer_size)
    pred <- fwd_prop$A2
    
    return (pred)
}

y_pred <- makePrediction(X_test, y_test, HIDDEN_NEURONS)
y_pred <- round(y_pred)


#Decision Boundaries
plotted our test-set predictions on top of the decision boundaries.
Plot of Predictions of all values of input.

NN show clear arear of class similar to data, logistic regression show only 2 area.


#Confusion Matrix
tb_nn <- table(y_test, y_pred)
tb_lr <- table(y_test, lr_pred)

#Accuracy Metrics
calculate_stats <- function(tb, model_name) {
  acc <- (tb[1] + tb[4])/(tb[1] + tb[2] + tb[3] + tb[4])
  recall <- tb[4]/(tb[4] + tb[3])
  precision <- tb[4]/(tb[4] + tb[2])
  f1 <- 2 * ((precision * recall) / (precision + recall))
  
  cat(model_name, ": \n")
  cat("\tAccuracy = ", acc*100, "%.")
  cat("\n\tPrecision = ", precision*100, "%.")
  cat("\n\tRecall = ", recall*100, "%.")
  cat("\n\tF1 Score = ", f1*100, "%.\n\n")
}

calculate_stats(tb_nn,"Neural Network")
calculate_stats(tb_lr,"Logistic Regression")



# my theory: 
#------------------------
(1)nueron network by creating multiple (number of neuron in hidden layer) linear planes (like in 2D with 2 input variable), and
search for the best planes arrangemnt, so that combination ouput of planes value (with activation function to add non-linearity) mimick ouput.
(2) with dicided NN structure, one can derive loss as fuction of weight W and intercept b like sigmoid(tanh(W*X=B))-Y etc depending on
the activation/loss fuction.  In theory one can find minimum of this function. But will it still work without layered propgation of parameters?
So strtucture may be the key ?  then can we contruct simple network without linear activation ?
(3) the f/b propgation process to fine tune the mutiple(hn) linear plane (reponse) by changing parameters to map it to multiple D space, so each class
find its location in space, which can be seperated by linear combination of mutiple D.  More layers will add more mapping for more complicated data.
(4) differnt cases, small D data to large D neuron ( planar flower data), or large D data to samll D neuron (gene expression, D reduction), intereting,
ie. search "neuron network gene expression dimention reduction"


#test
source("codes/ML/nn_funct.R")
set.seed(1)

#data
#2 class test, 1 hiddle neron should work
x=runif(1000)
y=runif(1000)
z=rep(1,1000)
z[x>0.5]=1
z[x<0.5]=0
data=as.data.frame(cbind(x,y,z))
names(data) <- c('X1','X2','Y')

hn=1
costfile="test/ML/cost_hist.2c"

# 3 class, 1 neuron will not work, 2 neurons works but not enough, 3 or more neuron works perfect
x=runif(1000)
y=runif(1000)
z=rep(1,1000)
z[x>0.5 & y>0.5]=1
z[x>0.5 & y<0.5]=0
z[x<0.5]=0
data=as.data.frame(cbind(x,y,z))
names(data) <- c('X1','X2','Y')

hn=4
costfile="test/ML/cost_hist.3c"

# 4 class, 1 neuron will not work, 2 neurons works but not enough, 3 or more neuron works perfect
x=runif(1000)
y=runif(1000)
z=rep(1,1000)
z[x>0.5 & y>0.5]=1
z[x>0.5 & y<0.5]=0
z[x<0.5 & y<0.5]=1
z[x<0.5 & y>0.5]=0
data=as.data.frame(cbind(x,y,z))
names(data) <- c('X1','X2','Y')

plot(x[z==1],y[z==1])
lines(x[z==0],y[z==0],type="p",col="red")

hn=4
costfile="test/ML/cost_hist.4c"

# train and test data
train_test <- getTrainTest(data)
X_train <- train_test$xtr
y_train <- train_test$ytr

#model
EPOCHS = 50000
HIDDEN_NEURONS = hn
LEARNING_RATE = 0.9

train_model <- trainModel(X_train, y_train, hidden_neurons = HIDDEN_NEURONS, num_iteration = EPOCHS, lr = LEARNING_RATE)
write(train_model$cost_hist, ncol=1,file=costfile)

data=read.csv(costfile)
plot(data[1:50000,1],type="b")

#prediction
X_test <- train_test$xte
y_test <- train_test$yte
y_pred <- makePrediction(X_test, y_test, HIDDEN_NEURONS, train_model)
y_pred <- round(y_pred)

#Confusion Matrix
tb_nn <- table(y_test, y_pred)

#Accuracy Metrics
calculate_stats(tb_nn,"Neural Network")


#why it works, check by step by step test
#------------------------
train_model$updated_params
$W1
             X1         X2
[1,] -1.9431535 -0.6433269
[2,]  0.6150587  2.5983547
[3,] -0.9413749  2.6050836
[4,] -1.8528691  0.8206101

$b1
          [,1]
[1,] -1.950319
[2,] -1.950319
[3,] -1.950319
[4,] -1.950319

$W2
         [,1]     [,2]      [,3]      [,4]
[1,] 25.09141 24.56948 -19.34447 -18.92123

$b2
         [,1]
[1,] 8.284779

# -----
layer_size <- getLayerSize(X_train, y_train, hidden_neurons = hn)

xtest=t(head(t(X_test),10))
ytest=t(head(t(y_test),10))
fwd_prop <- forwardPropagation(xtest, train_model$updated_params, layer_size)
