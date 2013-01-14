library("caret")
train <- read.csv("train.csv", header=TRUE, as.is=TRUE )
test <- read.csv("test.csv", header=TRUE, as.is=TRUE )
x <- train
test_x <- test
train_x <- x[,-1]
descrCorr <- cor(train_x)
highCorr <- findCorrelation(descrCorr, 0.90)
trainDescr <- train_x[, -highCorr]
testDescr <- test_x[, -highCorr]
ncol(trainDescr)
ncol(testDescr)
notrainDescr <- train_x[, highCorr]
ncol(notrainDescr)
xTrans <- preProcess(trainDescr)
trainDescr <- predict(xTrans, trainDescr)
testDescr <- predict(xTrans, testDescr)
ncol(trainDescr)
ncol(testDescr)
write.csv(trainDescr, "train1749.csv", col.names=TRUE)
write.csv(testDescr, "test1749.csv", col.names=TRUE)