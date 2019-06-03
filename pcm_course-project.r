##############################################
# load required libraries                    #
##############################################
library(caret)
library(rattle)

##############################################
# load training and testing data             #
##############################################
rawTrain <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"),header=TRUE)
dim(rawTrain)

rawTest <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"),header=TRUE)
dim(rawTest)

##############################################
# create and clean training/validation sets  #
##############################################
# rawTrain is 19622 observations of 160 variables
# rawTest is 120 observations of 160 variables
# (1) split train into training and testing sets
# (2) remove near zero variables
# (3) remove NA variables
# (4) remove ID and timestamp variables

# (1) split rawTrain data into training and testing data sets
inTrain  <- createDataPartition(rawTrain$classe, p=0.75, list=FALSE)
trainSet <- rawTrain[inTrain, ]
testSet  <- rawTrain[-inTrain, ]
# (1) copy rawTest to validateSet
validateSet <- rawTest

# (2) identify near zero variables in trainSet and validateSet
# (2) near zero variables = variables with 95%+ same value
nzvColsT <- nearZeroVar(trainSet, freqCut = 95/5)
nzvColsV <- nearZeroVar(validateSet, freqCut = 95/5)

# (2) remove near zero variable columns from trainSet, testSet & validateSet
nzv_trainSet <- trainSet[, -nzvColsT]
nzv_testSet <- testSet[, -nzvColsT]
nzv_validateSet <- validateSet[, -nzvColsV]

# (3) identify variable columns with 95%+ NA in nzv_trainSet & nzv_validateSet
colsNAT <- sapply(nzv_trainSet, function(x) mean(is.na(x))) > 0.95
colsNAV <- sapply(nzv_validateSet, function(x) mean(is.na(x))) > 0.95

# (3) remove identified colummns from nzv_trainSet, nzv_testSet & nzv_validateSet 
nzvna_trainSet <- nzv_trainSet[, colsNAT==FALSE]
nzvna_testSet <- nzv_testSet[, colsNAT==FALSE]
nzvna_validateSet <- nzv_validateSet[, colsNAV==FALSE]

# (4) the first 5 columns are X, username and timestamps
# (4) we can remove these are they are non-predictive for modeling
clean_trainSet <- nzvna_trainSet[, -(1:5)]
clean_testSet <- nzvna_testSet[, -(1:5)]
clean_validateSet <- nzvna_validateSet[, -(1:5)]

##############################################
# Modeling & Predicting                      #
##############################################
# (1) gbm - Gradient Boosting Machine
# (2) rf - Random Forest

# (1) generate gbm model from clean_trainSet
modelGBM <- train(classe ~ ., method = "gbm", data = clean_trainSet, verbose = FALSE)

# (1) view model results
modelGBM$finalModel
print(modelGBM)

# (1) predict classe of clean_testSet using modelGBM
predictGBM <- predict(modelGBM, newdata = clean_testSet)

# (1) check results of GBM prediction using confusion matrix
conmatGBM <- confusionMatrix(predictGBM, clean_testSet$classe)
conmatGBM

# (2) generate rf model from clean_trainSet
modelRF <- train(classe ~ ., method = "rf", data = clean_trainSet,
                 trControl = trainControl(method="cv"),number=3)

# (2) view model results
modelRF$finalModel
print(modelRF)

# (2) predict classe of clean_testSet using modelRF
predictRF <- predict(modelRF, newdata = clean_testSet)

# (2) check results of RF prediction using confusion matrix
conmatRF <- confusionMatrix(predictRF, clean_testSet$classe)
conmatRF

##############################################
# Apply RF model to clean_validateSet        #
# finalPreds = answers to final quiz         #
##############################################

finalPreds <- predict(modelRF, newdata = clean_validateSet)
finalPreds

