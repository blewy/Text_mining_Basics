# Text Minig based on the Chapter 4 of Machine Learning With R  -Second Edition

library(jsonlite)
library(dplyr)
library(ggplot2)
library(tm) # For NLP; creating bag-of-words
library(caret)
library(SnowballC)
library(Matrix)
library(xgboost)
library(plyr)
library(dplyr)
library(topicmodels)

# sudo R CMD javareconf
#install.packages("rJava",type='source')
#install.packages("RWeka")
library(RWeka)
library(rJava)
library(SOAR)
Sys.setenv(R_LOCAL_CACHE="cache")
ls()
getwd()
      
## ------- Import Data -----------

data_import <- fromJSON("./data/train.json", flatten = TRUE)
data_test <- fromJSON("./data/test.json", flatten = TRUE)


table(data_import$cuisine)
str(data_import)

#ggplot(data = data_import, aes(x = cuisine)) + 
#  geom_histogram() +
#  labs(title = "Cuisines", x = "Cuisine", y = "Number of Recipes")

ingredients_corpus <- Corpus(VectorSource(data_import$ingredients))
test_ingredients_corpus <- Corpus(VectorSource(data_test$ingredients))
  
all_ingredients_corpus <- c(ingredients_corpus, test_ingredients_corpus)
print(all_ingredients_corpus)

inspect(all_ingredients_corpus[1:2])

as.character(all_ingredients_corpus[[1]])

lapply(all_ingredients_corpus[1:2], as.character)
  
#n gram tokenizer usinh Rweka package
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2))

#BigramTokenizer <-
#  function(x)
#    unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)

#--  Create a document-term sparse matrix  ----
options(mc.cores=1) # to solve the problem with weka NGramTokenizer
ingredients_dtm_fast <- DocumentTermMatrix(all_ingredients_corpus, control = list(
    tolower = TRUE,
    removeNumbers = TRUE,
    stopwords = function(x) { removeWords(x, stopwords()) },
    removePunctuation = TRUE,
    stemming = TRUE,
    tokenize=BigramTokenizer,
    weighting = weightTfIdf
  ))

print(ingredients_dtm_fast)

findFreqTerms(ingredients_dtm_fast, lowfreq = 10)
  

##--- Removing Sparse ingredients ----  

  ingredients_dtm_freq <- removeSparseTerms(ingredients_dtm_fast, 0.99)
  
  # save frequently-appearing terms to a character vector
  ingredients_freq_words <- findFreqTerms(ingredients_dtm_freq, 0)
  str(ingredients_freq_words)
  
  # create DTMs with only the frequent terms
  ingredients_dtm_clean <- ingredients_dtm_freq[, ingredients_freq_words]
  str(ingredients_dtm_clean)
  #cresting a Matrix for models
  ingredientsDTM_matrix <-as.matrix(ingredients_dtm_clean)
  class(ingredientsDTM_matrix)
  
  nrow(ingredientsDTM_matrix)
  
  # Adding some features
  # head(rowsum(ingredientsDTM_matrix)) # simple count of ingredients per receipe
  
  # create DTMs trainf & teste
  #- split the DTM into TRAIN and SUBMIT sets
  
  ingredients_dtm_train  <- ingredientsDTM_matrix[1:nrow(data_import), ]
  ingredients_dtm_test <- ingredientsDTM_matrix[-(1:nrow(data_import)), ]
  
  Store(ingredients_dtm_test)
    #labels
  ingredients_Label_train <- as.factor(data_import$cuisine)
  

#clean memory
rm(ingredients_corpus, test_ingredients_corpus, all_ingredients_corpus,ingredients_dtm_fast,ingredients_dtm_freq,ingredients_dtm_clean,ingredientsDTM_matrix,data_test,data_import)

## -------  XgbTree Model -----------

cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 2, 
                        classProbs = TRUE,
                        allowParallel=T,
                        savePredictions = "final",
                        selectionFunction = "oneSE",
                        returnData = FALSE,
                        returnResamp = "final")

xgb.grid = expand.grid(nrounds = 200, # Boosting Iterations
                         eta = c(0.01, 0.03),#2/ntrees,  Step size shrinkage used in update to prevents overfitting 
                         max_depth = c(10), #Maximum depth of a tree
                         gamma = 0,  # Minimum Loss Reduction
                         subsample = c(0.6), #Subsample ratio of the training instance
                         colsample_bytree = c(0.8),#Subsample ratio of columns when constructing each tree
                         min_child_weight = seq(5,10,10) #Minimum Sum of Instance Weight
)

dim(xgb.grid)
xgb.grid


set.seed(45)
xgb_tune <-train(x=ingredients_dtm_train,
                 y=ingredients_Label_train,
                 method="xgbTree",
                 trControl=cv.ctrl,
                 tuneGrid=xgb.grid,
                 verbose=T,
                 metric="Accuracy",
                 nthread =3
)
xgb_tune
ggplot(xgb_tune)

predictions<-predict(xgb_tune,ingredients_dtm_test)

# build the submission file with the id's of the test file and the predictions of the h2o model
submission <- data.frame(id = data_test$id, cuisine = predictions)

#- write out the submission file
write.csv(submission, file = 'submission_xboost.csv', row.names = F)

Store(xgb_tune)




#----  Random forest ----------------

  ctrl <- trainControl(method = "repeatedcv", 
                       repeats = 1,number = 3, 
                       classProbs = TRUE,
                       allowParallel=T,
                       savePredictions = "final",
                       selectionFunction = "oneSE",
                       returnData = FALSE,
                       returnResamp = "final")
  
  grid_rf <- expand.grid(.mtry = c(16))
  
  rforest <- train(x=ingredients_dtm_train,
                   y=ingredients_Label_train,
                   method = "rf",
                   metric = "Accuracy",
                   ntree = 600,
                   trControl = ctrl,
                   tuneGrid = grid_rf
  )
rforest
plot(rforest)

predictions<-predict(rforest,ingredients_dtm_test)

# build the submission file with the id's of the test file and the predictions of the h2o model
submission <- data.frame(id = data_test$id, cuisine = predictions)

#- write out the submission file
write.csv(submission, file = 'submission_rf.csv', row.names = F)

Store(rforest)

##---- Random forest randomForest package ------


library(randomForest)
set.seed(300)
rf <- randomForest(x=ingredients_dtm_train,ntree=1000,mtry=16,
                   y=ingredients_Label_train)
rf

confusionMatrix(data = predict(rf,ingredients_dtm_train), reference = ingredients_Label_train)


predictions<-predict(rf,ingredients_dtm_test)

# build the submission file with the id's of the test file and the predictions of the h2o model
submission <- data.frame(id = data_test$id, cuisine = predictions)

#- write out the submission file
write.csv(submission, file = 'submission_rforest.csv', row.names = F)

Store(rf)

##----------  Naive Bays ---------


convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

ingredients_train <- apply(ingredients_dtm_train, MARGIN = 2,
                   convert_counts)
ingredients_test <- apply(ingredients_dtm_test, MARGIN = 2,
                    convert_counts)

Store(ingredients_test)
Objects()

ctrl <- trainControl(method = "repeatedcv", 
                     repeats = 1,number = 3, 
                     classProbs = TRUE,
                     allowParallel=T,
                     savePredictions = "final",
                     selectionFunction = "oneSE",
                     returnData = FALSE,
                     returnResamp = "final")

grid <- expand.grid(.fL=c(1,2),.usekernel=TRUE )


set.seed(998)
NBayes <- train(x=ingredients_dtm_train,
                y=ingredients_Label_train,
                method = "nb",
                metric = "Accuracy",
                trControl = ctrl,
                tuneGrid = grid
                )
NBayes
ggplot(NBayes)

Store(NBayes)

predipredictions<-predict(NBayes,ingredients_test)

head(predictions)

# build the submission file with the id's of the test file and the predictions of the h2o model
submission <- data.frame(id = data_test$id, cuisine = predictions)

#- write out the submission file
write.csv(submission, file = 'submission.csv', row.names = F)


##--------------------  Stocastic grading boosting ---------------
set.seed(998)

# Model
library(gbm)
model_gbm <- gbm.fit(x=ingredientsDTM,
                     y=as.factor(data_import$cuisine),
                     distribution = "multinomial",
                     n.trees = 20, interaction.depth = 1,
                     n.minobsinnode = 10, shrinkage = 0.001,
                     bag.fraction = 0.5, keep.data = TRUE, verbose = TRUE)

best.iter <- gbm.perf(model_gbm,method="OOB")
best.iter

predictions<-predict(model_gbm,test_ingredientsDTM,best.iter,)

head(predictions)

confusionMatrix(data = tf_test, reference = ingredients_Label_test)

# build the submission file with the id's of the test file and the predictions of the h2o model
submission <- data.frame(id = data_test$id, cuisine = predictions)

#- write out the submission file
write.csv(submission, file = 'submission.csv', row.names = F)

### 


#------------------ Logistic regression --------------

set.seed(998)
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 3, 
                        classProbs = TRUE,
                        allowParallel=T,
                        savePredictions = "final",
                        selectionFunction = "oneSE",
                        returnData = FALSE,
                        returnResamp = "final")

# Penalized model grid 

# L1 absolute value ("lasso")   - tends to result in many regression coeffients shrunk exactly to zero and a few other regression coefcients with comparatively little shrinkage
#  L2 quadratic ("ridge") penalty - Smmal but no zero

# .lambda  - 
# .alpha= elasticnet mixing parameter, alpha=1 is the lasso penalty, and alpha=0 the ridge penalty.
library(glmnet)
grid = expand.grid(.alpha=c(0),.lambda=seq(0,0.1,by=0.025))

set.seed(998)
glm <- train(x=ingredients_dtm_train,
                y=ingredients_Label_train,
                method = "glmnet",
                family="multinomial",
                metric = "Accuracy",
                tuneGrid= grid,
                trControl = cv.ctrl)
glm
plot(glm)

predipredictions<-predict(glm,ingredients_dtm_test)

head(predictions)

# build the submission file with the id's of the test file and the predictions of the h2o model
submission <- data.frame(id = data_test$id, cuisine = predictions)

#- write out the submission file
write.csv(submission, file = 'submission_glm.csv', row.names = F)


Store(glm)

##-------------  Suport Vect Machines. --------------


library(caret)

set.seed(998)
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 3, 
                        classProbs = TRUE,
                        allowParallel=T,
                        savePredictions = "final",
                        selectionFunction = "oneSE",
                        returnData = FALSE,
                        returnResamp = "final")

svmRGridReduced <- expand.grid(.lambda=0.25,.C = c(1,4,10,20))

#sigma os a parameter to be used on svmRadial type of svm, and c is the cost value 
#to create redularization of the model so it would not over fit.

getModelInfo("svmExpoString")

svmRModel <- train(x=ingredients_dtm_train,
                   y=ingredients_Label_train,
                   method = "svmRadial", 
                   metric = "Accuracy", 
                   tuneGrid = svmRGridReduced, 
                   fit = FALSE,
                   trControl = ctrl)
svmRModel
plot(svmRModel)

