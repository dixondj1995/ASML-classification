# install packages and load libraries 

install.packages('data.table', 'mlr3verse','precrec', 'mlr3', 'mlr3viz', 'skimr', 'DataExplorer', 'rpart')

install.packages("apcluster")

library("data.table")
library("mlr3verse")
library('precrec')
library('mlr3')
library('mlr3viz')

# load the dataset into R:

download.file("https://www.louisaslett.com/Courses/MISCADA/heart_failure.csv", "heart_failure.csv")

heart_failure <- read.csv("heart_failure.csv", stringsAsFactors = TRUE)


# Data exploration
# skim through the data
skimr::skim(heart_failure)

# a variety of plots 

DataExplorer::plot_bar(heart_failure, ncol = 3)

DataExplorer::plot_histogram(heart_failure, ncol = 3)

DataExplorer::plot_boxplot(heart_failure, by = "fatal_mi", ncol = 3)


### Task and resampling

# install all packages 

# load all packages 

library("data.table")
library("mlr3verse")

# set seeds and define task

set.seed(212) # set seed for reproducibility

heart_failure$fatal_mi <- as.factor(heart_failure$fatal_mi)

task_cardiac <- TaskClassif$new(id = "cardiac",
                               backend = na.omit(heart_failure),
                               target = "fatal_mi",
                               positive = '1')

task_cardiac

# 5-fold cross validation 
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(task_cardiac)

cv10 <- rsmp("cv", folds = 10)
cv10$instantiate(task_cardiac)


# install.packages("apcluster")
#mlr_learners
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart <- lrn("classif.rpart", predict_type = "prob")



# fitting the learners 

res_baseline <- resample(task_cardiac, lrn_baseline, cv5, store_models = TRUE)
res_cart <- resample(task_cardiac, lrn_cart, cv5, store_models = TRUE)

# Look at accuracy
res_baseline$aggregate()
res_cart$aggregate()



# benchmarking 

res <- benchmark(data.table(
  task       = list(task_cardiac),
  learner    = list(lrn_baseline,
                    lrn_cart),
  resampling = list(cv5)
), store_models = TRUE)
res
res$aggregate()

# aggregate many 

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

# get the trees (2nd model fitted), by asking for second set of resample
# results
trees <- res$resample_result(2)

# tree from first CV iteration:
tree1 <- trees$learners[[1]]

tree1_rpart <- tree1$model

# plot the tree that was fitted
plot(tree1_rpart, compress = TRUE, margin = 0.1)
text(tree1_rpart, use.n = TRUE, cex = 0.8)

#' We can see the other trees too.  Change the 5 in double brackets [[]] below to other values from 1 to 5 to see the model from each round of cross validation.
plot(res$resample_result(2)$learners[[4]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[4]]$model, use.n = TRUE, cex = 0.8)

plot(res$resample_result(2)$learners[[5]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[5]]$model, use.n = TRUE, cex = 0.8)


# Enable cross validation
lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)

res_cart_cv <- resample(task_cardiac, lrn_cart_cv, cv5, store_models = TRUE)
rpart::plotcp(res_cart_cv$learners[[1]]$model)
rpart::plotcp(res_cart_cv$learners[[2]]$model)
rpart::plotcp(res_cart_cv$learners[[3]]$model)
rpart::plotcp(res_cart_cv$learners[[4]]$model)
rpart::plotcp(res_cart_cv$learners[[5]]$model)


#' Now, choose a cost penalty and add this as a model to our benchmark set:
lrn_cart_cp <- lrn("classif.rpart", predict_type = "prob", cp = 0.02)

res <- benchmark(data.table(
  task       = list(task_cardiac),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_cart_cp),
  resampling = list(cv5)
), store_models = TRUE)

### check parameter choice has an impact on scores

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

### unpruned tree

plot(res$resample_result(2)$learners[[5]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[5]]$model, use.n = TRUE, cex = 0.8)

### pruned tree
plot(res$resample_result(3)$learners[[5]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(3)$learners[[5]]$model, use.n = TRUE, cex = 0.8)

# superlearner

set.seed(212) # set seed for reproducibility

task_cardiac <- TaskClassif$new(id = "cardiac",
                                backend = na.omit(heart_failure),
                                target = "fatal_mi",
                                positive = '1')

task_cardiac

# Cross validation resampling strategy
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(task_cardiac)

# Define a collection of base learners
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart     <- lrn("classif.rpart", predict_type = "prob")
lrn_cart_cp  <- lrn("classif.rpart", predict_type = "prob", cp = 0.02, id = "cartcp")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")

# Define a super learner
lrnsp_log_reg <- lrn("classif.log_reg", predict_type = "prob", id = "super")

# Missingness imputation pipeline
pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")

# Factors coding pipeline
pl_factor <- po("encode")

# Now define the full pipeline
spr_lrn <- gunion(list(
  # First group of learners requiring no modification to input
  gunion(list(
    po("learner_cv", lrn_baseline),
    po("learner_cv", lrn_cart),
    po("learner_cv", lrn_cart_cp)
  )),
  # Next group of learners requiring special treatment of missingness
  pl_missing %>>%
    gunion(list(
      po("learner_cv", lrn_ranger),
      po("learner_cv", lrn_log_reg) ,
      po("nop") # This passes through the original features adjusted for
      # missingness to the super learner
    )),
  # Last group needing factor encoding
  pl_factor %>>%
    po("learner_cv", lrn_xgboost)
)) %>>%
  po("featureunion") %>>%
  po(lrnsp_log_reg)


# Finally fit the base learners and super learner and evaluate
res_spr <- resample(task_cardiac, spr_lrn, cv5, store_models = TRUE)
res_spr$aggregate(list(msr("classif.ce"),
                       msr("classif.acc"),
                       msr("classif.fpr"),
                       msr("classif.fnr")))

# full benchmark without the superlearner 

res <- benchmark(data.table(
  task       = list(task_cardiac),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_cart_cp,
                    lrn_ranger,
                    lrn_log_reg,
                    lrn_xgboost),
  resampling = list(cv5)
), store_models = TRUE)


res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.fpr"),
                   msr("classif.fnr"),
                   msr("classif.auc"),
                   msr("classif.prauc"),
                   msr("classif.sensitivity"), 
                   msr("classif.specificity"),
                   msr("classif.ppv"),
                   msr("classif.npv")))


# flora - trees and forests only 

flora_res <- benchmark(data.table(
  task       = list(task_cardiac),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_cart_cp,
                    lrn_ranger),
  resampling = list(cv5)
), store_models = TRUE)


flora_res$aggregate(list(msr("classif.ce"),
                         msr("classif.acc"),
                         msr("classif.fpr"),
                         msr("classif.fnr"),
                         msr("classif.auc"),
                         msr("classif.prauc"),
                         msr("classif.sensitivity"), 
                         msr("classif.specificity"),
                         msr("classif.ppv"),
                         msr("classif.npv")))

# look at our results

head(fortify(flora_res))

# view box and whiskers

autoplot(flora_res)

# ROC curve

autoplot(flora_res$clone(deep = TRUE)$filter(task_ids = "cardiac"), type = "roc")

# PRC curve

autoplot(flora_res$clone(deep = TRUE)$filter(task_ids = "cardiac"), type = "prc")

# best three plots + baseline 
#### these are going in the report

final_res <- benchmark(data.table(
  task       = list(task_cardiac),
  learner    = list(lrn_baseline,
                    lrn_log_reg,
                    lrn_cart_cp,
                    lrn_ranger),
  resampling = list(cv5)
), store_models = TRUE)


final_res$aggregate(list(msr("classif.ce"),
                         msr("classif.acc"),
                         msr("classif.fpr"),
                         msr("classif.fnr"),
                         msr("classif.auc"),
                         msr("classif.prauc"),
                         msr("classif.sensitivity"), 
                         msr("classif.specificity"),
                         msr("classif.ppv"),
                         msr("classif.npv")))

# look at our results

head(fortify(final_res))

# view box and whiskers

autoplot(final_res)

# ROC curve

autoplot(final_res$clone(deep = TRUE)$filter(task_ids = "cardiac"), type = "roc")

# PRC curve

autoplot(final_res$clone(deep = TRUE)$filter(task_ids = "cardiac"), type = "prc")

###########
### log reg plots
###########

# ROC 
autoplot(final_res$clone(deep = TRUE)$filter(task_ids = "cardiac")$resample_result(2), type = "roc")

# PRC curve

autoplot(final_res$clone(deep = TRUE)$filter(task_ids = "cardiac")$resample_result(2), type = "prc")

###########
### CART CP plots
###########

# ROC 
autoplot(final_res$clone(deep = TRUE)$filter(task_ids = "cardiac")$resample_result(3), type = "roc")

# PRC curve

autoplot(final_res$clone(deep = TRUE)$filter(task_ids = "cardiac")$resample_result(3), type = "prc")

###########
### Random forest plots
###########

# ROC 
autoplot(final_res$clone(deep = TRUE)$filter(task_ids = "cardiac")$resample_result(4), type = "roc")

# PRC curve

autoplot(final_res$clone(deep = TRUE)$filter(task_ids = "cardiac")$resample_result(4), type = "prc")


################################
# tuning log reg probabilities 
################################


task = task_cardiac

# Explicitly instantiate the resampling for this task for reproduciblity
set.seed(212)
cv5$instantiate(task)

rr = resample(task, lrn_log_reg, cv5)
print(rr)

# Retrieve performance
rr$score(msr("classif.ce"))

rr$aggregate(msr("classif.ce"))
#> classif.ce 
#> 0.05285714 

# merged prediction objects of all resampling iterations
pred = rr$prediction()

pred$confusion


# Repeat resampling with featureless learner
rr_featureless = resample(task, lrn("classif.featureless"), cv5)


pred$confusion


# new predictions
pred$set_threshold(0.05)$response

pred$confusion

pred$score(measures = list(msr("classif.ce"),
                           msr("classif.acc"),
                           msr("classif.fpr"),
                           msr("classif.fnr")))
