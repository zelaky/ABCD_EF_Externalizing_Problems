#Title: Multivariate Clinical Analyses in the Training Set
#Author: Zoë E. Laky, M.A.
#Contact: zoe.laky@nih.gov
#Project Description: 
##Exploring multivariate associations between EF cluster membership and clinical diagnoses and symptoms in the training set for the baseline and 2-year follow-up.

#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")
#BiocManager::install("BiocParallel")
#install.packages("oosse") 
#install.packages("rlang")
#install.packages(c("dplyr", "tidyverse"))

library(oosse)
library(glmnet)
library(dplyr)
set.seed(843)
#set.seed(NULL)

#Functions
fitFun_lr = function(y, x){lm.fit(y = y, x = cbind(1, x))}
predFun_lr = function(mod, x) {cbind(1,x) %*% mod$coef} #is mod fit?
pvalue_est <- function(conf) {
  res <- buildConfInt(model, what = c("R2"), conf = conf)
  return(res[1]^2)
}

##All: Baseline to Baseline
#cbcl
all_t0_t0_cbcl_train <- read.csv('/user_path/all_t0_t0_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(all_t0_t0_cbcl_train[, c("cbcl_aggressive_t_t0", "cbcl_attention_t_t0", "cbcl_rulebreak_t_t0")])
y <- all_t0_t0_cbcl_train$tasks_t0_km_2_resid

cbcl_all_t0_t0_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                             methodMSE = c("bootstrap"), 
                             methodCor = c("jackknife"),
                             printTimeEstimate = TRUE,
                             cvReps = 200L,
                             nBootstraps = 200L,
                             nBootstrapsCor = 50L)
cbcl_all_t0_t0_lr$R2
#R2           R2SE 
#0.009826996  0.040043273
model <- cbcl_all_t0_t0_lr 
cbcl_all_t0_t0_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) #iterate through 0.95 until one end is 0, and then 1-the other end confidence is p-value
cbcl_all_t0_t0_ci
#2.5%           97.5% 
#-0.06865638    0.08831037
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 #only want to test if the R value is significantly positive, not negative (one-tailed)
pvalue
#0.4030789
rm(all_t0_t0_cbcl_train, cbcl_all_t0_t0_lr, cbcl_all_t0_t0_ci)

#ksads
all_t0_t0_ksads_train <- read.csv('/user_path/all_t0_t0_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(all_t0_t0_ksads_train[, c('ksads_odd_present_t0', 'ksads_cd_present_rc_t0', 'ksads_adhd_present_rc_t0')])
y <- all_t0_t0_ksads_train$tasks_t0_km_2_resid

ksads_all_t0_t0_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                             methodMSE = c("bootstrap"), 
                             methodCor = c("jackknife"),
                             printTimeEstimate = TRUE,
                             cvReps = 200L,
                             nBootstraps = 200L,
                             nBootstrapsCor = 50L)
ksads_all_t0_t0_lr$R2
#R2           R2SE 
#0.004006968  0.039058614
model <- ksads_all_t0_t0_lr 
ksads_all_t0_t0_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_all_t0_t0_ci
#2.5%           97.5% 
#-0.07254651  0.08056045
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
#0.4591436
rm(all_t0_t0_ksads_train, ksads_all_t0_t0_lr, ksads_all_t0_t0_ci)

##All: Baseline to 1-year Follow-up
#cbcl
all_t0_t1_cbcl_train <- read.csv('/user_path/all_t0_t1_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(all_t0_t1_cbcl_train[, c("cbcl_aggressive_t_t1", "cbcl_attention_t_t1", "cbcl_rulebreak_t_t1")])
y <- all_t0_t1_cbcl_train$tasks_t0_km_2_resid

cbcl_all_t0_t1_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                             methodMSE = c("bootstrap"), 
                             methodCor = c("jackknife"),
                             printTimeEstimate = TRUE,
                             cvReps = 200L,
                             nBootstraps = 200L,
                             nBootstrapsCor = 50L)
cbcl_all_t0_t1_lr$R2
#R2           R2SE 
#0.009783144  0.039563658  
model <- cbcl_all_t0_t1_lr 
cbcl_all_t0_t1_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) #iterate through 0.95 until one end is 0, and then 1-the other end confidence is p-value
cbcl_all_t0_t1_ci
#2.5%           97.5% 
#-0.06776020    0.08732649  
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 #only want to test if the R value is significantly positive, not negative (one-tailed)
pvalue
#0.402357
rm(all_t0_t1_cbcl_train, cbcl_all_t0_t1_lr, cbcl_all_t0_t1_ci)

#ksads
all_t0_t1_ksads_train <- read.csv('/user_path/all_t0_t1_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(all_t0_t1_ksads_train[, c('ksads_odd_present_t1', 'ksads_cd_present_rc_t1', 'ksads_adhd_present_rc_t1')])
y <- all_t0_t1_ksads_train$tasks_t0_km_2_resid

ksads_all_t0_t1_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                              methodMSE = c("bootstrap"), 
                              methodCor = c("jackknife"),
                              printTimeEstimate = TRUE,
                              cvReps = 200L,
                              nBootstraps = 200L,
                              nBootstrapsCor = 50L)
ksads_all_t0_t1_lr$R2
#R2           R2SE 
#0.004653923  0.039466469 
model <- ksads_all_t0_t1_lr 
ksads_all_t0_t1_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_all_t0_t1_ci
#2.5%           97.5% 
#-0.07269894    0.08200678
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
#0.4530641
rm(all_t0_t1_ksads_train, ksads_all_t0_t1_lr, ksads_all_t0_t1_ci)

##All: Baseline to 2-year Follow-up
#cbcl
all_t0_t2_cbcl_train <- read.csv('/user_path/all_t0_t2_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(all_t0_t2_cbcl_train[, c("cbcl_aggressive_t_t2", "cbcl_attention_t_t2", "cbcl_rulebreak_t_t2")])
y <- all_t0_t2_cbcl_train$tasks_t0_km_2_resid

cbcl_all_t0_t2_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                             methodMSE = c("bootstrap"), 
                             methodCor = c("jackknife"),
                             printTimeEstimate = TRUE,
                             cvReps = 200L,
                             nBootstraps = 200L,
                             nBootstrapsCor = 50L)
cbcl_all_t0_t2_lr$R2
#R2           R2SE 
#0.009061549  0.040365398  
model <- cbcl_all_t0_t2_lr 
cbcl_all_t0_t2_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) #iterate through 0.95 until one end is 0, and then 1-the other end confidence is p-value
cbcl_all_t0_t2_ci
#2.5%           97.5% 
#-0.07005318    0.08817628
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 #only want to test if the R value is significantly positive, not negative (one-tailed)
pvalue
#0.411192
rm(all_t0_t2_cbcl_train, cbcl_all_t0_t2_lr, cbcl_all_t0_t2_ci)

#ksads
all_t0_t2_ksads_train <- read.csv('/user_path/all_t0_t2_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(all_t0_t2_ksads_train[, c('ksads_odd_present_t2', 'ksads_cd_present_rc_t2', 'ksads_adhd_present_rc_t2')])
y <- all_t0_t2_ksads_train$tasks_t0_km_2_resid

ksads_all_t0_t2_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                              methodMSE = c("bootstrap"), 
                              methodCor = c("jackknife"),
                              printTimeEstimate = TRUE,
                              cvReps = 200L,
                              nBootstraps = 200L,
                              nBootstrapsCor = 50L)
ksads_all_t0_t2_lr$R2
#R2           R2SE 
#0.004046846  0.040133795 
model <- ksads_all_t0_t2_lr 
ksads_all_t0_t2_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_all_t0_t2_ci
#2.5%           97.5% 
#-0.07461395    0.08270764
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
#0.45984
rm(all_t0_t2_ksads_train, ksads_all_t0_t2_lr, ksads_all_t0_t2_ci)

##All: Baseline to 3-year Follow-up
#cbcl
all_t0_t3_cbcl_train <- read.csv('/user_path/all_t0_t3_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(all_t0_t3_cbcl_train[, c("cbcl_aggressive_t_t3", "cbcl_attention_t_t3", "cbcl_rulebreak_t_t3")])
y <- all_t0_t3_cbcl_train$tasks_t0_km_2_resid

cbcl_all_t0_t3_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                             methodMSE = c("bootstrap"), 
                             methodCor = c("jackknife"),
                             printTimeEstimate = TRUE,
                             cvReps = 200L,
                             nBootstraps = 200L,
                             nBootstrapsCor = 50L)
cbcl_all_t0_t3_lr$R2
#R2           R2SE 
#0.005117847  0.040672293
model <- cbcl_all_t0_t3_lr 
cbcl_all_t0_t3_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) #iterate through 0.95 until one end is 0, and then 1-the other end confidence is p-value
cbcl_all_t0_t3_ci
#2.5%           97.5% 
#-0.07459838    0.08483408
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 #only want to test if the R value is significantly positive, not negative (one-tailed)
pvalue
#0.4030789
rm(all_t0_t3_cbcl_train, cbcl_all_t0_t3_lr, cbcl_all_t0_t3_ci)

#ksads
all_t0_t3_ksads_train <- read.csv('/user_path/all_t0_t3_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(all_t0_t3_ksads_train[, c('ksads_cd_present_rc_t3', 'ksads_adhd_present_rc_t3')])
y <- all_t0_t3_ksads_train$tasks_t0_km_2_resid

ksads_all_t0_t3_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                              methodMSE = c("bootstrap"), 
                              methodCor = c("jackknife"),
                              printTimeEstimate = TRUE,
                              cvReps = 200L,
                              nBootstraps = 200L,
                              nBootstrapsCor = 50L)
ksads_all_t0_t3_lr$R2
#R2           R2SE 
#0.00125034   0.03997136
model <- ksads_all_t0_t3_lr 
ksads_all_t0_t3_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_all_t0_t3_ci
#2.5%           97.5% 
#-0.07709209    0.07959277
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
#0.4875177
rm(all_t0_t3_ksads_train, ksads_all_t0_t3_lr, ksads_all_t0_t3_ci)

##All: 2-year Follow-up to Baseline
#cbcl
all_t2_t0_cbcl_train <- read.csv('/user_path/all_t2_t0_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(all_t2_t0_cbcl_train[, c("cbcl_aggressive_t_t0", "cbcl_attention_t_t0", "cbcl_rulebreak_t_t0")])
y <- all_t2_t0_cbcl_train$tasks_t2_km_2_resid

cbcl_all_t2_t0_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                             methodMSE = c("bootstrap"), 
                             methodCor = c("jackknife"),
                             printTimeEstimate = TRUE,
                             cvReps = 200L,
                             nBootstraps = 200L,
                             nBootstrapsCor = 50L)
cbcl_all_t2_t0_lr$R2
#R2           R2SE 
#0.006466164  0.039871649
model <- cbcl_all_t2_t0_lr 
cbcl_all_t2_t0_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) #iterate through 0.95 until one end is 0, and then 1-the other end confidence is p-value
cbcl_all_t2_t0_ci
#2.5%           97.5% 
#-0.07168083    0.08461316
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 #only want to test if the R value is significantly positive, not negative (one-tailed)
pvalue
#0.4355836
rm(all_t2_t0_cbcl_train, cbcl_all_t2_t0_lr, cbcl_all_t2_t0_ci)

#ksads
all_t2_t0_ksads_train <- read.csv('/user_path/all_t2_t0_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(all_t2_t0_ksads_train[, c('ksads_odd_present_t0', 'ksads_cd_present_rc_t0', 'ksads_adhd_present_rc_t0')])
y <- all_t2_t0_ksads_train$tasks_t2_km_2_resid

ksads_all_t2_t0_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                              methodMSE = c("bootstrap"), 
                              methodCor = c("jackknife"),
                              printTimeEstimate = TRUE,
                              cvReps = 200L,
                              nBootstraps = 200L,
                              nBootstrapsCor = 50L)
ksads_all_t2_t0_lr$R2
#R2           R2SE 
#0.001621219  0.039455101
model <- ksads_all_t2_t0_lr 
ksads_all_t2_t0_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_all_t2_t0_ci
#2.5%           97.5% 
#-0.07570936    0.07895180
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
#0.4836072
rm(all_t2_t0_ksads_train, ksads_all_t2_t0_lr, ksads_all_t2_t0_ci)

##All: 2-year Follow-up to 1-year Follow-up
#cbcl
all_t2_t1_cbcl_train <- read.csv('/user_path/all_t2_t1_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(all_t2_t1_cbcl_train[, c("cbcl_aggressive_t_t1", "cbcl_attention_t_t1", "cbcl_rulebreak_t_t1")])
y <- all_t2_t1_cbcl_train$tasks_t2_km_2_resid

cbcl_all_t2_t1_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                             methodMSE = c("bootstrap"), 
                             methodCor = c("jackknife"),
                             printTimeEstimate = TRUE,
                             cvReps = 200L,
                             nBootstraps = 200L,
                             nBootstrapsCor = 50L)
cbcl_all_t2_t1_lr$R2
#R2           R2SE 
#0.007434913  0.039174385  
model <- cbcl_all_t2_t1_lr 
cbcl_all_t2_t1_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) #iterate through 0.95 until one end is 0, and then 1-the other end confidence is p-value
cbcl_all_t2_t1_ci
#2.5%           97.5% 
#-0.06934547    0.08421530   
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 #only want to test if the R value is significantly positive, not negative (one-tailed)
pvalue
#0.4247364
rm(all_t2_t1_cbcl_train, cbcl_all_t2_t1_lr, cbcl_all_t2_t1_ci)

#ksads
all_t2_t1_ksads_train <- read.csv('/user_path/all_t2_t1_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(all_t2_t1_ksads_train[, c('ksads_odd_present_t1', 'ksads_cd_present_rc_t1', 'ksads_adhd_present_rc_t1')])
y <- all_t2_t1_ksads_train$tasks_t2_km_2_resid

ksads_all_t2_t1_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                              methodMSE = c("bootstrap"), 
                              methodCor = c("jackknife"),
                              printTimeEstimate = TRUE,
                              cvReps = 200L,
                              nBootstraps = 200L,
                              nBootstrapsCor = 50L)
ksads_all_t2_t1_lr$R2
#R2           R2SE 
#0.005985576  0.039813710
model <- ksads_all_t2_t1_lr 
ksads_all_t2_t1_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_all_t2_t1_ci
#2.5%           97.5% 
#-0.07204786    0.08401901
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
#0.4402476
rm(all_t2_t1_ksads_train, ksads_all_t2_t1_lr, ksads_all_t2_t1_ci)

##All: 2-year Follow-up to 2-year Follow-up
#cbcl
all_t2_t2_cbcl_train <- read.csv('/user_path/all_t2_t2_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(all_t2_t2_cbcl_train[, c("cbcl_aggressive_t_t2", "cbcl_attention_t_t2", "cbcl_rulebreak_t_t2")])
y <- all_t2_t2_cbcl_train$tasks_t2_km_2_resid

cbcl_all_t2_t2_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                             methodMSE = c("bootstrap"), 
                             methodCor = c("jackknife"),
                             printTimeEstimate = TRUE,
                             cvReps = 200L,
                             nBootstraps = 200L,
                             nBootstrapsCor = 50L)
cbcl_all_t2_t2_lr$R2
#R2           R2SE 
#0.00811324   0.03983658 
model <- cbcl_all_t2_t2_lr 
cbcl_all_t2_t2_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) #iterate through 0.95 until one end is 0, and then 1-the other end confidence is p-value
cbcl_all_t2_t2_ci
#2.5%           97.5% 
#-0.06996502    0.08619150 
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 #only want to test if the R value is significantly positive, not negative (one-tailed)
pvalue
#0.4193081
rm(all_t2_t2_cbcl_train, cbcl_all_t2_t2_lr, cbcl_all_t2_t2_ci)

#ksads
all_t2_t2_ksads_train <- read.csv('/user_path/all_t2_t2_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(all_t2_t2_ksads_train[, c('ksads_odd_present_t2', 'ksads_cd_present_rc_t2', 'ksads_adhd_present_rc_t2')])
y <- all_t2_t2_ksads_train$tasks_t2_km_2_resid

ksads_all_t2_t2_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                              methodMSE = c("bootstrap"), 
                              methodCor = c("jackknife"),
                              printTimeEstimate = TRUE,
                              cvReps = 200L,
                              nBootstraps = 200L,
                              nBootstrapsCor = 50L)
ksads_all_t2_t2_lr$R2
#R2           R2SE 
#0.00267244   0.03967223 
model <- ksads_all_t2_t2_lr 
ksads_all_t2_t2_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_all_t2_t2_ci
#2.5%           97.5% 
#-0.07508369    0.08042857
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
#0.4731428
rm(all_t2_t2_ksads_train, ksads_all_t2_t2_lr, ksads_all_t2_t2_ci)

##All: 2-year Follow-up to 3-year Follow-up
#cbcl
all_t2_t3_cbcl_train <- read.csv('/user_path/all_t2_t3_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(all_t2_t3_cbcl_train[, c("cbcl_aggressive_t_t3", "cbcl_attention_t_t3", "cbcl_rulebreak_t_t3")])
y <- all_t2_t3_cbcl_train$tasks_t2_km_2_resid

cbcl_all_t2_t3_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                             methodMSE = c("bootstrap"), 
                             methodCor = c("jackknife"),
                             printTimeEstimate = TRUE,
                             cvReps = 200L,
                             nBootstraps = 200L,
                             nBootstrapsCor = 50L)
cbcl_all_t2_t3_lr$R2
#R2           R2SE 
#0.004303795  0.040536809
model <- cbcl_all_t2_t3_lr 
cbcl_all_t2_t3_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) #iterate through 0.95 until one end is 0, and then 1-the other end confidence is p-value
cbcl_all_t2_t3_ci
#2.5%           97.5% 
#-0.07514689    0.08375448
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 #only want to test if the R value is significantly positive, not negative (one-tailed)
pvalue
#0.4577225
rm(all_t2_t3_cbcl_train, cbcl_all_t2_t3_lr, cbcl_all_t2_t3_ci)

#ksads
all_t2_t3_ksads_train <- read.csv('/user_path/all_t2_t3_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(all_t2_t3_ksads_train[, c('ksads_cd_present_rc_t3', 'ksads_adhd_present_rc_t3')])
y <- all_t2_t3_ksads_train$tasks_t2_km_2_resid

ksads_all_t2_t3_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                              methodMSE = c("bootstrap"), 
                              methodCor = c("jackknife"),
                              printTimeEstimate = TRUE,
                              cvReps = 200L,
                              nBootstraps = 200L,
                              nBootstrapsCor = 50L)
ksads_all_t2_t3_lr$R2
#R2           R2SE 
#7.408842e-06 3.976524e-02 
model <- ksads_all_t2_t3_lr 
ksads_all_t2_t3_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_all_t2_t3_ci
#2.5%           97.5% 
#-0.07793104    0.07794585
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
#0.4999116
rm(all_t2_t3_ksads_train, ksads_all_t2_t3_lr, ksads_all_t2_t3_ci)

##Singleton: Baseline to Baseline
#cbcl
singleton_t0_t0_cbcl_train <- read.csv('/user_path/singleton_t0_t0_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(singleton_t0_t0_cbcl_train[, c("cbcl_aggressive_t_t0", "cbcl_attention_t_t0", "cbcl_rulebreak_t_t0")])
y <- singleton_t0_t0_cbcl_train$tasks_t0_km_2_resid

cbcl_singleton_t0_t0_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                                   methodMSE = c("bootstrap"), 
                                   methodCor = c("jackknife"),
                                   printTimeEstimate = TRUE,
                                   cvReps = 200L,
                                   nBootstraps = 200L,
                                   nBootstrapsCor = 50L)
cbcl_singleton_t0_t0_lr$R2
#R2           R2SE 
#0.01067957   0.04161415
model <- cbcl_singleton_t0_t0_lr 
cbcl_singleton_t0_t0_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) #iterate through 0.95 until one end is 0, and then 1-the other end confidence is p-value
cbcl_singleton_t0_t0_ci
#2.5%           97.5% 
#-0.07088266    0.09224180
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 #only want to test if the R value is significantly positive, not negative (one-tailed)
pvalue
#0.3987225
rm(singleton_t0_t0_cbcl_train, cbcl_singleton_t0_t0_lr, cbcl_singleton_t0_t0_ci)

#ksads
singleton_t0_t0_ksads_train <- read.csv('/user_path/singleton_t0_t0_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(singleton_t0_t0_ksads_train[, c('ksads_odd_present_t0', 'ksads_cd_present_rc_t0', 'ksads_adhd_present_rc_t0')])
y <- singleton_t0_t0_ksads_train$tasks_t0_km_2_resid

ksads_singleton_t0_t0_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                                    methodMSE = c("bootstrap"), 
                                    methodCor = c("jackknife"),
                                    printTimeEstimate = TRUE,
                                    cvReps = 200L,
                                    nBootstraps = 200L,
                                    nBootstrapsCor = 50L)
ksads_singleton_t0_t0_lr$R2
#R2           R2SE 
#0.002738504  0.041531400
model <- ksads_singleton_t0_t0_lr 
ksads_singleton_t0_t0_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_singleton_t0_t0_ci
#2.5%           97.5% 
#-0.07866154    0.08413855
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
#0.4737099
rm(singleton_t0_t0_ksads_train, ksads_singleton_t0_t0_lr, ksads_singleton_t0_t0_ci)

##Singleton: Baseline to 1-year Follow-up
#cbcl
singleton_t0_t1_cbcl_train <- read.csv('/user_path/singleton_t0_t1_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(singleton_t0_t1_cbcl_train[, c("cbcl_aggressive_t_t1", "cbcl_attention_t_t1", "cbcl_rulebreak_t_t1")])
y <- singleton_t0_t1_cbcl_train$tasks_t0_km_2_resid

cbcl_singleton_t0_t1_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                                   methodMSE = c("bootstrap"), 
                                   methodCor = c("jackknife"),
                                   printTimeEstimate = TRUE,
                                   cvReps = 200L,
                                   nBootstraps = 200L,
                                   nBootstrapsCor = 50L)
cbcl_singleton_t0_t1_lr$R2
#R2           R2SE 
#0.01190569   0.04173455  
model <- cbcl_singleton_t0_t1_lr 
cbcl_singleton_t0_t1_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) #iterate through 0.95 until one end is 0, and then 1-the other end confidence is p-value
cbcl_singleton_t0_t1_ci
#2.5%           97.5% 
#-0.06989253    0.09370392
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 #only want to test if the R value is significantly positive, not negative (one-tailed)
pvalue
#0.3877164
rm(singleton_t0_t1_cbcl_train, cbcl_singleton_t0_t1_lr, cbcl_singleton_t0_t1_ci)

#ksads
singleton_t0_t1_ksads_train <- read.csv('/user_path/singleton_t0_t1_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(singleton_t0_t1_ksads_train[, c('ksads_odd_present_t1', 'ksads_cd_present_rc_t1', 'ksads_adhd_present_rc_t1')])
y <- singleton_t0_t1_ksads_train$tasks_t0_km_2_resid

ksads_singleton_t0_t1_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                                    methodMSE = c("bootstrap"), 
                                    methodCor = c("jackknife"),
                                    printTimeEstimate = TRUE,
                                    cvReps = 200L,
                                    nBootstraps = 200L,
                                    nBootstrapsCor = 50L)
ksads_singleton_t0_t1_lr$R2
#R2           R2SE 
#0.005825467  0.041333778  
model <- ksads_singleton_t0_t1_lr 
ksads_singleton_t0_t1_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_singleton_t0_t1_ci
#2.5%           97.5% 
#-0.07518725    0.08683818 
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
#0.4439589
rm(singleton_t0_t1_ksads_train, ksads_singleton_t0_t1_lr, ksads_singleton_t0_t1_ci)

##Singleton: Baseline to 2-year Follow-up
#cbcl
singleton_t0_t2_cbcl_train <- read.csv('/user_path/singleton_t0_t2_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(singleton_t0_t2_cbcl_train[, c("cbcl_aggressive_t_t2", "cbcl_attention_t_t2", "cbcl_rulebreak_t_t2")])
y <- singleton_t0_t2_cbcl_train$tasks_t0_km_2_resid

cbcl_singleton_t0_t2_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                                   methodMSE = c("bootstrap"), 
                                   methodCor = c("jackknife"),
                                   printTimeEstimate = TRUE,
                                   cvReps = 200L,
                                   nBootstraps = 200L,
                                   nBootstrapsCor = 50L)
cbcl_singleton_t0_t2_lr$R2
#R2           R2SE 
#0.01055174   0.04170485 
model <- cbcl_singleton_t0_t2_lr 
cbcl_singleton_t0_t2_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) #iterate through 0.95 until one end is 0, and then 1-the other end confidence is p-value
cbcl_singleton_t0_t2_ci
#2.5%           97.5% 
#-0.07118827    0.09229175 
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 #only want to test if the R value is significantly positive, not negative (one-tailed)
pvalue
#0.400121
rm(singleton_t0_t2_cbcl_train, cbcl_singleton_t0_t2_lr, cbcl_singleton_t0_t2_ci)

#ksads
singleton_t0_t2_ksads_train <- read.csv('/user_path/singleton_t0_t2_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(singleton_t0_t2_ksads_train[, c('ksads_odd_present_t2', 'ksads_cd_present_rc_t2', 'ksads_adhd_present_rc_t2')])
y <- singleton_t0_t2_ksads_train$tasks_t0_km_2_resid

ksads_singleton_t0_t2_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                                    methodMSE = c("bootstrap"), 
                                    methodCor = c("jackknife"),
                                    printTimeEstimate = TRUE,
                                    cvReps = 200L,
                                    nBootstraps = 200L,
                                    nBootstrapsCor = 50L)
ksads_singleton_t0_t2_lr$R2
#R2           R2SE 
#0.004303626  0.041930626 
model <- ksads_singleton_t0_t2_lr 
ksads_singleton_t0_t2_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_singleton_t0_t2_ci
#2.5%           97.5% 
#-0.07787889    0.08648614 
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
#0.4591244
rm(singleton_t0_t2_ksads_train, ksads_singleton_t0_t2_lr, ksads_singleton_t0_t2_ci)

##Singleton: Baseline to 3-year Follow-up
#cbcl
singleton_t0_t3_cbcl_train <- read.csv('/user_path/singleton_t0_t3_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(singleton_t0_t3_cbcl_train[, c("cbcl_aggressive_t_t3", "cbcl_attention_t_t3", "cbcl_rulebreak_t_t3")])
y <- singleton_t0_t3_cbcl_train$tasks_t0_km_2_resid

cbcl_singleton_t0_t3_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                                   methodMSE = c("bootstrap"), 
                                   methodCor = c("jackknife"),
                                   printTimeEstimate = TRUE,
                                   cvReps = 200L,
                                   nBootstraps = 200L,
                                   nBootstrapsCor = 50L)
cbcl_singleton_t0_t3_lr$R2
#R2           R2SE 
#0.006675062  0.043213851 
model <- cbcl_singleton_t0_t3_lr 
cbcl_singleton_t0_t3_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) #iterate through 0.95 until one end is 0, and then 1-the other end confidence is p-value
cbcl_singleton_t0_t3_ci
#2.5%           97.5% 
#-0.07802253    0.09137265
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 #only want to test if the R value is significantly positive, not negative (one-tailed)
pvalue
#0.4386205
rm(singleton_t0_t3_cbcl_train, cbcl_singleton_t0_t3_lr, cbcl_singleton_t0_t3_ci)

#ksads
singleton_t0_t3_ksads_train <- read.csv('/user_path/singleton_t0_t3_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(singleton_t0_t3_ksads_train[, c('ksads_cd_present_rc_t3', 'ksads_adhd_present_rc_t3')])
y <- singleton_t0_t3_ksads_train$tasks_t0_km_2_resid

ksads_singleton_t0_t3_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                                    methodMSE = c("bootstrap"), 
                                    methodCor = c("jackknife"),
                                    printTimeEstimate = TRUE,
                                    cvReps = 200L,
                                    nBootstraps = 200L,
                                    nBootstrapsCor = 50L)
ksads_singleton_t0_t3_lr$R2
#R2           R2SE 
#0.001898816  0.041299077 
model <- ksads_singleton_t0_t3_lr 
ksads_singleton_t0_t3_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_singleton_t0_t3_ci
#2.5%           97.5% 
#-0.07904589    0.08284352
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
#0.4816596
rm(singleton_t0_t3_ksads_train, ksads_singleton_t0_t3_lr, ksads_singleton_t0_t3_ci)

##Singleton: 2-year Follow-up to Baseline
#cbcl
singleton_t2_t0_cbcl_train <- read.csv('/user_path/singleton_t2_t0_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(singleton_t2_t0_cbcl_train[, c("cbcl_aggressive_t_t0", "cbcl_attention_t_t0", "cbcl_rulebreak_t_t0")])
y <- singleton_t2_t0_cbcl_train$tasks_t2_km_2_resid

cbcl_singleton_t2_t0_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                                   methodMSE = c("bootstrap"), 
                                   methodCor = c("jackknife"),
                                   printTimeEstimate = TRUE,
                                   cvReps = 200L,
                                   nBootstraps = 200L,
                                   nBootstrapsCor = 50L)
cbcl_singleton_t2_t0_lr$R2
#R2           R2SE 
#0.009442232  0.041338222 
model <- cbcl_singleton_t2_t0_lr 
cbcl_singleton_t2_t0_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) #iterate through 0.95 until one end is 0, and then 1-the other end confidence is p-value
cbcl_singleton_t2_t0_ci
#2.5%           97.5% 
#-0.07157919    0.09046366 
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 #only want to test if the R value is significantly positive, not negative (one-tailed)
pvalue
#0.4096666
rm(singleton_t2_t0_cbcl_train, cbcl_singleton_t2_t0_lr, cbcl_singleton_t2_t0_ci)

#ksads
singleton_t2_t0_ksads_train <- read.csv('/user_path/singleton_t2_t0_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(singleton_t2_t0_ksads_train[, c('ksads_odd_present_t0', 'ksads_cd_present_rc_t0', 'ksads_adhd_present_rc_t0')])
y <- singleton_t2_t0_ksads_train$tasks_t2_km_2_resid

ksads_singleton_t2_t0_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                                    methodMSE = c("bootstrap"), 
                                    methodCor = c("jackknife"),
                                    printTimeEstimate = TRUE,
                                    cvReps = 200L,
                                    nBootstraps = 200L,
                                    nBootstrapsCor = 50L)
ksads_singleton_t2_t0_lr$R2
#R2           R2SE 
#0.002818307  0.041003481
model <- ksads_singleton_t2_t0_lr 
ksads_singleton_t2_t0_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_singleton_t2_t0_ci
#2.5%           97.5% 
#-0.07754704    0.08318365 
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
#0.4725975
rm(singleton_t2_t0_ksads_train, ksads_singleton_t2_t0_lr, ksads_singleton_t2_t0_ci)

##Singleton: 2-year Follow-up to 1-year Follow-up
#cbcl
singleton_t2_t1_cbcl_train <- read.csv('/user_path/singleton_t2_t1_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(singleton_t2_t1_cbcl_train[, c("cbcl_aggressive_t_t1", "cbcl_attention_t_t1", "cbcl_rulebreak_t_t1")])
y <- singleton_t2_t1_cbcl_train$tasks_t2_km_2_resid

cbcl_singleton_t2_t1_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                                   methodMSE = c("bootstrap"), 
                                   methodCor = c("jackknife"),
                                   printTimeEstimate = TRUE,
                                   cvReps = 200L,
                                   nBootstraps = 200L,
                                   nBootstrapsCor = 50L)
cbcl_singleton_t2_t1_lr$R2
#R2           R2SE 
#0.008971963  0.041397320  
model <- cbcl_singleton_t2_t1_lr 
cbcl_singleton_t2_t1_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
cbcl_singleton_t2_t1_ci
#2.5%           97.5% 
#-0.07216529    0.09010922
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
#0.4142108
rm(singleton_t2_t1_cbcl_train, cbcl_singleton_t2_t1_lr, cbcl_singleton_t2_t1_ci)

#ksads
singleton_t2_t1_ksads_train <- read.csv('/user_path/singleton_t2_t1_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(singleton_t2_t1_ksads_train[, c('ksads_odd_present_t1', 'ksads_cd_present_rc_t1', 'ksads_adhd_present_rc_t1')])
y <- singleton_t2_t1_ksads_train$tasks_t2_km_2_resid

ksads_singleton_t2_t1_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                                    methodMSE = c("bootstrap"), 
                                    methodCor = c("jackknife"),
                                    printTimeEstimate = TRUE,
                                    cvReps = 200L,
                                    nBootstraps = 200L,
                                    nBootstrapsCor = 50L)
ksads_singleton_t2_t1_lr$R2
#R2           R2SE 
#0.009456812  0.040417659 
model <- ksads_singleton_t2_t1_lr 
ksads_singleton_t2_t1_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_singleton_t2_t1_ci
#2.5%           97.5% 
#-0.06976034    0.08867397 
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
#0.4075074
rm(singleton_t2_t1_ksads_train, ksads_singleton_t2_t1_lr, ksads_singleton_t2_t1_ci)

##Singleton: 2-year Follow-up to 2-year Follow-up
#cbcl
singleton_t2_t2_cbcl_train <- read.csv('/user_path/singleton_t2_t2_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(singleton_t2_t2_cbcl_train[, c("cbcl_aggressive_t_t2", "cbcl_attention_t_t2", "cbcl_rulebreak_t_t2")])
y <- singleton_t2_t2_cbcl_train$tasks_t2_km_2_resid

cbcl_singleton_t2_t2_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                                   methodMSE = c("bootstrap"), 
                                   methodCor = c("jackknife"),
                                   printTimeEstimate = TRUE,
                                   cvReps = 200L,
                                   nBootstraps = 200L,
                                   nBootstrapsCor = 50L)
cbcl_singleton_t2_t2_lr$R2
#R2           R2SE 
#0.01226562   0.04131537 
model <- cbcl_singleton_t2_t2_lr 
cbcl_singleton_t2_t2_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
cbcl_singleton_t2_t2_ci
#2.5%           97.5% 
#-0.06871102    0.09324226  
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
#0.3832724
rm(singleton_t2_t2_cbcl_train, cbcl_singleton_t2_t2_lr, cbcl_singleton_t2_t2_ci)

#ksads
singleton_t2_t2_ksads_train <- read.csv('/user_path/singleton_t2_t2_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(singleton_t2_t2_ksads_train[, c('ksads_odd_present_t2', 'ksads_cd_present_rc_t2', 'ksads_adhd_present_rc_t2')])
y <- singleton_t2_t2_ksads_train$tasks_t2_km_2_resid

ksads_singleton_t2_t2_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                                    methodMSE = c("bootstrap"), 
                                    methodCor = c("jackknife"),
                                    printTimeEstimate = TRUE,
                                    cvReps = 200L,
                                    nBootstraps = 200L,
                                    nBootstrapsCor = 50L)
ksads_singleton_t2_t2_lr$R2
#R2           R2SE 
#0.002063332  0.041986432 
model <- ksads_singleton_t2_t2_lr 
ksads_singleton_t2_t2_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_singleton_t2_t2_ci
#2.5%           97.5% 
#-0.08022856    0.08435523 
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
#0.4803983
rm(singleton_t2_t2_ksads_train, ksads_singleton_t2_t2_lr, ksads_singleton_t2_t2_ci)

##Singleton: 2-year Follow-up to 3-year Follow-up
#cbcl
singleton_t2_t3_cbcl_train <- read.csv('/user_path/singleton_t2_t3_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(singleton_t2_t3_cbcl_train[, c("cbcl_aggressive_t_t3", "cbcl_attention_t_t3", "cbcl_rulebreak_t_t3")])
y <- singleton_t2_t3_cbcl_train$tasks_t2_km_2_resid

cbcl_singleton_t2_t3_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                                   methodMSE = c("bootstrap"), 
                                   methodCor = c("jackknife"),
                                   printTimeEstimate = TRUE,
                                   cvReps = 200L,
                                   nBootstraps = 200L,
                                   nBootstrapsCor = 50L)
cbcl_singleton_t2_t3_lr$R2
#R2           R2SE 
#0.007065896  0.043335956
model <- cbcl_singleton_t2_t3_lr 
cbcl_singleton_t2_t3_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) #iterate through 0.95 until one end is 0, and then 1-the other end confidence is p-value
cbcl_singleton_t2_t3_ci
#2.5%           97.5% 
#-0.07787102    0.09200281 
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 #only want to test if the R value is significantly positive, not negative (one-tailed)
pvalue
#0.4352392
rm(singleton_t2_t3_cbcl_train, cbcl_singleton_t2_t3_lr, cbcl_singleton_t2_t3_ci)

#ksads
singleton_t2_t3_ksads_train <- read.csv('/user_path/singleton_t2_t3_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(singleton_t2_t3_ksads_train[, c('ksads_cd_present_rc_t3', 'ksads_adhd_present_rc_t3')])
y <- singleton_t2_t3_ksads_train$tasks_t2_km_2_resid

ksads_singleton_t2_t3_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                                    methodMSE = c("bootstrap"), 
                                    methodCor = c("jackknife"),
                                    printTimeEstimate = TRUE,
                                    cvReps = 200L,
                                    nBootstraps = 200L,
                                    nBootstrapsCor = 50L)
ksads_singleton_t2_t3_lr$R2
#R2           R2SE 
#0.002694941  0.040970226 
model <- ksads_singleton_t2_t3_lr 
ksads_singleton_t2_t3_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_singleton_t2_t3_ci
#2.5%           97.5% 
#-0.07760523    0.08299511
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
#0.4737736
rm(singleton_t2_t3_ksads_train, ksads_singleton_t2_t3_lr, ksads_singleton_t2_t3_ci)

##Only: Baseline to Baseline
#cbcl
only_t0_t0_cbcl_train <- read.csv('/user_path/only_t0_t0_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(only_t0_t0_cbcl_train[, c("cbcl_aggressive_t_t0", "cbcl_attention_t_t0", "cbcl_rulebreak_t_t0")])
y <- only_t0_t0_cbcl_train$tasks_t0_km_2_resid

cbcl_only_t0_t0_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                              methodMSE = c("bootstrap"), 
                              methodCor = c("jackknife"),
                              printTimeEstimate = TRUE,
                              cvReps = 200L,
                              nBootstraps = 200L,
                              nBootstrapsCor = 50L)
cbcl_only_t0_t0_lr$R2
model <- cbcl_only_t0_t0_lr 
cbcl_only_t0_t0_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
cbcl_only_t0_t0_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(only_t0_t0_cbcl_train, cbcl_only_t0_t0_lr, cbcl_only_t0_t0_ci)

#ksads
only_t0_t0_ksads_train <- read.csv('/user_path/only_t0_t0_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(only_t0_t0_ksads_train[, c('ksads_odd_present_t0', 'ksads_cd_present_rc_t0', 'ksads_adhd_present_rc_t0')])
y <- only_t0_t0_ksads_train$tasks_t0_km_2_resid

ksads_only_t0_t0_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                               methodMSE = c("bootstrap"), 
                               methodCor = c("jackknife"),
                               printTimeEstimate = TRUE,
                               cvReps = 200L,
                               nBootstraps = 200L,
                               nBootstrapsCor = 50L)
ksads_only_t0_t0_lr$R2
model <- ksads_only_t0_t0_lr 
ksads_only_t0_t0_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_only_t0_t0_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(only_t0_t0_ksads_train, ksads_only_t0_t0_lr, ksads_only_t0_t0_ci)

##Only: Baseline to 1-year Follow-up
#cbcl
only_t0_t1_cbcl_train <- read.csv('/user_path/only_t0_t1_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(only_t0_t1_cbcl_train[, c("cbcl_aggressive_t_t1", "cbcl_attention_t_t1", "cbcl_rulebreak_t_t1")])
y <- only_t0_t1_cbcl_train$tasks_t0_km_2_resid

cbcl_only_t0_t1_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                              methodMSE = c("bootstrap"), 
                              methodCor = c("jackknife"),
                              printTimeEstimate = TRUE,
                              cvReps = 200L,
                              nBootstraps = 200L,
                              nBootstrapsCor = 50L)
cbcl_only_t0_t1_lr$R2
model <- cbcl_only_t0_t1_lr 
cbcl_only_t0_t1_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
cbcl_only_t0_t1_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(only_t0_t1_cbcl_train, cbcl_only_t0_t1_lr, cbcl_only_t0_t1_ci)

#ksads
only_t0_t1_ksads_train <- read.csv('/user_path/only_t0_t1_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(only_t0_t1_ksads_train[, c('ksads_odd_present_t1', 'ksads_cd_present_rc_t1', 'ksads_adhd_present_rc_t1')])
y <- only_t0_t1_ksads_train$tasks_t0_km_2_resid

ksads_only_t0_t1_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                               methodMSE = c("bootstrap"), 
                               methodCor = c("jackknife"),
                               printTimeEstimate = TRUE,
                               cvReps = 200L,
                               nBootstraps = 200L,
                               nBootstrapsCor = 50L)
ksads_only_t0_t1_lr$R2
model <- ksads_only_t0_t1_lr 
ksads_only_t0_t1_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_only_t0_t1_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(only_t0_t1_ksads_train, ksads_only_t0_t1_lr, ksads_only_t0_t1_ci)

##Only: Baseline to 2-year Follow-up
#cbcl
only_t0_t2_cbcl_train <- read.csv('/user_path/only_t0_t2_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(only_t0_t2_cbcl_train[, c("cbcl_aggressive_t_t2", "cbcl_attention_t_t2", "cbcl_rulebreak_t_t2")])
y <- only_t0_t2_cbcl_train$tasks_t0_km_2_resid

cbcl_only_t0_t2_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                              methodMSE = c("bootstrap"), 
                              methodCor = c("jackknife"),
                              printTimeEstimate = TRUE,
                              cvReps = 200L,
                              nBootstraps = 200L,
                              nBootstrapsCor = 50L)
cbcl_only_t0_t2_lr$R2
model <- cbcl_only_t0_t2_lr 
cbcl_only_t0_t2_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
cbcl_only_t0_t2_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(only_t0_t2_cbcl_train, cbcl_only_t0_t2_lr, cbcl_only_t0_t2_ci)

#ksads
only_t0_t2_ksads_train <- read.csv('/user_path/only_t0_t2_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(only_t0_t2_ksads_train[, c('ksads_odd_present_t2', 'ksads_cd_present_rc_t2', 'ksads_adhd_present_rc_t2')])
y <- only_t0_t2_ksads_train$tasks_t0_km_2_resid

ksads_only_t0_t2_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                               methodMSE = c("bootstrap"), 
                               methodCor = c("jackknife"),
                               printTimeEstimate = TRUE,
                               cvReps = 200L,
                               nBootstraps = 200L,
                               nBootstrapsCor = 50L)
ksads_only_t0_t2_lr$R2
model <- ksads_only_t0_t2_lr 
ksads_only_t0_t2_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_only_t0_t2_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(only_t0_t2_ksads_train, ksads_only_t0_t2_lr, ksads_only_t0_t2_ci)

##Only: Baseline to 3-year Follow-up
#cbcl
only_t0_t3_cbcl_train <- read.csv('/user_path/only_t0_t3_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(only_t0_t3_cbcl_train[, c("cbcl_aggressive_t_t3", "cbcl_attention_t_t3", "cbcl_rulebreak_t_t3")])
y <- only_t0_t3_cbcl_train$tasks_t0_km_2_resid

cbcl_only_t0_t3_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                              methodMSE = c("bootstrap"), 
                              methodCor = c("jackknife"),
                              printTimeEstimate = TRUE,
                              cvReps = 200L,
                              nBootstraps = 200L,
                              nBootstrapsCor = 50L)
cbcl_only_t0_t3_lr$R2
model <- cbcl_only_t0_t3_lr 
cbcl_only_t0_t3_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
cbcl_only_t0_t3_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(only_t0_t3_cbcl_train, cbcl_only_t0_t3_lr, cbcl_only_t0_t3_ci)

#ksads
only_t0_t3_ksads_train <- read.csv('/user_path/only_t0_t3_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(only_t0_t3_ksads_train[, c('ksads_cd_present_rc_t3', 'ksads_adhd_present_rc_t3')])
y <- only_t0_t3_ksads_train$tasks_t0_km_2_resid

ksads_only_t0_t3_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                               methodMSE = c("bootstrap"), 
                               methodCor = c("jackknife"),
                               printTimeEstimate = TRUE,
                               cvReps = 200L,
                               nBootstraps = 200L,
                               nBootstrapsCor = 50L)
ksads_only_t0_t3_lr$R2
model <- ksads_only_t0_t3_lr 
ksads_only_t0_t3_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_only_t0_t3_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(only_t0_t3_ksads_train, ksads_only_t0_t3_lr, ksads_only_t0_t3_ci)

##Only: 2-year Follow-up to Baseline
#cbcl
only_t2_t0_cbcl_train <- read.csv('/user_path/only_t2_t0_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(only_t2_t0_cbcl_train[, c("cbcl_aggressive_t_t0", "cbcl_attention_t_t0", "cbcl_rulebreak_t_t0")])
y <- only_t2_t0_cbcl_train$tasks_t2_km_2_resid

cbcl_only_t2_t0_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                              methodMSE = c("bootstrap"), 
                              methodCor = c("jackknife"),
                              printTimeEstimate = TRUE,
                              cvReps = 200L,
                              nBootstraps = 200L,
                              nBootstrapsCor = 50L)
cbcl_only_t2_t0_lr$R2
model <- cbcl_only_t2_t0_lr 
cbcl_only_t2_t0_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
cbcl_only_t2_t0_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(only_t2_t0_cbcl_train, cbcl_only_t2_t0_lr, cbcl_only_t2_t0_ci)

#ksads
only_t2_t0_ksads_train <- read.csv('/user_path/only_t2_t0_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(only_t2_t0_ksads_train[, c('ksads_odd_present_t0', 'ksads_cd_present_rc_t0', 'ksads_adhd_present_rc_t0')])
y <- only_t2_t0_ksads_train$tasks_t2_km_2_resid

ksads_only_t2_t0_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                               methodMSE = c("bootstrap"), 
                               methodCor = c("jackknife"),
                               printTimeEstimate = TRUE,
                               cvReps = 200L,
                               nBootstraps = 200L,
                               nBootstrapsCor = 50L)
ksads_only_t2_t0_lr$R2
model <- ksads_only_t2_t0_lr 
ksads_only_t2_t0_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_only_t2_t0_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(only_t2_t0_ksads_train, ksads_only_t2_t0_lr, ksads_only_t2_t0_ci)

##Only: 2-year Follow-up to 1-year Follow-up
#cbcl
only_t2_t1_cbcl_train <- read.csv('/user_path/only_t2_t1_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(only_t2_t1_cbcl_train[, c("cbcl_aggressive_t_t1", "cbcl_attention_t_t1", "cbcl_rulebreak_t_t1")])
y <- only_t2_t1_cbcl_train$tasks_t2_km_2_resid

cbcl_only_t2_t1_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                              methodMSE = c("bootstrap"), 
                              methodCor = c("jackknife"),
                              printTimeEstimate = TRUE,
                              cvReps = 200L,
                              nBootstraps = 200L,
                              nBootstrapsCor = 50L)
cbcl_only_t2_t1_lr$R2
model <- cbcl_only_t2_t1_lr 
cbcl_only_t2_t1_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
cbcl_only_t2_t1_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(only_t2_t1_cbcl_train, cbcl_only_t2_t1_lr, cbcl_only_t2_t1_ci)

#ksads
only_t2_t1_ksads_train <- read.csv('/user_path/only_t2_t1_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(only_t2_t1_ksads_train[, c('ksads_odd_present_t1', 'ksads_cd_present_rc_t1', 'ksads_adhd_present_rc_t1')])
y <- only_t2_t1_ksads_train$tasks_t2_km_2_resid

ksads_only_t2_t1_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                               methodMSE = c("bootstrap"), 
                               methodCor = c("jackknife"),
                               printTimeEstimate = TRUE,
                               cvReps = 200L,
                               nBootstraps = 200L,
                               nBootstrapsCor = 50L)
ksads_only_t2_t1_lr$R2
model <- ksads_only_t2_t1_lr 
ksads_only_t2_t1_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_only_t2_t1_ci 
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(only_t2_t1_ksads_train, ksads_only_t2_t1_lr, ksads_only_t2_t1_ci)

##Only: 2-year Follow-up to 2-year Follow-up
#cbcl
only_t2_t2_cbcl_train <- read.csv('/user_path/only_t2_t2_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(only_t2_t2_cbcl_train[, c("cbcl_aggressive_t_t2", "cbcl_attention_t_t2", "cbcl_rulebreak_t_t2")])
y <- only_t2_t2_cbcl_train$tasks_t2_km_2_resid

cbcl_only_t2_t2_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                              methodMSE = c("bootstrap"), 
                              methodCor = c("jackknife"),
                              printTimeEstimate = TRUE,
                              cvReps = 200L,
                              nBootstraps = 200L,
                              nBootstrapsCor = 50L)
cbcl_only_t2_t2_lr$R2
model <- cbcl_only_t2_t2_lr 
cbcl_only_t2_t2_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
cbcl_only_t2_t2_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(only_t2_t2_cbcl_train, cbcl_only_t2_t2_lr, cbcl_only_t2_t2_ci)

#ksads
only_t2_t2_ksads_train <- read.csv('/user_path/only_t2_t2_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(only_t2_t2_ksads_train[, c('ksads_odd_present_t2', 'ksads_cd_present_rc_t2', 'ksads_adhd_present_rc_t2')])
y <- only_t2_t2_ksads_train$tasks_t2_km_2_resid

ksads_only_t2_t2_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                               methodMSE = c("bootstrap"), 
                               methodCor = c("jackknife"),
                               printTimeEstimate = TRUE,
                               cvReps = 200L,
                               nBootstraps = 200L,
                               nBootstrapsCor = 50L)
ksads_only_t2_t2_lr$R2
model <- ksads_only_t2_t2_lr 
ksads_only_t2_t2_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_only_t2_t2_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(only_t2_t2_ksads_train, ksads_only_t2_t2_lr, ksads_only_t2_t2_ci)

##Only: 2-year Follow-up to 3-year Follow-up
#cbcl
only_t2_t3_cbcl_train <- read.csv('/user_path/only_t2_t3_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(only_t2_t3_cbcl_train[, c("cbcl_aggressive_t_t3", "cbcl_attention_t_t3", "cbcl_rulebreak_t_t3")])
y <- only_t2_t3_cbcl_train$tasks_t2_km_2_resid

cbcl_only_t2_t3_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                              methodMSE = c("bootstrap"), 
                              methodCor = c("jackknife"),
                              printTimeEstimate = TRUE,
                              cvReps = 200L,
                              nBootstraps = 200L,
                              nBootstrapsCor = 50L)
cbcl_only_t2_t3_lr$R2
model <- cbcl_only_t2_t3_lr 
cbcl_only_t2_t3_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
cbcl_only_t2_t3_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(only_t2_t3_cbcl_train, cbcl_only_t2_t3_lr, cbcl_only_t2_t3_ci)

#ksads
only_t2_t3_ksads_train <- read.csv('/user_path/only_t2_t3_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(only_t2_t3_ksads_train[, c('ksads_cd_present_rc_t3', 'ksads_adhd_present_rc_t3')])
y <- only_t2_t3_ksads_train$tasks_t2_km_2_resid

ksads_only_t2_t3_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                               methodMSE = c("bootstrap"), 
                               methodCor = c("jackknife"),
                               printTimeEstimate = TRUE,
                               cvReps = 200L,
                               nBootstraps = 200L,
                               nBootstrapsCor = 50L)
ksads_only_t2_t3_lr$R2
model <- ksads_only_t2_t3_lr 
ksads_only_t2_t3_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_only_t2_t3_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(only_t2_t3_ksads_train, ksads_only_t2_t3_lr, ksads_only_t2_t3_ci)

##RT: Baseline to Baseline
#cbcl
rt_t0_t0_cbcl_train <- read.csv('/user_path/rt_t0_t0_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(rt_t0_t0_cbcl_train[, c("cbcl_aggressive_t_t0", "cbcl_attention_t_t0", "cbcl_rulebreak_t_t0")])
y <- rt_t0_t0_cbcl_train$tasks_t0_km_2_resid

cbcl_rt_t0_t0_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                            methodMSE = c("bootstrap"), 
                            methodCor = c("jackknife"),
                            printTimeEstimate = TRUE,
                            cvReps = 200L,
                            nBootstraps = 200L,
                            nBootstrapsCor = 50L)
cbcl_rt_t0_t0_lr$R2
model <- cbcl_rt_t0_t0_lr 
cbcl_rt_t0_t0_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
cbcl_rt_t0_t0_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(rt_t0_t0_cbcl_train, cbcl_rt_t0_t0_lr, cbcl_rt_t0_t0_ci)

#ksads
rt_t0_t0_ksads_train <- read.csv('/user_path/rt_t0_t0_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(rt_t0_t0_ksads_train[, c('ksads_odd_present_t0', 'ksads_cd_present_rc_t0', 'ksads_adhd_present_rc_t0')])
y <- rt_t0_t0_ksads_train$tasks_t0_km_2_resid

ksads_rt_t0_t0_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                             methodMSE = c("bootstrap"), 
                             methodCor = c("jackknife"),
                             printTimeEstimate = TRUE,
                             cvReps = 200L,
                             nBootstraps = 200L,
                             nBootstrapsCor = 50L)
ksads_rt_t0_t0_lr$R2
model <- ksads_rt_t0_t0_lr 
ksads_rt_t0_t0_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_rt_t0_t0_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(rt_t0_t0_ksads_train, ksads_rt_t0_t0_lr, ksads_rt_t0_t0_ci)

##RT: Baseline to 1-year Follow-up
#cbcl
rt_t0_t1_cbcl_train <- read.csv('/user_path/rt_t0_t1_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(rt_t0_t1_cbcl_train[, c("cbcl_aggressive_t_t1", "cbcl_attention_t_t1", "cbcl_rulebreak_t_t1")])
y <- rt_t0_t1_cbcl_train$tasks_t0_km_2_resid

cbcl_rt_t0_t1_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                            methodMSE = c("bootstrap"), 
                            methodCor = c("jackknife"),
                            printTimeEstimate = TRUE,
                            cvReps = 200L,
                            nBootstraps = 200L,
                            nBootstrapsCor = 50L)
cbcl_rt_t0_t1_lr$R2
model <- cbcl_rt_t0_t1_lr 
cbcl_rt_t0_t1_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
cbcl_rt_t0_t1_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(rt_t0_t1_cbcl_train, cbcl_rt_t0_t1_lr, cbcl_rt_t0_t1_ci)

#ksads
rt_t0_t1_ksads_train <- read.csv('/user_path/rt_t0_t1_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(rt_t0_t1_ksads_train[, c('ksads_odd_present_t1', 'ksads_cd_present_rc_t1', 'ksads_adhd_present_rc_t1')])
y <- rt_t0_t1_ksads_train$tasks_t0_km_2_resid

ksads_rt_t0_t1_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                             methodMSE = c("bootstrap"), 
                             methodCor = c("jackknife"),
                             printTimeEstimate = TRUE,
                             cvReps = 200L,
                             nBootstraps = 200L,
                             nBootstrapsCor = 50L)
ksads_rt_t0_t1_lr$R2
model <- ksads_rt_t0_t1_lr 
ksads_rt_t0_t1_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_rt_t0_t1_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(rt_t0_t1_ksads_train, ksads_rt_t0_t1_lr, ksads_rt_t0_t1_ci)

##RT: Baseline to 2-year Follow-up
#cbcl
rt_t0_t2_cbcl_train <- read.csv('/user_path/rt_t0_t2_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(rt_t0_t2_cbcl_train[, c("cbcl_aggressive_t_t2", "cbcl_attention_t_t2", "cbcl_rulebreak_t_t2")])
y <- rt_t0_t2_cbcl_train$tasks_t0_km_2_resid

cbcl_rt_t0_t2_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                            methodMSE = c("bootstrap"), 
                            methodCor = c("jackknife"),
                            printTimeEstimate = TRUE,
                            cvReps = 200L,
                            nBootstraps = 200L,
                            nBootstrapsCor = 50L)
cbcl_rt_t0_t2_lr$R2
model <- cbcl_rt_t0_t2_lr 
cbcl_rt_t0_t2_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
cbcl_rt_t0_t2_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(rt_t0_t2_cbcl_train, cbcl_rt_t0_t2_lr, cbcl_rt_t0_t2_ci)

#ksads
rt_t0_t2_ksads_train <- read.csv('/user_path/rt_t0_t2_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(rt_t0_t2_ksads_train[, c('ksads_odd_present_t2', 'ksads_cd_present_rc_t2', 'ksads_adhd_present_rc_t2')])
y <- rt_t0_t2_ksads_train$tasks_t0_km_2_resid

ksads_rt_t0_t2_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                             methodMSE = c("bootstrap"), 
                             methodCor = c("jackknife"),
                             printTimeEstimate = TRUE,
                             cvReps = 200L,
                             nBootstraps = 200L,
                             nBootstrapsCor = 50L)
ksads_rt_t0_t2_lr$R2
model <- ksads_rt_t0_t2_lr 
ksads_rt_t0_t2_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_rt_t0_t2_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(rt_t0_t2_ksads_train, ksads_rt_t0_t2_lr, ksads_rt_t0_t2_ci)

##RT: Baseline to 3-year Follow-up
#cbcl
rt_t0_t3_cbcl_train <- read.csv('/user_path/rt_t0_t3_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(rt_t0_t3_cbcl_train[, c("cbcl_aggressive_t_t3", "cbcl_attention_t_t3", "cbcl_rulebreak_t_t3")])
y <- rt_t0_t3_cbcl_train$tasks_t0_km_2_resid

cbcl_rt_t0_t3_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                            methodMSE = c("bootstrap"), 
                            methodCor = c("jackknife"),
                            printTimeEstimate = TRUE,
                            cvReps = 200L,
                            nBootstraps = 200L,
                            nBootstrapsCor = 50L)
cbcl_rt_t0_t3_lr$R2
model <- cbcl_rt_t0_t3_lr 
cbcl_rt_t0_t3_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
cbcl_rt_t0_t3_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(rt_t0_t3_cbcl_train, cbcl_rt_t0_t3_lr, cbcl_rt_t0_t3_ci)

#ksads
rt_t0_t3_ksads_train <- read.csv('/user_path/rt_t0_t3_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(rt_t0_t3_ksads_train[, c('ksads_cd_present_rc_t3', 'ksads_adhd_present_rc_t3')])
y <- rt_t0_t3_ksads_train$tasks_t0_km_2_resid

ksads_rt_t0_t3_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                             methodMSE = c("bootstrap"), 
                             methodCor = c("jackknife"),
                             printTimeEstimate = TRUE,
                             cvReps = 200L,
                             nBootstraps = 200L,
                             nBootstrapsCor = 50L)
ksads_rt_t0_t3_lr$R2
model <- ksads_rt_t0_t3_lr 
ksads_rt_t0_t3_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_rt_t0_t3_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(rt_t0_t3_ksads_train, ksads_rt_t0_t3_lr, ksads_rt_t0_t3_ci)

##RT: 2-year Follow-up to Baseline
#cbcl
rt_t2_t0_cbcl_train <- read.csv('/user_path/rt_t2_t0_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(rt_t2_t0_cbcl_train[, c("cbcl_aggressive_t_t0", "cbcl_attention_t_t0", "cbcl_rulebreak_t_t0")])
y <- rt_t2_t0_cbcl_train$tasks_t2_km_2_resid

cbcl_rt_t2_t0_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                            methodMSE = c("bootstrap"), 
                            methodCor = c("jackknife"),
                            printTimeEstimate = TRUE,
                            cvReps = 200L,
                            nBootstraps = 200L,
                            nBootstrapsCor = 50L)
cbcl_rt_t2_t0_lr$R2
model <- cbcl_rt_t2_t0_lr 
cbcl_rt_t2_t0_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
cbcl_rt_t2_t0_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(rt_t2_t0_cbcl_train, cbcl_rt_t2_t0_lr, cbcl_rt_t2_t0_ci)

#ksads
rt_t2_t0_ksads_train <- read.csv('/user_path/rt_t2_t0_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(rt_t2_t0_ksads_train[, c('ksads_odd_present_t0', 'ksads_cd_present_rc_t0', 'ksads_adhd_present_rc_t0')])
y <- rt_t2_t0_ksads_train$tasks_t2_km_2_resid

ksads_rt_t2_t0_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                             methodMSE = c("bootstrap"), 
                             methodCor = c("jackknife"),
                             printTimeEstimate = TRUE,
                             cvReps = 200L,
                             nBootstraps = 200L,
                             nBootstrapsCor = 50L)
ksads_rt_t2_t0_lr$R2
model <- ksads_rt_t2_t0_lr 
ksads_rt_t2_t0_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_rt_t2_t0_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(rt_t2_t0_ksads_train, ksads_rt_t2_t0_lr, ksads_rt_t2_t0_ci)

##RT: 2-year Follow-up to 1-year Follow-up
#cbcl
rt_t2_t1_cbcl_train <- read.csv('/user_path/rt_t2_t1_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(rt_t2_t1_cbcl_train[, c("cbcl_aggressive_t_t1", "cbcl_attention_t_t1", "cbcl_rulebreak_t_t1")])
y <- rt_t2_t1_cbcl_train$tasks_t2_km_2_resid

cbcl_rt_t2_t1_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                            methodMSE = c("bootstrap"), 
                            methodCor = c("jackknife"),
                            printTimeEstimate = TRUE,
                            cvReps = 200L,
                            nBootstraps = 200L,
                            nBootstrapsCor = 50L)
cbcl_rt_t2_t1_lr$R2
model <- cbcl_rt_t2_t1_lr 
cbcl_rt_t2_t1_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
cbcl_rt_t2_t1_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(rt_t2_t1_cbcl_train, cbcl_rt_t2_t1_lr, cbcl_rt_t2_t1_ci)

#ksads
rt_t2_t1_ksads_train <- read.csv('/user_path/rt_t2_t1_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(rt_t2_t1_ksads_train[, c('ksads_odd_present_t1', 'ksads_cd_present_rc_t1', 'ksads_adhd_present_rc_t1')])
y <- rt_t2_t1_ksads_train$tasks_t2_km_2_resid

ksads_rt_t2_t1_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                             methodMSE = c("bootstrap"), 
                             methodCor = c("jackknife"),
                             printTimeEstimate = TRUE,
                             cvReps = 200L,
                             nBootstraps = 200L,
                             nBootstrapsCor = 50L)
ksads_rt_t2_t1_lr$R2
model <- ksads_rt_t2_t1_lr 
ksads_rt_t2_t1_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_rt_t2_t1_ci 
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(rt_t2_t1_ksads_train, ksads_rt_t2_t1_lr, ksads_rt_t2_t1_ci)

##RT: 2-year Follow-up to 2-year Follow-up
#cbcl
rt_t2_t2_cbcl_train <- read.csv('/user_path/rt_t2_t2_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(rt_t2_t2_cbcl_train[, c("cbcl_aggressive_t_t2", "cbcl_attention_t_t2", "cbcl_rulebreak_t_t2")])
y <- rt_t2_t2_cbcl_train$tasks_t2_km_2_resid

cbcl_rt_t2_t2_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                            methodMSE = c("bootstrap"), 
                            methodCor = c("jackknife"),
                            printTimeEstimate = TRUE,
                            cvReps = 200L,
                            nBootstraps = 200L,
                            nBootstrapsCor = 50L)
cbcl_rt_t2_t2_lr$R2
model <- cbcl_rt_t2_t2_lr 
cbcl_rt_t2_t2_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
cbcl_rt_t2_t2_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(rt_t2_t2_cbcl_train, cbcl_rt_t2_t2_lr, cbcl_rt_t2_t2_ci)

#ksads
rt_t2_t2_ksads_train <- read.csv('/user_path/rt_t2_t2_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(rt_t2_t2_ksads_train[, c('ksads_odd_present_t2', 'ksads_cd_present_rc_t2', 'ksads_adhd_present_rc_t2')])
y <- rt_t2_t2_ksads_train$tasks_t2_km_2_resid

ksads_rt_t2_t2_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                             methodMSE = c("bootstrap"), 
                             methodCor = c("jackknife"),
                             printTimeEstimate = TRUE,
                             cvReps = 200L,
                             nBootstraps = 200L,
                             nBootstrapsCor = 50L)
ksads_rt_t2_t2_lr$R2
model <- ksads_rt_t2_t2_lr 
ksads_rt_t2_t2_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_rt_t2_t2_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(rt_t2_t2_ksads_train, ksads_rt_t2_t2_lr, ksads_rt_t2_t2_ci)

##RT: 2-year Follow-up to 3-year Follow-up
#cbcl
rt_t2_t3_cbcl_train <- read.csv('/user_path/rt_t2_t3_cbcl_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(rt_t2_t3_cbcl_train[, c("cbcl_aggressive_t_t3", "cbcl_attention_t_t3", "cbcl_rulebreak_t_t3")])
y <- rt_t2_t3_cbcl_train$tasks_t2_km_2_resid

cbcl_rt_t2_t3_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                            methodMSE = c("bootstrap"), 
                            methodCor = c("jackknife"),
                            printTimeEstimate = TRUE,
                            cvReps = 200L,
                            nBootstraps = 200L,
                            nBootstrapsCor = 50L)
cbcl_rt_t2_t3_lr$R2
model <- cbcl_rt_t2_t3_lr 
cbcl_rt_t2_t3_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
cbcl_rt_t2_t3_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(rt_t2_t3_cbcl_train, cbcl_rt_t2_t3_lr, cbcl_rt_t2_t3_ci)

#ksads
rt_t2_t3_ksads_train <- read.csv('/user_path/rt_t2_t3_ksads_train.csv', header = TRUE, stringsAsFactors = FALSE)
X <- as.matrix(rt_t2_t3_ksads_train[, c('ksads_cd_present_rc_t3', 'ksads_adhd_present_rc_t3')])
y <- rt_t2_t3_ksads_train$tasks_t2_km_2_resid

ksads_rt_t2_t3_lr <- R2oosse(y, X, fitFun_lr, predFun_lr, 
                             methodMSE = c("bootstrap"), 
                             methodCor = c("jackknife"),
                             printTimeEstimate = TRUE,
                             cvReps = 200L,
                             nBootstraps = 200L,
                             nBootstrapsCor = 50L)
ksads_rt_t2_t3_lr$R2
model <- ksads_rt_t2_t3_lr 
ksads_rt_t2_t3_ci <- buildConfInt(model, what = c("R2"), conf = 0.95) 
ksads_rt_t2_t3_ci
optimize_pvalue <- optimize(pvalue_est, c(0.0001, 0.99999999))
pvalue <- (1 - optimize_pvalue$minimum)/2 
pvalue
rm(rt_t2_t3_ksads_train, ksads_rt_t2_t3_lr, ksads_rt_t2_t3_ci)