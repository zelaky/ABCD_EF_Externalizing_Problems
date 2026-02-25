#Title: Cluster Characterization with Best Subset Linear Regression  
#Author: Zoë E. Laky, M.A.
#Contact: zoe.laky@nih.gov

#install.packages("L0Learn") 
library(L0Learn)
library(dplyr)
set.seed(843)

citation()

abcd_all <- read.csv(all, header = TRUE, stringsAsFactors = FALSE)
abcd_singleton = read.csv(singleton, header = TRUE, stringsAsFactors = FALSE)
abcd_only = read.csv(only, header = TRUE, stringsAsFactors = FALSE)
abcd_rt = read.csv(rt, header = TRUE, stringsAsFactors = FALSE)
abcd_split = read.csv(split, header = TRUE, stringsAsFactors = FALSE)

# -----------------------------
# Functions
# -----------------------------
binarize_clusters_R <- function(df, col_name, prefix = NULL, max_levels = 15) {
  levels <- sort(unique(df[[col_name]]))
  n_levels <- length(levels)
  if (n_levels > max_levels) {
    stop(paste("Too many unique levels in", col_name, "(", n_levels, "). Limit is", max_levels))
  }
  if (is.null(prefix)) {
    prefix <- col_name
  }
  for (level in levels) {
    new_col <- paste0(prefix, "_", level, "_level")
    df[[new_col]] <- as.integer(df[[col_name]] == level)
  }
  return(df)
}

characterize_clusters_R <- function(X_train, y_train, feature_names = colnames(X_train), max_subset = 7) {
  fit <- L0Learn.fit(
    as.matrix(X_train),
    as.numeric(y_train),
    loss = "Logistic",
    nGamma = 20, 
    penalty = "L0",
    algorithm = "CDPSI",
    maxSuppSize = max_subset,
    maxIters = 1000,
    maxSwaps = 500,
    intercept = TRUE
  )
  n <- nrow(X_train)
  results <- list()
  for (i in seq_along(fit$beta)) {
    gam_fit <- fit$beta[[i]]
    intercepts <- fit$a0[[i]]  
    for (j in seq_len(ncol(gam_fit))) {
      beta <- gam_fit[, j]
      intercept <- intercepts[j]  
      # predicted probabilities
      prob_pred <- 1 / (1 + exp(-(as.matrix(X_train) %*% beta + intercept)))
      prob_pred <- pmin(pmax(prob_pred, 1e-15), 1 - 1e-15)
      # log-likelihood & BIC
      log_likelihood <- sum(y_train * log(prob_pred) + (1 - y_train) * log(1 - prob_pred))
      k <- sum(beta != 0) + 1
      bic <- k * log(n) - 2 * log_likelihood
      retained_features <- feature_names[which(beta != 0)]
      results[[length(results) + 1]] <- list(
        bic = bic,
        intercept = intercept,
        beta = beta,
        retained_features = retained_features,
        lambda = fit$lambda[[i]][j],
        gamma = fit$gamma[i]
      )
    }
  }
  results_df <- data.frame(
    bic = sapply(results, function(x) ifelse(is.null(x$bic), NA, x$bic)),
    lambda = sapply(results, function(x) ifelse(is.null(x$lambda), NA, x$lambda)),
    gamma = sapply(results, function(x) ifelse(is.null(x$gamma), NA, x$gamma)),
    intercept = sapply(results, function(x) ifelse(is.null(x$intercept), NA, x$intercept)),
    stringsAsFactors = FALSE
  )
  results_df$coef <- lapply(results, function(x) if (!is.null(x$beta)) x$beta else NA)
  results_df$retained_features <- lapply(results, function(x) {
    if (is.null(x$retained_features) || length(x$retained_features) == 0) {
      NA
    } else {
      x$retained_features
    }
  })
  return(results_df)
}

flatten_lists <- function(model_df) {
  df <- model_df
  if ("coef" %in% colnames(df)) {
    df$coef <- sapply(df$coef, function(x) {
      if (is.null(x) || all(is.na(x))) return(NA)
      paste(round(x, 4), collapse = ";")
    })
  }
  if ("retained_features" %in% colnames(df)) {
    df$retained_features <- sapply(df$retained_features, function(x) {
      if (is.null(x) || all(is.na(x))) return(NA)
      paste(x, collapse = ";")
    })
  }
  return(df)
}

# -----------------------------
# Prepare Dataframes
# -----------------------------
#ids
train_ids <- abcd_split %>% 
  filter(train_ids == 1) %>% 
  pull(src_subject_id)
test_ids <- abcd_split %>% 
  filter(test_ids == 1) %>% 
  pull(src_subject_id)
train_ids_df <- data.frame(src_subject_id = train_ids, stringsAsFactors = FALSE)
test_ids_df <- data.frame(src_subject_id = test_ids, stringsAsFactors = FALSE)

#variables
tasks_t0_cols <- colnames(abcd_all)[
  (startsWith(colnames(abcd_all), "nback_t0") | startsWith(colnames(abcd_all), "sst_t0")) &
    !grepl("pc", colnames(abcd_all))
]
tasks_t2_cols <- colnames(abcd_all)[
  (startsWith(colnames(abcd_all), "nback_t2") | startsWith(colnames(abcd_all), "sst_t2")) &
    !grepl("pc", colnames(abcd_all))
]
clusters_t0_col <- c("tasks_t0_km_2")
clusters_t2_col <- c("tasks_t2_km_2")

#all
all_train <- abcd_all %>% dplyr::filter(src_subject_id %in% train_ids_df$src_subject_id)
all_test  <- abcd_all %>% dplyr::filter(src_subject_id %in% test_ids_df$src_subject_id)

#singleton
singleton_train <- abcd_singleton %>% dplyr::filter(src_subject_id %in% train_ids_df$src_subject_id)
singleton_test  <- abcd_singleton %>% dplyr::filter(src_subject_id %in% test_ids_df$src_subject_id)

#only
only_train <- abcd_only %>% dplyr::filter(src_subject_id %in% train_ids_df$src_subject_id)
only_test  <- abcd_only %>% dplyr::filter(src_subject_id %in% test_ids_df$src_subject_id)

#rt
rt_train <- abcd_rt %>% dplyr::filter(src_subject_id %in% train_ids_df$src_subject_id)
rt_test  <- abcd_rt %>% dplyr::filter(src_subject_id %in% test_ids_df$src_subject_id)

#dataframes
all_tasks_t0_train <- all_train[, tasks_t0_cols]
all_tasks_t0_test  <- all_test[, tasks_t0_cols]
all_tasks_t2_train <- all_train[, tasks_t2_cols]
all_tasks_t2_test  <- all_test[, tasks_t2_cols]
all_clusters_t0_train <- all_train[, clusters_t0_col]
all_clusters_t0_test  <- all_test[, clusters_t0_col]
all_clusters_t2_train <- all_train[, clusters_t2_col]
all_clusters_t2_test  <- all_test[, clusters_t2_col]

singleton_tasks_t0_train <- singleton_train[, tasks_t0_cols]
singleton_tasks_t0_test  <- singleton_test[, tasks_t0_cols]
singleton_tasks_t2_train <- singleton_train[, tasks_t2_cols]
singleton_tasks_t2_test  <- singleton_test[, tasks_t2_cols]
singleton_clusters_t0_train <- singleton_train[, clusters_t0_col]
singleton_clusters_t0_test  <- singleton_test[, clusters_t0_col]
singleton_clusters_t2_train <- singleton_train[, clusters_t2_col]
singleton_clusters_t2_test  <- singleton_test[, clusters_t2_col]

only_tasks_t0_train <- only_train[, tasks_t0_cols]
only_tasks_t0_test  <- only_test[, tasks_t0_cols]
only_tasks_t2_train <- only_train[, tasks_t2_cols]
only_tasks_t2_test  <- only_test[, tasks_t2_cols]
only_clusters_t0_train <- only_train[, clusters_t0_col]
only_clusters_t0_test  <- only_test[, clusters_t0_col]
only_clusters_t2_train <- only_train[, clusters_t2_col]
only_clusters_t2_test  <- only_test[, clusters_t2_col]

rt_tasks_t0_train <- rt_train[, tasks_t0_cols]
rt_tasks_t0_test  <- rt_test[, tasks_t0_cols]
rt_tasks_t2_train <- rt_train[, tasks_t2_cols]
rt_tasks_t2_test  <- rt_test[, tasks_t2_cols]
rt_clusters_t0_train <- rt_train[, clusters_t0_col]
rt_clusters_t0_test  <- rt_test[, clusters_t0_col]
rt_clusters_t2_train <- rt_train[, clusters_t2_col]
rt_clusters_t2_test  <- rt_test[, clusters_t2_col]

rm(abcd_all, abcd_singleton, abcd_only, abcd_rt, abcd_split, all_train, all_test, singleton_train, singleton_test, only_train, only_test, rt_train, rt_test)

# -----------------------------
# All Baseline Clusters N=2
# -----------------------------
X_train <- all_tasks_t0_train
X_train_scaled <- scale(X_train)
y_train <- all_clusters_t0_train

X_test <- all_tasks_t0_test
X_test_scaled <- scale(X_test)
y_test <- all_clusters_t0_test

all_train_t0_model <- characterize_clusters_R(X_train_scaled, y_train)

# identify best training model 
best_row <- all_train_t0_model[which.min(all_train_t0_model$bic), ]

best_coef <- best_row$coef[[1]]                  
best_intercept <- best_row$intercept             
best_features <- best_row$retained_features[[1]] 
names(best_coef) <- colnames(X_train_scaled)
coef_selected <- best_coef[best_features]

# apply to training and testing data
X_train_mat <- as.matrix(X_train_scaled[, best_features, drop = FALSE])
X_test_mat  <- as.matrix(X_test_scaled[, best_features, drop = FALSE])

prob_train <- 1 / (1 + exp(-(X_train_mat %*% coef_selected + best_intercept))) # calculate predicted probabilities
prob_test  <- 1 / (1 + exp(-(X_test_mat %*% coef_selected + best_intercept)))

y_train_pred <- ifelse(prob_train > 0.5, 1, 0) # convert probabilities to class labels
y_test_pred  <- ifelse(prob_test > 0.5, 1, 0)

# calculate accuracy
train_accuracy <- mean(y_train_pred == y_train)
test_accuracy  <- mean(y_test_pred == y_test)

conf_mat_train <- table(Predicted = y_train_pred, Actual = y_train)
print(conf_mat_train)
#     0    1
#0 1416   87
#1   81 1174

conf_mat_test <- table(Predicted = y_test_pred, Actual = y_test)
print(conf_mat_test)
#     0    1
#0 1345  104
#1   85 1209

best_features
# "sst_t0_crgo_stdrt"  "sst_t0_crlg_rate"   "nback_t0_c0b_rate"  
# "nback_t0_c0b_mrt"   "nback_t0_c2b_rate"  "nback_t0_c2b_stdrt"
train_accuracy 
# 0.9390863
test_accuracy      
# 0.9310973

# -----------------------------
# All 2-year Follow-up Clusters N=2
# -----------------------------
X_train <- all_tasks_t2_train
X_train_scaled <- scale(X_train)
y_train <- all_clusters_t2_train

X_test <- all_tasks_t2_test
X_test_scaled <- scale(X_test)
y_test <- all_clusters_t2_test

all_train_t2_model <- characterize_clusters_R(X_train_scaled, y_train)

# identify best training model 
best_row <- all_train_t2_model[which.min(all_train_t2_model$bic), ]

best_coef <- best_row$coef[[1]]                  
best_intercept <- best_row$intercept             
best_features <- best_row$retained_features[[1]] 
names(best_coef) <- colnames(X_train_scaled)
coef_selected <- best_coef[best_features]

# apply to training and testing data
X_train_mat <- as.matrix(X_train_scaled[, best_features, drop = FALSE])
X_test_mat  <- as.matrix(X_test_scaled[, best_features, drop = FALSE])

prob_train <- 1 / (1 + exp(-(X_train_mat %*% coef_selected + best_intercept))) # calculate predicted probabilities
prob_test  <- 1 / (1 + exp(-(X_test_mat %*% coef_selected + best_intercept)))

y_train_pred <- ifelse(prob_train > 0.5, 1, 0) # convert probabilities to class labels
y_test_pred  <- ifelse(prob_test > 0.5, 1, 0)

# calculate accuracy
train_accuracy <- mean(y_train_pred == y_train)
test_accuracy  <- mean(y_test_pred == y_test)

conf_mat_train <- table(Predicted = y_train_pred, Actual = y_train)
print(conf_mat_train)
#     0    1
#0 1525   57
#1   52 1124
conf_mat_test <- table(Predicted = y_test_pred, Actual = y_test)
print(conf_mat_test)
#     0    1
#0 1476   98
#1   36 1133

best_features       
# "sst_t2_crgo_stdrt"  "nback_t2_c0b_rate"  "nback_t2_c0b_mrt"   
# "nback_t2_c0b_stdrt" "nback_t2_c2b_rate"  "nback_t2_c2b_mrt"   "nback_t2_c2b_stdrt"
train_accuracy      
# 0.9604786
test_accuracy       
# 0.9511484

# -----------------------------
# Singleton Baseline Clusters N=2
# -----------------------------
X_train <- singleton_tasks_t0_train
X_train_scaled <- scale(X_train)
y_train <- singleton_clusters_t0_train

X_test <- singleton_tasks_t0_test
X_test_scaled <- scale(X_test)
y_test <- singleton_clusters_t0_test

singleton_train_t0_model <- characterize_clusters_R(X_train_scaled, y_train)
best_row <- singleton_train_t0_model[which.min(singleton_train_t0_model$bic), ]
best_coef <- best_row$coef[[1]]                  
best_intercept <- best_row$intercept             
best_features <- best_row$retained_features[[1]] 
names(best_coef) <- colnames(X_train_scaled)
coef_selected <- best_coef[best_features]

X_train_mat <- as.matrix(X_train_scaled[, best_features, drop = FALSE])
X_test_mat  <- as.matrix(X_test_scaled[, best_features, drop = FALSE])
prob_train <- 1 / (1 + exp(-(X_train_mat %*% coef_selected + best_intercept))) 
prob_test  <- 1 / (1 + exp(-(X_test_mat %*% coef_selected + best_intercept)))

y_train_pred <- ifelse(prob_train > 0.5, 1, 0) 
y_test_pred  <- ifelse(prob_test > 0.5, 1, 0)
train_accuracy <- mean(y_train_pred == y_train)
test_accuracy  <- mean(y_test_pred == y_test)

conf_mat_train <- table(Predicted = y_train_pred, Actual = y_train)
print(conf_mat_train)
#     0    1
#0 1055   77
#1   76 1223
conf_mat_test <- table(Predicted = y_test_pred, Actual = y_test)
print(conf_mat_test)
#     0    1
#0 1106  71
#1   96 1158
best_features
# "sst_t0_crlg_rate"  "sst_t0_nrgo_rate"   "nback_t0_c0b_rate"  
# "nback_t0_c0b_mrt"   "nback_t0_c2b_rate"  "nback_t0_c2b_stdrt"
train_accuracy 
# 0.9370629
test_accuracy      
# 0.931304

# -----------------------------
# Singleton 2-year Follow-up Clusters N=2
# -----------------------------
X_train <- singleton_tasks_t2_train
X_train_scaled <- scale(X_train)
y_train <- singleton_clusters_t2_train

X_test <- singleton_tasks_t2_test
X_test_scaled <- scale(X_test)
y_test <- singleton_clusters_t2_test

singleton_train_t2_model <- characterize_clusters_R(X_train_scaled, y_train)
best_row <- singleton_train_t2_model[which.min(singleton_train_t2_model$bic), ]

best_coef <- best_row$coef[[1]]                  
best_intercept <- best_row$intercept             
best_features <- best_row$retained_features[[1]] 
names(best_coef) <- colnames(X_train_scaled)
coef_selected <- best_coef[best_features]

X_train_mat <- as.matrix(X_train_scaled[, best_features, drop = FALSE])
X_test_mat  <- as.matrix(X_test_scaled[, best_features, drop = FALSE])
prob_train <- 1 / (1 + exp(-(X_train_mat %*% coef_selected + best_intercept))) 
prob_test  <- 1 / (1 + exp(-(X_test_mat %*% coef_selected + best_intercept)))

y_train_pred <- ifelse(prob_train > 0.5, 1, 0) 
y_test_pred  <- ifelse(prob_test > 0.5, 1, 0)
train_accuracy <- mean(y_train_pred == y_train)
test_accuracy  <- mean(y_test_pred == y_test)

conf_mat_train <- table(Predicted = y_train_pred, Actual = y_train)
print(conf_mat_train)
#     0    1
#0  998   45
#1   47 1341
conf_mat_test <- table(Predicted = y_test_pred, Actual = y_test)
print(conf_mat_test)
#     0    1
#0 1010   31
#1   78 1312
best_features       
# "sst_t2_crgo_stdrt"  "nback_t2_c0b_rate"  "nback_t2_c0b_mrt"   
# "nback_t2_c0b_stdrt" "nback_t2_c2b_rate"  "nback_t2_c2b_mrt"   "nback_t2_c2b_stdrt"
train_accuracy      
# 0.9621555
test_accuracy       
# 0.9551625

# -----------------------------
# Only Baseline Clusters N=2
# -----------------------------
X_train <- only_tasks_t0_train
X_train_scaled <- scale(X_train)
y_train <- only_clusters_t0_train

X_test <- only_tasks_t0_test
X_test_scaled <- scale(X_test)
y_test <- only_clusters_t0_test

only_train_t0_model <- characterize_clusters_R(X_train_scaled, y_train)
best_row <- only_train_t0_model[which.min(only_train_t0_model$bic), ]
best_coef <- best_row$coef[[1]]                  
best_intercept <- best_row$intercept             
best_features <- best_row$retained_features[[1]] 
names(best_coef) <- colnames(X_train_scaled)
coef_selected <- best_coef[best_features]

X_train_mat <- as.matrix(X_train_scaled[, best_features, drop = FALSE])
X_test_mat  <- as.matrix(X_test_scaled[, best_features, drop = FALSE])
prob_train <- 1 / (1 + exp(-(X_train_mat %*% coef_selected + best_intercept))) 
prob_test  <- 1 / (1 + exp(-(X_test_mat %*% coef_selected + best_intercept)))

y_train_pred <- ifelse(prob_train > 0.5, 1, 0) 
y_test_pred  <- ifelse(prob_test > 0.5, 1, 0)
train_accuracy <- mean(y_train_pred == y_train)
test_accuracy  <- mean(y_test_pred == y_test)

conf_mat_train <- table(Predicted = y_train_pred, Actual = y_train)
print(conf_mat_train)
#     0    1
#0 1416   87
#1   81 1174
conf_mat_test <- table(Predicted = y_test_pred, Actual = y_test)
print(conf_mat_test)
#     0    1
#0 1345  104
#1   85 1209
best_features
# "sst_t0_crgo_stdrt"  "sst_t0_crlg_rate"   "nback_t0_c0b_rate"  
# "nback_t0_c0b_mrt"   "nback_t0_c2b_rate"  "nback_t0_c2b_stdrt"
train_accuracy 
# 0.9390863
test_accuracy      
# 0.9310973

# -----------------------------
# Only 2-year Follow-up Clusters N=2
# -----------------------------
X_train <- only_tasks_t2_train
X_train_scaled <- scale(X_train)
y_train <- only_clusters_t2_train

X_test <- only_tasks_t2_test
X_test_scaled <- scale(X_test)
y_test <- only_clusters_t2_test

only_train_t2_model <- characterize_clusters_R(X_train_scaled, y_train)
best_row <- only_train_t2_model[which.min(only_train_t2_model$bic), ]

best_coef <- best_row$coef[[1]]                  
best_intercept <- best_row$intercept             
best_features <- best_row$retained_features[[1]] 
names(best_coef) <- colnames(X_train_scaled)
coef_selected <- best_coef[best_features]

X_train_mat <- as.matrix(X_train_scaled[, best_features, drop = FALSE])
X_test_mat  <- as.matrix(X_test_scaled[, best_features, drop = FALSE])
prob_train <- 1 / (1 + exp(-(X_train_mat %*% coef_selected + best_intercept))) 
prob_test  <- 1 / (1 + exp(-(X_test_mat %*% coef_selected + best_intercept)))

y_train_pred <- ifelse(prob_train > 0.5, 1, 0) 
y_test_pred  <- ifelse(prob_test > 0.5, 1, 0)
train_accuracy <- mean(y_train_pred == y_train)
test_accuracy  <- mean(y_test_pred == y_test)

conf_mat_train <- table(Predicted = y_train_pred, Actual = y_train)
print(conf_mat_train)
#     0    1
#0  1525   57
#1   52   1124
conf_mat_test <- table(Predicted = y_test_pred, Actual = y_test)
print(conf_mat_test)
#     0    1
#0 1476   98
#1   36 1133
best_features       
# "sst_t2_crgo_stdrt"  "nback_t2_c0b_rate"  "nback_t2_c0b_mrt"   
# "nback_t2_c0b_stdrt" "nback_t2_c2b_rate"  "nback_t2_c2b_mrt"   "nback_t2_c2b_stdrt"
train_accuracy      
# 0.9604786
test_accuracy       
# 0.9511484

# -----------------------------
# RT Baseline Clusters N=2
# -----------------------------
X_train <- rt_tasks_t0_train
X_train_scaled <- scale(X_train)
y_train <- rt_clusters_t0_train

X_test <- rt_tasks_t0_test
X_test_scaled <- scale(X_test)
y_test <- rt_clusters_t0_test

rt_train_t0_model <- characterize_clusters_R(X_train_scaled, y_train)
best_row <- rt_train_t0_model[which.min(rt_train_t0_model$bic), ]
best_coef <- best_row$coef[[1]]                  
best_intercept <- best_row$intercept             
best_features <- best_row$retained_features[[1]] 
names(best_coef) <- colnames(X_train_scaled)
coef_selected <- best_coef[best_features]

X_train_mat <- as.matrix(X_train_scaled[, best_features, drop = FALSE])
X_test_mat  <- as.matrix(X_test_scaled[, best_features, drop = FALSE])
prob_train <- 1 / (1 + exp(-(X_train_mat %*% coef_selected + best_intercept))) 
prob_test  <- 1 / (1 + exp(-(X_test_mat %*% coef_selected + best_intercept)))

y_train_pred <- ifelse(prob_train > 0.5, 1, 0) 
y_test_pred  <- ifelse(prob_test > 0.5, 1, 0)
train_accuracy <- mean(y_train_pred == y_train)
test_accuracy  <- mean(y_test_pred == y_test)

conf_mat_train <- table(Predicted = y_train_pred, Actual = y_train)
print(conf_mat_train)
#     0    1
#0 1174   81
#1   87 1416
conf_mat_test <- table(Predicted = y_test_pred, Actual = y_test)
print(conf_mat_test)
#     0    1
#0 1208  86
#1  103  1346
best_features
# "sst_t0_crgo_stdrt"  "sst_t0_crlg_rate"   "nback_t0_c0b_rate"  
# "nback_t0_c0b_mrt"   "nback_t0_c2b_rate"  "nback_t0_c2b_stdrt"
train_accuracy 
# 0.9390863
test_accuracy      
# 0.9310973

# -----------------------------
# RT 2-year Follow-up Clusters N=2
# -----------------------------
X_train <- rt_tasks_t2_train
X_train_scaled <- scale(X_train)
y_train <- rt_clusters_t2_train

X_test <- rt_tasks_t2_test
X_test_scaled <- scale(X_test)
y_test <- rt_clusters_t2_test

rt_train_t2_model <- characterize_clusters_R(X_train_scaled, y_train)
best_row <- rt_train_t2_model[which.min(rt_train_t2_model$bic), ]

best_coef <- best_row$coef[[1]]                  
best_intercept <- best_row$intercept             
best_features <- best_row$retained_features[[1]] 
names(best_coef) <- colnames(X_train_scaled)
coef_selected <- best_coef[best_features]

X_train_mat <- as.matrix(X_train_scaled[, best_features, drop = FALSE])
X_test_mat  <- as.matrix(X_test_scaled[, best_features, drop = FALSE])
prob_train <- 1 / (1 + exp(-(X_train_mat %*% coef_selected + best_intercept))) 
prob_test  <- 1 / (1 + exp(-(X_test_mat %*% coef_selected + best_intercept)))

y_train_pred <- ifelse(prob_train > 0.5, 1, 0) 
y_test_pred  <- ifelse(prob_test > 0.5, 1, 0)
train_accuracy <- mean(y_train_pred == y_train)
test_accuracy  <- mean(y_test_pred == y_test)

conf_mat_train <- table(Predicted = y_train_pred, Actual = y_train)
print(conf_mat_train)
#     0    1
#0  1525   57
#1   52   1124
conf_mat_test <- table(Predicted = y_test_pred, Actual = y_test)
print(conf_mat_test)
#     0    1
#0 1476   98
#1   35 1134
best_features       
# "sst_t2_crgo_stdrt"  "nback_t2_c0b_rate"  "nback_t2_c0b_mrt"   
# "nback_t2_c0b_stdrt" "nback_t2_c2b_rate"  "nback_t2_c2b_mrt"   "nback_t2_c2b_stdrt"
train_accuracy      
# 0.9604786
test_accuracy       
# 0.9515129

# -----------------------------
# Export Files
# -----------------------------

all_train_t0_model <- flatten_lists(all_train_t0_model)
write.csv(all_train_t0_model, "/Users/lakyzf/Desktop/EF_/10_Analyses/Cluster_Characterization/all_train_t0_model.csv", row.names = FALSE)

all_train_t2_model <- flatten_lists(all_train_t2_model)
write.csv(all_train_t2_model, "/Users/lakyzf/Desktop/EF_/10_Analyses/Cluster_Characterization/all_train_t2_model.csv", row.names = FALSE)

singleton_train_t0_model <- flatten_lists(singleton_train_t0_model)
write.csv(singleton_train_t0_model, "/Users/lakyzf/Desktop/EF_/10_Analyses/Cluster_Characterization/singleton_train_t0_model.csv", row.names = FALSE)

singleton_train_t2_model <- flatten_lists(singleton_train_t2_model)
write.csv(singleton_train_t2_model, "/Users/lakyzf/Desktop/EF_/10_Analyses/Cluster_Characterization/singleton_train_t2_model.csv", row.names = FALSE)

only_train_t0_model <- flatten_lists(only_train_t0_model)
write.csv(only_train_t0_model, "/Users/lakyzf/Desktop/EF_/10_Analyses/Cluster_Characterization/only_train_t0_model.csv", row.names = FALSE)

only_train_t2_model <- flatten_lists(only_train_t2_model)
write.csv(only_train_t2_model, "/Users/lakyzf/Desktop/EF_/10_Analyses/Cluster_Characterization/only_train_t2_model.csv", row.names = FALSE)

rt_train_t0_model <- flatten_lists(rt_train_t0_model)
write.csv(rt_train_t0_model, "/Users/lakyzf/Desktop/EF_/10_Analyses/Cluster_Characterization/rt_train_t0_model.csv", row.names = FALSE)

rt_train_t2_model <- flatten_lists(rt_train_t2_model)
write.csv(rt_train_t2_model, "/Users/lakyzf/Desktop/EF_/10_Analyses/Cluster_Characterization/rt_train_t2_model.csv", row.names = FALSE)

