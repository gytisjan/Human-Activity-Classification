# Human activity classification using Random Forest and Naive Bayes.
# Run this script from the project directory containing the data folder.

library(caret)
library(randomForest)
library(e1071)
library(ggcorrplot)

train <- read.csv("C://Users//gytis//OneDrive//Dokumentai//R programinė kalba//activity_train.csv")
test <- read.csv("C://Users//gytis//OneDrive//Dokumentai//R programinė kalba//activity_test.csv")

# Keep the same predictors and their order in both datasets.
stopifnot(identical(names(train), names(test)))
stopifnot("activity" %in% names(train))

# Explore class frequencies and the distribution of selected features.
freq_table <- table(train$activity)
barplot(
  freq_table,
  main = "Class frequencies",
  xlab = "Activity class",
  ylab = "Frequency",
  col = "skyblue",
  border = "black"
)

plot(
  train$tBodyAcc_Mean_1, train$tBodyAcc_Mean_2,
  col = train$activity,
  xlab = "tBodyAcc_Mean_1", ylab = "tBodyAcc_Mean_2"
)
plot(
  train$tBodyAcc_Mean_1, train$tBodyAccJerk_Mean_1,
  col = train$activity,
  xlab = "tBodyAcc_Mean_1", ylab = "tBodyAccJerk_Mean_1"
)

# Calculate correlations using training predictors only, excluding class labels.
train_predictors <- train[, names(train) != "activity", drop = FALSE]
test_predictors <- test[, names(train_predictors), drop = FALSE]

correlation_matrix <- cor(train_predictors)
threshold <- 0.8
high_cor_vars <- findCorrelation(correlation_matrix, cutoff = threshold)

# Display the same selected correlations as in the original project.
plot_columns <- c(2, 3, 5, 7, 8, 21, 54, 61, 84, 91, 111)
show_cor <- cor(train_predictors[, plot_columns, drop = FALSE])
print(ggcorrplot(show_cor))

# Apply the feature selection learned from training data to both datasets.
selected_features <- names(train_predictors)[-high_cor_vars]
train_clean <- train[, c(selected_features, "activity"), drop = FALSE]
test_clean <- test[, c(selected_features, "activity"), drop = FALSE]

train_labels <- factor(train_clean$activity)
test_labels <- factor(test_clean$activity, levels = levels(train_labels))
train_features <- train_clean[, selected_features, drop = FALSE]
test_features <- test_clean[, selected_features, drop = FALSE]

cat("Selected predictors:", ncol(train_features), "\n")
cat("Removed predictors:", length(high_cor_vars), "\n")

# Train a small Random Forest as an initial experiment.
set.seed(123)
rf_small <- randomForest(
  x = train_features,
  y = train_labels,
  ntree = 2,
  mtry = round(sqrt(ncol(train_features))),
  importance = TRUE
)
print(rf_small)

small_predictions <- predict(rf_small, train_features)
small_cm <- confusionMatrix(small_predictions, train_labels)
cat("Random Forest training accuracy (2 trees):",
    as.numeric(small_cm$overall["Accuracy"]), "\n")

# Balanced accuracy summarizes each class against all other classes.
train_balanced_accuracy <- small_cm$byClass[, "Balanced Accuracy"]
barplot(
  train_balanced_accuracy,
  main = "Training balanced accuracy (2 trees)",
  xlab = "Activity class", ylab = "Balanced accuracy",
  col = "skyblue", border = "black", ylim = c(0, 1)
)

# Train the larger Random Forest from the original experiments.
set.seed(123)
rf_model <- randomForest(
  x = train_features,
  y = train_labels,
  ntree = 500,
  mtry = 27,
  importance = TRUE
)
print(rf_model)

# Check the out-of-bag error as trees are added.
oob_error <- rf_model$err.rate[, "OOB"]
plot(
  seq_along(oob_error), oob_error,
  type = "l", col = "blue", lwd = 2,
  xlab = "Number of trees", ylab = "OOB error",
  main = "Random Forest out-of-bag error"
)

rf_predictions <- predict(rf_model, test_features)
rf_cm <- confusionMatrix(rf_predictions, test_labels)
cat("Random Forest test accuracy (500 trees):",
    as.numeric(rf_cm$overall["Accuracy"]), "\n")
print(rf_cm$table)

# Per-class sensitivity measures correctly classified observations within each class.
rf_class_accuracy <- rf_cm$byClass[, "Sensitivity"]
barplot(
  rf_class_accuracy,
  main = "Random Forest class-wise accuracy (500 trees)",
  xlab = "Activity class", ylab = "Recall",
  col = "skyblue", border = "black", ylim = c(0, 1)
)

varImpPlot(rf_model)

# Compare OOB error for different numbers of candidate split variables.
mtry_values <- unique(as.integer(round(c(3, 9, 27, sqrt(ncol(train_features))))))
oob_errors <- vector("list", length(mtry_values))

for (i in seq_along(mtry_values)) {
  set.seed(123)
  rf_temp <- randomForest(
    x = train_features,
    y = train_labels,
    ntree = 500,
    mtry = mtry_values[i],
    importance = FALSE
  )
  oob_errors[[i]] <- rf_temp$err.rate[, "OOB"]
}

line_colors <- c("red", "blue", "black", "orange")
plot(
  seq_along(oob_errors[[1]]), oob_errors[[1]],
  type = "l", col = line_colors[1], lwd = 2,
  xlab = "Number of trees", ylab = "OOB error",
  ylim = range(unlist(oob_errors)),
  main = "OOB error for different mtry values"
)
for (i in 2:length(mtry_values)) {
  lines(seq_along(oob_errors[[i]]), oob_errors[[i]],
        col = line_colors[i], lwd = 2)
}
legend(
  "topright",
  legend = paste("mtry =", mtry_values),
  col = line_colors[seq_along(mtry_values)], lwd = 2
)

# Use the same selected features and factor labels for Naive Bayes.
# This function reports test performance for each parameter setting.
evaluate_nb <- function(model, label) {
  predictions <- predict(model, test_features)
  result <- confusionMatrix(predictions, test_labels)
  cat(label, "test accuracy:", as.numeric(result$overall["Accuracy"]), "\n")
  return(result)
}

nb_default <- naiveBayes(x = train_features, y = train_labels)
nb_default_cm <- evaluate_nb(nb_default, "Naive Bayes (default)")

# Check the effect of Laplace smoothing as in the original code.
laplace_values <- c(0, 0.5, 1, 2, 5)
for (laplace in laplace_values) {
  nb_model <- naiveBayes(
    x = train_features,
    y = train_labels,
    laplace = laplace
  )
  evaluate_nb(nb_model, paste("Naive Bayes (Laplace =", laplace, ")"))
}

# Explore kernel density estimation with the original bandwidth settings.
# Laplace smoothing mainly affects categorical features; predictors here are numeric.
nb_kernel <- naiveBayes(
  x = train_features,
  y = train_labels,
  laplace = 50,
  usekernel = TRUE,
  adjust = 3
)
nb_kernel_cm <- evaluate_nb(nb_kernel, "Naive Bayes (kernel, adjust = 3)")

nb_kernel_adjusted <- naiveBayes(
  x = train_features,
  y = train_labels,
  laplace = 1,
  usekernel = TRUE,
  adjust = 1
)
nb_adjusted_cm <- evaluate_nb(nb_kernel_adjusted, "Naive Bayes (kernel, adjust = 1)")

# Display the class-wise recall for the last Naive Bayes experiment.
nb_class_accuracy <- nb_adjusted_cm$byClass[, "Sensitivity"]
barplot(
  nb_class_accuracy,
  main = "Naive Bayes class-wise accuracy",
  xlab = "Activity class", ylab = "Recall",
  col = "skyblue", border = "black", ylim = c(0, 1)
)
