##### CLASSIFICATION ON IRIS DATASET

### Importing Data
iris <- read.csv("D:/VIA/DATA SCIENCE INDONESIA (EAST JAVA CHAPTER)/DSC UNIPA 2020/2020-12-05 Classification using R/iris.csv", row.names=1) # import data
features <- iris[,1:4] # define features

### EDA
View(iris) # to view the iris dataset
str(iris) # to view the structure of the iris dataset
unique(iris[c("Species")]) # to check the class on the Species column
table(iris$Species) # to count the number of observations in each class ONLY, the results also can be seen in the summary(iris)

### Preprocessing Data
# Detecting Missing Value
summary(iris) # if there is no NA's in the result then there is no missing value on the dataset

# Detecting Outlier
boxplot(features, main = "Detecting Outliers on Each Variable") # to check wether there are outliers on each variable or not

# Detecting Multivariate Outlier
#install.packages("outliers")
library(outliers)
z <- abs(scores(features, type = 'z')) # to calculate the absolute z-score
outlier <- subset(z, SepalLengthCm > 3 | SepalWidthCm > 3 | PetalLengthCm > 3 | PetalWidthCm > 3) # selecting observations with z > 3, | means or
outlier_df <- data.frame("Notes" = c("Not Outlier","Outlier"), "Number of Observations" = c(dim(iris)[1]-dim(outlier)[1], dim(outlier)[1])) # create dataframe of outliers and not outliers
outlier_df
barplot(outlier_df$Number.of.Observations, names.arg = outlier_df$Notes, main = "Detecting Multivariate Outliers", ylab = "Number of Observations")

#Normalization using L2 Norm
l2_norm_func <- function(x) sqrt(sum(x^2)) # to build the l2 norm function
l2_norm <- apply(features,1, l2_norm_func) # to apply the function on the features
features_normalized <- features/l2_norm # to normalize data using the l2 norm value

# Detecting Outlier After Normalization
boxplot(features_normalized, main = "Detecting Outliers on Each Variable After Normalization") # to check wether there are still outliers or not

# Detecting Multivariate Outlier After Normalization
z_norm <- abs(scores(features_normalized, type = 'z')) # to calculate the absolute z-score
outlier_norm <- subset(z_norm, SepalLengthCm > 3 | SepalWidthCm > 3 | PetalLengthCm > 3 | PetalWidthCm > 3) # selecting observations with z > 3, | means or
outlier_norm_df <- data.frame("Notes" = c("Not Outlier","Outlier"), "Number of Observations" = c(dim(iris)[1]-dim(outlier_norm)[1], dim(outlier_norm)[1])) # create dataframe of outliers and not outliers
outlier_norm_df
barplot(outlier_norm_df$Number.of.Observations, names.arg = outlier_norm_df$Notes, main = "Detecting Multivariate Outliers After Normalization", ylab = "Number of Observations")

### Classification using K-Nearest Neighbor
# Combining Features Normalized and Label
iris_normalized <- iris
iris_normalized[1:4] <- features_normalized

# Splitting Data into Training and Testing
set.seed(0)
#install.packages("caTools")
library(caTools)
split <- sample.split(iris_normalized, SplitRatio = 0.8)
train_set <- subset(iris_normalized, split == TRUE)
test_set <- subset(iris_normalized, split == FALSE)
dim(train_set)
dim(test_set)
x_train <- train_set[,1:4]
x_test <- test_set[,1:4]
y_train <- train_set[,5]
y_test <- test_set[,5]

# Classification using K-Nearest Neighbor
set.seed(0)
#install.packages("class")
library(class)
k = 3
knn.pred = knn(x_train, x_test, y_train, k = k)

# Confusion Matrix and Model Performance
#install.packages("caret")
#install.packages("e1071")
library(caret)
library(e1071)
cm <- table(knn.pred, y_test)
confusionMatrix(cm)