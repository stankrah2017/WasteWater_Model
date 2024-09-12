library(caret)
ESData1=Statistical_Methods_Data_final1
attach(ESData1)
set.seed(123)  # Set seed for reproducibility
##Scaling data
scaled_data <- as.data.frame(scale(ESData1[, c("Flow_Speed", "Depth_wastewater", "Width_wastewater", "pH",
                                                     "dissolved_oxygen", "Electrical_conductivity", 
                                                     "Salinity", "Seawater_specific_gravity", 
                                                     "Catchment_population", "HF183_cat")]))
All_scaled_data <- cbind(scaled_data, as.data.frame(ESData1[, c("Site_ID", "VISIT", "S_Typhi")]))
##########################
All_data <- as.data.frame(ESData1[, c("Flow_Speed", "Depth_wastewater", "Width_wastewater", "pH",
                                               "dissolved_oxygen", "Electrical_conductivity", 
                                               "Salinity", "Seawater_specific_gravity", 
                                               "Catchment_population", "HF183_cat","Site_ID", "VISIT", "S_Typhi")])

#####Splitting data
trainIndex <- createDataPartition(All_scaled_data$S_Typhi, p = 0.8, list = FALSE)
training_data <- All_scaled_data[trainIndex, ]
testing_data <- All_scaled_data[-trainIndex, ]

############################
library(caret)
# Specify the number of repetitions
num_samples <- 10000
all_training_sets <- list()  # Initialize a list to store training sets

# Generate training sets
for (i in 1:num_samples) {
  trainIndex <- createDataPartition(All_scaled_data$S_Typhi, p = 0.8, list = FALSE)
  training_data <- All_scaled_data[trainIndex, ]
  testing_data <- All_scaled_data[-trainIndex, ]
  
  # Store the training set in the list
  all_training_sets[[i]] <- training_data
}

# To check how many training sets were generated
num_generated_training_sets <- length(all_training_sets)
print(paste("Number of training sets generated:", num_generated_training_sets))

training_data1 <- all_training_sets[[1]]  # Get the first training set

#############################

###Logistic
GLM1 <- glm(S_Typhi ~ Flow_Speed+Depth_wastewater+Width_wastewater+pH+dissolved_oxygen+Electrical_conductivity+Salinity+Seawater_specific_gravity+Catchment_population + HF183_cat+factor(VISIT) , data = training_data, family = "binomial")##better
GLM2 <- glm(S_Typhi ~ Flow_Speed+Depth_wastewater+Width_wastewater+pH+dissolved_oxygen+Electrical_conductivity+Salinity+Seawater_specific_gravity+Catchment_population + HF183_cat + (1 | Site_ID), data = training_data, family = "binomial")
summary(GLM1)

##prediction
# Accuracy
predlogt <- predict(GLM1, newdata = testing_data, type = "response")
# Confusion matrix
conf_matrix <- table(testing_data$S_Typhi, ifelse(predlogt > 0.5, 1, 0))
conf_matrix
# Sensitivity and specificity
sensitivity <- conf_matrix[2,2] / sum(conf_matrix[2,])
specificity <- conf_matrix[1,1] / sum(conf_matrix[1,])
sensitivity
specificity

# Precision
precision <- conf_matrix[2,2] / sum(conf_matrix[,2])
precision

# ROC curve
library(pROC)
roc_curve <- roc(testing_data$S_Typhi, predlogt)
plot(roc_curve)

library(pROC)
roc_obj <- roc(testing_data$S_Typhi, predlogt)
AUC <- auc(roc_obj)
AUC
# RMSE function
calculate_RMSE <- function(observed, predicted) {
  return(sqrt(mean((observed - predicted)^2)))
}

# MAPE function
calculate_MAPE <- function(observed, predicted) {
  return(mean(abs(observed - predicted) / observed) * 100)
}
MAPE_logt <- calculate_MAPE(testing_data$S_Typhi, predlogt)
RMSE_logt <- calculate_RMSE(testing_data$S_Typhi, predlogt)
RMSE_logt
MAPE_logt

#########################################################################3

# For GLMM###MIXED EFFECT MODEL
library(lme4)
library(Matrix)
library(caret)
GLMM <- glmer(S_Typhi ~ Flow_Speed + Depth_wastewater + Width_wastewater + pH + dissolved_oxygen + 
                 Electrical_conductivity + Salinity + Seawater_specific_gravity + Catchment_population + 
                 HF183_cat + (1 | Site_ID) + (1 | VISIT), data = All_scaled_data, family = binomial)
summary(GLMM)
########model2 for GLMM
training_data <- all_training_sets[[1]]  # Get the first training set
GLMM2 <- glmer(S_Typhi ~ Flow_Speed + Depth_wastewater + Width_wastewater + pH + dissolved_oxygen + 
                Electrical_conductivity + Salinity + Seawater_specific_gravity + Catchment_population + 
                HF183_cat + (1 | Site_ID) + (1 | VISIT), data = training_data1, family = binomial, 
              control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000)))
summary(GLMM2)
GLMM_lasso <- glmer(S_Typhi ~ Flow_Speed + Width_wastewater + pH + dissolved_oxygen + 
                 Seawater_specific_gravity + Catchment_population + 
                 HF183_cat + (1 | Site_ID) + (1 | VISIT), data = training_data, family = binomial, 
               control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000)))
summary(GLMM_lasso)
GLMM3 <- glmer(S_Typhi ~ Flow_Speed + Width_wastewater + pH + (1 | Site_ID) + (1 | VISIT), data = All_data, family = binomial, 
               control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000)))
summary(GLMM3)
summary_model=summary(GLMM3)
GLMM4 <- glmer(S_Typhi ~ Flow_Speed + pH + (1 | Site_ID) + (1 | VISIT), data = All_scaled_data, family = binomial, 
               control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000)))
summary(GLMM4)


##prediction
# Accuracy
predglmm1 <- predict(GLMM2, newdata = testing_data, type = "response")
# Confusion matrix
conf_matrixglmm <- table(testing_data$S_Typhi, ifelse(predglmm1 > 0.5, 1, 0))
conf_matrixglmm
# Sensitivity and specificity
sensitivityglmm <- conf_matrixglmm[2,2] / sum(conf_matrixglmm[2,])
specificityglmm <- conf_matrixglmm[1,1] / sum(conf_matrixglmm[1,])
sensitivityglmm
specificityglmm

# Precision
precisionglmm <- conf_matrixglmm[2,2] / sum(conf_matrixglmm[,2])
precisionglmm
# ROC curve
library(pROC)
roc_curveglmm <- roc(testing_data$S_Typhi, predglmm1)
plot(roc_curveglmm)
roc_curveglmm
#######
RMSE_glmm1 <- calculate_RMSE(testing_data$S_Typhi, predglmm1)
RMSE_glmm1

################################################
##RANDOM FOREST
library(randomForest)
ran1=randomForest(S_Typhi ~ Flow_Speed + Depth_wastewater + Width_wastewater + pH + dissolved_oxygen +
                    Electrical_conductivity + Salinity + Seawater_specific_gravity + 
                    Catchment_population + HF183_cat + VISIT, 
                  data = training_data1)
summary(ran1)
##prediction
# Accuracy
predran <- predict(ran1, newdata = testing_data, type = "response")
# Confusion matrix
conf_matrixran <- table(testing_data$S_Typhi, ifelse(predran > 0.5, 1, 0))
conf_matrixran
# Sensitivity and specificity
sensitivityran <- conf_matrixran[2,2] / sum(conf_matrixran[2,])
specificityran <- conf_matrixran[1,1] / sum(conf_matrixran[1,])
sensitivityran
specificityran

# Precision
precisionran <- conf_matrixran[2,2] / sum(conf_matrixran[,2])
precisionran
# ROC curve
library(pROC)
roc_curveran <- roc(testing_data$S_Typhi, predran)
plot(roc_curveran)
roc_curveran
######
MAPE_ran <- calculate_MAPE(testing_data$S_Typhi, predran)
RMSE_ran <- calculate_RMSE(testing_data$S_Typhi, predran)
RMSE_ran
MAPE_ran
################################################
######SUPPORT VECTOR
library(e1071)
sup1=svm(S_Typhi ~ Flow_Speed + Depth_wastewater + Width_wastewater + pH + dissolved_oxygen +
    Electrical_conductivity + Salinity + Seawater_specific_gravity + 
    Catchment_population + HF183_cat + VISIT, 
  data = training_data,probability = TRUE)
summary(sup1)

##prediction
# Accuracy
predsup <- predict(sup1, newdata = testing_data, type = "response")
# Confusion matrix
conf_matrixsup <- table(testing_data$S_Typhi, ifelse(predsup > 0.5, 1, 0))
conf_matrixsup
# Sensitivity and specificity
sensitivitysup <- conf_matrixsup[2,2] / sum(conf_matrixsup[2,])
specificitysup <- conf_matrixsup[1,1] / sum(conf_matrixsup[1,])
sensitivitysup
specificitysup

# Precision
precisionsup <- conf_matrixsup[2,2] / sum(conf_matrixsup[,2])
precisionsup
# ROC curve
library(pROC)
roc_curvesup <- roc(testing_data$S_Typhi, predsup)
plot(roc_curvesup)
roc_curvesup
#########
MAPE_sup <- calculate_MAPE(testing_data$S_Typhi, predsup)
RMSE_sup <- calculate_RMSE(testing_data$S_Typhi, predsup)
RMSE_sup
################
###neural network
library(neuralnet)
nnr <- neuralnet(S_Typhi ~ Flow_Speed + Depth_wastewater + Width_wastewater + pH + dissolved_oxygen +
                        Electrical_conductivity + Salinity + Seawater_specific_gravity + Catchment_population + 
                        HF183_cat + VISIT,data = training_data, hidden = c(5, 3), linear.output = FALSE)
print(nnr)
##prediction
# Accuracy
prednnr <- predict(nnr, newdata = testing_data, type = "response")
# Confusion matrix
conf_matrixnnr <- table(testing_data$S_Typhi, ifelse(prednnr > 0.5, 1, 0))
conf_matrixnnr
# Sensitivity and specificity
sensitivitynnr <- conf_matrixnnr[2,2] / sum(conf_matrixnnr[2,])
specificitynnr <- conf_matrixnnr[1,1] / sum(conf_matrixnnr[1,])
sensitivitynnr
specificitynnr

# Precision
precisionnnr <- conf_matrixnnr[2,2] / sum(conf_matrixnnr[,2])
precisionnnr
# ROC curve
library(pROC)
roc_curvennr <- roc(testing_data$S_Typhi, prednnr)
plot(roc_curvennr)
roc_curvennr
#######
MAPE_nnr <- calculate_MAPE(testing_data$S_Typhi, prednnr)
RMSE_nnr <- calculate_RMSE(testing_data$S_Typhi, prednnr)
RMSE_nnr
MAPE_nnr

####################################
GBM
install.packages("gbm")
library(gbm)
gbm <- gbm(S_Typhi ~ Flow_Speed + Depth_wastewater + Width_wastewater + pH + dissolved_oxygen +
                   Electrical_conductivity + Salinity + Seawater_specific_gravity + Catchment_population + 
                   HF183_cat, data = training_data, distribution = "bernoulli", n.trees = 100, interaction.depth = 3)

##prediction
# Accuracy
predgbm <- predict(gbm, newdata = testing_data, type = "response")
# Confusion matrix
conf_matrixgbm <- table(testing_data$S_Typhi, ifelse(predgbm > 0.5, 1, 0))
conf_matrixgbm
# Sensitivity and specificity
sensitivitygbm <- conf_matrixgbm[2,2] / sum(conf_matrixgbm[2,])
specificitygbm <- conf_matrixgbm[1,1] / sum(conf_matrixgbm[1,])
sensitivitygbm
specificitygbm

# Precision
precisiongbm <- conf_matrixgbm[2,2] / sum(conf_matrixgbm[,2])
precisiongbm
# ROC curve
library(pROC)
roc_curvegbm <- roc(testing_data$S_Typhi, predgbm)
plot(roc_curvegbm)
roc_curvegbm
########################################
MAPE_gbm <- calculate_MAPE(testing_data$S_Typhi, predgbm)
RMSE_gbm <- calculate_RMSE(testing_data$S_Typhi, predgbm)
RMSE_gbm
MAPE_gbm


##############################
# Given fixed effect estimates
intercept_est <- -0.4115
Flow_Speed_est <- -0.2668
Width_wastewater_est <- 0.2693
pH_est <- 0.5450

# Calculate odds ratios
odds_ratios <- exp(c(intercept_est, Flow_Speed_est, Width_wastewater_est, pH_est))

# Naming the predictors
predictors <- c("Intercept", "Flow_Speed", "Width_wastewater", "pH")

# Creating a data frame with predictors and their odds ratios
odds_ratios_df <- data.frame(Predictor = predictors, Odds_Ratio = odds_ratios)

# Output the odds ratios
print(odds_ratios_df)

###########
##ALL DATA
ran=randomForest(S_Typhi ~ Flow_Speed + Depth_wastewater + Width_wastewater + pH + dissolved_oxygen +
                    Electrical_conductivity + Salinity + Seawater_specific_gravity + 
                    Catchment_population + HF183_cat + VISIT, 
                  data = All_scaled_data)
#FEATURE SELECTION
fect=importance(ran)
fect
varImpPlot(ran)
mean_importance <- mean(fect[, "IncNodePurity"])
###########MODEL FITTING

##DATA SPLIT
library(caret)
##Scaling data
scaled_data2 <- as.data.frame(scale(ESData1[, c("pH","dissolved_oxygen", "Electrical_conductivity", 
                                               "Salinity", "Catchment_population")]))
All_scaled_data2 <- cbind(scaled_data2, as.data.frame(ESData1[, c("Site_ID", "VISIT", "S_Typhi")]))


# Specify the number of repetitions
num_samples <- 10000
all_training_sets <- list()  # Initialize a list to store training sets

# Generate training sets
for (i in 1:num_samples) {
  trainIndex <- createDataPartition(All_scaled_data2$S_Typhi, p = 0.8, list = FALSE)
  training_data <- All_scaled_data2[trainIndex, ]
  testing_data <- All_scaled_data2[-trainIndex, ]
  
  # Store the training set in the list
  all_training_sets[[i]] <- training_data
}

# To check how many training sets were generated
num_generated_training_sets <- length(all_training_sets)
print(paste("Number of training sets generated:", num_generated_training_sets))

training_data1 <- all_training_sets[[1]]  # Get the first training set

##RANDOM FOREST
library(randomForest)
ranfect=randomForest(S_Typhi ~ pH + dissolved_oxygen +
                    Electrical_conductivity + Salinity  + 
                    Catchment_population + VISIT, 
                  data = training_data1)
summary(ranfect)
##prediction
# Accuracy
predran <- predict(ranfect, newdata = testing_data, type = "response")
# Confusion matrix
conf_matrixran <- table(testing_data$S_Typhi, ifelse(predran > 0.5, 1, 0))
conf_matrixran
# Sensitivity and specificity
sensitivityran <- conf_matrixran[2,2] / sum(conf_matrixran[2,])
specificityran <- conf_matrixran[1,1] / sum(conf_matrixran[1,])
sensitivityran
specificityran

# Precision
precisionran <- conf_matrixran[2,2] / sum(conf_matrixran[,2])
precisionran
# ROC curve
library(pROC)
roc_curveran <- roc(testing_data$S_Typhi, predran)
plot(roc_curveran)
roc_curveran

#######GRAPH IMPORTANCE
# Load necessary libraries
library(randomForest)
library(ggplot2)

# Assuming you have your random forest model named 'ran'
# Extract variable importance
importance_df <- as.data.frame(importance(ran))

# Add variable names as a column
importance_df$Variable <- rownames(importance_df)

# Rename columns for clarity (optional)
colnames(importance_df) <- c("IncNodePurity", "Variable")

# Calculate the mean of IncNodePurity
mean_importance <- mean(importance_df$IncNodePurity)

# Create the plot
ggplot(importance_df, aes(x = reorder(Variable, IncNodePurity), y = IncNodePurity)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() + # Flip coordinates for better readability
  labs(title = "Variable Importance",
       x = "Variables",
       y = "IncNodePurity") +
  geom_hline(yintercept = mean_importance, color = "red", linetype = "dashed", linewidth = 1) + # Changed size to linewidth
  annotate("text", x = Inf, y = mean_importance, label = paste("Mean:", round(mean_importance, 2)), 
           color = "red", vjust = -0.5, hjust = 1) + # Adding text annotation for mean
  theme_minimal() + # Using a minimal theme
  theme(axis.text.y = element_text(size = 10), # Adjust text size
        plot.title = element_text(hjust = 0.5, size = 14)) # Center title

