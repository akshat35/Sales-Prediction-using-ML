library(data.table)
library(dplyr)
library(ggplot2)
library(caret)
library(corrplot)
library(xgboost)
library(cowplot)

#reading the data
train = fread("Train_UWu5bXk.csv")
test = fread("Test_u94Q5KV.csv")

#checking the data's dimensions
dim(train)
dim(test)

#taking a look at all the features
names(test)
names(train) #it can be noticed that Item_Outlet_Sales is present in train but not test because we have to predict that for test data

#taking a look at structure of data to understand better
str(train)
str(test)

#combining the training and testing data to avoid doing all the feature engineering twice
test[,Item_Outlet_Sales := NA]
combi = rbind(train, test) #combining training and testing data set
dim(combi)

##########Exploratory Data Analysis########

########Univariate Analysis########
#Target Variable
ggplot(train) + geom_histogram(aes(train$Item_Outlet_Sales), binwidth = 100, fill = "darkgreen") + xlab("Item_Outlet_Sales")
#Independent Variable - continous
p1 = ggplot(combi) + geom_histogram(aes(Item_Weight), binwidth = 0.5, fill = 'blue')
p2 = ggplot(combi) + geom_histogram(aes(Item_Visibility), binwidth = 0.005, fill = 'blue')
p3 = ggplot(combi) + geom_histogram(aes(Item_MRP), binwidth = 1, fill = 'blue')
plot_grid(p1,p2,p3, nrow=1)

#Independent variables categorical

ggplot(combi %>% group_by(Item_Fat_Content) %>% summarise(Count = n())) + 
  geom_bar(aes(Item_Fat_Content, Count), stat = 'identity')

#LF and low fat are Low Fat and reg is Regular, so lets change the names and plot again
combi$Item_Fat_Content[combi$Item_Fat_Content == 'LF'] = "Low Fat"
combi$Item_Fat_Content[combi$Item_Fat_Content == 'low fat'] = "Low Fat"
combi$Item_Fat_Content[combi$Item_Fat_Content == 'reg'] = 'Regular'

#plotting the frequencies again
ggplot(combi %>% group_by(Item_Fat_Content) %>% summarise(Count = n())) +
  geom_bar(aes(Item_Fat_Content, Count), stat = 'identity')

#plot for item type
p4 = ggplot(combi %>% group_by(Item_Type) %>% summarise(Count = n())) + 
  geom_bar(aes(Item_Type, Count), stat = 'identity') +
  xlab("") +
  geom_label(aes(Item_Type, Count, label = Count), vjust = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1 )) +
  ggtitle("Item Type")

#plot for Outlet_Identifier
p5 = ggplot(combi %>% group_by(Outlet_Identifier) %>% summarise(Count = n())) +
  geom_bar(aes(Outlet_Identifier, Count), stat = 'identity') +
  geom_label(aes(Outlet_Identifier, Count, label = Count), vjust = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#plot for Outlet_Size
p6 = ggplot(combi %>% group_by(Outlet_Size) %>% summarise(Count = n())) +
  geom_bar(aes(Outlet_Size, Count), stat = 'identity') +
  geom_label(aes(Outlet_Size, Count, label = Count), vjust = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

second_row = plot_grid(p5,p6, nrow = 1)
plot_grid(p4, second_row, ncol = 1)

# plot for Outlet_Establishment_Year
p7 = ggplot(combi %>% group_by(Outlet_Establishment_Year) %>% summarise(Count = n())) + 
  geom_bar(aes(factor(Outlet_Establishment_Year), Count), stat = "identity", fill = "coral1") +
  geom_label(aes(factor(Outlet_Establishment_Year), Count, label = Count), vjust = 0.5) +
  xlab("Outlet_Establishment_Year") +
  theme(axis.text.x = element_text(size = 8.5))

# plot for Outlet_Type
p8 = ggplot(combi %>% group_by(Outlet_Type) %>% summarise(Count = n())) + 
  geom_bar(aes(Outlet_Type, Count), stat = "identity", fill = "coral1") +
  geom_label(aes(factor(Outlet_Type), Count, label = Count), vjust = 0.5) +
  theme(axis.text.x = element_text(angle= 45,size = 8.5))

# ploting both plots together
plot_grid(p7, p8, ncol = 2)

##########Bivariate Analysis###########

train = combi[1:nrow(train)] #extracting train from combined dataset

#Scatter Plots for Continous Independent Variable vs Target Variable
# Item_Weight vs Item_Outlet_Sales
p9 = ggplot(train)+
  geom_point(aes(Item_Weight, Item_Outlet_Sales), colour = "violet", alpha = 0.3) +
  theme(axis.title = element_text(size = 8.5))

#Item_Visibility vs Item_Outlet_Sales
p10 = ggplot(train)+
  geom_point(aes(Item_Visibility, Item_Outlet_Sales), colour = "violet", alpha = 0.3) +
  theme(axis.title = element_text(size = 8.5))

#Item_MRP vs Item_Outlet_Sales
p11 = ggplot(train) + 
  geom_point(aes(Item_MRP, Item_Outlet_Sales), colour = "violet", alpha = 0.3)+
  theme(axis.title = element_text(size = 8.5))

#Violin Plots for Continous Independent Variable vs Target Variable

#Item_Type vs Item_Outlet_Sales
p12 = ggplot(train)+
  geom_violin(aes(Item_Type, Item_Outlet_Sales), fill = "blue") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Item_Fat_Content vs Item_Outlet_Sales
p13 = ggplot(train) +
  geom_violin(aes(Item_Fat_Content, Item_Outlet_Sales), fill = 'blue') +
  theme(axis.text.x = element_text(angle=45, hjust = 1))

#Outlet_Identifier vs Item_Outlet_Sales
p14 = ggplot(train) +
  geom_violin(aes(Outlet_Identifier, Item_Outlet_Sales), fill = 'blue') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


p15 = ggplot(train) + geom_violin(aes(Outlet_Location_Type, Item_Outlet_Sales), fill = "blue")
p16 = ggplot(train) + geom_violin(aes(Outlet_Type, Item_Outlet_Sales), fill = "blue")

#########Missing Values########
#checking the number of missing values
sum(is.na(combi)) # there are a total of 8120 missing values
sum(is.na(combi$Item_Weight)) #2439 are from item weight and the rest are the predictions that we have to make

# imputing the missing values using mean imputation grouped by Item_Identifier
missing_indexes = which(is.na(combi$Item_Weight))

for(i in missing_indexes){
  
  item = combi$Item_Identifier[i]
  combi$Item_Weight = mean(combi$Item_Weight[combi$Item_Identifier == item], na.rm = T)
}

#check for missing values
sum(is.na(combi$Item_Weight))

#replacing the zeros in Item_Visibility with mean grouped by Item_Identifier

zero_indexes = which(combi$Item_Visibility == 0)

for(i in zero_indexes){
  
  item = combi$Item_Identifier[i]
  combi$Item_Visibility[i] = mean(combi$Item_Visibility[combi$Item_Identifier == item], na.rm = T)
}

############Feature engineering#########

#creating a new feature that classifies item type into perishable or non-perishable
perishable = c("Breads", "Breakfast", "Dairy", "Fruits and Vegetables", "Meat", "Seafood")
non_perishable = c("Baking Goods", "Canned", "Frozen Foods", "Hard Drinks", "Health and Hygiene", "Household", "Soft Drinks")

combi[,Item_Type_New := ifelse(Item_Type %in% perishable, "perishable", ifelse(Item_Type %in% non_perishable, "non-perishable", "not sure"))]

# creating new feature Item_Category based on Item_Identifier
#first lets have a look at item_identifier 
table(combi$Item_Type, substr(combi$Item_Identifier,1,2)) #it can be noticed that there are three categories DR, FD, NC

combi[,Item_Category := substr(combi$Item_Identifier,1,2)]

#changing Item_Fat_Content of non consumables to Not Edible
combi$Item_Fat_Content[combi$Item_Category == "NC"] = "Non-Edible"

# years of operation for outlets
combi[,Outlet_Years := 2013 - Outlet_Establishment_Year]
combi$Outlet_Establishment_Year = as.factor(combi$Outlet_Establishment_Year)

# Price per unit weight
combi[,price_per_unit_wt := Item_MRP/Item_Weight]

# creating new independent variable - Item_MRP_clusters
combi[,Item_MRP_clusters := ifelse(Item_MRP < 69, "1st", 
                                   ifelse(Item_MRP >= 69 & Item_MRP < 136, "2nd",
                                          ifelse(Item_MRP >= 136 & Item_MRP < 203, "3rd", "4th")))]


#########Encoding Catrgorical Variables##########

# Label encoding Outlet_Size

combi[,Outlet_Size_Num := ifelse(Outlet_Size == "Small",0,
                                 ifelse(Outlet_Size == "Medium",1,2))]


#Label encoding Outlet_Location_Type

combi[, Outlet_Location_Type_Num := ifelse(Outlet_Location_Type == 'Tier 3', 0,
                                           ifelse(Outlet_Location_Type == 'Tier 2',1,2))]


#deleting the original categorical variables that were label encoded
combi[, c("Outlet_Size", "Outlet_Location_Type") := NULL]

# one hot encoding
ohe = dummyVars("~.", data = combi[,-c("Item_Identifier", "Outlet_Establishment_Year", "Item_Type")], fullRank = T)
ohe_df = data.table(predict(ohe, combi[,-c("Item_Identifier", "Outlet_Establishment_Year", "Item_Type")]))
combi = cbind(combi[,"Item_Identifier"], ohe_df)

#####preprocessing data######

# removing skewness

#log transforming Item_Visibility
combi[,Item_Visibility := log(Item_Visibility + 1)]

#log transforming Item_Visibility
combi[,price_per_unit_wt := log(price_per_unit_wt + 1)]

#Scaling numeric predictors

num_vars = which(sapply(combi, is.numeric))
num_vars_names = names(num_vars)
combi_numeric = combi[,setdiff(num_vars_names, "Item_Outlet_Sales"), with = F]
prep_num = preProcess(combi_numeric, method=c("center", "scale"))
combi_numeric_norm = predict(prep_num, combi_numeric)

combi[,setdiff(num_vars_names, "Item_Outlet_Sales") := NULL] # removing numeric independent variables
combi = cbind(combi, combi_numeric_norm)


train = combi[1:nrow(train)]
test = combi[(nrow(train) + 1):nrow(combi)]
test[,Item_Outlet_Sales := NULL] # removing Item_Outlet_Sales as it contains only NA for test dataset

# checking for correlation
cor_train = cor(train[,-c("Item_Identifier")])
corrplot(cor_train, method = "pie", type = "lower", tl.cex = 0.9)

##########Modeling########

#######Lasso Regression#######

set.seed(1235)
cross_validation = trainControl(method = "cv", number = 5)
GridL =expand.grid(alpha = 1, lambda = seq(0.001,0.1, by = 0.0002))

lasso_linear_regression_model = train(x = train[,-c("Item_Identifier","Item_Outlet_Sales")], y = train$Item_Outlet_Sales, method = 'glmnet',trControl = cross_validation, tuneGrid = GridL )

Item_Outlet_Sales_Lasso_Linear_Regression = predict(lasso_linear_regression_model, test[,-c("Item_Identifier")])

dtest_mean = mean(train$Item_Outlet_Sales)
# Calculate total sum of squares
TSS = sum((train$Item_Outlet_Sales - dtest_mean)^2 )
# Calculate residual sum of squares
residuals = train$Item_Outlet_Sales - Item_Outlet_Sales_Lasso_Linear_Regression
RSS = sum(residuals^2)
# Calculate R-squared
R_squared_lasso= sqrt((1 - (RSS/TSS))^2)

#########Ridge Regression#######
set.seed(1236)
cross_validation = trainControl(method = "cv", number = 5)
GridR = expand.grid(alpha = 0, lambda = seq(0.001,0.1, by = 0.0002))

ridge_linear_regression_model = train(x = train[,-c("Item_Identifier", "Item_Outlet_Sales")], y = train$Item_Outlet_Sales, method = 'glmnet', trControl = cross_validation, tuneGrid = GridR)

Item_Outlet_Sales_Ridge_Linear_Regression = predict(ridge_linear_regression_model, test[,-c("Item_Identifier")])

dtest_mean = mean(train$Item_Outlet_Sales)
# Calculate total sum of squares
TSS = sum((train$Item_Outlet_Sales - dtest_mean)^2 )
# Calculate residual sum of squares
residuals = train$Item_Outlet_Sales - Item_Outlet_Sales_Ridge_Linear_Regression
RSS = sum(residuals^2)
# Calculate R-squared
R_squared_ridge= sqrt((1 - (RSS/TSS))^2)

######RandomForest#########

set.seed(1237)
cross_validation = trainControl(method="cv", number=5) # 5-fold CV
GridRF = expand.grid(
  .mtry = c(3:10),
  .splitrule = "variance",
  .min.node.size = c(10,15,20)
)

randomforest_regression_model = train(x = train[, -c("Item_Identifier", "Item_Outlet_Sales")], 
                                      y = train$Item_Outlet_Sales,
                                      method='ranger', 
                                      trControl= cross_validation, 
                                      tuneGrid = GridRF,
                                      num.trees = 400,
                                      importance = "permutation")

Item_Outlet_Sales_RandomForest_Regression = predict(randomforest_regression_model, test[,-c("Item_Identifier")])

plot(randomforest_regression_model)
partialPlot(randomforest_regression_model,train, Price, "YES")
dtest_mean = mean(train$Item_Outlet_Sales)
# Calculate total sum of squares
TSS = sum((train$Item_Outlet_Sales - dtest_mean)^2 )
# Calculate residual sum of squares
residuals = train$Item_Outlet_Sales - Item_Outlet_Sales_RandomForest_Regression
RSS = sum(residuals^2)
# Calculate R-squared
R_squared_rf= sqrt((1 - (RSS/TSS))^2)

########XGBoost#############

param_list = list(
  objective = "reg:linear",
  eta = 0.01,
  gamma  =1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.5
)

dtrain = xgb.DMatrix(data = as.matrix(train[,-c("Item_Identifier","Item_Outlet_Sales")]), label = train$Item_Outlet_Sales)

dtest = xgb.DMatrix(data = as.matrix(test[,-c("Item_Identifier")]))

set.seed(112)
xgbcv = xgb.cv(params = param_list, 
               data = dtrain,
               nrounds = 1000,
               nfold = 5,
               print_every_n = 10,
               early_stopping_rounds = 30,
               maximize = F)

xgb_model = xgb.train(data = dtrain, params = param_list, nrounds = 885)

var_importance = xgb.importance(feature_names = setdiff(names(train), c("Item_Identifier", "Item_Outlet_Sales")),
                                model = xgb_model)
xgb.plot.importance(var_importance)

Item_Outlet_Sales_XGBoost_Regression = predict(xgb_model, dtest)

dtest_mean = mean(train$Item_Outlet_Sales)
# Calculate total sum of squares
TSS = sum((train$Item_Outlet_Sales - dtest_mean)^2 )
# Calculate residual sum of squares
residuals = train$Item_Outlet_Sales - Item_Outlet_Sales_XGBoost_Regression
RSS = sum(residuals^2)
# Calculate R-squared
R_squared_xgboost= sqrt((1 - (RSS/TSS))^2)
train1=fread("Train_UWu5bXk.csv")
train1=train1[,Item_Outlet_Sales]
test1 = fread("Test_u94Q5KV.csv")
Predictions  = cbind(test1,train1,Item_Outlet_Sales_Lasso_Linear_Regression,Item_Outlet_Sales_Ridge_Linear_Regression,
                     Item_Outlet_Sales_RandomForest_Regression, Item_Outlet_Sales_XGBoost_Regression)
#RMSE(train$Item_Outlet_Sales,Item_Outlet_Sales_Linear_Regression)
write.csv(Predictions, "predictions.csv")

print(R_squared_lasso)
print(R_squared_ridge)
print(R_squared_rf)
print(R_squared_xgboost)