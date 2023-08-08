# Cargar y utilizar funci?n IPAK.  Esta funci?n carga todas las librer?as que
# necesitar?amos para esta sentencia propuesta.
#ver v?deo https://www.youtube.com/watch?v=UjQz9SxG9rk

ipak <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

packages <- c(
  "tidyverse", 
  "MASS",
  "car",
  "e1071",
  "caret",
  "cowplot",
  "caTools",
  "pROC",
  "ggcorrplot",
  "rpart",
  "readr",
  "caTools",
  "dplyr",
  "party",
  "partykit",
  "rattle",
  "rpart.plot",
  "RColorBrewer",
  "multiROC",
  "dummies",
  "kableExtra",
  "ggplot2",
  "xgboost",
  "purrr",
  "Matrix",
  "randomForest")
  
ipak(packages)


# Verificamos el directorio de trabajo en el que estamos
getwd()

# cambiamos el directorio de trabajo al que me interesa
setwd("~/3 - Maestria en Estad?stica UNTREF/OPM20. Estad?stica Aplicada a la Investigaci?n de Mercado/TP Final")
getwd()


# levantamos la matriz de datos que nos interesa
DATOS <- read.csv2("t01-2019_Aglo33_individuos_RESUMEN_v2_R.csv",header = TRUE,sep = ";",dec = ",")
DATOS <- as.data.frame(DATOS)
DATOS
nrow(DATOS)
colnames(DATOS)
str(DATOS)


# Vamos a llamar a nuestra base df (data frame), para facilitar el uso de las
# sentencias provistas por la p?gina

df <- DATOS
df
str(df)

glimpse(df)

df$CH03 <- as.factor(df$CH03)
df$CH09 <- as.factor(df$CH09)
df$NIVEL_ED <- as.factor(df$NIVEL_ED)
df$NI.OS_ED <- as.factor(df$NI.OS_ED)
df$ESTADO <- as.factor(df$ESTADO)
df$CAT_OCUP <- as.factor(df$CAT_OCUP)
df$CAT_INAC <- as.factor(df$CAT_INAC)
df$IV1 <- as.factor(df$IV1)
df$IV3 <- as.factor(df$IV3)
df$IV4 <- as.factor(df$IV4)
df$IV5 <- as.factor(df$IV5)
df$IV6 <- as.factor(df$IV6)
df$BA.O_1 <- as.factor(df$BA.O_1)
df$IV10 <- as.factor(df$IV10)
df$IV11 <- as.factor(df$IV11)
df$IV12_1 <- as.factor(df$IV12_1)
df$IV12_2 <- as.factor(df$IV12_2)
df$IV12_3 <- as.factor(df$IV12_3)
df$II7 <- as.factor(df$II7)
df$II8 <- as.factor(df$II8)
df$II9 <- as.factor(df$II9)


glimpse(df)


# Vamos a estandarizar las variables continua
# Inicialmente creamos un subset de las variables cuantitativas continuas que vamos a estandarizar
df_int <- df[,c(3,10,22)]
df_int <- data.frame(scale(df_int))
df_int


# Ahora creamos las variables dummy
# Inicialmente crea un subset con las variables categ?ricas al que llama df_cat
str(df)
df_cat <- df[,-c(1,3,10,22,26)]
str(df_cat)

#Creating Dummy Variables
dummy<- data.frame(sapply(df_cat,function(x) data.frame(model.matrix(~x-1,data =df_cat))))
head(dummy)
str(dummy)


# Combinamos los sets de datos
POBREZA = as.integer(df$POBREZA_MULTID-1)
table(POBREZA)
KEY_INDIV <- df$KEY_INDIV 
df_final <- cbind(KEY_INDIV,df_int,POBREZA,dummy)
head(df_final)
glimpse(df_final)
str(df_final)

# Separamos el data set en Entretamiento y Validaci?n 
set.seed(123)
indices = sample.split(df_final$POBREZA, SplitRatio = 0.7)
train_data = df_final[indices,]
test_data = df_final[!(indices),]

str(train_data)
str(test_data)


# La implementaci?n XGBoost de R requiere que los datos que usemos sean matrices, 
# espec?ficamente de tipo DMatrix, as? que necesitamos convertir nuestros sets 
# de entrenamiento y prueba a este tipo de estructura.
# Al usar esta funci?n es muy importante que los datos no incluyan la columna con la variable
# objetivo, de lo contrario, obtendremos una precisi?n perfecta en las  predicciones, que ser? 
# in?til con datos nuevos.

trainFea <- train_data[,-c(1,5)]
testFea <- test_data[,-c(1,5)]


# Generamos los vectores de la variable POBREZA, para train y test
labeltrain = train_data$POBREZA
labeltest = test_data$POBREZA
str(labeltrain)
str(labeltest)
table(labeltest)

#generamos los features y excluimos los Ids
train_ind = train_data$KEY_INDIV
test_ind = test_data$KEY_INDIV

columns =  names(trainFea)
str(columns)


# Pasamos las matrices al formato requerido por XGBoost

xgb.train = xgb.DMatrix(data=as.matrix(trainFea),label=labeltrain)
xgb.test = xgb.DMatrix(data=as.matrix(testFea),label=labeltest)
dim(xgb.train)
dim(xgb.test)


# Define the parameters for multinomial classification
num_class = 4
params = list(
  booster="gbtree",
  eta=0.05,
  max_depth=6,
  gamma=3,
  subsample=0.75,
  colsample_bytree=1,
  objective="multi:softprob",
  eval_metric="mlogloss",
  num_class=num_class
)

# Breve explicaci?n de los par?metros seleccionados
# objective: 
#El tipo de tarea de clasificaci?n que realizaremos. 
# Para clasificaci?n binaria especificamos "binary:logistic".
# eta: 
#La tasa de aprendizaje del modelo. Un mayor valor llega m?s r?pidamente al m?nimo 
# de la funci?n objetivo, es decir, a un "mejor modelo", pero puede "pasarse" de su valor 
# ?ptimo. En cambio, un valor peque?o puede nunca llegar al valor ?ptimo de la funci?n 
# objetivo, incluso despu?s de muchas iteraciones. 
# En ambos casos, esto afecta el desempe?o de nuestros modelos con nuevos datos.
# subsample:
#
# colsample:
#
# max.depth: 
# "Profundidad" o n?mero de nodos de bifurcaci?n de los ?rboles de de decisi?n usados 
# en el entrenamiento. Aunque una mayor profundidad puede devolver mejores resultados, 
# tambi?n puede resultar en overfitting (sobre ajuste).
# scale_pos_weight:
#
# nround: 
# El n?mero de iteraciones que se realizar?n antes de detener el proceso de ajuste. 
# Un mayor n?mero de iteraciones generalmente devuelve mejores resultados de predicci?n 
# pero necesita m?s tiempo de entrenamiento.

# nthread: 
# El n?mero de hilos computacionales que ser?n usados en el proceso de entrenamiento. 
# Generalmente se refiere a los n?cleos del procesador de tu equipo de c?mputo, 
# local o remoto, pero tambi?n pueden ser los n?cleos de un GPU.



# Entrenamos el MODELO XGBoost classifer
xgb.fit=xgb.train(
  params=params,
  data=xgb.train,
  nrounds=1000,
  nthreads=1,
  early_stopping_rounds=10,
  watchlist=list(val_train=xgb.train,val_test=xgb.test),
  verbose=0,)


# Review the final model and results
xgb.fit
xgb.fit$params


# Predicci?n con el DATA SET ENTRENAMIENTO

pred_train_1 <- predict(xgb.fit, as.matrix(trainFea))
str(pred_train_1)

pred_train_1 <- matrix(pred_train_1, ncol=4, byrow=TRUE)  # Remodelando la matriz
head(pred_train_1)

pred_train_1_labels <- max.col(pred_train_1) -1  # Convertir las probabilidades a softmax labels
head(pred_train_1_labels)

sum(pred_train_1_labels!= labeltrain)/length(labeltrain) # lo siguiente deber?a resultar en el mismo error que se vio en la ?ltima iteraci?n


# Construyo tabla de confusi?n para el resultado del train
labeltrain
table_mat_pred_train_1 <- table(labeltrain, pred_train_1_labels)
table_mat_pred_train_1

accuracy_pred_train_1 <- sum(diag(table_mat_pred_train_1)) / sum(table_mat_pred_train_1)
print(paste('Accuracy for train', accuracy_pred_train_1))


# Predicci?n con Data de Testing
pred_test_1 <- predict(xgb.fit, as.matrix(testFea))
str(pred_test_1)


pred_test_1 <- matrix(pred_test_1, ncol=4, byrow=TRUE)  # Remodelando la matriz


pred_test_1_labels <- max.col(pred_test_1) -1  # Convertir las probabilidades a softmax labels
head(pred_test_1_labels)
str(pred_test_1_labels)

sum(pred_test_1_labels!= labeltest)/length(labeltest) # lo siguiente deber?a resultar en el mismo error que se vio en la ?ltima iteraci?n


# Construyo tabla de confusi?n para el testing
labeltest
table_mat_pred_test_1 <- table(labeltest, pred_test_1_labels)
table_mat_pred_test_1

accuracy_pred_test_1 <- sum(diag(table_mat_pred_test_1)) / sum(table_mat_pred_test_1)
print(paste('Accuracy for test', accuracy_pred_test_1))


# Graficamos la importancia de las variables en el modelo
importance_matrix <- xgb.importance(colnames(trainFea$data), model = xgb.fit)
importance_matrix2 <- subset(importance_matrix,importance_matrix$Frequency>0.01)
xgb.plot.importance(importance_matrix2, rel_to_first = TRUE, xlab = "Relative importance",cex=1)



#############
#############
#
# Curvas ROC - MultiniveL
#
#############
#############

# La construimos para el MODELO 1
#################################

pred_test_1
predict_test_p <- as.data.frame(pred_test_1)
predict_test_p$X1._pred_M1 <- pred_test_1[,1] 
predict_test_p$X2._pred_M1 <- pred_test_1[,2] 
predict_test_p$X3._pred_M1 <- pred_test_1[,3] 
predict_test_p$X4._pred_M1 <- pred_test_1[,4] 
predict_test_p <- predict_test_p[,-c(1:4)]

str(test_data)
test_data2 <- test_data[,-c(1)]
test_data2$POBREZA <- test_data2$POBREZA+1 
test_data2$POBREZA <- as.factor(test_data2$POBREZA)
str(test_data2)

true_label <- dummies::dummy(test_data2$POBREZA, sep = ".")
true_label <- data.frame(true_label)
colnames(true_label) <- gsub(".*?\\.", "", colnames(true_label))
colnames(true_label) <- paste(colnames(true_label), "_true")
df_final1 <- cbind(true_label, predict_test_p)

df_final1 <- data.frame(df_final1)
str(df_final1)

roc_res <- multi_roc(df_final1, force_diag=T)
pr_res <- multi_pr(df_final1, force_diag=T)

plot_roc_df <- plot_roc_data(roc_res)
plot_pr_df <- plot_pr_data(pr_res)


(ggplot2)
ggplot(plot_roc_df, aes(x = 1-Specificity, y=Sensitivity)) +
  geom_path(aes(color = Group, linetype=Method), size=1.5) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), 
               colour='grey', linetype = 'dotdash') +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5),
        legend.justification=c(1, 0), legend.position=c(.95, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=0.5, 
                                         linetype="solid", colour ="black"))


ggplot(plot_pr_df, aes(x=Recall, y=Precision)) + 
  geom_path(aes(color = Group, linetype=Method), size=1.5) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), 
        legend.justification=c(1, 0), legend.position=c(.95, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=0.5, 
                                         linetype="solid", colour ="black"))

str(pred_test_1_labels)
pred_test_1_labels <- data.frame(pred_test_1_labels)
colnames(pred_test_1_labels)
predict_test <- pred_test_1_labels
colnames(predict_test)
predict_test$POBREZA <- predict_test$pred_test_1_labels
colnames(predict_test)
predict_test <- predict_test[,-c(1)]
y_pred<-as.ordered(predict_test)
auc <- multiclass.roc(test_data2$POBREZA, y_pred, levels = c(1,2,3,4) )
print(auc)
