# Cargar librerías necesarias
library(MASS) # Para LDA y QDA
library(caret) # Para métricas de clasificación y matriz de confusión

# Cargamos los datos de las especies, que hemos separado en 3 ficheros distintos
getwd()
setwd("C:/Users/belen/OneDrive/Escritorio")

######################################################################################
######################################################################################
# Importación y tratamiento de datos
######################################################################################
######################################################################################


# Cargar los datos, separados en 3 archivos para facilitar su lectura en el programa
especie1 <- read.table("especie1.txt", header = TRUE, sep="", strip.white = TRUE)
View(especie1)
especie2 <- read.table("especie2.txt", header = TRUE, sep="", strip.white = TRUE)
View(especie2)
especie3 <- read.table("especie3.txt", header = TRUE, sep="", strip.white = TRUE)
View(especie3)

# Añadir una columna que indique al tipo de especie que representa codificando mediante:
# 1=especie1, 2=especie2, 3=especie3
especie1$Species <- "1"
especie2$Species <- "2"
especie3$Species <- "3"

# Combinamos los datos en un único conjunto de datos, creamos así un dataframe con los datos
data <- rbind(especie1, especie2, especie3)
str(data())
class(data) # es un dataframe

# Convertir la variable Species en factor
data$Species <- as.factor(data$Species)

# Nuevas observaciones
nuevas_obs <- data.frame(
  X1 = c(4.6, 6.8, 7.2),
  X2 = c(3.6, 2.8, 3.2),
  X3 = c(1.0, 4.8, 6.0),
  X4 = c(0.2, 1.4, 1.8)
)

#####################################################################################
#####################################################################################
# (i) Clasificación con función discriminante lineal (LDA)
#####################################################################################
#####################################################################################

lda_model <- lda(Species ~ X1 + X2 + X3 + X4, data = data)
lda_pred <- predict(lda_model, nuevas_obs)

# Análisis de discriminantes de LDA
cat("\nAnálisis de discriminantes de LDA:\n")
print(lda_model$scaling)
cat("Los discriminantes de LDA indican las combinaciones lineales de las variables predictoras que maximizan la separación entre clases. Sirven para comprender cómo las variables contribuyen a la clasificación.\n")

# Matriz de confusión para LDA
lda_pred_all <- predict(lda_model, data)$class
lda_conf_matrix <- confusionMatrix(lda_pred_all, data$Species)
cat("\nMatriz de confusión para LDA:\n")
print(lda_conf_matrix$table)

#####################################################################################
#####################################################################################
# (ii) Clasificación con función discriminante cuadrática (QDA)
#####################################################################################
#####################################################################################

qda_model <- qda(Species ~ X1 + X2 + X3 + X4, data = data)
qda_pred <- predict(qda_model, nuevas_obs)

# Análisis de discriminantes de QDA
cat("\nAnálisis de discriminantes de QDA:\n")
cat("El modelo de QDA considera la varianza y covarianza específicas de cada clase, por lo que no utiliza escalamiento lineal.\n")

# Inspeccionar las medias por clase
cat("\nMedias de las variables predictoras por clase:\n")
print(qda_model$means)

# Inspeccionar matrices de covarianza por clase
cat("\nMatrices de covarianza por clase:\n")
print(qda_model$scaling)
cat("Estas matrices representan la estructura de covarianza de las variables dentro de cada clase, lo que permite clasificaciones más flexibles.\n")


# Matriz de confusión para QDA
qda_pred_all <- predict(qda_model, data)$class
qda_conf_matrix <- confusionMatrix(qda_pred_all, data$Species)
cat("\nMatriz de confusión para QDA:\n")
print(qda_conf_matrix$table)

######################################################################################
# Métricas principales para evaluar modelos
######################################################################################
cat("\nMétricas principales:\n")
cat("1. Exactitud (accuracy): Proporción de observaciones clasificadas correctamente.\n")
cat("   LDA Exactitud: ", lda_conf_matrix$overall["Accuracy"], "\n")
cat("   QDA Exactitud: ", qda_conf_matrix$overall["Accuracy"], "\n")
cat("2. Sensibilidad y especificidad: Miden el rendimiento para cada clase.\n")
cat("3. Matriz de confusión: Detalla la comparación entre las predicciones y las clases reales.\n")
cat("4. Valores discriminantes: Ayudan a interpretar las contribuciones de cada variable.\n")


######################################################################################
######################################################################################
# Comparativa de los modelos
######################################################################################
######################################################################################

# Crear una tabla comparativa
resultado_tabla <- data.frame(
  LDA_Prediccion = lda_pred$class,
  QDA_Prediccion = qda_pred$class,
  X1 = nuevas_obs$X1,
  X2 = nuevas_obs$X2,
  X3 = nuevas_obs$X3,
  X4 = nuevas_obs$X4
)

# Mostrar la tabla
cat("\nTabla comparativa de resultados:\n")
print(resultado_tabla)

# Representación gráfica de las funciones discriminantes (para LDA)
plot(lda_model, col = as.numeric(data$Species), pch = 19, main = "LDA: Separación entre clases")

#Representación gráfica de las predicciones:
plot(data$X1, data$X2, col = as.numeric(data$Species), pch = 19, main = "Clasificación QDA")
points(nuevas_obs$X1, nuevas_obs$X2, col = "red", pch = 4, cex = 2)

#Kappa: Proporciona una medida de la concordancia ajustada al azar
cat("   LDA Kappa: ", lda_conf_matrix$overall["Kappa"], "\n")
cat("   QDA Kappa: ", qda_conf_matrix$overall["Kappa"], "\n")

# Identificar los errores de clasificación y explorar patrones comunes:
errores_lda <- data[data$Species != lda_pred_all, ]
errores_qda <- data[data$Species != qda_pred_all, ]
cat("\nErrores de clasificación LDA:\n")
print(errores_lda)
cat("\nErrores de clasificación QDA:\n")
print(errores_qda)

install.packages("heplots")
library(heplots)
boxM(data[, 1:4], data$Species)


# Comentarios sobre los análisis
# La función discriminante lineal (LDA) asume que las covarianzas de las clases son iguales. Por tanto, los resultados son más robustos cuando esta suposición es válida.
# La función discriminante cuadrática (QDA) permite diferentes covarianzas para cada clase, lo que la hace más flexible, pero también más sensible a variaciones en los datos.
# Comparando las predicciones de LDA y QDA, se puede observar si las suposiciones de covarianza homogénea o heterogénea tienen un impacto significativo en las clasificaciones.
# Las características de las observaciones nuevas (x1, x2, x3, x4) pueden dar pistas sobre por qué fueron clasificadas de manera diferente.
