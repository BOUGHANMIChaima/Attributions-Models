
#~~~~~~~~~~~~~~ Régression logistique régularisé nfolds = 5  :::: ~~~~~~~~~~ 
##~~~~~~ I. Amélioration de la précision par feature engineerig ::::
###~~~~~~~  RL avec feature engineering (canal départ/dernier + fréquence des canaux dans le chemin
##### ~~~~     ) & train/test :

# Load libraries
library(data.table)
library(dplyr)
library(stringr)
library(caret)
library(lubridate)
library(Matrix)
library(glmnet)
library(pROC)
library(kableExtra)

# Charger les données
df <- read.csv("path/attribution data.csv")

# Trier les données par 'cookie' et 'time' ==> Ordre croissant
df <- df[order(df$cookie, df$time), ]

head(df) %>% kable() %>%
  kable_styling(bootstrap_options = "striped",
                full_width = F, 
                position="center")


# Convertir le data frame en un objet data.table
setDT(df)

# Appliquer les opérations sur les colonnes
df = df[order(cookie, time),time:=ymd_hms(time)][,id := seq_len(.N), by = cookie]


dt_wide = dcast(data = df, formula = cookie ~ id, value.var = "channel")
dt_wide = dt_wide[, path:=do.call(paste,c(.SD, sep=' > ')), .SDcols=-1]
dt_wide = dt_wide[, path:=word(path, 1, sep = " > NA")]

conversion = df[, .(conversion=sum(conversion), conversion_value=sum(conversion_value)), by=cookie]

setkey(conversion, cookie)
setkey(dt_wide, cookie)

dt_wide = merge(dt_wide, conversion)

dt_WIDE= dt_wide[, .(path, conversion, conversion_value)]

## Affichage
head(dt_WIDE) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped",
                full_width = F, 
                position="center")

data=dt_WIDE[, c("path","conversion")]

#~~~~~~~~

# 1. Découper les chemins en listes de canaux
data$path_split <- str_split(data$path, " > ")

# 2. Feature Engineering: Canal de départ et dernier canal
data$start_channel <- sapply(data$path_split, function(x) x[1])
data$end_channel <- sapply(data$path_split, function(x) x[length(x)])

# 3. Feature Engineering: Fréquence des canaux
unique_channels <- unique(unlist(data$path_split))
for (channel in unique_channels) {
  data[[paste0("freq_", channel)]] <- sapply(data$path_split, function(x) sum(x == channel))
}

# 4. Feature Engineering: Interactions entre canaux (toutes les interactions sans limitation)
data$channel_interactions <- sapply(data$path_split, function(x) paste(sort(unique(x)), collapse = " > "))

# Créer des variables binaires pour toutes les interactions
interaction_features <- unique(data$channel_interactions)
for (interaction in interaction_features) {
  data[[paste0("interaction_", interaction)]] <- as.integer(data$channel_interactions == interaction)
}

# 5. Encodage One-Hot pour le canal de départ et le dernier canal
data <- cbind(data, model.matrix(~ start_channel + end_channel - 1, data = data))

# 6. Préparation des données : Suppression des colonnes inutiles
data_clean <- data %>% 
  select(-c(path, path_split, channel_interactions, start_channel, end_channel))

# Convertir la variable de conversion en facteur pour le modèle
data_clean$conversion <- as.factor(data_clean$conversion)

# Nettoyage des noms de colonnes pour éviter les problèmes dans la modélisation
colnames(data_clean) <- make.names(colnames(data_clean))

# 7. Séparation en données d'entraînement et de test
set.seed(123)
trainIndex <- createDataPartition(data_clean$conversion, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)
train_data <- data_clean[trainIndex,]
test_data <- data_clean[-trainIndex,]

# 8. Création de matrices sparse pour accélérer le modèle
# Cela permet de gérer plus efficacement les variables avec beaucoup de zéros
train_matrix <- sparse.model.matrix(conversion ~ .-1, data = train_data)
test_matrix <- sparse.model.matrix(conversion ~ .-1, data = test_data)

# 9. Entraînement d'un modèle de régression logistique régularisée avec glmnet (plus rapide)
# alpha = 1 pour la régularisation LASSO (sélection de variables automatique)

#### execution_time <- system.time({ ###

model_glmnet <- cv.glmnet(train_matrix, as.numeric(train_data$conversion) - 1, 
                          family = "binomial", alpha = 1, 
                          type.measure = "auc", nfolds = 5)

# 10. Prédictions sur l'ensemble de test

predictions <- predict(model_glmnet, newx = test_matrix, type = "response", s = "lambda.min")

# Convertir les prédictions en classes binaires (0 ou 1)
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# 11. Évaluation des performances du modèle (Accuracy, AUC, etc.)
confusion_matrix <- confusionMatrix(as.factor(predicted_classes), test_data$conversion)
print(confusion_matrix)

# Calcul de l'AUC
roc_curve <- roc(as.numeric(test_data$conversion) - 1, as.vector(predictions))
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))

~#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## ~~~ # Fonction pour prédire la conversion à partir d'un chemin donné
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Fonction pour prédire la conversion à partir d'un chemin donné
  predict_conversion <- function(path, model_glmnet, train_data) {
    
    # Étape 1 : Nettoyer le chemin donné et appliquer le même Feature Engineering
    
    # Diviser le chemin en une liste de canaux
    path_split <- unlist(strsplit(path, " > "))
    
    # Canal de départ et dernier canal
    start_channel <- path_split[1]
    end_channel <- path_split[length(path_split)]
    
    # Calcul de la fréquence des canaux dans le chemin
    unique_channels <- colnames(train_data)[grepl("freq_", colnames(train_data))]
    
    # Créer un dataframe vide avec les mêmes colonnes que train_data
    new_data <- as.data.frame(matrix(0, nrow = 1, ncol = ncol(train_data) - 1))  # -1 pour exclure la colonne 'conversion'
    colnames(new_data) <- colnames(train_data)[-ncol(train_data)]
    
    # Remplir le dataframe avec les bonnes valeurs
    for (channel in unique_channels) {
      # Retirer le préfixe "freq_" du nom de la colonne pour obtenir le nom du canal
      channel_name <- gsub("freq_", "", channel)
      
      # Compter la fréquence d'apparition du canal dans le chemin
      new_data[, channel] <- sum(path_split == channel_name)
    }
    
    # Ajouter les interactions canaux si elles existent dans les données d'entraînement
    channel_interaction <- paste(sort(unique(path_split)), collapse = " > ")
    interaction_column <- paste0("interaction_", channel_interaction)
    if (interaction_column %in% colnames(new_data)) {
      new_data[, interaction_column] <- 1
    }
    
    # One-Hot Encoding pour le canal de départ et le dernier canal
    start_channel_col <- paste0("start_channel", start_channel)
    end_channel_col <- paste0("end_channel", end_channel)
    
    if (start_channel_col %in% colnames(new_data)) {
      new_data[, start_channel_col] <- 1
    }
    if (end_channel_col %in% colnames(new_data)) {
      new_data[, end_channel_col] <- 1
    }
    
    # Étape 2 : Forcer la Consistance des Colonnes entre train_data et new_data
    
    # Ajouter des colonnes manquantes dans new_data (les remplir avec des zéros)
    missing_cols <- setdiff(colnames(train_data)[-ncol(train_data)], colnames(new_data))
    for (col in missing_cols) {
      new_data[[col]] <- 0
    }
    
    # Supprimer les colonnes excédentaires dans new_data qui n'existent pas dans train_data
    extra_cols <- setdiff(colnames(new_data), colnames(train_data)[-ncol(train_data)])
    new_data <- new_data[, !(colnames(new_data) %in% extra_cols)]
    
    # Étape 3 : Conversion en matrice sparse
    tryCatch({
      new_matrix <- sparse.model.matrix(~ . -1, data = new_data)
      
      # Étape 4 : Faire la prédiction avec le modèle glmnet
      prediction <- predict(model_glmnet, newx = new_matrix, type = "response", s = "lambda.min")
      
      # Étape 5 : Décision finale de conversion ou non
      predicted_class <- ifelse(prediction > 0.5, 1, 0)
      
      return(predicted_class)
      
    }, error = function(e) {
      print("Erreur dans la prédiction :")
      print(e)
    })
  }

# Exemple d'utilisation
path_example <- "Instagram > Online Display > Online Display > Online Display"
result <- predict_conversion(path_example, model_glmnet, train_data)
print(paste("Prédiction pour le chemin:", path_example, "=> Conversion prédite:", result))

#~~~~~~~~~~ AUC = 0,58 !!!!!!!
#~~~~~~~~~~~

###~~~~~~~ Régression logistique standard avec + de feature engineering & train+test !!!


# Charger les données
df <- read.csv("path/attribution data.csv")

# Trier les données par 'cookie' et 'time' ==> Ordre croissant
df <- df[order(df$cookie, df$time), ]

head(df) %>% kable() %>%
  kable_styling(bootstrap_options = "striped",
                full_width = F, 
                position="center")


# Convertir le data frame en un objet data.table
setDT(df)

# Appliquer les opérations sur les colonnes
df = df[order(cookie, time),time:=ymd_hms(time)][,id := seq_len(.N), by = cookie]


dt_wide = dcast(data = df, formula = cookie ~ id, value.var = "channel")
dt_wide = dt_wide[, path:=do.call(paste,c(.SD, sep=' > ')), .SDcols=-1]
dt_wide = dt_wide[, path:=word(path, 1, sep = " > NA")]

conversion = df[, .(conversion=sum(conversion), conversion_value=sum(conversion_value)), by=cookie]

setkey(conversion, cookie)
setkey(dt_wide, cookie)

dt_wide = merge(dt_wide, conversion)

dt_WIDE= dt_wide[, .(path, conversion, conversion_value)]

## Affichage
head(dt_WIDE) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped",
                full_width = F, 
                position="center")

data=dt_WIDE[, c("path","conversion")]

# 2. Découpage des chemins en listes
data$path_split <- strsplit(as.character(data$path), " > ")

# 3. Profondeur du parcours (nombre d'étapes)
data$path_length <- sapply(data$path_split, length)

# 4. Canal de départ et dernier canal
data$start_channel <- sapply(data$path_split, function(x) x[1])
data$last_channel <- sapply(data$path_split, function(x) x[length(x)])

# 5. Fréquence des canaux
unique_channels <- unique(unlist(data$path_split))

# Créer des colonnes de fréquence pour chaque canal
for (channel in unique_channels) {
  data[[paste0("freq_", channel)]] <- sapply(data$path_split, function(x) sum(x == channel))
}

# 6. One-Hot Encoding pour les canaux de départ et dernier canal
one_hot_start <- model.matrix(~ start_channel - 1, data = data)
one_hot_last <- model.matrix(~ last_channel - 1, data = data)

# 7. Ajout des One-Hot Encodings au data.frame d'origine
data <- cbind(data, one_hot_start, one_hot_last)

# 8. Ajout de poids au dernier canal (poids = 1.5 pour la fréquence du dernier canal)
for (channel in unique_channels) {
  data[[paste0("weight_last_", channel)]] <- ifelse(data$last_channel == channel, 
                                                    1.5 * data[[paste0("freq_", channel)]], 
                                                    data[[paste0("freq_", channel)]])
}

# 9. Nettoyage des colonnes inutiles
columns_to_remove <- c("path", "path_split", "start_channel", "last_channel")
data <- data %>% select(-any_of(columns_to_remove))

# Vérification que data est toujours un data.frame
if (!is.data.frame(data)) {
  stop("Erreur: les données doivent rester un data.frame après nettoyage.")
}

# 10. Partionnement des données en Train et Test
set.seed(123)
train_index <- createDataPartition(data$conversion, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Assurez-vous que la variable de sortie 'conversion' est un facteur à deux niveaux
train_data$conversion <- as.factor(train_data$conversion)
test_data$conversion <- as.factor(test_data$conversion)

# Mesurer le temps d'exécution de l'entraînement du modèle de classification logistique
execution_time <- system.time({
  logistic_model <- train(conversion ~ ., data = train_data, method = "glm", family = "binomial")
})

# Afficher le temps d'exécution
print(paste("Temps d'exécution de l'entraînement du modèle (en secondes) :", execution_time['elapsed']))


# 12. Prédiction sur les données de test
predictions <- predict(logistic_model, newdata = test_data)

# 13. Évaluation du modèle
conf_matrix <- confusionMatrix(predictions, test_data$conversion)
print(conf_matrix)

# Vous pouvez également vérifier l'AUC si nécessaire
library(pROC)
roc_curve <- roc(test_data$conversion, as.numeric(predictions))
auc(roc_curve)  
                #### 0.5026 !!!!!

################## RL Standard + feature engineering + Train & Test + 
################   OVER SAMPLNG (ROSE) !!!!!!!!!!!!!

# Charger les données
df <- read.csv("path/attribution data.csv")

# Trier les données par 'cookie' et 'time' ==> Ordre croissant
df <- df[order(df$cookie, df$time), ]

# head(df) %>% kable() %>%
#   kable_styling(bootstrap_options = "striped",
#                 full_width = F, 
#                 position="center")


# Convertir le data frame en un objet data.table
setDT(df)

# Appliquer les opérations sur les colonnes
df = df[order(cookie, time),time:=ymd_hms(time)][,id := seq_len(.N), by = cookie]


dt_wide = dcast(data = df, formula = cookie ~ id, value.var = "channel")
dt_wide = dt_wide[, path:=do.call(paste,c(.SD, sep=' > ')), .SDcols=-1]
dt_wide = dt_wide[, path:=word(path, 1, sep = " > NA")]

conversion = df[, .(conversion=sum(conversion), conversion_value=sum(conversion_value)), by=cookie]

setkey(conversion, cookie)
setkey(dt_wide, cookie)

dt_wide = merge(dt_wide, conversion)

dt_WIDE= dt_wide[, .(path, conversion, conversion_value)]

# ## Affichage
# head(dt_WIDE) %>%
#   kable() %>%
#   kable_styling(bootstrap_options = "striped",
#                 full_width = F, 
#                 position="center")

data=dt_WIDE[, c("path","conversion")]

install.packages("ROSE")   # Pour l'oversampling
library(ROSE)

# 1. Découper les chemins en listes de canaux
data$path_split <- str_split(data$path, " > ")

# 2. Feature Engineering: Canal de départ et dernier canal
data$start_channel <- sapply(data$path_split, function(x) x[1])
data$end_channel <- sapply(data$path_split, function(x) x[length(x)])

# 3. Feature Engineering: Fréquence des canaux
unique_channels <- unique(unlist(data$path_split))
for (channel in unique_channels) {
  data[[paste0("freq_", channel)]] <- sapply(data$path_split, function(x) sum(x == channel))
}

# 4. Feature Engineering: Interactions entre canaux (toutes les interactions sans limitation)
data$channel_interactions <- sapply(data$path_split, function(x) paste(sort(unique(x)), collapse = " > "))

# Créer des variables binaires pour toutes les interactions
interaction_features <- unique(data$channel_interactions)
for (interaction in interaction_features) {
  data[[paste0("interaction_", interaction)]] <- as.integer(data$channel_interactions == interaction)
}

# 5. Encodage One-Hot pour le canal de départ et le dernier canal
data <- cbind(data, model.matrix(~ start_channel + end_channel - 1, data = data))

# 6. Préparation des données : Suppression des colonnes inutiles
data_clean <- data %>% 
  select(-c(path, path_split, channel_interactions, start_channel, end_channel))

# Convertir la variable de conversion en facteur pour le modèle
data_clean$conversion <- as.factor(data_clean$conversion)

# Nettoyage des noms de colonnes pour éviter les problèmes dans la modélisation
colnames(data_clean) <- make.names(colnames(data_clean))

# 7. Séparation en données d'entraînement et de test
set.seed(123)
trainIndex <- createDataPartition(data_clean$conversion, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)
train_data <- data_clean[trainIndex,]
test_data <- data_clean[-trainIndex,]

###### Réequilibrage des classes ::::::: AUC 0,56 NULLL !!!

# Effectuer l'oversampling sur les données d'entraînement
train_data_balanced <- ROSE(conversion ~ ., data = train_data, seed = 123)$data

# Vérifier la distribution après oversampling
table(train_data_balanced$conversion)

# Entraînement du modèle de régression logistique sur les données équilibrées
set.seed(123)
execution_time <- system.time({
  logistic_model <- train(conversion ~ ., data = train_data_balanced, method = "glm", family = "binomial")
})
  
# Résumé du modèle
summary(logistic_model)

####
# Prédictions sur l'ensemble de test
predictions <- predict(logistic_model, newdata = test_data)

# Conversion des probabilités en classes
predicted_classes <- ifelse(predictions == "1", 1, 0)


# Calcul et affichage de la courbe ROC
roc_curve <- roc(test_data$conversion, as.numeric(predicted_classes))
plot(roc_curve, main = "Courbe ROC")

# Affichage de l'AUC
auc(roc_curve)

### Validation croisée :: 
set.seed(123)
logistic_model_cv <- train(conversion ~ ., data = train_data_balanced, method = "glm", family = "binomial", 
                           trControl = trainControl(method = "cv", number = 10))

# Affichage du modèle avec validation croisée
print(logistic_model_cv)

# Prédictions sur l'ensemble de test
predictions <- predict(logistic_model_cv, newdata = test_data)

# Conversion des probabilités en classes
predicted_classes <- ifelse(predictions == "1", 1, 0)

# Calcul et affichage de la courbe ROC
roc_curve <- roc(test_data$conversion, as.numeric(predicted_classes))
plot(roc_curve, main = "Courbe ROC")

# Affichage de l'AUC
auc(roc_curve)

############ Random Forest avec ROSE :==> NUL == AUC = 0,56 !!! 

# Charger le package randomForest
library(randomForest)
library(caret)
library(ROSE)

# Remplacer les espaces dans les noms de colonnes pour éviter les erreurs de formule
colnames(train_data) <- make.names(colnames(train_data))
colnames(test_data) <- make.names(colnames(test_data))

# Assurez-vous que 'conversion' est un facteur
train_data$conversion <- as.factor(train_data$conversion)
test_data$conversion <- as.factor(test_data$conversion)

# Appliquer ROSE pour équilibrer les classes
set.seed(123)
balanced_data <- ROSE(conversion ~ ., data = train_data, seed = 123)$data

# Formation du modèle Random Forest sur les données équilibrées
set.seed(123)
rf_model <- randomForest(conversion ~ ., data = balanced_data, ntree = 100, mtry = 3)

# Affichage des résultats du modèle
print(rf_model)

# Faire des prédictions sur les données de test
predictions <- predict(rf_model, newdata = test_data)

# Évaluer les performances du modèle
conf_matrix <- confusionMatrix(predictions, test_data$conversion)
print(conf_matrix)

# Optionnel : Calculer AUC ROC
library(pROC)
roc_curve <- roc(test_data$conversion, as.numeric(predictions))
print(auc(roc_curve))

####################### XGBoost standard (sans features) avec Grid Search !!! ~~~~~~~~~~~~~~~
#####   ~~~~~~~~~~~~~~~~ ça n'a pas marché !!!!!!!!!!!!!
 
# Charger les packages nécessaires
library(caret)
library(xgboost)
library(Matrix)
library(pROC)


# Charger les données
df <- read.csv("path/attribution data.csv")

# Trier les données par 'cookie' et 'time' ==> Ordre croissant
df <- df[order(df$cookie, df$time), ]

# Convertir le data frame en un objet data.table
setDT(df)

# Appliquer les opérations sur les colonnes
df = df[order(cookie, time),time:=ymd_hms(time)][,id := seq_len(.N), by = cookie]


dt_wide = dcast(data = df, formula = cookie ~ id, value.var = "channel")
dt_wide = dt_wide[, path:=do.call(paste,c(.SD, sep=' > ')), .SDcols=-1]
dt_wide = dt_wide[, path:=word(path, 1, sep = " > NA")]

conversion = df[, .(conversion=sum(conversion), conversion_value=sum(conversion_value)), by=cookie]

setkey(conversion, cookie)
setkey(dt_wide, cookie)

dt_wide = merge(dt_wide, conversion)

dt_WIDE= dt_wide[, .(path, conversion, conversion_value)]

data=dt_WIDE[, c("path","conversion")]

###### 
# 1. Découper les chemins en listes de canaux
data$path_split <- str_split(data$path, " > ")

# 2. Feature Engineering: Canal de départ et dernier canal
data$start_channel <- sapply(data$path_split, function(x) x[1])
data$end_channel <- sapply(data$path_split, function(x) x[length(x)])

# 3. Feature Engineering: Fréquence des canaux
unique_channels <- unique(unlist(data$path_split))
for (channel in unique_channels) {
  data[[paste0("freq_", channel)]] <- sapply(data$path_split, function(x) sum(x == channel))
}

# 4. Feature Engineering: Interactions entre canaux (toutes les interactions sans limitation)
data$channel_interactions <- sapply(data$path_split, function(x) paste(sort(unique(x)), collapse = " > "))

# Créer des variables binaires pour toutes les interactions
interaction_features <- unique(data$channel_interactions)
for (interaction in interaction_features) {
  data[[paste0("interaction_", interaction)]] <- as.integer(data$channel_interactions == interaction)
}

# 5. Encodage One-Hot pour le canal de départ et le dernier canal
data <- cbind(data, model.matrix(~ start_channel + end_channel - 1, data = data))

# 6. Préparation des données : Suppression des colonnes inutiles
data_clean <- data %>% 
  select(-c(path, path_split, channel_interactions, start_channel, end_channel))

# Convertir la variable de conversion en facteur pour le modèle
data_clean$conversion <- as.factor(data_clean$conversion)

# Nettoyage des noms de colonnes pour éviter les problèmes dans la modélisation
colnames(data_clean) <- make.names(colnames(data_clean))

#########################

data= data_clean
# Étape 1 : Préparation des données
# Convertir 'conversion' en facteur et nettoyer les niveaux
data$conversion <- as.factor(data$conversion)
levels(data$conversion) <- make.names(levels(data$conversion), unique = TRUE)

# Étape 2 : Diviser les données en ensemble d'entraînement et de test
set.seed(123)
train_index <- createDataPartition(data$conversion, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Étape 3 : Conversion des données en matrices pour XGBoost
train_matrix <- sparse.model.matrix(conversion ~ . - 1, data = train_data)
test_matrix <- sparse.model.matrix(conversion ~ . - 1, data = test_data)

# Labels pour XGBoost
train_labels <- as.numeric(train_data$conversion) - 1
test_labels <- as.numeric(test_data$conversion) - 1

# Créer des DMatrix pour XGBoost
dtrain <- xgb.DMatrix(data = train_matrix, label = train_labels)
dtest <- xgb.DMatrix(data = test_matrix, label = test_labels)

# Étape 4 : Définir la grille des hyperparamètres pour la recherche sur grille
grid <- expand.grid(
  nrounds = c(100, 200),  # Nombre d'itérations
  max_depth = c(4, 6, 8),
  eta = c(0.01, 0.1, 0.3),  # Taux d'apprentissage
  gamma = c(0, 1, 5),  # Régularisation
  colsample_bytree = c(0.5, 0.7, 1),
  min_child_weight = c(1, 5, 10),
  subsample = c(0.5, 0.7, 1)
)

# Étape 5 : Configuration de la validation croisée avec caret
control <- trainControl(
  method = "cv", 
  number = 5, 
  verboseIter = TRUE, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

# Étape 6 : Entraîner le modèle XGBoost avec Grid Search
xgb_model <- train(
  x = train_matrix,
  y = as.factor(train_labels),
  trControl = control,
  tuneGrid = grid,
  method = "xgbTree",
  metric = "ROC",
  verbose = TRUE
)

################################### XGBoost avec Feature engineering :::::::

# Charger les données
df <- read.csv("path/attribution data.csv")

# Trier les données par 'cookie' et 'time' ==> Ordre croissant
df <- df[order(df$cookie, df$time), ]

# Convertir le data frame en un objet data.table
setDT(df)

# Appliquer les opérations sur les colonnes
df = df[order(cookie, time),time:=ymd_hms(time)][,id := seq_len(.N), by = cookie]


dt_wide = dcast(data = df, formula = cookie ~ id, value.var = "channel")
dt_wide = dt_wide[, path:=do.call(paste,c(.SD, sep=' > ')), .SDcols=-1]
dt_wide = dt_wide[, path:=word(path, 1, sep = " > NA")]

conversion = df[, .(conversion=sum(conversion), conversion_value=sum(conversion_value)), by=cookie]

setkey(conversion, cookie)
setkey(dt_wide, cookie)

dt_wide = merge(dt_wide, conversion)

dt_WIDE= dt_wide[, .(path, conversion, conversion_value)]

data=dt_WIDE[, c("path","conversion")]

###########################

# Découper la colonne Path en une liste de canaux
data$path_split <- strsplit(as.character(data$Path), " > ")

# Profondeur du parcours (nombre d'étapes)
data$path_length <- sapply(data$path_split, length)

# Canal de départ et dernier canal
data$start_channel <- sapply(data$path_split, function(x) x[1])
data$last_channel <- sapply(data$path_split, function(x) x[length(x)])

# Fréquence des canaux
unique_channels <- unique(unlist(data$path_split))

# Créer des colonnes de fréquence pour chaque canal
for (channel in unique_channels) {
  data[[paste0("freq_", channel)]] <- sapply(data$path_split, function(x) sum(x == channel))
}

# One-Hot Encoding pour les canaux de départ et dernier canal
one_hot_start <- model.matrix(~ start_channel - 1, data = data)
one_hot_last <- model.matrix(~ last_channel - 1, data = data)

# Ajouter les encodages au data.frame
data <- cbind(data, one_hot_start, one_hot_last)

# Ajouter de nouveaux poids au dernier canal
for (channel in unique_channels) {
  data[[paste0("weight_last_", channel)]] <- ifelse(data$last_channel == channel, 1.5 * data[[paste0("freq_", channel)]], data[[paste0("freq_", channel)]])
}

# Supprimer les colonnes non nécessaires
columns_to_remove <- c("Path", "path_split", "start_channel", "last_channel")
data_clean <- data %>% select(-all_of(columns_to_remove))

# Convertir en matrices pour XGBoost
train_index <- createDataPartition(data_clean$Conversion, p = 0.8, list = FALSE)
train_data <- data_clean[train_index, ]
test_data <- data_clean[-train_index, ]

# Convertir en matrices éparses
train_matrix <- sparse.model.matrix(Conversion ~ . - 1, data = train_data)
test_matrix <- sparse.model.matrix(Conversion ~ . - 1, data = test_data)

# Labels pour XGBoost
train_labels <- as.numeric(train_data$Conversion) - 1
test_labels <- as.numeric(test_data$Conversion) - 1

# Créer des DMatrix pour XGBoost
dtrain <- xgb.DMatrix(data = train_matrix, label = train_labels)
dtest <- xgb.DMatrix(data = test_matrix, label = test_labels)

# Définir la grille des hyperparamètres
grid <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(4, 6, 8),
  eta = c(0.01, 0.1, 0.3),
  gamma = c(0, 1, 5),
  colsample_bytree = c(0.5, 0.7, 1),
  min_child_weight = c(1, 5, 10),
  subsample = c(0.5, 0.7, 1)
)

# Configuration de la validation croisée
control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Entraîner le modèle XGBoost avec Grid Search
xgb_model <- train(
  x = train_matrix,
  y = as.factor(train_labels),
  trControl = control,
  tuneGrid = grid,
  method = "xgbTree",
  metric = "ROC",
  verbose = TRUE
)

# Afficher les meilleurs hyperparamètres trouvés
print(xgb_model$bestTune)

# Faire des prédictions sur les données de test
predictions <- predict(xgb_model, newdata = test_matrix)

# Évaluation des performances
conf_matrix <- confusionMatrix(predictions, as.factor(test_labels))
print(conf_matrix)

# Courbe ROC et AUC
roc_curve <- roc(test_labels, as.numeric(predictions))
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))









