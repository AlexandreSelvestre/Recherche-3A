library(COUNT)
data(mdvis)

# Extraire les noms de colonnes
col_names <- colnames(mdvis)

# Enregistrer le tableau mdvis au format CSV avec les noms de colonnes
write.csv(mdvis, file = "mdvis.csv", row.names = FALSE)

# Lire le fichier CSV avec les noms de colonnes
write.table(mdvis, file="mdvis.csv", sep=",", col.names=col_names, row.names=FALSE)