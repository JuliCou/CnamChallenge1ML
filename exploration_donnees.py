#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 09:48:53 2019

@author: courgibet
"""
import pandas as pd
from scipy.stats import normaltest, shapiro, f_oneway, kruskal, chi2_contingency
# from matplotlib.pyplot import boxplot, figure, hist, plot, ylabel,\
#                               xlabel, title, text, xticks, bar, savefig
import matplotlib.pyplot as plt
import fpdf
import os
import warnings
# Cette instruction est à éviter...
warnings.filterwarnings("ignore")


def descQuanti(dataframe, var, risque, f_pdf, nb):
    """
    Fonction permettant l'étude complète d'une variable quantitative.
    Avec comme paramètres en entrée :
        un dataframe ;
        une variable quantitative (chaine de caractères ;
        correspondant au nom d'une colonne du dataframe)
        le nombre de figures déjà affichées ;
        une probabilité entre 0 et 1.
        
    Elle calcule et retourne dans un dictionnaire
    
    la moyenne, l'écart-type, la médiane, le min et le max.

    Elle affiche un histogramme de distribution
    (+ comparaison à une gaussienne) et une boite à moustache.

    La fonction réalise aussi un test de normalité.
    """
    donnees = dataframe[var]

    analyse = {}
    analyse["moyenne"] = donnees.mean()
    analyse["ecart-type"] = donnees.std()
    analyse["mediane"] = donnees.median()
    analyse["minimum"] = donnees.min()
    analyse["maximum"] = donnees.max()
    
    texte = "La variable " + var + " est quantitative : "
    print(texte)
    # Ecriture fichier pdf
    f_pdf.write(5, texte)
    f_pdf.ln()
    
    for key, value in analyse.items():
        texte = key + " : " + str(value)
        print(texte)
        # Ecriture fichier pdf
        f_pdf.write(5, texte)
        f_pdf.ln()
    
    # Histogramme
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
    
    # hist(x=donnees, density=True)
    donnees.plot(kind="kde")
    plt.ylabel("Densité")
    plt.xlabel("Valeurs")
    plt.title("Distribution de la variable " + var)

    # Test de normalité
    if risque < 0 or risque > 1:
        texte = "Le risque en entrée n'est pas une probabilité."
        print(texte)
    else:
        k2, p_pearson = normaltest(donnees)
        W, p_shapiro = shapiro(donnees)
        if p_pearson < risque and p_shapiro < risque:
            texte = "La variable ne peut pas être considérée comme normale au risque " + str(risque)
            print(texte)
        elif p_pearson < risque or p_shapiro < risque:
            texte = "Selon la méthode (Shapiro vs Pearson), la variable peut être considérée comme normale."
            print(texte)
        else:
            texte = "La variable est considérée comme normale au risque " + str(risque)
            print(texte)

    # Ajout p-valeur sur histogramme
    texte_hist = "Normalité (p-valeur) : " + str(p_pearson)
    plt.text(0.6, 0.6, texte_hist, transform=ax.transAxes)
    
    # Ecriture fichier pdf
    # Image
    p = directory + "image" + str(nb) + ".png"
    plt.savefig(p)
    f_pdf.image(p, w=100)

    # Texte
    f_pdf.write(5, "p-valeur du test de Normalité de Pearson : " + str(p_pearson))
    f_pdf.ln()
    f_pdf.write(5, "p-valeur du test de Normalité de Shapiro : " + str(p_shapiro))
    f_pdf.ln()
    f_pdf.write(5, texte + "\n")
    
    # Boite à moustache
    fig = plt.figure()
    fig.add_axes([0.15, 0.15, 0.75, 0.75])
    plt.boxplot(donnees)
    plt.ylabel("Valeurs")
    plt.xticks([1], [var])
    plt.title("Boîte à moustache de " + var)
    
    # Fichier pdf
    p = directory + "image" + str(nb+1) + ".png"
    plt.savefig(p)
    f_pdf.image(p, w=100)

    return analyse, f_pdf, nb+2


def descQuali(dataframe, var, f_pdf, nb):
    """
    Fonction permettant l'étude complète d'une variable qualitative.
    Avec comme paramètres en entrée :
        un dataframe ;
        une chaine de caractères (nom d'une variable
        d'une colonne du dataframe)
        le nombre de figures déjà affichées.

    Elle calcule et affiche la table d'effectifs et de pourcentages.

    Elle affiche un diagramme en barres.
    """
    donnees = dataframe[var]
    texte = "La variable " + var + " est qualitative :"
    print(texte)

    # Ecriture fichier pdf
    f_pdf.write(5, texte + "\n")

    # Tables d'effectifs et de pourcentage
    t = pd.crosstab(donnees, "freq")
    t_n = pd.crosstab(donnees, "freq", normalize=True)

    # Pandas dataframe des tables à afficher
    crosstables= [t, t_n]
    ct = pd.concat(crosstables, axis=1)
    ct.columns = ["frequence", "frequence norm"]

    modalites = list(ct.index.values)

    if len(modalites)>20:
        print("Trop nombreuses")
        return f_pdf, nb
    else:
        # Enregistrement fichier pdf
        texte = "modalités | fréquence | fréquence normalisée \n"
        for mod, val_1, val_2 in zip(modalites, ct["frequence"], ct["frequence norm"]):
            texte = mod + " | " + str(val_1) + " | " + str(round(val_2, 4))
        f_pdf.write(5, texte + "\n")

        # Affichage des résultats
        print(ct)
        fig = plt.figure()
        fig.add_axes([0.15, 0.15, 0.75, 0.75])
        ct["frequence"].plot.bar()
    
        # Fichier pdf
        p = directory + "image" + str(nb) + ".png"
        plt.savefig(p)
        f_pdf.image(p, w=100)
    
        return f_pdf, nb+1


def lienQuantiQuanti(dataframe, var, f_pdf, nb):
    """
    Fonction qui étudie les corrélations entre une variable quantitative
    nommée en entrée et le reste des variables quantitatives du dataframe.
    
    Elle affiche les coefficients de corrélation entre la variable d'entrée
    et le reste des variables quantitatives.
    
    Elle trace un graphique nuage de points entre la variable d'entrée et
    la variable d'intérêt "price".
    """
    texte = "Coefficient de corrélation entre les variables quantitatives et " + var + " :"
    print(texte)
    f_pdf.write(5, texte + "\n")
    
    # Tableau de corrélation et affichage
    tableau_corr = dataframe.corr()
    col_1 = []
    col_2 = []
    for elt in tableau_corr:
        if elt != var:
            col_1.append(elt)
            col_2.append(tableau_corr.at[var, elt])
            texte = elt + " : " + str(round(tableau_corr.at[var, elt], 3))
            print(texte)
            f_pdf.write(5, texte + "\n")
    
    df = pd.DataFrame({"col_1" : col_1, "col_2" : col_2})
    
    # Diagramme en barres
    fig = plt.figure()
    fig.add_axes([0.15, 0.15, 0.75, 0.75])
    df["col_2"].plot.bar()
    plt.title("Diagramme en barre des corrélations de la variable " + var)
    plt.xticks(list(range(len(col_1))), col_1)
    
    # Ecriture pdf
    p = directory + "image" + str(nb) + ".png"
    plt.savefig(p)
    f_pdf.image(p, w=100)
    
    return f_pdf, nb+1


def lienQuantiQuali(dataframe, var, var_interet, risque, f_pdf, nb):
    """
    Fonction qui étudie la corrélation entre une variable quantitative
    nommée en entrée et la variable qualitative d'intérêt.
    
    La fonction réalise un test de la variance ANOVA, puis affiche une boîte
    à moustache pour chaque modalité de la variable qualitative donnée en entrée.
    """
    
    # Modalités
    modalites = dataframe[var_interet].unique()
    
    # ANOVA
    donnees_modales = [dataframe[var][dataframe[var_interet] == m] for m in modalites]
    statistique, p_val = f_oneway(*donnees_modales)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
    # Required hypothesis for f_oneway :
    # 2. Each sample is from a normally distributed population.
    # It is not true here...
    # Un autre test si distribution non normale :
    s_2, p_2 = kruskal(*donnees_modales)
    texte = "Statistique ANOVA normalité : " + str(statistique) + " p-valeur : " + str(p_val)
    texte += "\nStatistique ANOVA non normalité : "
    texte +=  str(s_2) + " p-valeur : " + str(p_2)
    print(texte)
    f_pdf.write(5, texte + "\n")
    
    if risque < 0 or risque > 1:
        texte = "Le risque en entrée n'est pas une probabilité."
    else:
        # On rejete H0
        if p_val < risque:
            texte = "La modalité n'a pas d'influence sur la variable " + var
            texte += " au risque " + str(risque)
        # On accepte H0
        else:
            texte = "La modalité a une influence sur la variable " + var
            texte += " au risque " + str(risque)
    print(texte)
    f_pdf.write(5, texte)
    f_pdf.ln()

    # Boite à moustache
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
    plt.boxplot(donnees_modales)
    plt.ylabel("Valeurs")
    plt.xticks(list(range(1,len(modalites)+1)), modalites)
    plt.title("Boîte à moustache des modalités de la variable " + var_interet + "selon " + var)
    texte_boxplot = "ANOVA (p-valeur) : " + str(p_val)
    plt.text(0.6, 0.6, texte_boxplot, transform=ax.transAxes)

    # Fichier pdf
    p = directory + "image" + str(nb) + ".png"
    plt.savefig(p)
    f_pdf.image(p, w=100)
    
    return f_pdf, nb+1


def lienQualiQuali(dataframe, var, var_interet, risque, f_pdf, nb):
    """
    Fonction qui étudie la corrélation entre une variable qualitative
    nommée en entrée et la variable qualitative d'intérêt.
    
    La fonction réalise un test de la variance ANOVA, puis affiche une boîte
    à moustache pour chaque modalité de la variable qualitative donnée en entrée.
    """
    # Tableau croisé
    t_croise = pd.crosstab(dataframe[var_interet], dataframe[var], margins = True)
    
    # Enregistrement fichier pdf
    modalites_x = list(t_croise.index.values)
    modalites_y = list(t_croise.columns.values)
    texte = "Il y a " + str(len(modalites_y)) + " modalités."
    
    print("1")
    if len(modalites_y) > 20:
        print("2")
        f_pdf.write(5, texte + "\n")
        return f_pdf, nb
    else:
        print("3")
        texte += "Les modalites de la variable " + var
        f_pdf.write(5, texte + "\n")
        for mod_y in modalites_y:
            texte += mod_y + " | " 
        
        
        texte_t = "modalites | "
        for mod_y in modalites_y:
            texte_t += mod_y + " | "
        texte_t += "\n"
        for x, mod_x in enumerate(modalites_x):
            txt = mod_x + " | "
            for y, mod_y in enumerate(modalites_y):
                txt += str(round(t_croise.at[mod_x, mod_y], 4)) + " | "
            texte_t += txt + "\n"
        
        f_pdf.write(5, texte_t + "\n")
    
        # Test d'indépendance Chi-2
        chi2, p, dof, ex = chi2_contingency(t_croise)
        if p < risque:
            texte = "Hypothèse nulle : indépendance des 2 variables"
        else:
            texte = "Hypothèse nulle rejetée : il n'y a pas indépendance entre "
            texte +=  var_interet + " et " + var
        f_pdf.write(5, texte + "\n")
    
        # Diagramme en barres
        fig = plt.figure()
        fig.add_axes([0.15, 0.15, 0.75, 0.75])
        t_croise.plot.bar()
        plt.title("Diagramme en barre des données croisées " + var)
        
        # Fichier pdf
        p = directory + "image" + str(nb) + ".png"
        plt.savefig(p)
        f_pdf.image(p, w=100)
    
        return f_pdf, nb+1


if __name__ == '__main__':
    path_train = 'train.csv'
    path_test = 'test.csv'
    train = pd.read_csv('train.csv', header=0, sep=",")
    test = pd.read_csv(path_test, header=0, sep=",")

    # Fichiers résultats PDF
    directory = "pictures/"
    fichier = "resultat.pdf"
    pdf = fpdf.FPDF()
    pdf.set_font("Arial", size=12)
    
    
    liste_caracteristiques = list(train.columns)
    risque = 0.05
    var_interet = "status_group" # Variable qualitative
    nb_pic = 0
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for variable in liste_caracteristiques:
        try:
            # Pour les variables quantitatives
            pdf.add_page()
            analyse, pdf, nb_pic = descQuanti(train, variable, risque, pdf, nb_pic)
            print("\n\n")
            # Pas d'analyse pour la variable d'intérêt
            if variable != var_interet:
                pdf.add_page()
                # Lien entre les autres variables quantitatives
                pdf, nb_pic = lienQuantiQuanti(train, variable, pdf, nb_pic)
                # Lien avec la variable d'intérêt qualitative
                pdf, nb_pic = lienQuantiQuali(train, variable, var_interet, risque, pdf, nb_pic)
            print("\n\n")
        except:
            try:
                # Pour les variables qualitatives - dont variable d'intérêt
                pdf, nb_pic = descQuali(train, variable, pdf, nb_pic)
                print("\n\n")
                pdf.add_page()
                if variable != var_interet:
                    pdf, nb_pic = lienQualiQuali(train, variable, var_interet, risque, pdf, nb_pic)
                print("\n\n")
            except:
                print("Variable booléenne")
    # Ecriture des données dans le fichier pdf
    pdf.output(fichier)
    
    # Suppression images temporaires
    for fs in os.listdir(directory):
        file_path = os.path.join(directory, fs)
        os.remove(file_path)
    os.rmdir(directory)
