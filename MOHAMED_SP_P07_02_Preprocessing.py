#!/usr/bin/env python
# coding: utf-8

# # Preprocessing

# Ce notebook comprendra le nettoyage et le feature engeneering sur la base des conclusions de l'EDA 
#     

# In[1]:


# chargement librairies
import numpy as np
import pandas as pd 
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
import sys
import time
from datetime import datetime
# Warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


def load_all_tables(directory_path='', verbose=True):
    '''
    Function to load all the tables required
    Input:
        directory_path: str, default = ''
            Path of directory in which tables are stored in
        verbose: bool, default = True
            Whether to keep verbosity or not
    '''
    if verbose:
        print("Chargement des jeux de donnees...")
        print("--------------------------------------------------------------------")
        start = datetime.now()

    application_train = pd.read_csv(directory_path + 'application_train.csv')
    if verbose:
        print("Fichier application_train.csv chargé -> dataframe : application_train")

    application_test = pd.read_csv(directory_path + 'application_test.csv')
    if verbose:
        print("Fichier application_test.csv chargé -> dataframe : application_test")

    bureau = pd.read_csv(directory_path + 'bureau.csv')
    if verbose:
        print("Fichier bureau.csv chargé -> dataframe : bureau")

    bureau_balance = pd.read_csv(directory_path + 'bureau_balance.csv')
    if verbose:
        print("Fichier bureau_balance.csv chargé -> dataframe : bureau_balance")

    cc_balance = pd.read_csv(directory_path + 'credit_card_balance.csv')
    if verbose:
        print("Fichier credit_card_balance.csv chargé -> dataframe : cc_balance")

    installments_payments = pd.read_csv(
        directory_path + 'installments_payments.csv')
    if verbose:
        print(
            "Fichier installments_payments.csv chargé -> dataframe : installments_payments")

    POS_CASH_balance = pd.read_csv(directory_path + 'POS_CASH_balance.csv')
    if verbose:
        print("Fichier POS_CASH_balance.csv chargé -> dataframe : POS_CASH_balance")

    HomeCredit_columns_description = pd.read_csv(
        directory_path +
        'HomeCredit_columns_description.csv',
        encoding='cp1252')
    del HomeCredit_columns_description['Unnamed: 0']
    if verbose:
        print("Fichier HomeCredit_columns_description.csv chargé -> dataframe : HomeCredit_columns_description")

    previous_application = pd.read_csv(
        directory_path + 'previous_application.csv')
    if verbose:
        print("Fichier previous_application.csv chargé -> dataframe : previous_application")

    if verbose:
        print("--------------------------------------------------------------------")
        print(
            f'Chargement des 9 jeux de donnees termineéen {datetime.now() - start} secondes')

    return application_train, application_test, bureau, bureau_balance,         cc_balance, installments_payments, POS_CASH_balance, previous_application,         HomeCredit_columns_description


# In[3]:


path=r'C:\\Users\\P7\\'
application_train, application_test, bureau, bureau_balance, cc_balance,     installments_payments, POS_CASH_balance, previous_application,     HomeCredit_columns_description =load_all_tables(path)


# ### traintement fichier principal : application_train

# In[4]:


application_train.shape


# In[5]:


# variables quanti
cols_num=application_train.select_dtypes(include=[np.number]).columns.to_list()


# In[6]:


# variables quali
cols_cat=application_train.select_dtypes(exclude=[np.number]).columns.to_list()


# In[7]:


# reduction mémoire 
# # source: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(data, verbose=True):
   
    '''
    This function is used to reduce the memory usage by converting the datatypes of a pandas
    DataFrame withing required limits.
    '''

    start_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('-' * 79)
        print('Memory usage du dataframe: {:.2f} MB'.format(start_mem))

    for col in data.columns:
        col_type = data[col].dtype

        #  Float et int
        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(
                        np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)
            else:
                if c_min > np.finfo(
                        np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

        # # Boolean : pas à faire car pour machine learning il faut des int 0/1
        # et pas False/True
        # if list(data[col].unique()) == [0, 1] or list(data[col].unique()) == [1, 0]:
        #     data[col] = data[col].astype(bool)

    end_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage après optimization: {:.2f} MB'.format(end_mem))
        print('Diminution de {:.1f}%'.format(
            100 * (start_mem - end_mem) / start_mem))
        print('-' * 79)

    return data


# In[8]:


reduce_mem_usage(application_train, verbose=True)


# In[9]:


cols_num


# In[10]:


# transformer les variables dans la bonne catégorie
application_train['REGION_RATING_CLIENT'].unique()


# In[11]:


application_train['REGION_RATING_CLIENT']=application_train['REGION_RATING_CLIENT'].astype('object')


# In[12]:


application_train['REGION_RATING_CLIENT_W_CITY'].unique()


# In[13]:


application_train['REGION_RATING_CLIENT_W_CITY']=application_train['REGION_RATING_CLIENT_W_CITY'].astype('object')


# In[14]:


cols_cat


# In[15]:


#correction Gender
application_train['CODE_GENDER'].unique()


# In[16]:


application_train[application_train['CODE_GENDER'] =='XNA']['CODE_GENDER'].count()


# In[17]:


# Correction : difficile d'imputer le sexe par le mode de cette catégorie
# Comme il n'y a que 4 clients avec un sexe non renseigné, on supprime ces
# valeurs
application_train =  application_train[application_train['CODE_GENDER'] != 'XNA']


# In[18]:


# correction Name family statut
application_train['NAME_FAMILY_STATUS'].unique()


# In[19]:


# Nombre de lignes ayant la valeur 'Maternity leave' ?
application_train[application_train['NAME_FAMILY_STATUS'] ==
                  'Unknown']['NAME_FAMILY_STATUS'].count()


# In[20]:


# Correction : remplacer 'Unknown' par np.nan
application_train['NAME_FAMILY_STATUS'] =     [row if row != 'Unknown' else np.nan for row in
     application_train['NAME_FAMILY_STATUS']]


# ## 2-Merge des fichiers, nettoyage et feature engeneering 
On va merger ensemble les fichiers application_train.csv et application_test.csv, pour y effectuer un nettoyage, un encodage des variables, et la création de nouvelles variables, afin de rendre le tout homogène.


# In[21]:


print('The shape of data before:',application_train.shape)


# In[22]:


# Creation of test and train dataframe
train_set=application_train
test_set = application_test


# In[23]:


#Create a simple dataset with the train / test merge app
train_set = test_set.append(train_set)


# In[24]:


print('Train:' + str(train_set.shape))
print('Test:' + str(test_set.shape))


# In[25]:


#Now just in case, let's check if we've got it right
# Les deux jeux de données ont exactement le même format avec une seule différence, la TARGET dispo dans le train.
train_set.TARGET.isna().sum()


# In[26]:


sum(train_set.SK_ID_CURR[train_set.TARGET.isna()] == application_test.SK_ID_CURR) 

A partir du fichier bureau, il est possible d'extraire un historique sur les précédents crédits enregistrés par les clients. Il peut donc être intéressant d'enrichir l'échantillon avec ce type de données.
# In[27]:


#Total number of previous credits taken by each customer
previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(
                                       columns ={'SK_ID_BUREAU': 'PREVIOUS_LOANS_COUNT'})
previous_loan_counts.head()


# In[28]:


train_set = train_set.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')
# Fill the missing values with 0 
train_set['PREVIOUS_LOANS_COUNT'] = train_set['PREVIOUS_LOANS_COUNT'].fillna(0)
print("The shape of train_set after merging:")
display(train_set.shape)
print("=="*50)
display(train_set.head())


# In[29]:


test_set = test_set.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')
# Fill the missing values with 0 
test_set['PREVIOUS_LOANS_COUNT'] = test_set['PREVIOUS_LOANS_COUNT'].fillna(0)
print("The shape of test_set after merging:")
display(test_set.shape)
print("=="*50)
display(test_set.head())

Bureau Balance : rajouter des informations concernant la position cash a travers la jointure sk_id_bureau 
nous procederons à une transformation en moyenne de cette variable
# In[30]:


#Monthly average balances of previous credits in Credit Bureau.
bureau_bal_mean = bureau_balance.groupby('SK_ID_BUREAU', as_index=False).mean().rename(columns = 
                                        {'MONTHS_BALANCE': 'MONTHS_BALANCE_MEAN'})
bureau_bal_mean.head()


# In[31]:


bureau_full = bureau.merge(bureau_bal_mean, on='SK_ID_BUREAU', how='left')
bureau_full.drop('SK_ID_BUREAU', axis=1, inplace=True)
display(bureau_full.head())
display(bureau_full.shape)


# In[32]:


bureau_mean = bureau_full.groupby('SK_ID_CURR', as_index=False).mean().add_prefix('PREV_BUR_MEAN_')
bureau_mean = bureau_mean.rename(columns = {'PREV_BUR_MEAN_SK_ID_CURR' : 'SK_ID_CURR'})
bureau_mean.shape


# In[33]:


train_set = train_set.merge(bureau_mean, on='SK_ID_CURR', how='left')
# Fill the missing values with 0 
train_set['PREVIOUS_LOANS_COUNT'] = train_set['PREVIOUS_LOANS_COUNT'].fillna(0)
print("The shape of train_set after merging:")
display(train_set.shape)
print("=="*50)
display(train_set.head())


# In[34]:


test_set = test_set.merge(bureau_mean, on='SK_ID_CURR', how='left')
# Fill the missing values with 0 
test_set['PREVIOUS_LOANS_COUNT'] = test_set['PREVIOUS_LOANS_COUNT'].fillna(0)
print("The shape of test_set after merging:")
display(test_set.shape)
print("=="*50)
display(test_set.head())

previous_application, le nombre de demandes précédentes des clients au crédit immobilier à ajouter à l'échantillon.
# In[35]:


#Number of previous applications of the clients to Home Credit
previous_application_counts = previous_application.groupby('SK_ID_CURR', 
                                                           as_index=False)['SK_ID_PREV'].count().rename(
                                                           columns = {'SK_ID_PREV': 'PREVIOUS_APPLICATION_COUNT'})
previous_application_counts.head()


# In[36]:


train_set = train_set.merge(previous_application_counts, on='SK_ID_CURR', how='left')
# Fill the missing values with 0 
train_set['PREVIOUS_LOANS_COUNT'] = train_set['PREVIOUS_LOANS_COUNT'].fillna(0)
print("The shape of train_set after merging:")
display(train_set.shape)
print("=="*50)
display(train_set.head())


# In[37]:


test_set = test_set.merge(previous_application_counts, on='SK_ID_CURR', how='left')
# Fill the missing values with 0 
test_set['PREVIOUS_LOANS_COUNT'] = test_set['PREVIOUS_LOANS_COUNT'].fillna(0)
print("The shape of test_set after merging:")
display(test_set.shape)
print("=="*50)
display(test_set.head())

credit_card_balance
# In[38]:


# Remove SK_ID_CURR :
cc_balance.drop('SK_ID_CURR', axis=1, inplace=True)


# In[39]:


cc_balance_mean = cc_balance.groupby('SK_ID_PREV', as_index=False).mean().add_prefix('CARD_MEAN_')
cc_balance_mean.rename(columns = {'CARD_MEAN_SK_ID_PREV' : 'SK_ID_PREV'}, inplace=True)
cc_balance_mean.shape


# In[40]:


#Merge with previous_application
previous_application = previous_application.merge(cc_balance_mean, on='SK_ID_PREV', how='left')
previous_application.shape

installments_payments
# In[41]:


installments_payments.drop('SK_ID_CURR', axis=1, inplace=True)
install_pay_mean = installments_payments.groupby('SK_ID_PREV', as_index=False).mean().add_prefix('INSTALL_MEAN_')
install_pay_mean.rename(columns = {'INSTALL_MEAN_SK_ID_PREV' : 'SK_ID_PREV'}, inplace=True)
install_pay_mean.shape


# In[42]:


#Merge with previous_application
previous_application = previous_application.merge(install_pay_mean, on='SK_ID_PREV', how='left')
previous_application.shape

POS_CASH_balance
# In[43]:


POS_CASH_balance.drop('SK_ID_CURR', axis=1, inplace=True)
POS_mean = installments_payments.groupby('SK_ID_PREV', as_index=False).mean().add_prefix('POS_MEAN_')
POS_mean.rename(columns = {'POS_MEAN_SK_ID_PREV' : 'SK_ID_PREV'}, inplace=True)
POS_mean.shape


# In[44]:


#Merge with previous_application
previous_application = previous_application.merge(POS_mean, on='SK_ID_PREV', how='left')
previous_application.shape

Retour sur previous_application pour assembles les lignes d'observation selon SK_ID_CURR.
# In[45]:


prev_appl_mean = previous_application.groupby('SK_ID_CURR', as_index=False).mean().add_prefix('PREV_APPL_MEAN_')
prev_appl_mean.rename(columns = {'PREV_APPL_MEAN_SK_ID_CURR' : 'SK_ID_CURR'}, inplace=True)
prev_appl_mean = prev_appl_mean.drop('PREV_APPL_MEAN_SK_ID_PREV', axis=1)


# In[46]:


train_set = train_set.merge(prev_appl_mean, on='SK_ID_CURR', how='left')
# Fill the missing values with 0 
train_set['PREVIOUS_LOANS_COUNT'] = train_set['PREVIOUS_LOANS_COUNT'].fillna(0)
print("The shape of train_set after merging:")
display(train_set.shape)
print("=="*50)
display(train_set.head())


# In[47]:


test_set = test_set.merge(prev_appl_mean, on='SK_ID_CURR', how='left')
# Fill the missing values with 0 
test_set['PREVIOUS_LOANS_COUNT'] = test_set['PREVIOUS_LOANS_COUNT'].fillna(0)
print("The shape of test_set after merging:")
display(test_set.shape)
print("=="*50)
display(test_set.head())


# In[48]:


# Comparaison variables catégorielles du train set et du test set


# In[49]:


for var in cols_cat :
    var_train = application_train[var].unique()
    var_test = application_test[var].unique()
    diff = [val for val in var_train if val not in var_test]
    if len(diff) > 0 and diff != 'nan':
        print(f'Variable {var} - catégories différentes : {diff}')
      


# In[50]:


# NAME_INCOME_TYPE
train_set[train_set['NAME_INCOME_TYPE'] ==
                  'Maternity leave']['NAME_INCOME_TYPE'].count()


# In[51]:


# Correction : remplacer 'Maternity leave' par np.nan
train_set['NAME_INCOME_TYPE'] =     [row if row != 'Maternity leave' else np.nan for row in
    train_set['NAME_INCOME_TYPE']]
# Vérification
train_set[train_set['NAME_INCOME_TYPE'] ==
                  'Maternity leave']['NAME_INCOME_TYPE'].count()


# #### Valeurs manquantes

# In[52]:


def get_missing_values(df_work, pourcentage, affiche_heatmap, retour=False):
    """Indicateurs sur les variables manquantes
       @param in : df_work dataframe obligatoire
                   pourcentage : boolean si True affiche le nombre heatmap
                   affiche_heatmap : boolean si True affiche la heatmap
       @param out : none
    """

    # 1. Nombre de valeurs manquantes totales
    nb_nan_tot = df_work.isna().sum().sum()
    nb_donnees_tot = np.product(df_work.shape)
    pourc_nan_tot = round((nb_nan_tot / nb_donnees_tot) * 100, 2)
    print(
        f'Valeurs manquantes : {nb_nan_tot} NaN pour {nb_donnees_tot} données ({pourc_nan_tot} %)')

    if pourcentage:
        print("-------------------------------------------------------------")
        print("Nombre et pourcentage de valeurs manquantes par variable\n")
        # 2. Visualisation du nombre et du pourcentage de valeurs manquantes
        # par variable
        values = df_work.isnull().sum()
        percentage = 100 * values / len(df_work)
        table = pd.concat([values, percentage.round(2)], axis=1)
        table.columns = [
            'Nombres de valeurs manquantes',
            '% de valeurs manquantes']
        display(table[table['Nombres de valeurs manquantes'] != 0]
                .sort_values('% de valeurs manquantes', ascending=False)
                .style.background_gradient('seismic'))

    if affiche_heatmap:
        print("-------------------------------------------------------------")
        print("Heatmap de visualisation des valeurs manquantes")
        # 3. Heatmap de visualisation des valeurs manquantes
        plt.figure(figsize=(20, 10))
        sns.heatmap(df_work.isna(), cbar=False)
        plt.show()

    if retour:
        return table


# In[53]:


# Valeurs manquantes du dataframe
df_nan_applitrain = get_missing_values(train_set,
                                                   True, False, True)


# In[54]:


# Liste des variables ayant plus de 70% de valeurs manquantes
cols_nan_a_suppr =     df_nan_applitrain[df_nan_applitrain['% de valeurs manquantes'] > 70]     .index.to_list()
print(f'Nombre de variables à supprimer : {len(cols_nan_a_suppr)}')
cols_nan_a_suppr


# In[55]:


# Suppression des variables avec un seuil de nan > 70%
train_set.drop(columns=cols_nan_a_suppr, inplace=True)
# Variables catégorielles
cols_cat = train_set.select_dtypes(exclude=[np.number]).columns     .to_list()
# Variables quantitatives
cols_num = train_set.select_dtypes(include=[np.number]).columns     .to_list()
# Taille : nombre de lignes/colonnes
nRow, nVar = train_set.shape
print(f'Le jeu de données contient {nRow} lignes et {nVar} variables.')


# In[56]:


# meme traitement sur ke jeu de test
test_set.drop(columns=cols_nan_a_suppr, inplace=True)
# Variables catégorielles
cols_cat = test_set.select_dtypes(exclude=[np.number]).columns     .to_list()
# Variables quantitatives
cols_num = test_set.select_dtypes(include=[np.number]).columns     .to_list()
# Taille : nombre de lignes/colonnes
nRow, nVar = test_set.shape
print(f'Le jeu de données contient {nRow} lignes et {nVar} variables.')


# ### imputation
# variables quantitatives imputées par median.
# variables qualitatives imputées par mode.

# In[57]:


# Variables quantitatives - imputation par médiane
nb_nan_median =train_set[cols_num].isna().sum().sum()
print(f'Nombre de nan avant imputation par median : {nb_nan_median}')
train_set.fillna(train_set[cols_num].median(), inplace=True)
# Vérification
nb_nan_median = train_set[cols_num].isna().sum().sum()
print(f'Nombre de nan après imputation par median : {nb_nan_median}')


# In[58]:


# Variables quantitatives - imputation par médiane
nb_nan_median_test =test_set[cols_num].isna().sum().sum()
print(f'Nombre de nan avant imputation par median : {nb_nan_median}')
test_set.fillna(test_set[cols_num].median(), inplace=True)
# Vérification
nb_nan_median_test = test_set[cols_num].isna().sum().sum()
print(f'Nombre de nan après imputation par median : {nb_nan_median}')


# In[59]:


# Variables qualitatives - imputation par mode de la variable
nb_nan_cat = train_set[cols_cat].isna().sum().sum()
print(f'Nombre de nan avant imputation par mode : {nb_nan_cat}')
for var in cols_cat:
    mode = train_set[var].mode()[0]
    train_set[var].fillna(mode, inplace=True)
nb_nan_cat =train_set[cols_cat].isna().sum().sum()
print(f'Nombre de nan après imputation par mode : {nb_nan_cat}')


# In[60]:


# Variables qualitatives - imputation par mode de la variable
nb_nan_cat_test = test_set[cols_cat].isna().sum().sum()
print(f'Nombre de nan avant imputation par mode : {nb_nan_cat}')
for var in cols_cat:
    mode = test_set[var].mode()[0]
    test_set[var].fillna(mode, inplace=True)
nb_nan_cat_test =test_set[cols_cat].isna().sum().sum()
print(f'Nombre de nan après imputation par mode : {nb_nan_cat}')


# In[61]:


get_missing_values(train_set,True,False,True)


# In[62]:


get_missing_values(test_set,True,False,True)


# In[63]:


# Vérification : plus de nan
# Valeurs manquantes du dataframe train_set
print(f'Nombre nan train_set : {train_set.isna().sum().sum()}')
print(f'Nombre nan train_set : {test_set.isna().sum().sum()}')


# #### FEATURES ENGINEERING
# 
# nous nous inspirons des variables crées sur le concours Kaggle pour les répliquer 
# source : https://github.com/rishabhrao1997/Home-Credit-Default-Risk
#    

# In[64]:


def feature_engineering_application(data):
   '''
   FEATURE ENGINEERING : création de nouvelles variables.
   Extrait de : https://github.com/rishabhrao1997/Home-Credit-Default-Risk
   Parameters
   ----------
   data : dataframe pour ajout de nouvelles variables, obligatoire.
   Returns
   -------
   None.
   '''
   
   # Ratio : Montant du crédit du prêt / Revenu du demandeur
   data['CREDIT_INCOME_RATIO'] = data['AMT_CREDIT'] /        (data['AMT_INCOME_TOTAL'] + 0.00001)
   # Ratio : Montant du crédit du prêt / Annuité de prêt
   data['CREDIT_ANNUITY_RATIO'] = data['AMT_CREDIT'] /        (data['AMT_ANNUITY'] + 0.00001)
   # Ratio : Annuité de prêt / Revenu du demandeur
   data['ANNUITY_INCOME_RATIO'] = data['AMT_ANNUITY'] /        (data['AMT_INCOME_TOTAL'] + 0.00001)
   # Différence : Revenu du demandeur - Annuité de prêt
   data['INCOME_ANNUITY_DIFF'] = data['AMT_INCOME_TOTAL'] -        data['AMT_ANNUITY']
   # Ratio : Montant du crédit du prêt / prix des biens pour lesquels le prêt est accordé
   # Crédit est supérieur au prix des biens ?
   data['CREDIT_GOODS_RATIO'] = data['AMT_CREDIT'] /        (data['AMT_GOODS_PRICE'] + 0.00001)
   # Différence : Revenu du demandeur - prix des biens pour lesquels le prêt
   # est accordé
   data['INCOME_GOODS_DIFF'] = data['AMT_INCOME_TOTAL'] /        data['AMT_GOODS_PRICE']
   # Ratio : Annuité de prêt / Âge du demandeur au moment de la demande
   data['INCOME_AGE_RATIO'] = data['AMT_INCOME_TOTAL'] / (
       data['DAYS_BIRTH'] + 0.00001)
   # Ratio : Montant du crédit du prêt / Âge du demandeur au moment de la
   # demande
   data['CREDIT_AGE_RATIO'] = data['AMT_CREDIT'] / (
       data['DAYS_BIRTH'] + 0.00001)
   # Ratio : Revenu du demandeur / Score normalisé de la source de données
   # externe 3
   data['INCOME_EXT_RATIO'] = data['AMT_INCOME_TOTAL'] /        (data['EXT_SOURCE_3'] + 0.00001)
   # Ratio : Montant du crédit du prêt / Score normalisé de la source de
   # données externe
   data['CREDIT_EXT_RATIO'] = data['AMT_CREDIT'] /        (data['EXT_SOURCE_3'] + 0.00001)
   # Multiplication : Revenu du demandeur
   #                  * heure à laquelle le demandeur à fait sa demande de prêt
   data['HOUR_PROCESS_CREDIT_MUL'] = data['AMT_CREDIT'] *        data['HOUR_APPR_PROCESS_START']

   # -----------------------------------------------------------------------
   # Variables sur l'âge
   # -----------------------------------------------------------------------
   # YEARS_BIRTH - Âge du demandeur au moment de la demande DAYS_BIRTH en
   # années
   data['YEARS_BIRTH'] = data['DAYS_BIRTH'] * -1 / 365
   # Différence : Âge du demandeur - Ancienneté dans l'emploi à date demande
   data['AGE_EMPLOYED_DIFF'] = data['DAYS_BIRTH'] - data['DAYS_EMPLOYED']
   # Ratio : Ancienneté dans l'emploi à date demande / Âge du demandeur
   data['EMPLOYED_AGE_RATIO'] = data['DAYS_EMPLOYED'] /        (data['DAYS_BIRTH'] + 0.00001)
   # Ratio : nombre de jours avant la demande où le demandeur a changé de téléphone \
   #         äge du client
   data['LAST_PHONE_BIRTH_RATIO'] = data[
       'DAYS_LAST_PHONE_CHANGE'] / (data['DAYS_BIRTH'] + 0.00001)
   # Ratio : nombre de jours avant la demande où le demandeur a changé de téléphone \
   #         ancienneté dans l'emploi
   data['LAST_PHONE_EMPLOYED_RATIO'] = data[
       'DAYS_LAST_PHONE_CHANGE'] / (data['DAYS_EMPLOYED'] + 0.00001)

   # -----------------------------------------------------------------------
   # Variables sur la voiture
   # -----------------------------------------------------------------------
   # Différence : Âge de la voiture du demandeur -  Ancienneté dans l'emploi
   # à date demande
   data['CAR_EMPLOYED_DIFF'] = data['OWN_CAR_AGE'] - data['DAYS_EMPLOYED']
   # Ratio : Âge de la voiture du demandeur / Ancienneté dans l'emploi à date
   # demande
   data['CAR_EMPLOYED_RATIO'] = data['OWN_CAR_AGE'] /        (data['DAYS_EMPLOYED'] + 0.00001)
   # Différence : Âge du demandeur - Âge de la voiture du demandeur
   data['CAR_AGE_DIFF'] = data['DAYS_BIRTH'] - data['OWN_CAR_AGE']
   # Ratio : Âge de la voiture du demandeur / Âge du demandeur
   data['CAR_AGE_RATIO'] = data['OWN_CAR_AGE'] /        (data['DAYS_BIRTH'] + 0.00001)

   # -----------------------------------------------------------------------
   # Variables sur les contacts
   # -----------------------------------------------------------------------
   # Somme : téléphone portable? + téléphone professionnel? + téléphone
   #         professionnel fixe? + téléphone portable joignable? +
   #         adresse de messagerie électronique?
   data['FLAG_CONTACTS_SUM'] = data['FLAG_MOBIL'] + data['FLAG_EMP_PHONE'] +        data['FLAG_WORK_PHONE'] + data['FLAG_CONT_MOBILE'] +        data['FLAG_PHONE'] + data['FLAG_EMAIL']

   # -----------------------------------------------------------------------
   # Variables sur les membres de la famille
   # -----------------------------------------------------------------------
   # Différence : membres de la famille - enfants (adultes)
   data['CNT_NON_CHILDREN'] = data['CNT_FAM_MEMBERS'] - data['CNT_CHILDREN']
   # Ratio : nombre d'enfants / Revenu du demandeur
   data['CHILDREN_INCOME_RATIO'] = data['CNT_CHILDREN'] /        (data['AMT_INCOME_TOTAL'] + 0.00001)
   # Ratio : Revenu du demandeur / membres de la famille : revenu par tête
   data['PER_CAPITA_INCOME'] = data['AMT_INCOME_TOTAL'] /        (data['CNT_FAM_MEMBERS'] + 1)

   # -----------------------------------------------------------------------
   # Variables sur la région
   # -----------------------------------------------------------------------
   # Moyenne : moyenne de notes de la région/ville où vit le client * revenu
   # du demandeur
   data['REGIONS_INCOME_MOY'] = (data['REGION_RATING_CLIENT'] +
                                 data['REGION_RATING_CLIENT_W_CITY']) * data['AMT_INCOME_TOTAL'] / 2
   # Max : meilleure note de la région/ville où vit le client
   data['REGION_RATING_MAX'] = [max(ele1, ele2) for ele1, ele2 in zip(
       data['REGION_RATING_CLIENT'], data['REGION_RATING_CLIENT_W_CITY'])]
   # Min : plus faible note de la région/ville où vit le client
   data['REGION_RATING_MIN'] = [min(ele1, ele2) for ele1, ele2 in zip(
       data['REGION_RATING_CLIENT'], data['REGION_RATING_CLIENT_W_CITY'])]
   # Moyenne : des notes de la région et de la ville où vit le client
   data['REGION_RATING_MEAN'] = (
       data['REGION_RATING_CLIENT'] + data['REGION_RATING_CLIENT_W_CITY']) / 2
   # Multipication : note de la région/ note de la ville où vit le client
   data['REGION_RATING_MUL'] = data['REGION_RATING_CLIENT'] *        data['REGION_RATING_CLIENT_W_CITY']
   # Somme : des indicateurs  :
   # Indicateur si l'adresse permanente du client ne correspond pas à l'adresse de contact (1=différent ou 0=identique - au niveau de la région)
   # Indicateur si l'adresse permanente du client ne correspond pas à l'adresse professionnelle (1=différent ou 0=identique - au niveau de la région)
   # Indicateur si l'adresse de contact du client ne correspond pas à l'adresse de travail (1=différent ou 0=identique - au niveau de la région).
   # Indicateur si l'adresse permanente du client ne correspond pas à l'adresse de contact (1=différent ou 0=identique - au niveau de la ville)
   # Indicateur si l'adresse permanente du client ne correspond pas à l'adresse professionnelle (1=différent ou 0=même - au niveau de la ville).
   # Indicateur si l'adresse de contact du client ne correspond pas à
   # l'adresse de travail (1=différent ou 0=identique - au niveau de la
   # ville).
   data['FLAG_REGIONS_SUM'] = data['REG_REGION_NOT_LIVE_REGION'] +        data['REG_REGION_NOT_WORK_REGION'] +        data['LIVE_REGION_NOT_WORK_REGION'] +        data['REG_CITY_NOT_LIVE_CITY'] +        data['REG_CITY_NOT_WORK_CITY'] +        data['LIVE_CITY_NOT_WORK_CITY']

   # -----------------------------------------------------------------------
   # Variables sur les sources externes : sum, min, multiplication, max, var, scoring
   # -----------------------------------------------------------------------
   # Somme : somme des scores des 3 sources externes
   data['EXT_SOURCE_SUM'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                  'EXT_SOURCE_3']].sum(axis=1)
   # Moyenne : moyenne des scores des 3 sources externes
   data['EXT_SOURCE_MEAN'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                   'EXT_SOURCE_3']].mean(axis=1)
   # Multiplication : des scores des 3 sources externes
   data['EXT_SOURCE_MUL'] = data['EXT_SOURCE_1'] *        data['EXT_SOURCE_2'] * data['EXT_SOURCE_3']
   # Max : Max parmi les 3 scores des 3 sources externes
   data['EXT_SOURCE_MAX'] = [max(ele1, ele2, ele3) for ele1, ele2, ele3 in zip(
       data['EXT_SOURCE_1'], data['EXT_SOURCE_2'], data['EXT_SOURCE_3'])]
   # Min : Min parmi les 3 scores des 3 sources externes
   data['EXT_SOURCE_MIN'] = [min(ele1, ele2, ele3) for ele1, ele2, ele3 in zip(
       data['EXT_SOURCE_1'], data['EXT_SOURCE_2'], data['EXT_SOURCE_3'])]
   # Variance : variance des scores des 3 sources externes
   data['EXT_SOURCE_VAR'] = [np.var([ele1, ele2, ele3]) for ele1, ele2, ele3 in zip(
       data['EXT_SOURCE_1'], data['EXT_SOURCE_2'], data['EXT_SOURCE_3'])]
   # Scoring : scoring des scores des 3 sources externes, score 1 poids 2...
   data['WEIGHTED_EXT_SOURCE'] = data.EXT_SOURCE_1 *        2 + data.EXT_SOURCE_2 * 3 + data.EXT_SOURCE_3 * 4

   # -----------------------------------------------------------------------
   # Variables sur le bâtiment
   # -----------------------------------------------------------------------
   # Somme : Informations normalisées sur l'immeuble où vit le demandeur des moyennes
   # de la taille de l'appartement, de la surface commune, de la surface habitable,
   # de l'âge de l'immeuble, du nombre d'ascenseurs, du nombre d'entrées,
   # de l'état de l'immeuble et du nombre d'étages.
   data['APARTMENTS_SUM_AVG'] = data['APARTMENTS_AVG'] + data['BASEMENTAREA_AVG'] + data['YEARS_BEGINEXPLUATATION_AVG'] + data[
       'YEARS_BUILD_AVG'] + data['ELEVATORS_AVG'] + data['ENTRANCES_AVG'] + data[
       'FLOORSMAX_AVG'] + data['FLOORSMIN_AVG'] + data['LANDAREA_AVG'] + data[
       'LIVINGAREA_AVG'] + data['NONLIVINGAREA_AVG']
   # Somme : Informations normalisées sur l'immeuble où vit le demandeur des modes
   # de la taille de l'appartement, de la surface commune, de la surface habitable,
   # de l'âge de l'immeuble, du nombre d'ascenseurs, du nombre d'entrées,
   # de l'état de l'immeuble et du nombre d'étages.
   data['APARTMENTS_SUM_MODE'] = data['APARTMENTS_MODE'] + data['BASEMENTAREA_MODE'] + data['YEARS_BEGINEXPLUATATION_MODE'] + data[
       'YEARS_BUILD_MODE'] + data['ELEVATORS_MODE'] + data['ENTRANCES_MODE'] + data[
       'FLOORSMAX_MODE'] + data['FLOORSMIN_MODE'] + data['LANDAREA_MODE'] + data[
       'LIVINGAREA_MODE'] + data['NONLIVINGAREA_MODE'] + data['TOTALAREA_MODE']
   # Somme : Informations normalisées sur l'immeuble où vit le demandeur des médianes
   # de la taille de l'appartement, de la surface commune, de la surface habitable,
   # de l'âge de l'immeuble, du nombre d'ascenseurs, du nombre d'entrées,
   # de l'état de l'immeuble et du nombre d'étages.
   data['APARTMENTS_SUM_MEDI'] = data['APARTMENTS_MEDI'] + data['BASEMENTAREA_MEDI'] + data['YEARS_BEGINEXPLUATATION_MEDI'] + data[
       'YEARS_BUILD_MEDI'] + data['ELEVATORS_MEDI'] + data['ENTRANCES_MEDI'] + data[
       'FLOORSMAX_MEDI'] + data['FLOORSMIN_MEDI'] + data['LANDAREA_MEDI'] + \
       data['NONLIVINGAREA_MEDI']
   # Multiplication : somme des moyennes des infos sur le bâtiment * revenu
   # du demandeur
   data['INCOME_APARTMENT_AVG_MUL'] = data['APARTMENTS_SUM_AVG'] *        data['AMT_INCOME_TOTAL']
   # Multiplication : somme des modes des infos sur le bâtiment * revenu du
   # demandeur
   data['INCOME_APARTMENT_MODE_MUL'] = data['APARTMENTS_SUM_MODE'] *        data['AMT_INCOME_TOTAL']
   # Multiplication : somme des médianes des infos sur le bâtiment * revenu
   # du demandeur
   data['INCOME_APARTMENT_MEDI_MUL'] = data['APARTMENTS_SUM_MEDI'] *        data['AMT_INCOME_TOTAL']

   # -----------------------------------------------------------------------
   # Variables sur les défauts de paiements et les défauts observables
   # -----------------------------------------------------------------------
   # Somme : nombre d'observations de l'environnement social du demandeur
   #         avec des défauts observables de 30 DPD (jours de retard) +
   #        nombre d'observations de l'environnement social du demandeur
   #         avec des défauts observables de 60 DPD (jours de retard)
   data['OBS_30_60_SUM'] = data['OBS_30_CNT_SOCIAL_CIRCLE'] +        data['OBS_60_CNT_SOCIAL_CIRCLE']
   # Somme : nombre d'observations de l'environnement social du demandeur
   #         avec des défauts de paiement de 30 DPD (jours de retard) +
   #        nombre d'observations de l'environnement social du demandeur
   #         avec des défauts de paiement de 60 DPD (jours de retard)
   data['DEF_30_60_SUM'] = data['DEF_30_CNT_SOCIAL_CIRCLE'] +        data['DEF_60_CNT_SOCIAL_CIRCLE']
   # Multiplication : nombre d'observations de l'environnement social du demandeur
   #         avec des défauts observables de 30 DPD (jours de retard) *
   #        nombre d'observations de l'environnement social du demandeur
   #         avec des défauts observables de 60 DPD (jours de retard)
   data['OBS_DEF_30_MUL'] = data['OBS_30_CNT_SOCIAL_CIRCLE'] *        data['DEF_30_CNT_SOCIAL_CIRCLE']
   # Multiplication : nombre d'observations de l'environnement social du demandeur
   #         avec des défauts de paiement de 30 DPD (jours de retard) *
   #        nombre d'observations de l'environnement social du demandeur
   #         avec des défauts de paiement de 60 DPD (jours de retard)
   data['OBS_DEF_60_MUL'] = data['OBS_60_CNT_SOCIAL_CIRCLE'] *        data['DEF_60_CNT_SOCIAL_CIRCLE']
   # Somme : nombre d'observations de l'environnement social du demandeur
   #         avec des défauts de paiement ou des défauts observables avec 30
   #         DPD (jours de retard) et 60 DPD.
   data['SUM_OBS_DEF_ALL'] = data['OBS_30_CNT_SOCIAL_CIRCLE'] + data['DEF_30_CNT_SOCIAL_CIRCLE'] +        data['OBS_60_CNT_SOCIAL_CIRCLE'] + data['DEF_60_CNT_SOCIAL_CIRCLE']
   # Ratio : Montant du crédit du prêt /
   #         nombre d'observations de l'environnement social du demandeur
   #         avec des défauts observables de 30 DPD (jours de retard)
   data['OBS_30_CREDIT_RATIO'] = data['AMT_CREDIT'] /        (data['OBS_30_CNT_SOCIAL_CIRCLE'] + 0.00001)
   # Ratio : Montant du crédit du prêt /
   #         nombre d'observations de l'environnement social du demandeur
   #         avec des défauts observables de 60 DPD (jours de retard)
   data['OBS_60_CREDIT_RATIO'] = data['AMT_CREDIT'] /        (data['OBS_60_CNT_SOCIAL_CIRCLE'] + 0.00001)
   # Ratio : Montant du crédit du prêt /
   #         nombre d'observations de l'environnement social du demandeur
   #         avec des défauts de paiement de 30 DPD (jours de retard)
   data['DEF_30_CREDIT_RATIO'] = data['AMT_CREDIT'] /        (data['DEF_30_CNT_SOCIAL_CIRCLE'] + 0.00001)
   # Ratio : Montant du crédit du prêt /
   #         nombre d'observations de l'environnement social du demandeur
   #         avec des défauts de paiement de 60 DPD (jours de retard)
   data['DEF_60_CREDIT_RATIO'] = data['AMT_CREDIT'] /        (data['DEF_60_CNT_SOCIAL_CIRCLE'] + 0.00001)

   # -----------------------------------------------------------------------
   # Variables sur les indicateurs des documents fournis ou non
   # -----------------------------------------------------------------------
   # Toutes les variables DOCUMENT_
   cols_flag_doc = [flag for flag in data.columns if 'FLAG_DOC' in flag]
   # Somme : tous les indicateurs des documents fournis ou non
   data['FLAGS_DOCUMENTS_SUM'] = data[cols_flag_doc].sum(axis=1)
   # Moyenne : tous les indicateurs des documents fournis ou non
   data['FLAGS_DOCUMENTS_AVG'] = data[cols_flag_doc].mean(axis=1)
   # Variance : tous les indicateurs des documents fournis ou non
   data['FLAGS_DOCUMENTS_VAR'] = data[cols_flag_doc].var(axis=1)
   # Ecart-type : tous les indicateurs des documents fournis ou non
   data['FLAGS_DOCUMENTS_STD'] = data[cols_flag_doc].std(axis=1)

   # -----------------------------------------------------------------------
   # Variables sur le détail des modifications du demandeur : jour/heure...
   # -----------------------------------------------------------------------
   # Somme : nombre de jours avant la demande de changement de téléphone
   #         + nombre de jours avant la demande de changement enregistré sur la demande
   #         + nombre de jours avant la demande le client où il à
   #           changé la pièce d'identité avec laquelle il a demandé le prêt
   data['DAYS_DETAILS_CHANGE_SUM'] = data['DAYS_LAST_PHONE_CHANGE'] +        data['DAYS_REGISTRATION'] + data['DAYS_ID_PUBLISH']
   # Somme : nombre de demandes de renseignements sur le client adressées au Bureau de crédit
   # une heure + 1 jour + 1 mois + 3 mois + 1 an et 1 jour avant la demande
   data['AMT_ENQ_SUM'] = data['AMT_REQ_CREDIT_BUREAU_HOUR'] + data['AMT_REQ_CREDIT_BUREAU_DAY'] + data['AMT_REQ_CREDIT_BUREAU_WEEK'] +        data['AMT_REQ_CREDIT_BUREAU_MON'] +            data['AMT_REQ_CREDIT_BUREAU_QRT'] +                data['AMT_REQ_CREDIT_BUREAU_YEAR']
   # Ratio : somme du nombre de demandes de renseignements sur le client adressées au Bureau de crédit
   #         une heure + 1 jour + 1 mois + 3 mois + 1 an et 1 jour avant la demande \
   #         Montant du crédit du prêt
   data['ENQ_CREDIT_RATIO'] = data['AMT_ENQ_SUM'] /        (data['AMT_CREDIT'] + 0.00001)

   return data


# In[65]:


train_set = feature_engineering_application(train_set)
train_set.shape


# In[66]:


test_set = feature_engineering_application(test_set)
test_set.shape


# ####  ENCODAGE 
# on va encoder les variables qualitatives afin de transformer les données en valeurs numériques, car les modèles que l'on va entraîner ne peuvent pas travailler sur des données qualitatives.
# 

# In[67]:


# Liste des variables catégorielles
cols_cat


# In[68]:


data_train = train_set[train_set['SK_ID_CURR'].isin(application_train.SK_ID_CURR)]
data_test = test_set[test_set['SK_ID_CURR'].isin(application_test.SK_ID_CURR)]


# In[69]:


print('Training Features shape with categorical columns: ', data_train.shape)
print('Testing Features shape with categorical columns: ', data_test.shape)


# In[70]:


data_object_infos = data_train.select_dtypes("object").describe().T
data_object_infos["unique"] = data_train.select_dtypes("object").apply( pd.Series.unique, axis=0)
data_object_infos["nunique"] = data_train.select_dtypes("object").apply( pd.Series.nunique, axis=0)
data_object_infos


# In[71]:


test_data_object_infos = data_test.select_dtypes("object").describe().T
test_data_object_infos["unique"] = data_test.select_dtypes("object").apply( pd.Series.unique, axis=0)
test_data_object_infos["nunique"] = data_test.select_dtypes("object").apply( pd.Series.nunique, axis=0)
test_data_object_infos


# In[72]:


def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# In[73]:


data_train, cols_one_hot_encoded = one_hot_encoder(data_train, False)
print(f'{len(cols_one_hot_encoded)} colonnes on été "one hot encoded" :')
for col in cols_one_hot_encoded:
    print(f"- {col}")


# In[74]:


data_test, cols_one_hot_encoded = one_hot_encoder(data_test, False)
print(f'{len(cols_one_hot_encoded)} colonnes on été "one hot encoded" :')
for col in cols_one_hot_encoded:
    print(f"- {col}")


# In[75]:


print(data_train.shape)
print(data_test.shape)


# In[76]:


# Préparation de la matrice de corrélation
# ---------------------------------------------------------------------
# Variables fortement corrélées : si le coef de Pearson est :
# > 0.8 ou < -0.8
# et inférieur à 1 ou > -1 (corrélée avec elle-même)
seuil = 0.8
# Matrice de corrélation avec valeur absolue pour ne pas avoir à gérer
# les corrélations positives et négatives séparément
corr = data_train.corr().abs()
# On ne conserve que la partie supérieur à la diagonale pour n'avoir
# qu'une seule fois les corrélations prisent en compte (symétrie axiale)
corr_triangle = corr.where(np.triu(np.ones(corr.shape), k=1)
                           .astype(np.bool))


# In[77]:


# Variables avec un coef de Pearson > 0.8?
cols_corr_a_supp = [var for var in corr_triangle.columns
                    if any(corr_triangle[var] > seuil)]
print(f'{len(cols_corr_a_supp)} variables fortement corrélées à supprimer :\n')
for var in cols_corr_a_supp:
    print(var)


# In[78]:


# Suppression des variables fortement corrélées
print(f'data_train : {data_train.shape}')
data_train.drop(columns=cols_corr_a_supp,  inplace=True)
print(f'data_train : {data_train.shape}')


# In[79]:


# Suppression des variables fortement corrélées
print(f'data_test : {data_test.shape}')
data_test.drop(columns=cols_corr_a_supp,  inplace=True)
print(f'data_test : {data_test.shape}')


# In[80]:


# Remplacer :, espaces... par '_' et mettre en majuscules le nom des variables
import re
data_train = data_train.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+',
                                                     '_', x.upper()))


# In[81]:


data_test = data_test.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+',
                                                   '_', x.upper()))


# In[82]:


# save data_train et data set pour utilisation ultérieure
data_train.to_csv('base_train.csv')


# In[83]:


data_test.to_csv('base_test.csv')


# In[84]:


#save our TARGET variable
target = data_train.TARGET 
data_train.drop('TARGET', axis=1, inplace=True) #remove TARGET from train

#Align the datasets
data_train, data_test = data_train.align(data_test, join='inner', axis=1)


# In[85]:


print(target.shape)
print(data_train.shape)
print(data_test.shape)


# In[86]:


msno.bar(data_train)


# In[87]:


msno.bar(data_test)


# In[88]:


msno.bar(pd.DataFrame(target))


# In[89]:


# enregistrement des données 


# In[90]:


data_train.to_csv('data.csv', index=False)
data_test.to_csv('test_set.csv', index=False)
target.to_csv('target.csv', index=False)

Les dataframes  sont nettoyées, encodées et contiennent les nouvelles variables métiers et automatiques.
Mais il va  falloir diminuer le nombre de variables avec des techniques de features engineering.
La suite dans le notebook_features_selection
# In[91]:


import os
os.getcwd()


# In[92]:


len(target)


# In[93]:


len(data_train)


# In[ ]:




