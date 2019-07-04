
# coding: utf-8

# # <center><font color = 'orange'> Random pandas dataframe </font></center>

# In[1]:


import random

import string

import pandas as pd

import numpy as np


# ### Random character strings

# In[2]:


def rand_str ( n ) :
    
    """
     n : number of random characters in the returned string
    """
    
    rand_string = ''.join( random.SystemRandom().choices( string.ascii_uppercase + string.digits , k = n ) )
    return rand_string


# ### Liste de chaînes de caractères aléatoires

# In[3]:


def rand_str_lst( N , n ) :
    
    """
     N : number of string values
     n : number of characters for each string values ( each string value is built with the same number of characters )
    """
    
    rand_str_lst = []
    for i in range( N ) :
        rand_str_lst.append( rand_str( n ) )
    return rand_str_lst


# ### Dataframe aléatoire

# In[4]:


def rand_df ( n_col_num , lst_par_col_str , n_rows ) :
    
    """
     n_col_num : number of numeric columns
     
     lst_par_col_str : tuple ( a , b ) where a is the number of values in a hat of values and b the number of characters 
                       of each values ( all values are built with the same number of characters )
                       
     n_rows : number of rows in the dataframe
    """
    
    df = pd.DataFrame()
    
    for i in range( n_col_num ) :
        # name of new numeric column :
        col = 'col_num_' + str( i )
        
        # values for the new numeric column 
        df[ col ] = np.random.normal( 0 , 1 , n_rows )
           
    for i , par in enumerate( lst_par_col_str ) :
        # name of new string column :
        col = 'col_str_' + str( i )
        
        # values for sampling with replacement :
        rand_val_hat = rand_str_lst( par[0] , par[1] )
        
        # values for the new string column :
        df[ col ] = pd.Series( [ random.choice( rand_val_hat ) for _ in range( n_rows ) ] )
        
    return df


# ### NaN dans dataframe

# In[7]:


def add_nan( df , dic_col_pct_nan ) :
    
    """
     df_in : dataframe in which to replace values by nan
     dic_col_pct_nan : dictionary where key is column name and value is percentage of rows with nan values
    """
    
    for item in dic_col_pct_nan.items() :
        prob_nan = item[ 1 ]
        col = item[ 0 ]
        
        df[ col ] = df[ col ].mask( np.random.random( df.shape[0] ) < prob_nan )
    


# ## <font color = 'green'>Random grouping</font>

# **Algorithm :**
# 
# 
# X categorical variables, values x(1) , ... , x(N)
# 
# y target variable
# 
#     [ 0 ] baseline predictive model
# 
# y predicted by untransformed X -> model performance
# 
# 
# "while" loop :
# 
# 	[ 1 ] alternative models : transforming X by grouping values
# 
#   		[ 1.0 ] choose randomly number of groups ( 1 < n < N )
# 
#   		[ 1.1 ] choose randomly number N(k) ( <= N - k ) of values for each group ( k )
# 
#   				/!\ k-th group cannot have more than N - k elements
# 
#   				exemple : N = 10 distinct values for X
#   						  n = 2 groups of X values
# 
#   						  k = 1 : firt group can put together at most 9 X values
# 
# 
#   		[ 1.2 ] alternative model
# 
# 		y predicted by transformed X -> model performance 
# 
# 	[ 2 ] Comparing baseline vs alternative models
# 
# 		if alternative model improves performance beyond a fixed treshold => keep the alternative model
# 		else try another alternative model
# 

# [ I ] Random dataframe :

# ### Regroupements de modalités suivant variable cible

df = rand_df( n_col_num = 3 , 
              lst_par_col_str = [ ( 25 , 3 ) , ( 7 , 4 ) , ( 2 , 1 ) ] ,
              n_rows = 500 )

# !!! FONCTION RAND_GRP : regroupements aléatoires de modalités suivant variable cible !!!


def rand_grp( df , col , target , n_trial_max , min_improve ):
    """
     df : dataframe
     col : col to group
     target : target column
     n_trial_max : max number of trials
     min_improve : min accuracy improvement
    """
    
    # baseline model :
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn import preprocessing
    
    df_blm = df[ [ col , target ] ]
    lb = preprocessing.LabelBinarizer()
    df_blm[ 'y' ] = lb.fit_transform( df_blm[ target ] )
    df_blm = df_blm.drop( [ target ] , axis = 1 )
    df_blm = pd.get_dummies( df_blm , columns = [ col ] )
    
    y = df_blm[ 'y' ].values
    X = df_blm.drop( [ 'y' ] , axis = 1 ).values
    
    clf = RandomForestClassifier()
    clf.fit( X , y )
    y_pred = clf.predict( X )
    
    blm_perf = accuracy_score( y , y_pred )
    print( 'blm_perf : {0}'.format( blm_perf ) )
    
    # point à voir : traiter regroupement déjà fait ( ne pas incrémenter n_trial )
    
    n_trial = 0 
    perf_var = 0
    
    while n_trial < n_trial_max and perf_var < min_improve :
        print( '[[[[[[[[[[[ trial n° ]]]]]]]]]]] : {0}'.format( n ) )
        
        # value list from the string column :
        set_val = set( df[ col ].value_counts().index )
        #print( 'set_val:{0}'.format( set_val ) )
        
        # number of distinct values from string column ( function col argument ) :
        nbr_val = len( set_val )
        #print( 'nbr_val:{0}'.format( nbr_val ) )
        
        # number of value groups to built :
        nbr_grp = random.randint( 2 , nbr_val - 1 )
        #print( 'nbr_grp:{0}'.format( nbr_grp ) )
        
        # [ * ] building groups :

        # values left for grouping :
        nbr_grp_left = nbr_grp
        set_val_left = set_val
        nbr_val_left = len( set_val_left )

        # empty dictionary for mapping :
        dic_4_map = { }
        
        # building group of values :
        for i in range( nbr_grp ) :
            #print( '[ i ] : {0}'.format( i ) )
    
            # group name :
            grp_name = 'grp_' + str( i )
            #print( 'grp_name:{0}'.format( grp_name ) )
    
            # group number of values :
            if nbr_grp_left == 1 :
                nbr_val_grp = nbr_val_left
            else :
                nbr_val_grp = random.randint( 1 , nbr_val_left - nbr_grp_left + 1  )
            #print( 'nbr_val_grp:{0}'.format( nbr_val_grp ) )
    
            set_grp_val = set( random.choices( list( set_val_left ) , k = nbr_val_grp  ) )
            #print( '******* set_grp_val:{0} *******'.format( set_grp_val ) )
    
            # dictionary value : group for mapping :
            # !!! vérifier que ce groupe n'a pas déjà été testé !!! 
            for val in set_grp_val :
                dic_4_map.update( { val : grp_name } )
    
            #print( 'dic_4_map : {0}'.format( dic_4_map ) )
    
            nbr_grp_left = nbr_grp_left - 1
            #print( 'nbr_grp_left : {0}'.format( nbr_grp_left ) )
    
            set_val_left = set_val_left - set_grp_val
            #print( 'set_val_left : {0}'.format( set_val_left ) )
    
            nbr_val_left = nbr_val_left - len( set_grp_val )
            #print( 'nbr_val_left : {0}'.format( nbr_val_left ) )
        
        # mapping :
        df_copy = df.copy()
        df_copy[ col ] = df_copy[ col ].map( dic_4_map )
        df_copy.head()
        
        # alternative model :
        df_copy = df_copy[ [ col , target ] ]
        lb = preprocessing.LabelBinarizer()
        df_copy[ 'y' ] = lb.fit_transform( df_copy[ target ] )
        df_copy = df_copy.drop( [ target ] , axis = 1 )
        df_copy = pd.get_dummies( df_copy , columns = [ col ] )
    
        y = df_copy[ 'y' ].values
        X = df_copy.drop( [ 'y' ] , axis = 1 ).values
    
        clf = RandomForestClassifier( n_estimators = 100 )
        clf.fit( X , y )
        y_pred = clf.predict( X )
    
        m_perf = accuracy_score( y , y_pred )
        print( 'm_perf : {0}'.format( m_perf ) )
        
        # performance base line vs new model
        perf_var = ( m_perf - blm_perf )  / blm_perf
        print( '------- perf_var : {0} -------'.format( perf_var ) )
        
        # trial counter :
        n_trial = n_trial + 1 
    
    if perf_var >= min_improve :
        return dic_4_map
    else :
        print( 'perf not improved' )


# ### Test 

rand_grp( df , 'col_str_0' , 'col_str_2' , 300 , 0.03 )

