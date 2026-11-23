from factor_analyzer import FactorAnalyzer
import numpy as np 
from scipy.linalg import inv
from scipy.stats import chi2 
import matplotlib.pyplot as plt 
from scipy.stats import shapiro,f_oneway,kruskal
import re
def clean_filename(name):
    name = name.lower()
    name = re.sub(r"[^\w\s/-]", "", name)   # ahora permite /
    name = name.replace(" ", "_")
    return name
def alpha_cronbach(df,variable_cols):
    variable=df[variable_cols]
    items=variable.var(axis=0,ddof=1)
    totales=variable.sum(axis=1).var(ddof=1)
    k=len(variable_cols)
    alpha=(k/(k-1))*(1-(items.sum())/(totales))
    return alpha
def mannwhitney_effect(stat, n1, n2):
    N = n1 + n2
    mean_U = n1*n2/2
    std_U = ((n1*n2*(n1+n2+1))/12)**0.5
    z = (stat - mean_U)/std_U
    r = abs(z)/np.sqrt(N)
    return r
def cohens_d(x,y):
    nx=len(x)
    ny=len(y)

    pooled=((nx-1)*x.var()+(ny-1)*y.var())/(nx+ny-2)
    pooled=pooled**0.5

    d=(x.mean()-y.mean())/pooled
    return d
def eta_squared_kruskal(H,n,k):
    return (H-k+1)/(n-k)
    
def kmo(df):
    corr =df.corr().values
    inv_corr= inv(corr)
    partial_corr = - inv_corr/np.outer(np.sqrt(np.diag(inv_corr)),np.sqrt(np.diag(inv_corr)))
    np.fill_diagonal(partial_corr,0)
    #SUma los cuadrados
    a =np.sum(corr**2)-np.sum(np.diag(corr**2))
    b=np.sum(partial_corr**2)
    kmo_global= a/(a+b)
    kmo_per_val={}
    for i in range(len(df.columns)):
        col_corr=np.sum(corr[i,:]**2)-corr[i,i]**2
        col_partial = np.sum(partial_corr[i,:]**2)
        kmo_per_val[df.columns[i]]=col_corr/(col_partial+col_corr)
    return kmo_global,kmo_per_val
def bartlett_sphericity(df):
    n =df.shape[0]
    p =df.shape[1]
    corr_matrix= df.corr().values
    chi2_val = -(n-1-(2*p+5)/6)*np.log(np.linalg.det(corr_matrix))
    df_chi= p*(p-1)/2 
    p_value = 1- chi2.cdf(chi2_val,df_chi)
    return chi2_val, p_value 
from scikit_posthocs import posthoc_dunn

def posthoc_kruskal(df, variable, group_col, p_adjust="bonferroni"):   
    df_clean = df.dropna(subset=[group_col, variable])
    return posthoc_dunn(df_clean, val_col=variable, group_col=group_col, p_adjust=p_adjust)
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def posthoc_anova(df, variable, group_col):
    df_clean = df.dropna(subset=[group_col, variable])
    tukey = pairwise_tukeyhsd(endog=df_clean[variable], groups=df[group_col], alpha=0.05)
    return tukey.summary()
def eigen_analysis(df,variable_name_cols,save_path):
    df_items=df[variable_name_cols]
    fa=FactorAnalyzer(rotation=None)
    fa.fit(df_items)
    eigenvalues,_=fa.get_eigenvalues()
    plt.figure()
    plt.plot(range(1,len(eigenvalues)+1),eigenvalues)
    plt.axhline(1, linestyle="--")
    plt.xlabel("Factor")
    plt.ylabel("Eigenvalue")
    plt.title("Scree Plot")
    if save_path is not None:
        plt.savefig(clean_filename(save_path)+".png",dpi=300,bbox_inches="tight")
    else:
        plt.show()
    total_variance=sum(eigenvalues)
    cumulative=np.cumsum(eigenvalues)/total_variance
    return {
        "eigenvalues":eigenvalues,
        "cumulative_variance":cumulative
    }    
def factor_analysis(df,cols_variable_name, n_factors=3, method="principal", rotation="varimax"):
    df_items = df[cols_variable_name]
    fa = FactorAnalyzer(n_factors=n_factors, method=method, rotation=rotation)
    fa.fit(df_items)
    loadings = fa.loadings_
    variance = fa.get_factor_variance()
    results = {
            "loadings": loadings,
            "variance_explained": variance
    }
    return results

def normality_by_group(df,variable,group_col):
    groups=df[group_col].unique()
    results= {}
    for g in groups :
        data =df[df[group_col]==g][variable]
        stat,p=shapiro(data)
        results[g]={
            "statistics":stat,
            "p_value":p 
        }
    return results
def anova_test(df,variable,group_col):
    groups =[g[variable].values for _,g in df.groupby(group_col)]
    F,p =f_oneway(*groups)
    return F,p
def kruskal_test(df,variable,group_col):
    groups =[g[variable].values for _,g in df.groupby(group_col)]
    H,p =kruskal(*groups)
    return H,p
