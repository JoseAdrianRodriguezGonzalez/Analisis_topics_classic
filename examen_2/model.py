from variables_ import setVariables
from scipy.stats import ttest_ind, mannwhitneyu
import pandas as pd 
from tests import normality_by_group,anova_test,kruskal_test,cohens_d,mannwhitney_effect,eta_squared_kruskal,posthoc_kruskal,posthoc_anova
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
class GenderAnalysis: 
    def __init__(self,df,gender_col="genero"):
        self.df=df 
        self.gender_col=gender_col
    def normality_test(self,variable):
        return normality_by_group(self.df,variable,self.gender_col)   
    def compare(self,variable):
        normality =self.normality_test(variable)
        normal=all(v["p_value"]>0.05 for v in normality.values())
        if normal :
            stat,p=anova_test(self.df,variable,self.gender_col)
            test="ANOVA"
        else:
            stat,p=kruskal_test(self.df,variable,self.gender_col)
            test="Kruskal-Wallis"
        results={
            "variable":variable,
            "test":test,
            "statistic":stat,
            "p_value":p,
            "normality":normality
        }
        return results
    def posthoc(self, variable):
        normality = self.normality_test(variable)
        normal = all(v["p_value"] > 0.05 for v in normality.values())
        if normal:
            return posthoc_anova(self.df, variable, self.gender_col)
        else:
            return posthoc_kruskal(self.df, variable, self.gender_col)


class AgeAnalysis:
    def __init__(self,df,age_col="edad") -> None:
        self.df=df
        self.age_col=age_col
        pass
    def build_age_groups(self):
        bins=[17,20,23,28]
        labels=["18-20","21-23","24-28"]
        self.df["age_group"]=pd.cut(self.df[self.age_col],bins=bins,labels=labels)
        return self.df 
    def normality_test(self,variable):
        return normality_by_group(self.df,variable,self.age_col)   
    def compare(self,variable):
        normality =self.normality_test(variable)
        normal=all(v["p_value"]>0.05 for v in normality.values())
        k=self.df["age_group"].nunique()
        n=len(self.df)
        if normal :
            stat,p=anova_test(self.df,variable,"age_group")
            test="ANOVA"
            effect=stat/(stat+n-k)
        else:
            stat,p=kruskal_test(self.df,variable,"age_group")
            test="Kruskal-Wallis"
            effect=eta_squared_kruskal(stat,n,k)
        means=self.df.groupby("age_group")[variable].mean()
        results={
            "variable":variable,
            "test":test,
            "statistic":stat,
            "p_value":p,
            "effect_size":effect,
            "group_means":means.to_dict()
        }
        return results
    def posthoc(self, variable):
        normality = self.normality_test(variable)
        normal = all(v["p_value"] > 0.05 for v in normality.values())
        if normal:
            return posthoc_anova(self.df, variable, "age_group")
        else:
            return posthoc_kruskal(self.df, variable, "age_group")

class LocationAnalysis:
    def __init__(self, df, location_col="location"):
        self.df = df
        self.location_col = location_col
    def build_location_group(self):
        guanajuato_municipios = [
            "irapuato","leon","salamanca","salvatierra",
            "valle de santiago","guanajuato","abasolo",
            "villagran","cortazar","cueramaro","dolores hidalgo"
        ]
        self.df["location_group"] = self.df[self.location_col].apply(
            lambda x: "Guanajuato" if str(x).lower() in guanajuato_municipios else "Otros estados"
        )
        return self.df
    def normality_test(self, variable):
        return normality_by_group(self.df, variable, "location_group")
    def compare(self, variable):
        normality = self.normality_test(variable)
        normal = all(v["p_value"] > 0.05 for v in normality.values())
        g1 = self.df[self.df["location_group"]=="Guanajuato"][variable]
        g2 = self.df[self.df["location_group"]=="Otros estados"][variable]
        if normal:
            stat, p = ttest_ind(g1, g2)
            test = "t-test"
            effect=cohens_d(g1,g2)
        else:
            stat, p = mannwhitneyu(g1, g2)
            test = "Mann-Whitney"
            effect=mannwhitney_effect(stat,len(g1),len(g2))
        direction=g1.mean()-g2.mean()
        return {
            "variable": variable,
            "test": test,
            "statistic": stat,
            "p_value": p,
            "effect_size":effect,
            "direction":direction
        }
class OriginAnalysis:
    def __init__(self, df, city_col="ciudad"):
        self.df = df
        self.city_col = city_col
    def build_origin_group(self):
        self.df["origin_group"] = self.df[self.city_col].apply(
            lambda x: "Salamanca" if str(x).lower()=="salamanca" else "No Salamanca"
        )
        return self.df

    def normality_test(self, variable):
        return normality_by_group(self.df, variable, "origin_group")

    def compare(self, variable):

        normality = self.normality_test(variable)
        normal = all(v["p_value"] > 0.05 for v in normality.values())

        g1 = self.df[self.df["origin_group"]=="Salamanca"][variable]
        g2 = self.df[self.df["origin_group"]=="No Salamanca"][variable]

        if normal:
            stat, p = ttest_ind(g1, g2)
            test = "t-test"
            effect=cohens_d(g1,g2)
        else:
            stat, p = mannwhitneyu(g1, g2)
            test = "Mann-Whitney"
            effect=mannwhitney_effect(stat,len(g1),len(g2))
        direction=g1.mean()-g2.mean()
        return {
            "variable": variable,
            "test": test,
            "statistic": stat,
            "p_value": p,
            "effect_size":effect,
            "direction":direction
        }
class RegressionModel:
    def __init__(self,df):
        self.df=df 
        self.X=self.df[["AV","CV"]]
        self.Y=self.df["GV"]
    def model(self):
        X=sm.add_constant(self.X)
        model=sm.OLS(self.Y,X).fit()
        self.sumary=model.summary()
    def __str__(self):
        return str(self.sumary)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
def cluster_kmean(df): 
# Probar varios k para ver cuál es el óptimo
    sil_scores = []
    X=df[["AV","CV"]]
    for k in range(2,15):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        sil_scores.append((score,k)) 
    print("Silhouette scores:", sil_scores)
    sil_scores_sorted=sorted(sil_scores)
    print(sil_scores_sorted)
    kmeans = KMeans(n_clusters=sil_scores[-1][1], random_state=42)
    df['cluster_kmeans'] = kmeans.fit_predict(X)
    return df
def Hierarchical(df):
    X=df[["AV","CV","GV"]]
# linkage con método ward (minimiza varianza)
    Z = linkage(X, method='ward')
# Dendrograma
    plt.figure(figsize=(10,5))
    dendrogram(Z, labels=df.index, color_threshold=0)
    plt.title("Dendrograma Hierarchical Clustering")
    plt.show()
    df['cluster_hier'] = fcluster(Z, t=3, criterion='maxclust')
    return df
def dbsc(df):
    X=df[["AV","CV","GV"]]
    dbscan = DBSCAN(eps=0.8, min_samples=5)
    df['cluster_dbscan'] = dbscan.fit_predict(X)
    return df
def visualization(df, cluster_column):
    pca = PCA(n_components=2)
    X = df[["AV","CV","GV"]]
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    clusters = df[cluster_column].unique()
    cmap = plt.get_cmap('Set1')
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']  # círculo, cuadrado, triángulo, diamante, etc.
    for i, cluster in enumerate(clusters):
        cluster_points = X_pca[df[cluster_column] == cluster]
        plt.scatter(
            cluster_points[:,0], cluster_points[:,1],
            color=cmap(i % cmap.N),          # color por cluster
            marker=markers[i % len(markers)], # forma por cluster
            label=f'Cluster {cluster}',
            s=80
        )
    plt.title(f"Clusters {cluster_column} (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Clusters")
    plt.show()
