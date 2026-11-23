import pandas as pd 
from tests import alpha_cronbach,kmo,bartlett_sphericity,eigen_analysis,factor_analysis
from factor_analyzer import FactorAnalyzer
from descriptivos import DA
class BaseAnalysis:
    def __init__(self,df:pd.DataFrame):
        self.df=df 

class setVariables(BaseAnalysis):
    def __init__(self, df: pd.DataFrame,col_va:list[str],col_cv:list[str],col_gv:list[str]):
        super().__init__(df)
        self.AV=col_va
        self.CV=col_cv
        self.GV=col_gv
        self.dictionary_name={"AV": self.AV,
                              "CV": self.CV,
                                "GV": self.GV
                              }
    def info_variable_complete(self,variable_name:str):
        descr=DA(self.df)
        results={col:descr.info_column(col) for col in self.dictionary_name[variable_name] }
        return pd.DataFrame(results).T 
    def build_score(self, variable_name: str, method="mean"):
        cols = self.dictionary_name[variable_name]
        if method=="mean":
            return self.df[cols].mean(axis=1)
        elif method=="median":
            return self.df[cols].median(axis=1)
    def get_all(self):
        
        self.df["AV"] = self.build_score("AV")
        self.df["CV"] = self.build_score("CV")
        self.df["GV"] = self.build_score("GV")

        self.score_variable = {
            "AV": self.df["AV"],
            "CV": self.df["CV"],
            "GV": self.df["GV"]
        }

        return self.df
    def get_info(self,variable_name):
        df=pd.DataFrame(self.score_variable)
        self.variables_score=DA(df)
        metrics=self.variables_score.info_column(variable_name)
        return metrics
    def plot_histo_variable(self,variable_name,path):
        self.variables_score.plot_histogram(variable_name,save_path=path)

    def plot_boxplot_variable(self,variable_name,path):
        self.variables_score.plot_boxplots(variable_name,save_path=path)
    def get_alpha(self):
        self.alphas={col:alpha_cronbach(self.df,val) for col,val in self.dictionary_name.items() }
        df_alpha=pd.DataFrame(list(self.alphas.items()),
                              columns=["variable","cronbach_alpha"])
        return df_alpha
    def correlations(self,without):
        df_items=self.df 
        if without!=None:
            pre_dict={col: self.df[col] for col in df_items.columns if col not in without}
            df_items=pd.DataFrame(pre_dict) 
        correlation={"pearson":df_items.corr(method="pearson"),
                     "spearman":df_items.corr(method="spearman")}
        return correlation
    def correlations_scores(self):
        return {
            "pearson":self.variables_score.df.corr(method="pearson"),
            "spearman":self.variables_score.df.corr(method="spearman")
        }
    def save_correlations(self,correlations,folder,):
        pearson=correlations["pearson"]
        spearman=correlations["spearman"]
        pearson.to_csv(f"{folder}correlation_pearson.csv")
        spearman.to_csv(f"{folder}correlation_spearman.csv")
        
    def adequacy_tests(self,variable_name):
        cols = self.dictionary_name[variable_name]
        df_items = self.df[cols]
        kmo_global,kmo_per_var = kmo(df_items)
        chi2_val,p_value =bartlett_sphericity(df_items)
        results = {
            "KMO_global":kmo_global,
            "KMO_per_variable":kmo_per_var,
            "bartlett_sphericity":chi2_val,
            "Bartlett_p":p_value,
        }
        return results
    def save_adequacy_tests(self, variable_name, path):
        results = self.adequacy_tests(variable_name)
        kmo_global = results["KMO_global"]
        bartlett_chi2 = results["bartlett_sphericity"]
        bartlett_p = results["Bartlett_p"]
        factorable = (kmo_global >= 0.6) and (bartlett_p < 0.05)
        df_global = pd.DataFrame({
            "metric": [
                "KMO_global",
                "Bartlett_chi2",
                "Bartlett_p",
                "Factor_analysis_feasible"
            ],
            "value": [
                kmo_global,
                bartlett_chi2,
                bartlett_p,
                factorable
            ]
        })

        df_kmo_items = pd.DataFrame.from_dict(
            results["KMO_per_variable"],
            orient="index",
            columns=["KMO"]
        )
        df_kmo_items["quality"] = pd.cut(
            df_kmo_items["KMO"],
            bins=[0,0.5,0.6,0.7,0.8,1],
           labels=["unacceptable","poor","mediocre","good","excellent"]
        )
        df_kmo_items["recommended_remove"] = df_kmo_items["KMO"] < 0.5
        df_global.to_csv(f"{path}/{variable_name}_adequacy_global.csv", index=False)
        df_kmo_items.to_csv(f"{path}/{variable_name}_kmo_per_item.csv")

        return factorable
    def eigen_analysis(self,variable_name,path):
        return eigen_analysis(self.df,self.dictionary_name[variable_name],path)

    def factorial_analysis(self,variable_name):
        return factor_analysis(self.df,self.dictionary_name[variable_name])
