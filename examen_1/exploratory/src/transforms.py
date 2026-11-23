def normalize_counts(df,emotions):
    df_bebidas_norms=df.copy()
    df_bebidas_norms[emotions]=df[emotions].div(
        df[emotions].sum(axis=1),axis=0)
    return df_bebidas_norms

