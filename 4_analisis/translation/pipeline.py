from translation.translation import Translator  
def normalize_language(df,text_col="comentario_clean",lang_col="lang"):
    translator=Translator()
    df=df.copy()
    mask_translate=df[lang_col].isin(["en","mix","mixed"])
    texts_to_translate=(df.loc[mask_translate,text_col].fillna("").astype(str).tolist())
    if len(texts_to_translate)>0:

        print(f"Traduciendo {len(texts_to_translate)} textos...")
        translated=translator.translate_batch(texts_to_translate)
        df.loc[mask_translate,text_col]=translated
    return df
