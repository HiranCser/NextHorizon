from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Optional
import sys
import os
# Ensure project root is on sys.path so local packages like `utils` can be imported when
# Streamlit runs from the `ui` directory.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.feature_engineering import run_pipeline, spacy_model_available
import json
import os
import plotly.express as px
import collections

def _load_df(preferred_clean_path: str, fallback_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(preferred_clean_path)
    except Exception:
        try:
            return pd.read_csv(fallback_path)
        except Exception:
            return pd.DataFrame()

def render():
    st.title("Diagnostics: Data Quality & EDA")
    st.markdown("Use this page to inspect cleaned datasets, missingness, and outliers.")

    jd_clean = _load_df('build_jd_dataset/jd_database.clean.csv', 'build_jd_dataset/jd_database.csv')
    train_clean = _load_df('build_training_dataset/training_database.clean.csv', 'build_training_dataset/training_database.csv')

    st.header("Datasets Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("JD rows", len(jd_clean))
        st.write(jd_clean.columns.tolist()[:12])
    with col2:
        st.metric("Training rows", len(train_clean))
        st.write(train_clean.columns.tolist()[:12])

    if not jd_clean.empty:
        st.header("Job Description Dataset")
        st.subheader("Missingness (top columns)")
        miss = jd_clean.isna().sum().sort_values(ascending=False).head(20)
        st.bar_chart(miss)

        st.subheader("Top source domains")
        if 'source_domain_norm' in jd_clean.columns:
            dfdom = jd_clean['source_domain_norm'].fillna('')
        else:
            dfdom = jd_clean['source_domain'].fillna('')
        domvc = dfdom.value_counts().reset_index()
        domvc.columns = ['domain','count']
        st.plotly_chart(px.bar(domvc.head(25), x='domain', y='count', title='Top source domains'))

        st.subheader("Experience numeric distribution")
        numcols = [c for c in ['exp_min_years','exp_max_years'] if c in jd_clean.columns]
        for c in numcols:
            fig = px.histogram(jd_clean, x=c, nbins=30, title=f'{c} distribution')
            st.plotly_chart(fig)

        st.subheader("Sample extreme JD text lengths")
        jd_clean['text_len'] = jd_clean.get('jd_text','').astype(str).str.len()
        short = jd_clean.nsmallest(3, 'text_len')[['role_title','source_url','text_len','jd_text']]
        long = jd_clean.nlargest(3, 'text_len')[['role_title','source_url','text_len','jd_text']]
        st.write("Shortest examples:")
        for _, r in short.iterrows():
            st.write(r.to_dict())
        st.write("Longest examples:")
        for _, r in long.iterrows():
            st.write(r.to_dict())

    if not train_clean.empty:
        st.header("Training Dataset")
        st.subheader("Missingness (top columns)")
        miss = train_clean.isna().sum().sort_values(ascending=False).head(20)
        st.bar_chart(miss)

        st.subheader("Provider distribution")
        prov = train_clean.get('provider_normalized') if 'provider_normalized' in train_clean.columns else train_clean.get('provider')
        if prov is not None:
            pv = prov.fillna('Unknown').value_counts().reset_index()
            pv.columns = ['provider','count']
            st.plotly_chart(px.bar(pv.head(30), x='provider', y='count', title='Top providers'))

        st.subheader("Hours & Rating distributions")
        for c in ['hours','rating']:
            if c in train_clean.columns:
                fig = px.histogram(train_clean, x=c, nbins=40, title=f'{c} distribution')
                st.plotly_chart(fig)

        st.subheader("Sample course entries with extreme description length")
        train_clean['desc_len'] = train_clean.get('description','').astype(str).str.len()
        shortc = train_clean.nsmallest(3,'desc_len')[['training_id','title','provider','desc_len','link']]
        longc = train_clean.nlargest(3,'desc_len')[['training_id','title','provider','desc_len','link']]
        st.write("Shortest descriptions:")
        st.write(shortc.to_dict(orient='records'))
        st.write("Longest descriptions (trimmed):")
        for _, r in longc.iterrows():
            d = r.to_dict()
            d['link'] = d.get('link','')
            st.write({k: (v if k!='description' else str(v)[:400]) for k,v in d.items()})

        st.markdown("---")
        st.subheader("Top Entities")
        st.write("Visualize the most frequent named entities found in course descriptions. Uses the pipeline report if available, or computes entities with spaCy when the model is installed.")

        report_path = 'build_training_dataset/training_database.report.json'
        top_entities = None
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    rp = json.load(f)
                top_entities = rp.get('top_entities') or []
            except Exception:
                top_entities = None

        if not top_entities and spacy_model_available():
            # compute entities live (may be slow)
            if st.button('Compute Top Entities (live)'):
                with st.spinner('Computing entities, this may take a while...'):
                    try:
                        import spacy
                        nlp = spacy.load('en_core_web_sm')
                        ctr = collections.Counter()
                        for txt in train_clean.get('description','').fillna('').astype(str):
                            if not txt:
                                continue
                            doc = nlp(txt)
                            for e in doc.ents:
                                ctr[(e.label_, e.text.strip())] += 1
                        top = ctr.most_common(50)
                        top_entities = [{'label': l[0][0], 'text': l[0][1], 'count': l[1]} for l in top]
                    except Exception as e:
                        st.error(f'Live entity computation failed: {e}')
                        top_entities = []
        elif not top_entities:
            st.info('No report found and spaCy model not available. Run the feature pipeline with enrichment or install spaCy.')

        if top_entities:
            # aggregate by text (or label+text) and plot top 25
            try:
                df_ent = pd.DataFrame(top_entities)
                # create a display label
                df_ent['display'] = df_ent.apply(lambda r: f"{r['text']} ({r['label']})", axis=1)
                df_ent = df_ent.sort_values('count', ascending=False).head(25)
                fig = px.bar(df_ent, x='display', y='count', title='Top Entities (text (label))')
                fig.update_layout(xaxis_tickangle=-45, height=500)
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f'Failed to render top entities: {e}')

        st.markdown("---")
        st.subheader("Feature Engineering")
        st.write("Run feature pipeline to compute TF-IDF and derived features for the training dataset.")
        col_a, col_b = st.columns([1,2])
        with col_a:
            enrich_toggle = st.checkbox('Enable NER/POS enrichment', value=True)
        with col_b:
            model_status = 'Available' if spacy_model_available() else 'Missing'
            st.markdown(f"**spaCy model:** {model_status}")

        if st.button('Run Feature Pipeline (training)'):
            with st.spinner('Running feature pipeline...'):
                out_csv = 'build_training_dataset/training_database.features.csv'
                out_prefix = 'build_training_dataset/training_database'
                try:
                    res = run_pipeline('build_training_dataset/training_database.csv', out_csv, out_prefix, enrich=enrich_toggle)
                    st.success('Feature pipeline completed')
                    st.write(res)
                    df_feat = pd.read_csv(out_csv)
                    st.write('Sample rows with new feature columns:')
                    cols_show = [c for c in ['title_len','description_len','skill_count','hours_bucket','rating_bucket','readability','ents_total'] if c in df_feat.columns]
                    st.dataframe(df_feat[cols_show].head(10))
                    # show report if exists
                    try:
                        import json
                        report_path = out_prefix + '.report.json'
                        with open(report_path, 'r', encoding='utf-8') as f:
                            rp = json.load(f)
                        st.write('Run report:')
                        st.json(rp)
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f'Feature pipeline failed: {e}')

    st.markdown("---")
    st.markdown("Diagnostics page - generated from cleaned CSVs if available, else original CSVs.")

    # Evaluation reports
    st.markdown('---')
    st.subheader('Evaluation Reports')
    import glob
    report_files = sorted(glob.glob('reports/*.json'))
    if not report_files:
        st.info('No evaluation reports found in `reports/`. Run the evaluation script to generate reports.')
    else:
        sel = st.selectbox('Choose evaluation report', options=report_files)
        if sel:
            try:
                with open(sel, 'r', encoding='utf-8') as f:
                    rpt = json.load(f)
                # Show key metrics if present
                if 'precision_recall_at_k' in rpt:
                    st.markdown('**Retrieval metrics**')
                    pr = rpt['precision_recall_at_k']
                    for k, v in pr.items():
                        st.write(f'Precision@{k}: {v.get("precision_mean")} — Recall@{k}: {v.get("recall_mean")}')
                if 'mrr_at_k' in rpt:
                    st.write(f"**MRR@k:** {rpt.get('mrr_at_k')}")
                if 'ndcg_at_k' in rpt:
                    st.write(f"**NDCG@k:** {rpt.get('ndcg_at_k')}")
                if 'embedding_health' in rpt:
                    st.markdown('**Embedding health**')
                    eh = rpt['embedding_health']
                    st.write(f"Mean cosine: {eh.get('mean_cosine')}, std: {eh.get('std_cosine')}, zero-norm rate: {eh.get('zero_norm_rate')}")
                st.markdown('**Full report**')
                st.json(rpt)
            except Exception as e:
                st.error(f'Failed to load evaluation report: {e}')

if __name__ == '__main__':
    render()
