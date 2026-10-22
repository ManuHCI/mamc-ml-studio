import io

import numpy as np
import pandas as pd
import streamlit as st

from mlstudio import clustering, plots, ui
from mlstudio.data_check import column_kind, is_num

ui.page_head(None, "Find patient groups (clustering)",
             "Unsupervised learning: no outcome column - the computer looks for natural groups of similar patients.")
ui.need_data()
ss = st.session_state
df = ss.df

ui.simple("In supervised learning (pages 4-7) we give the model the right answers to learn from. In **clustering** there "
          "is no answer column: the algorithm only looks at how similar patients are to each other and forms groups. "
          "Doctors then study the groups and decide whether they mean something (e.g. 'phenotypes' of diabetes or sepsis).",
          "Supervised vs unsupervised")

usable = [c for c in df.columns if column_kind(df[c]) in ("numeric", "binary", "categorical")]
default = [c for c in usable if is_num(df[c]) and c != ss.target and column_kind(df[c]) == "numeric"]
feats = st.multiselect("Variables used to measure similarity", usable, default=default,
                       help="Usually the clinical measurements. Do not include an outcome you want to compare with afterwards.")
if len(feats) < 2:
    ui.box("warn", "Choose at least two variables.")
    st.stop()
Xall, Z = clustering.prepare(df, feats)
ui.box("note", f"{len(df)} patients, {Z.shape[1]} columns after turning categories into 0/1. All variables are **scaled** "
               "(mean 0, SD 1) so that, e.g., triglycerides in hundreds do not outweigh BMI in tens. Blank cells were filled "
               "with the median.")

methods = list(clustering.CLUSTER_METHODS)
method = st.radio("Method", methods, horizontal=True, format_func=lambda m: clustering.CLUSTER_METHODS[m]["name"])
info = clustering.CLUSTER_METHODS[method]
with st.container(border=True):
    st.markdown(f"**{info['name']}:** {info['one_line']}")
    ui.simple(info["analogy"], "Think of it like this")
    st.markdown("\n".join(f"{i + 1}. {s}" for i, s in enumerate(info["how"])))
    st.markdown(f"**Good:** {info['good']}  \n**Watch out:** {info['bad']}")

if method in ("kmeans", "hier"):
    st.subheader("How many groups?")
    ks, inert, sils = clustering.scan_k(Z, 8, 42)
    c1, c2 = st.columns(2)
    with c1:
        ui.show(plots.elbow(ks, inert, sils), key="sil")
    with c2:
        ui.show(plots.inertia_plot(ks, inert), key="elb")
    ui.simple("The **silhouette score** (-1 to 1) measures how well each patient sits in its own group compared with the "
              "nearest other group: above 0.5 = clear groups, 0.25-0.5 = some structure, below 0.25 = weak. In the **elbow "
              "plot**, look for the bend after which adding groups gains little. Always check that the groups make "
              "clinical sense.", "Choosing k")
    best_k = ks[int(np.nanargmax(sils))]
    k = st.slider("Number of groups (k)", 2, 8, int(best_k))
    linkage = "ward"
    if method == "hier":
        linkage = st.selectbox("Linkage (how distance between groups is measured)", ["ward", "average", "complete", "single"],
                               help="Ward joins the groups that increase the within-group spread the least - usually the best start.")
    labels, sil = clustering.run(method, Z, k=k, linkage=linkage)
else:
    c1, c2 = st.columns(2)
    eps = c1.slider("eps - neighbourhood size (in SD units)", 0.3, 5.0, 1.5, 0.1,
                    help="How close two patients must be to count as neighbours.")
    ms = c2.slider("min_samples - neighbours needed to form a crowd", 2, 30, 5)
    labels, sil = clustering.run("dbscan", Z, eps=eps, min_samples=ms)

ngroups = len(set(labels) - {"Noise"})
noise = int((labels == "Noise").sum())
q = "clear" if sil >= 0.5 else "moderate" if sil >= 0.25 else "weak" if np.isfinite(sil) else "not available"
c = st.columns(3)
with c[0]:
    ui.tile("Groups found", str(ngroups))
with c[1]:
    ui.tile("Silhouette score", ui.num(sil, 2), f"{q} group structure")
with c[2]:
    ui.tile("Noise (unassigned)", str(noise))
if ngroups < 1:
    ui.box("warn", "No groups were found. Increase eps or lower min_samples.")
    st.stop()

coords, var = clustering.pca_coords(Z)
ui.show(plots.cluster_scatter(coords, labels, axis_titles=(f"Component 1 ({var[0] * 100:.0f}% of variation)",
                                                           f"Component 2 ({var[1] * 100:.0f}% of variation)")), key="clsc")
ui.simple("Patients have many measurements, so they live in a many-dimensional space. **PCA** squeezes that space onto a "
          "flat map while keeping as much of the spread as possible. Patients close together on the map are similar.",
          "What is this map?")

means, z = clustering.profile(df, Xall, labels)
ui.show(plots.cluster_profile(z), key="prof")
desc = clustering.describe_groups(z)
for g in z.index:
    st.markdown(f"- **{g}** ({int(means.loc[g, 'Patients'])} patients): {desc[g]}")
with st.expander("Average values of each group (original units)"):
    st.dataframe(means.round(2), width="stretch")

if method == "hier":
    with st.expander("Dendrogram (family tree)"):
        st.pyplot(clustering.dendrogram_figure(Z), width="stretch")

if ss.target and ss.target in df.columns and ss.target not in feats:
    st.subheader(f"Do the groups differ in {ss.target}?")
    if ss.task == "classification":
        ct = pd.crosstab(pd.Series(labels, name="Group"), df[ss.target].astype(str).values, normalize="index") * 100
        st.dataframe(ct.round(1).astype(str) + "%", width="stretch")
    else:
        st.dataframe(pd.DataFrame({"Group": labels, ss.target: df[ss.target].values}).groupby("Group")[ss.target]
                     .describe().round(2), width="stretch")
    st.caption("The outcome was NOT used to form the groups - so a difference here is a genuine finding worth exploring.")

c1, c2 = st.columns(2)
out = df.copy()
out["Cluster_group"] = labels
buf = io.BytesIO()
out.to_excel(buf, index=False)
c1.download_button("Download data with the group of each patient", buf.getvalue(), "data_with_groups.xlsx",
                   icon=":material/download:")
if c2.button("Add the group as a new column to the working data", icon=":material/add:"):
    ui.update_data(out, [{"text": f"Added column 'Cluster_group' ({clustering.CLUSTER_METHODS[method]['name']}, "
                                  f"{ngroups} groups)", "code": None, "key": "cluster"}])
    st.success("Added. You can now use 'Cluster_group' in Explore or as a predictor.")

with st.expander("Watch K-means at work (animation)"):
    nums = [c for c in feats if is_num(df[c])][:2]
    if len(nums) == 2:
        sub = df[nums].dropna().astype(float)
        sub = (sub - sub.mean()) / sub.std()
        fig = plots.kmeans_animation(sub.to_numpy()[:400], k=min(max(ngroups, 2), 6))
        fig.update_xaxes(title=dict(text=f"{nums[0]} (scaled)"))
        fig.update_yaxes(title=dict(text=f"{nums[1]} (scaled)"))
        ui.show(fig, key="kmanim")
        st.caption("Press Play. Each step: every patient joins the nearest centre (X), then each centre moves to the middle of its patients.")
ui.footer()
