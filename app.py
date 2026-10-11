import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Segmentation Dashboard")
st.caption("Explore customer groups, analyze patterns, and filter customers interactively.")

# ---------- Helpers ----------
ID_NAMES = {
    "id", "customerid", "customer_id", "customer id",
    "userid", "user_id", "transactionid", "transaction_id"
}

def is_identifier(column_name):
    normalized = str(column_name).strip().lower().replace("-", "_")
    compact = normalized.replace("_", "").replace(" ", "")
    return normalized in ID_NAMES or compact.endswith("id")

def create_segments(data, requested_clusters=4):
    """Return a copy with a Segment column, using K-Means when needed."""
    result = data.copy()

    # Preserve user-provided labels or an existing cluster column.
    existing = next(
        (c for c in result.columns
         if str(c).strip().lower() in {"segment", "cluster", "cluster_label", "clusterlabel"}),
        None
    )
    if existing is not None:
        if existing != "Segment":
            result = result.rename(columns={existing: "Segment"})
        result["Segment"] = result["Segment"].fillna("Unknown").astype(str)
        return result, None

    numeric_columns = result.select_dtypes(include=np.number).columns.tolist()
    clustering_features = [c for c in numeric_columns if not is_identifier(c)]

    # Prefer familiar customer-segmentation features when available.
    preferred = ["Age", "AnnualIncome", "SpendingScore"]
    preferred_found = [c for c in preferred if c in clustering_features]
    if len(preferred_found) >= 2:
        clustering_features = preferred_found

    if len(clustering_features) < 2:
        return result, "At least two numeric non-ID columns are needed to generate segments."

    # Prepare finite numeric values and impute missing values with medians.
    X = result[clustering_features].replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).fillna(0)

    # Remove constant columns, which add no information to clustering.
    varying = [c for c in X.columns if X[c].nunique(dropna=False) > 1]
    if len(varying) < 2:
        return result, "At least two numeric features must contain varying values."

    X = X[varying]
    if len(X) < 2:
        return result, "At least two customer rows are required to generate segments."

    scaled = StandardScaler().fit_transform(X)
    n_clusters = min(requested_clusters, len(result), len(np.unique(scaled, axis=0)))
    if n_clusters < 2:
        return result, "The available rows do not contain enough distinct values for clustering."

    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(scaled)
    result["Segment"] = [f"Segment {label + 1}" for label in labels]
    return result, None

def numeric_columns_for_filters(data):
    return data.select_dtypes(include=np.number).columns.tolist()

# ---------- Upload ----------
uploaded_file = st.file_uploader(
    "Upload your customer CSV",
    type=["csv"],
    help="Upload a CSV containing customer records. If no Segment/Cluster column exists, K-Means segments are generated automatically."
)

if uploaded_file is None:
    st.info("Upload a customer CSV to open the dashboard. You can use the sample file `customer_segments.csv`.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if raw_df.empty:
    st.error("The uploaded CSV has no data rows.")
    st.stop()

# Clean column labels and duplicate names.
raw_df.columns = [str(c).strip() for c in raw_df.columns]
if raw_df.columns.duplicated().any():
    st.error("Your CSV contains duplicate column names. Rename duplicate columns and upload it again.")
    st.stop()

# Generate segments on the complete dataset, before filtering, so filters do not change cluster assignments.
df, segment_note = create_segments(raw_df)

if segment_note:
    st.warning(f"Segment generation note: {segment_note}")

if "Segment" in df.columns and "Segment" not in raw_df.columns:
    st.success("Customer segments were generated automatically with K-Means.")

# ---------- Sidebar filters ----------
st.sidebar.header("🔎 Customer Filters")
filtered_df = df.copy()

categorical_columns = [
    c for c in df.columns
    if c != "Segment" and (
        pd.api.types.is_object_dtype(df[c])
        or pd.api.types.is_categorical_dtype(df[c])
        or pd.api.types.is_bool_dtype(df[c])
    )
]
numeric_filter_columns = numeric_columns_for_filters(df)

for col in categorical_columns:
    options = sorted(df[col].dropna().astype(str).unique().tolist())
    selected = st.sidebar.multiselect(
        f"{col}",
        options=options,
        default=options,
        key=f"cat_filter_{col}",
    )
    if len(selected) < len(options):
        filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)]

for col in numeric_filter_columns:
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        continue
    minimum = float(series.min())
    maximum = float(series.max())
    if minimum < maximum:
        selected_range = st.sidebar.slider(
            f"{col} range",
            min_value=minimum,
            max_value=maximum,
            value=(minimum, maximum),
            key=f"num_filter_{col}",
        )
        filtered_df = filtered_df[
            pd.to_numeric(filtered_df[col], errors="coerce").between(
                selected_range[0], selected_range[1], inclusive="both"
            )
        ]

if st.sidebar.button("Reset filters"):
    st.rerun()

# ---------- Overview metrics ----------
total_customers = len(df)
filtered_customers = len(filtered_df)
numeric_feature_count = len(df.select_dtypes(include=np.number).columns)
categorical_feature_count = len([
    c for c in df.columns
    if c != "Segment" and (
        pd.api.types.is_object_dtype(df[c])
        or pd.api.types.is_categorical_dtype(df[c])
        or pd.api.types.is_bool_dtype(df[c])
    )
])

st.subheader("📌 Customer Overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Customers", f"{total_customers:,}")
m2.metric("Filtered Customers", f"{filtered_customers:,}")
m3.metric("Numeric Features", numeric_feature_count)
m4.metric("Categorical Features", categorical_feature_count)

if filtered_df.empty:
    st.warning("No customers match the selected filters. Adjust the filters in the sidebar.")
    st.stop()

# ---------- Tabs ----------
tab_segments, tab_features, tab_corr, tab_data = st.tabs([
    "📊 Segment Analysis",
    "📈 Feature Explorer",
    "🔥 Correlation Analysis",
    "📋 Customer Data",
])

with tab_segments:
    st.subheader("Customer Segment Analysis")
    if "Segment" not in filtered_df.columns:
        st.warning("Segments could not be created. Ensure your CSV has at least two varying numeric customer features.")
    else:
        segment_counts = (
            filtered_df["Segment"].fillna("Unknown").astype(str)
            .value_counts()
            .rename_axis("Segment")
            .reset_index(name="Customers")
        )

        count_col, share_col = st.columns(2)
        with count_col:
            st.markdown("**Customers in Each Segment**")
            fig = px.bar(
                segment_counts, x="Segment", y="Customers",
                color="Segment", text="Customers",
                title="Segment Size",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with share_col:
            st.markdown("**Segment Distribution**")
            fig = px.pie(
                segment_counts, names="Segment", values="Customers",
                hole=0.45, title="Share of Filtered Customers",
            )
            st.plotly_chart(fig, use_container_width=True)

        summary_metrics = {"Customers": ("Segment", "size")}
        for col, label in [
            ("Age", "Average Age"),
            ("AnnualIncome", "Average Annual Income"),
            ("SpendingScore", "Average Spending Score"),
        ]:
            if col in filtered_df.columns:
                summary_metrics[label] = (col, "mean")

        summary = (
            filtered_df.groupby("Segment", dropna=False)
            .agg(**summary_metrics)
            .reset_index()
        )
        st.markdown("**Segment Summary**")
        st.dataframe(summary.round(2), use_container_width=True, hide_index=True)

        if {"AnnualIncome", "SpendingScore"}.issubset(filtered_df.columns):
            st.markdown("**Income vs Spending by Segment**")
            hover_cols = [c for c in ["CustomerID", "Age", "Gender"] if c in filtered_df.columns]
            fig = px.scatter(
                filtered_df,
                x="AnnualIncome",
                y="SpendingScore",
                color="Segment",
                hover_data=hover_cols or None,
                title="Customer Groups by Income and Spending",
            )
            st.plotly_chart(fig, use_container_width=True)

        segment_numeric = [
            c for c in ["Age", "AnnualIncome", "SpendingScore"]
            if c in filtered_df.select_dtypes(include=np.number).columns
        ]
        if segment_numeric:
            averages = (
                filtered_df.groupby("Segment")[segment_numeric]
                .mean()
                .reset_index()
                .melt(id_vars="Segment", var_name="Feature", value_name="Average")
            )
            fig = px.bar(
                averages, x="Feature", y="Average", color="Segment",
                barmode="group", title="Average Feature Values by Segment",
            )
            st.plotly_chart(fig, use_container_width=True)

with tab_features:
    st.subheader("Feature Explorer")
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns are available to plot.")
    else:
        left, right = st.columns(2)
        with left:
            x_feature = st.selectbox("X-axis feature", numeric_cols, index=0)
        with right:
            default_y = 1 if len(numeric_cols) > 1 else 0
            y_feature = st.selectbox("Y-axis feature", numeric_cols, index=default_y)

        if x_feature == y_feature:
            fig = px.histogram(
                filtered_df, x=x_feature, color="Segment" if "Segment" in filtered_df.columns else None,
                marginal="box", title=f"Distribution of {x_feature}",
            )
        else:
            fig = px.scatter(
                filtered_df, x=x_feature, y=y_feature,
                color="Segment" if "Segment" in filtered_df.columns else None,
                hover_data=[c for c in ["CustomerID", "Gender"] if c in filtered_df.columns],
                title=f"{y_feature} vs {x_feature}",
            )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Selected Feature Statistics**")
        st.dataframe(
            filtered_df[[x_feature] if x_feature == y_feature else [x_feature, y_feature]].describe().round(2),
            use_container_width=True,
        )

with tab_corr:
    st.subheader("Correlation Analysis")
    numeric_df = filtered_df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        st.info("At least two numeric columns are required for correlation analysis.")
    else:
        corr = numeric_df.corr(numeric_only=True)
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Numeric Feature Correlation Heatmap",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Correlation measures linear association; it does not establish causation.")

with tab_data:
    st.subheader("Customer Data")
    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} rows after filters.")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Customer CSV",
        data=csv_bytes,
        file_name="filtered_customer_segments.csv",
        mime="text/csv",
    )

with st.expander("About this dashboard"):
    st.write(
        "If your CSV does not already contain a Segment or Cluster column, "
        "the app uses K-Means on scaled numeric customer features to generate "
        "up to four groups. Cluster numbers are arbitrary labels, not rankings. "
        "Review the segment averages to understand the groups."
    )
