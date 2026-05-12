import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Hotel Booking Demand Dashboard",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

BRAND_GREEN = "#2D6A4F"
BRAND_LIGHT = "#95D5B2"
BRAND_ORANGE = "#E76F51"
BG = "#F8F9FA"
MONTH_ORDER = list(calendar.month_name)[1:]
MONTH_TO_NUM = {m: i for i, m in enumerate(calendar.month_name) if m}
SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

st.markdown(
    """
    <style>
        .main { background-color: #F8F9FA; }
        div[data-testid="stMetric"] {
            background: grey;
            border: 1px solid #E8ECEF;
            padding: 18px;
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.04);
        }
        .section-title {
            font-size: 1.25rem;
            font-weight: 800;
            color: #1B4332;
            margin-top: 1.2rem;
        }
        .small-note {
            color: #5f6b66;
            font-size: 0.95rem;
        }
        .ml-metric-card {
            background: white;
            border: 1px solid #E8ECEF;
            border-radius: 14px;
            padding: 16px 20px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.05);
            text-align: center;
        }
        .ml-metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #2D6A4F;
        }
        .ml-metric-label {
            font-size: 0.85rem;
            color: #6c757d;
            margin-top: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        local_csv = Path(__file__).with_name("hotel_bookings.csv")
        fallback_csv = Path("hotel_bookings.csv")
        if local_csv.exists():
            df = pd.read_csv(local_csv)
        elif fallback_csv.exists():
            df = pd.read_csv(fallback_csv)
        else:
            raise FileNotFoundError("hotel_bookings.csv not found. Upload it from the sidebar.")

    df = df.copy()
    df["children"] = df["children"].fillna(0)
    df["country"] = df["country"].fillna("Unknown")
    df["agent"] = df["agent"].fillna(0)
    df["company"] = df["company"].fillna(0)
    df = df[df["adults"] > 0]
    for col in ["children", "babies", "stays_in_weekend_nights", "stays_in_week_nights", "lead_time", "adr"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    df["arrival_month"] = df["arrival_date_month"].map(MONTH_TO_NUM)
    df["arrival_date"] = pd.to_datetime(
        dict(year=df["arrival_date_year"], month=df["arrival_month"], day=df["arrival_date_day_of_month"]),
        errors="coerce",
    )
    df["arrival_weekday"] = df["arrival_date"].dt.day_name()
    df["arrival_date_month_name"] = pd.Categorical(df["arrival_date_month"], categories=MONTH_ORDER, ordered=True)
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["total_revenue"] = df["adr"] * df["total_nights"]
    df["has_agent"] = (df["agent"] > 0).astype(int)
    df["has_company"] = (df["company"] > 0).astype(int)
    df["was_on_waiting_list"] = (df["days_in_waiting_list"] > 0).astype(int)
    df["has_prev_cancellations"] = (df["previous_cancellations"] > 0).astype(int)
    df["is_loyal_guest"] = (df["previous_bookings_not_canceled"] > 0).astype(int)
    df["has_baby"] = (df["babies"] > 0).astype(int)
    df["has_booking_changes"] = (df["booking_changes"] > 0).astype(int)
    df["is_free_room"] = (df["adr"] == 0).astype(int)
    df["has_parking"] = (df["required_car_parking_spaces"] > 0).astype(int)
    df["is_family"] = (df["children"] > 0).astype(int)
    df["is_high_season"] = df["arrival_month"].isin([6, 7, 8]).astype(int)
    df["is_long_lead"] = (df["lead_time"] > 90).astype(int)
    df["season"] = df["arrival_month"].map(SEASON_MAP)
    bins = [0, 7, 30, 90, 180, 365, 9999]
    labels = ["0–7d", "8–30d", "31–90d", "91–180d", "181–365d", "365d+"]
    df["lead_bucket"] = pd.cut(df["lead_time"], bins=bins, labels=labels, include_lowest=True)
    return df


@st.cache_data(show_spinner=True)
def run_ml_pipeline(_df):
    """Train models and return all results needed for the ML tab."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import (
        classification_report, f1_score,
        precision_recall_curve, roc_auc_score,
        roc_curve, average_precision_score, confusion_matrix,
    )
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier

    df = _df.copy()

    # Build model dataframe
    viz_cols = ['arrival_date_month_name', 'lead_bucket', 'season', 'total_revenue',
                'arrival_date', 'arrival_weekday']
    drop_cols = [c for c in viz_cols if c in df.columns]
    df_model = df.drop(columns=drop_cols)

    # Keep only columns that exist and are useful
    keep_cols = [
        'hotel', 'meal', 'country', 'market_segment', 'distribution_channel',
        'reserved_room_type', 'customer_type', 'lead_time', 'arrival_date_week_number',
        'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children',
        'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
        'booking_changes', 'adr', 'total_of_special_requests', 'has_agent',
        'arrival_month', 'arrival_weekday' if 'arrival_weekday' in df_model.columns else None,
        'was_on_waiting_list', 'has_prev_cancellations', 'is_loyal_guest', 'has_baby',
        'has_booking_changes', 'is_free_room', 'has_parking', 'total_nights',
        'total_guests', 'is_family', 'is_high_season', 'is_long_lead', 'is_canceled',
    ]
    keep_cols = [c for c in keep_cols if c is not None and c in df_model.columns]
    df_model = df_model[keep_cols].dropna()

    X = df_model.drop(columns=['is_canceled'])
    y = df_model['is_canceled']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # Group rare countries
    def group_rare(train, val, test, col, top_n=20):
        train, val, test = train.copy(), val.copy(), test.copy()
        for d in [train, val, test]:
            d[col] = d[col].astype(str)
        top_vals = train[col].value_counts().nlargest(top_n).index
        for d in [train, val, test]:
            d[col] = d[col].where(d[col].isin(top_vals), 'Other')
        return train, val, test

    X_train, X_val, X_test = group_rare(X_train, X_val, X_test, 'country', 20)

    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
    ])
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc = preprocessor.transform(X_val)
    X_test_enc = preprocessor.transform(X_test)

    neg, pos = np.bincount(y_train)
    class_ratio = neg / pos

    def best_threshold(model, Xtr, ytr, Xv, yv, min_recall=0.70):
        model.fit(Xtr, ytr)
        y_prob = model.predict_proba(Xv)[:, 1]
        precision, recall, thresholds = precision_recall_curve(yv, y_prob)
        f1 = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
        df_thr = pd.DataFrame({'threshold': thresholds, 'precision': precision[:-1],
                               'recall': recall[:-1], 'f1': f1})
        valid = df_thr[df_thr['recall'] >= min_recall]
        if valid.empty:
            valid = df_thr
        best = valid.sort_values('f1', ascending=False).iloc[0]
        return model, y_prob, best

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'LightGBM': LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                   scale_pos_weight=class_ratio, random_state=42, verbose=-1),
        'LightGBM (Tuned)': LGBMClassifier(n_estimators=500, learning_rate=0.03, max_depth=7,
                                            num_leaves=63, min_child_samples=20, subsample=0.8,
                                            colsample_bytree=0.8, scale_pos_weight=class_ratio,
                                            random_state=42, verbose=-1),
    }

    val_results = {}
    fitted_models = {}
    for name, model in models.items():
        fitted, y_prob, best = best_threshold(model, X_train_enc, y_train, X_val_enc, y_val)
        val_results[name] = {
            'threshold': best['threshold'],
            'precision': best['precision'],
            'recall': best['recall'],
            'f1': best['f1'],
        }
        fitted_models[name] = (fitted, best['threshold'])

    results_df = pd.DataFrame(val_results).T.sort_values('f1', ascending=False)
    best_name = results_df.index[0]
    best_thr = results_df.loc[best_name, 'threshold']

    # Retrain best model on train+val, evaluate on test
    from scipy.sparse import vstack
    best_model = models[best_name]
    X_full_enc = vstack([X_train_enc, X_val_enc])
    y_full = pd.concat([y_train, y_val])
    best_model.fit(X_full_enc, y_full)
    y_test_prob = best_model.predict_proba(X_test_enc)[:, 1]
    y_test_pred = (y_test_prob >= best_thr).astype(int)

    roc_auc = roc_auc_score(y_test, y_test_prob)
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_test_prob)
    ap = average_precision_score(y_test, y_test_prob)
    cm = confusion_matrix(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred, output_dict=True)

    # Feature importances
    feat_names = num_cols + list(
        preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    )
    importances = best_model.feature_importances_
    feat_imp = pd.DataFrame({'feature': feat_names, 'importance': importances}) \
        .sort_values('importance', ascending=False).head(20)

    # SMOTE
    try:
        from imblearn.over_sampling import SMOTE
        from scipy.sparse import issparse
        X_tr_dense = X_train_enc.toarray() if issparse(X_train_enc) else X_train_enc
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_sm, y_sm = sm.fit_resample(X_tr_dense, y_train)
        lgbm_smote = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
        X_val_dense = X_val_enc.toarray() if issparse(X_val_enc) else X_val_enc
        _, y_prob_sm, best_sm = best_threshold(lgbm_smote, X_sm, y_sm, X_val_dense, y_val)
        smote_results = {
            'threshold': best_sm['threshold'],
            'precision': best_sm['precision'],
            'recall': best_sm['recall'],
            'f1': best_sm['f1'],
            'before': list(np.bincount(y_train)),
            'after': list(np.bincount(y_sm)),
        }
        smote_available = True
    except Exception as e:
        smote_results = {}
        smote_available = False

    return {
        'results_df': results_df,
        'best_name': best_name,
        'best_thr': best_thr,
        'roc_auc': roc_auc,
        'fpr': fpr, 'tpr': tpr,
        'precision_vals': precision_vals, 'recall_vals': recall_vals,
        'ap': ap,
        'cm': cm,
        'report': report,
        'feat_imp': feat_imp,
        'cancel_mean': y.mean(),
        'smote_results': smote_results,
        'smote_available': smote_available,
        'test_size': len(y_test),
        'train_size': len(y_train),
    }


def pct(x):
    return f"{x:.1%}" if pd.notna(x) else "0.0%"

def number(x):
    return f"{x:,.0f}"

def money(x):
    return f"${x:,.0f}"


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🏨 Hotel Dashboard")
st.sidebar.caption("Upload the CSV or keep `hotel_bookings.csv` beside this app.")
uploaded_file = st.sidebar.file_uploader("Upload hotel_bookings.csv", type=["csv"])

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(str(e))
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

hotel_options = sorted(df["hotel"].dropna().unique())
hotels = st.sidebar.multiselect("Hotel type", hotel_options, default=hotel_options)

year_options = sorted(df["arrival_date_year"].dropna().unique())
years = st.sidebar.multiselect("Arrival year", year_options, default=year_options)

month_options = [m for m in MONTH_ORDER if m in set(df["arrival_date_month"].dropna())]
months = st.sidebar.multiselect("Arrival month", month_options, default=month_options)

segment_options = sorted(df["market_segment"].dropna().unique())
segments = st.sidebar.multiselect("Market segment", segment_options, default=segment_options)

customer_options = sorted(df["customer_type"].dropna().unique())
customer_types = st.sidebar.multiselect("Customer type", customer_options, default=customer_options)

min_adr, max_adr = float(df["adr"].min()), float(df["adr"].quantile(0.99))
adr_range = st.sidebar.slider("ADR range", min_value=min_adr, max_value=max_adr, value=(min_adr, max_adr))

cancel_filter = st.sidebar.radio("Booking status", ["All", "Canceled only", "Not canceled only"], horizontal=False)

filtered = df[
    df["hotel"].isin(hotels)
    & df["arrival_date_year"].isin(years)
    & df["arrival_date_month"].isin(months)
    & df["market_segment"].isin(segments)
    & df["customer_type"].isin(customer_types)
    & df["adr"].between(adr_range[0], adr_range[1])
].copy()

if cancel_filter == "Canceled only":
    filtered = filtered[filtered["is_canceled"] == 1]
elif cancel_filter == "Not canceled only":
    filtered = filtered[filtered["is_canceled"] == 0]

# ── Header ───────────────────────────────────────────────────────────────────
st.title("Hotel Booking Demand — Visualization & Analysis Dashboard")
st.markdown(
    "<div class='small-note'>Interactive Streamlit dashboard based on the notebook visualization section: revenue, cancellations, ADR, guest behavior, seasonality, engineered feature analysis, and ML model results.</div>",
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning("No rows match the selected filters. Change the sidebar filters.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Bookings", number(len(filtered)))
with k2:
    st.metric("Cancellation Rate", pct(filtered["is_canceled"].mean()))
with k3:
    st.metric("Total Revenue", money(filtered["total_revenue"].sum()))
with k4:
    st.metric("Average ADR", money(filtered["adr"].mean()))
with k5:
    st.metric("Avg. Lead Time", f"{filtered['lead_time'].mean():.0f} days")


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview",
    "Revenue & ADR",
    "Cancellation Analysis",
    "Guests & Segments",
    "Correlation & Features",
    "Data Explorer",
    "🤖 ML Results",
])

# ── Tab 1 — Overview ─────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-title'>Hotel Overview</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    hotel_counts = filtered["hotel"].value_counts().reset_index()
    hotel_counts.columns = ["hotel", "count"]
    fig = px.pie(hotel_counts, values="count", names="hotel", title="Hotel Type Proportion", hole=0.45,
                 color_discrete_sequence=[BRAND_GREEN, BRAND_ORANGE])
    fig.update_layout(template="plotly_white", height=430)
    c1.plotly_chart(fig, use_container_width=True)

    night_vals = pd.DataFrame({
        "Night Type": ["Weekend Nights", "Week Nights"],
        "Total Nights": [filtered["stays_in_weekend_nights"].sum(), filtered["stays_in_week_nights"].sum()],
    })
    fig = px.pie(night_vals, values="Total Nights", names="Night Type", title="Weekend vs Week Nights", hole=0.45,
                 color_discrete_sequence=[BRAND_LIGHT, BRAND_GREEN])
    fig.update_layout(template="plotly_white", height=430)
    c2.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    cancel_dist = filtered["is_canceled"].map({0: "Not Canceled", 1: "Canceled"}).value_counts().reset_index()
    cancel_dist.columns = ["Status", "Bookings"]
    fig = px.pie(cancel_dist, values="Bookings", names="Status", title="Cancellation Distribution", hole=0.45,
                 color="Status", color_discrete_map={"Not Canceled": BRAND_GREEN, "Canceled": BRAND_ORANGE})
    fig.update_layout(template="plotly_white", height=430)
    c3.plotly_chart(fig, use_container_width=True)

    season_count = filtered["season"].value_counts().reindex(["Spring", "Summer", "Autumn", "Winter"]).dropna().reset_index()
    season_count.columns = ["Season", "Bookings"]
    fig = px.bar(season_count, x="Season", y="Bookings", text="Bookings", title="Total Bookings by Season",
                 color_discrete_sequence=[BRAND_GREEN])
    fig.update_layout(template="plotly_white", height=430)
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    c4.plotly_chart(fig, use_container_width=True)

# ── Tab 2 — Revenue & ADR ────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-title'>Revenue & Price Behavior</div>", unsafe_allow_html=True)

    rev_year = filtered.groupby(["arrival_date_year", "hotel"], as_index=False)["total_revenue"].sum()
    fig = px.bar(rev_year, x="arrival_date_year", y="total_revenue", color="hotel", barmode="group",
                 title="Total Revenue per Hotel Yearly", text="total_revenue",
                 color_discrete_sequence=[BRAND_GREEN, BRAND_ORANGE])
    fig.update_layout(template="plotly_white", height=460, xaxis_title="Year", yaxis_title="Total Revenue")
    fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)

    adr_month = filtered.groupby(["arrival_date_month_name", "hotel"], observed=True, as_index=False)["adr"].mean()
    fig = px.line(adr_month, x="arrival_date_month_name", y="adr", color="hotel", markers=True,
                  title="Average ADR per Month by Hotel Type", color_discrete_sequence=[BRAND_GREEN, BRAND_ORANGE])
    fig.update_layout(template="plotly_white", height=460, xaxis_title="Month", yaxis_title="Average Daily Rate")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    fig = px.histogram(filtered, x="adr", color="is_canceled", nbins=60, marginal="box",
                       title="ADR Distribution: Canceled vs Not Canceled",
                       color_discrete_map={0: BRAND_GREEN, 1: BRAND_ORANGE})
    fig.update_layout(template="plotly_white", height=430, legend_title_text="Canceled")
    c1.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(filtered, x="lead_time", color="is_canceled", nbins=60, marginal="box",
                       title="Lead Time Distribution: Canceled vs Not Canceled",
                       color_discrete_map={0: BRAND_GREEN, 1: BRAND_ORANGE})
    fig.update_layout(template="plotly_white", height=430, legend_title_text="Canceled")
    c2.plotly_chart(fig, use_container_width=True)

# ── Tab 3 — Cancellation Analysis ────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-title'>Cancellation Analysis</div>", unsafe_allow_html=True)

    cancel_month = filtered.groupby("arrival_month", as_index=False)["is_canceled"].mean()
    cancel_month["month_name"] = cancel_month["arrival_month"].map(lambda x: calendar.month_abbr[int(x)])
    cancel_month["rate_text"] = cancel_month["is_canceled"].map(lambda x: f"{x:.1%}")
    fig = px.bar(cancel_month, x="month_name", y="is_canceled", text="rate_text",
                 title="Cancellation Rate by Month", color_discrete_sequence=[BRAND_ORANGE])
    fig.add_hline(y=filtered["is_canceled"].mean(), line_dash="dash", line_color="gray", annotation_text="Average")
    fig.update_layout(template="plotly_white", height=430, yaxis_tickformat=".0%", yaxis_title="Cancellation Rate")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    lead_cancel = filtered.groupby("lead_bucket", observed=True, as_index=False)["is_canceled"].mean()
    lead_cancel["rate_text"] = lead_cancel["is_canceled"].map(lambda x: f"{x:.1%}")
    fig = px.bar(lead_cancel, x="lead_bucket", y="is_canceled", text="rate_text",
                 title="Cancellation Rate by Lead Time Bucket", color_discrete_sequence=[BRAND_GREEN])
    fig.update_layout(template="plotly_white", height=430, yaxis_tickformat=".0%", yaxis_title="Cancellation Rate")
    c1.plotly_chart(fig, use_container_width=True)

    cust_cancel = filtered.groupby("customer_type", as_index=False)["is_canceled"].mean().sort_values("is_canceled")
    cust_cancel["rate_text"] = cust_cancel["is_canceled"].map(lambda x: f"{x:.1%}")
    fig = px.bar(cust_cancel, x="is_canceled", y="customer_type", orientation="h", text="rate_text",
                 title="Cancellation Rate by Customer Type", color_discrete_sequence=[BRAND_ORANGE])
    fig.update_layout(template="plotly_white", height=430, xaxis_tickformat=".0%", xaxis_title="Cancellation Rate", yaxis_title="")
    c2.plotly_chart(fig, use_container_width=True)

    pivot = filtered.groupby(["hotel", "market_segment"], as_index=False)["is_canceled"].mean()
    heatmap_data = pivot.pivot(index="hotel", columns="market_segment", values="is_canceled")
    fig = px.imshow(heatmap_data, text_auto=".0%", aspect="auto", title="Cancellation Rate by Hotel and Market Segment",
                    color_continuous_scale="YlGn")
    fig.update_layout(template="plotly_white", height=430)
    st.plotly_chart(fig, use_container_width=True)

    top_countries = filtered["country"].value_counts().nlargest(15).index
    country_cancel = filtered[filtered["country"].isin(top_countries)].groupby("country", as_index=False)["is_canceled"].mean().sort_values("is_canceled")
    country_cancel["rate_text"] = country_cancel["is_canceled"].map(lambda x: f"{x:.1%}")
    fig = px.bar(country_cancel, x="is_canceled", y="country", orientation="h", text="rate_text",
                 title="Cancellation Rate by Country — Top 15 Booking Countries", color_discrete_sequence=[BRAND_GREEN])
    fig.add_vline(x=filtered["is_canceled"].mean(), line_dash="dash", line_color="gray", annotation_text="Overall avg")
    fig.update_layout(template="plotly_white", height=560, xaxis_tickformat=".0%", xaxis_title="Cancellation Rate", yaxis_title="Country")
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 4 — Guests & Segments ─────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-title'>Guest & Segment Behavior</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    sr_cancel = filtered.groupby("total_of_special_requests", as_index=False)["is_canceled"].mean()
    sr_cancel["rate_text"] = sr_cancel["is_canceled"].map(lambda x: f"{x:.1%}")
    fig = px.bar(sr_cancel, x="total_of_special_requests", y="is_canceled", text="rate_text",
                 title="Cancellation Rate by Number of Special Requests", color_discrete_sequence=[BRAND_GREEN])
    fig.update_layout(template="plotly_white", height=430, yaxis_tickformat=".0%")
    c1.plotly_chart(fig, use_container_width=True)

    bc_cancel = filtered.groupby("booking_changes", as_index=False)["is_canceled"].mean()
    bc_cancel = bc_cancel[bc_cancel["booking_changes"] <= bc_cancel["booking_changes"].quantile(0.98)]
    bc_cancel["rate_text"] = bc_cancel["is_canceled"].map(lambda x: f"{x:.1%}")
    fig = px.bar(bc_cancel, x="booking_changes", y="is_canceled", text="rate_text",
                 title="Cancellation Rate by Number of Booking Changes", color_discrete_sequence=[BRAND_ORANGE])
    fig.update_layout(template="plotly_white", height=430, yaxis_tickformat=".0%")
    c2.plotly_chart(fig, use_container_width=True)

    guest_comp = filtered.groupby("hotel", as_index=False)[["adults", "children", "babies"]].mean()
    fig = go.Figure()
    for col, color in [("adults", BRAND_GREEN), ("children", BRAND_LIGHT), ("babies", BRAND_ORANGE)]:
        fig.add_bar(x=guest_comp["hotel"], y=guest_comp[col], name=col.title(), marker_color=color)
    fig.update_layout(barmode="stack", template="plotly_white", height=450,
                      title="Average Guest Composition by Hotel Type", yaxis_title="Average Count per Booking")
    st.plotly_chart(fig, use_container_width=True)

    segment_summary = filtered.groupby("market_segment", as_index=False).agg(
        bookings=("is_canceled", "size"),
        cancel_rate=("is_canceled", "mean"),
        avg_adr=("adr", "mean"),
        revenue=("total_revenue", "sum"),
    ).sort_values("bookings", ascending=False)
    segment_summary["cancel_rate"] = segment_summary["cancel_rate"].map(lambda x: f"{x:.1%}")
    segment_summary["avg_adr"] = segment_summary["avg_adr"].map(lambda x: f"${x:,.0f}")
    segment_summary["revenue"] = segment_summary["revenue"].map(lambda x: f"${x:,.0f}")
    st.dataframe(segment_summary, use_container_width=True, hide_index=True)

# ── Tab 5 — Correlation & Features ───────────────────────────────────────────
with tab5:
    st.markdown("<div class='section-title'>Correlation & Engineered Feature Analysis</div>", unsafe_allow_html=True)

    num_cols_corr = [
        "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
        "adults", "children", "is_repeated_guest", "previous_cancellations",
        "previous_bookings_not_canceled", "booking_changes", "adr",
        "total_of_special_requests", "total_nights", "total_guests", "is_canceled",
    ]
    available_corr = [c for c in num_cols_corr if c in filtered.columns]
    corr = filtered[available_corr].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Heatmap — Numeric Features",
                    color_continuous_scale="RdYlGn", zmin=-1, zmax=1)
    fig.update_layout(template="plotly_white", height=700)
    st.plotly_chart(fig, use_container_width=True)

    binary_features = {
        "Was on Waitlist": "was_on_waiting_list",
        "Has Prev Cancellations": "has_prev_cancellations",
        "Is Loyal Guest": "is_loyal_guest",
        "Has Baby": "has_baby",
        "Has Booking Changes": "has_booking_changes",
        "Is Free Room": "is_free_room",
        "Has Parking": "has_parking",
        "Is Family": "is_family",
        "High Season": "is_high_season",
        "Long Lead (>90d)": "is_long_lead",
        "Has Agent": "has_agent",
        "Has Company": "has_company",
    }
    rows = []
    for label, col in binary_features.items():
        if col in filtered.columns:
            for flag, group in [(0, "No"), (1, "Yes")]:
                sub = filtered[filtered[col] == flag]
                rows.append({
                    "Feature": label,
                    "Group": group,
                    "Cancel Rate": sub["is_canceled"].mean() if len(sub) else np.nan,
                    "Bookings": len(sub),
                })
    feat_df = pd.DataFrame(rows).dropna()
    fig = px.bar(feat_df, x="Feature", y="Cancel Rate", color="Group", barmode="group",
                 text=feat_df["Cancel Rate"].map(lambda x: f"{x:.1%}"),
                 title="Cancellation Rate: Engineered Binary Features — Yes vs No",
                 color_discrete_map={"No": BRAND_GREEN, "Yes": BRAND_ORANGE})
    fig.update_layout(template="plotly_white", height=560, yaxis_tickformat=".0%", xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 6 — Data Explorer ─────────────────────────────────────────────────────
with tab6:
    st.markdown("<div class='section-title'>Filtered Data Explorer</div>", unsafe_allow_html=True)
    st.caption("Use this tab to inspect the exact rows after applying the sidebar filters.")

    display_cols = [
        "hotel", "is_canceled", "arrival_date_year", "arrival_date_month", "lead_time",
        "market_segment", "customer_type", "country", "adr", "total_nights",
        "total_guests", "total_revenue", "reservation_status",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[display_cols].head(1000), use_container_width=True, hide_index=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="filtered_hotel_bookings.csv",
        mime="text/csv",
    )

    st.markdown("<div class='section-title'>Quick Analysis Notes</div>", unsafe_allow_html=True)
    most_cancel_month = cancel_month.sort_values("is_canceled", ascending=False).iloc[0]
    highest_segment = filtered.groupby("market_segment")["is_canceled"].mean().sort_values(ascending=False).head(1)
    top_rev_hotel = filtered.groupby("hotel")["total_revenue"].sum().sort_values(ascending=False).head(1)
    st.write(
        f"- Highest cancellation month in the current filters: **{calendar.month_name[int(most_cancel_month['arrival_month'])]}** "
        f"with **{most_cancel_month['is_canceled']:.1%}** cancellation rate."
    )
    st.write(
        f"- Highest-risk market segment: **{highest_segment.index[0]}** with **{highest_segment.iloc[0]:.1%}** cancellation rate."
    )
    st.write(
        f"- Top revenue hotel type: **{top_rev_hotel.index[0]}** with **${top_rev_hotel.iloc[0]:,.0f}** estimated revenue."
    )

# ── Tab 7 — ML Results ────────────────────────────────────────────────────────
with tab7:
    st.markdown("<div class='section-title'>🤖 Machine Learning — Cancellation Prediction</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='small-note'>Models trained on the full dataset (70/15/15 split). "
        "Threshold tuned on validation set to ensure ≥ 70% recall. "
        "Best model retrained on train+val and evaluated on held-out test set.</div>",
        unsafe_allow_html=True,
    )

    run_ml = st.button("▶ Train Models & Show Results", type="primary", use_container_width=False)
    if "ml_results" not in st.session_state:
        st.session_state["ml_results"] = None

    if run_ml:
        with st.spinner("Training Logistic Regression, LightGBM, and LightGBM (Tuned) — this may take ~30s …"):
            st.session_state["ml_results"] = run_ml_pipeline(df)

    ml = st.session_state.get("ml_results")

    if ml is None:
        st.info("Click the button above to train and evaluate all models. Results are cached after the first run.")
        st.stop()

    # ── Section A: Model Comparison ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Model Comparison (Validation Set)")

    results_df = ml['results_df'].copy()
    results_df_display = results_df.copy()
    results_df_display['threshold'] = results_df_display['threshold'].map(lambda x: f"{x:.3f}")
    results_df_display['precision'] = results_df_display['precision'].map(lambda x: f"{x:.3f}")
    results_df_display['recall'] = results_df_display['recall'].map(lambda x: f"{x:.3f}")
    results_df_display['f1'] = results_df_display['f1'].map(lambda x: f"{x:.3f}")
    results_df_display.index.name = "Model"
    results_df_display = results_df_display.reset_index()
    st.dataframe(results_df_display, use_container_width=True, hide_index=True)

    # Bar chart: F1, Precision, Recall per model
    melted = ml['results_df'].reset_index().melt(
        id_vars='index', value_vars=['precision', 'recall', 'f1'],
        var_name='Metric', value_name='Score'
    ).rename(columns={'index': 'Model'})
    fig = px.bar(
        melted, x='Model', y='Score', color='Metric', barmode='group',
        title='Precision / Recall / F1 by Model (Validation Set)',
        text=melted['Score'].map(lambda x: f"{x:.3f}"),
        color_discrete_sequence=[BRAND_GREEN, BRAND_ORANGE, BRAND_LIGHT],
    )
    fig.update_layout(template='plotly_white', height=420, yaxis_range=[0, 1])
    fig.update_traces(textposition='outside', cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── Section B: Test Set KPIs ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 🏆 Test Set Results — Best Model: `{ml['best_name']}`")

    report = ml['report']
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("ROC-AUC", f"{ml['roc_auc']:.4f}")
    k2.metric("F1 (Canceled)", f"{report['1']['f1-score']:.3f}")
    k3.metric("Precision (Canceled)", f"{report['1']['precision']:.3f}")
    k4.metric("Recall (Canceled)", f"{report['1']['recall']:.3f}")
    k5.metric("Accuracy", f"{report['accuracy']:.3f}")

    # ── Section C: ROC + PR curves ───────────────────────────────────────────
    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=ml['fpr'], y=ml['tpr'], mode='lines', name=f"{ml['best_name']} (AUC={ml['roc_auc']:.3f})",
            line=dict(color=BRAND_ORANGE, width=2.5),
            fill='tozeroy', fillcolor='rgba(231,111,81,0.08)',
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier',
            line=dict(color='gray', dash='dash', width=1.2),
        ))
        fig_roc.update_layout(
            template='plotly_white', title='ROC Curve', height=420,
            xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
            legend=dict(x=0.55, y=0.08),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with c2:
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=ml['recall_vals'], y=ml['precision_vals'], mode='lines',
            name=f"{ml['best_name']} (AP={ml['ap']:.3f})",
            line=dict(color=BRAND_GREEN, width=2.5),
            fill='tozeroy', fillcolor='rgba(45,106,79,0.08)',
        ))
        fig_pr.add_hline(
            y=ml['cancel_mean'], line_dash='dash', line_color='gray',
            annotation_text='Baseline (random)',
        )
        fig_pr.update_layout(
            template='plotly_white', title='Precision-Recall Curve', height=420,
            xaxis_title='Recall', yaxis_title='Precision',
            legend=dict(x=0.55, y=0.92),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # ── Section D: Confusion Matrix ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔢 Confusion Matrix (Test Set)")

    cm = ml['cm']
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels = ['Not Canceled', 'Canceled']

    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=j, y=i,
                text=f"<b>{cm[i, j]:,}</b><br><span style='font-size:11px'>({cm_pct[i, j]:.1%})</span>",
                showarrow=False,
                font=dict(size=16, color='black'),
            ))

    fig_cm = go.Figure(go.Heatmap(
        z=cm_pct,
        x=labels, y=labels,
        colorscale='YlGn',
        showscale=True,
        zmin=0, zmax=1,
    ))
    fig_cm.update_layout(
        template='plotly_white',
        title=f'Confusion Matrix — {ml["best_name"]} (threshold={ml["best_thr"]:.3f})',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=420,
        annotations=annotations,
        xaxis=dict(tickvals=[0, 1], ticktext=labels),
        yaxis=dict(tickvals=[0, 1], ticktext=labels),
    )
    col_cm, col_report = st.columns([1, 1])
    col_cm.plotly_chart(fig_cm, use_container_width=True)

    with col_report:
        st.markdown("**Classification Report**")
        report_rows = []
        for cls, lbl in [('0', 'Not Canceled'), ('1', 'Canceled'), ('macro avg', 'Macro Avg'), ('weighted avg', 'Weighted Avg')]:
            if cls in report:
                r = report[cls]
                report_rows.append({
                    'Class': lbl,
                    'Precision': f"{r['precision']:.3f}",
                    'Recall': f"{r['recall']:.3f}",
                    'F1-Score': f"{r['f1-score']:.3f}",
                    'Support': f"{int(r['support']):,}" if 'support' in r else '—',
                })
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)
        st.markdown(
            f"**Test set:** {ml['test_size']:,} samples &nbsp;|&nbsp; "
            f"**Train set:** {ml['train_size']:,} samples"
        )

    # ── Section E: Feature Importances ──────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 🔍 Top 20 Feature Importances — {ml['best_name']}")

    feat_imp = ml['feat_imp']
    colors = [BRAND_ORANGE if i < 5 else BRAND_GREEN for i in range(len(feat_imp))]

    fig_fi = go.Figure(go.Bar(
        x=feat_imp['importance'][::-1],
        y=feat_imp['feature'][::-1],
        orientation='h',
        marker_color=colors[::-1],
        text=[f"{v:,.0f}" for v in feat_imp['importance'][::-1]],
        textposition='outside',
    ))
    fig_fi.update_layout(
        template='plotly_white',
        title=f'Top 20 Feature Importances — {ml["best_name"]}',
        xaxis_title='Importance Score',
        height=600,
        margin=dict(l=220),
    )
    st.plotly_chart(fig_fi, use_container_width=True)
    st.caption("🟠 Top 5 most important features highlighted in orange.")

    # ── Section F: SMOTE Results ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚖️ Class Imbalance — LightGBM + SMOTE vs Baseline")

    if ml['smote_available']:
        sr = ml['smote_results']
        c1, c2 = st.columns(2)

        # Before / After SMOTE bar chart
        before_after = pd.DataFrame({
            'Class': ['Not Canceled', 'Canceled', 'Not Canceled (SMOTE)', 'Canceled (SMOTE)'],
            'Count': [sr['before'][0], sr['before'][1], sr['after'][0], sr['after'][1]],
            'Set': ['Original', 'Original', 'After SMOTE', 'After SMOTE'],
        })
        fig_smote = px.bar(
            before_after, x='Class', y='Count', color='Set', barmode='group',
            title='Class Distribution Before and After SMOTE',
            color_discrete_map={'Original': BRAND_ORANGE, 'After SMOTE': BRAND_GREEN},
            text='Count',
        )
        fig_smote.update_layout(template='plotly_white', height=400)
        fig_smote.update_traces(texttemplate='%{text:,}', textposition='outside')
        c1.plotly_chart(fig_smote, use_container_width=True)

        # Metric comparison table
        baseline = ml['results_df'].loc[ml['best_name']] if ml['best_name'] in ml['results_df'].index else ml['results_df'].iloc[0]
        compare_df = pd.DataFrame({
            'Strategy': ['LightGBM (scale_pos_weight)', 'LightGBM + SMOTE'],
            'Threshold': [f"{baseline['threshold']:.3f}", f"{sr['threshold']:.3f}"],
            'Precision': [f"{baseline['precision']:.3f}", f"{sr['precision']:.3f}"],
            'Recall': [f"{baseline['recall']:.3f}", f"{sr['recall']:.3f}"],
            'F1': [f"{baseline['f1']:.3f}", f"{sr['f1']:.3f}"],
        })
        c2.markdown("**Validation Set Comparison**")
        c2.dataframe(compare_df, use_container_width=True, hide_index=True)
        c2.markdown(
            f"""
            | | Before SMOTE | After SMOTE |
            |---|---|---|
            | Not Canceled | {sr['before'][0]:,} | {sr['after'][0]:,} |
            | Canceled | {sr['before'][1]:,} | {sr['after'][1]:,} |
            | Ratio | {sr['before'][0]/sr['before'][1]:.2f}:1 | 1:1 |
            """
        )
        st.markdown(
            "> **Takeaway:** SMOTE synthetically oversamples the minority class (Canceled) to achieve a 1:1 ratio. "
            "Compare the F1 scores above to see whether oversampling or `scale_pos_weight` performs better on your data."
        )
    else:
        st.warning(
            "`imbalanced-learn` is not installed. Install it with `pip install imbalanced-learn` to see SMOTE results."
        )
