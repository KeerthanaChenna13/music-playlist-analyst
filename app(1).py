import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Spotify Playlist Analyst",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎵 Spotify Playlist Analyst")
st.caption("Analyze playlist rankings, popularity, artist performance, and predict the next playlist position.")


# =========================================================
# DATA LOADING
# =========================================================
REQUIRED_COLUMNS = {
    "date",
    "position",
    "song",
    "artist",
    "popularity",
    "duration_ms",
    "total_tracks",
    "is_explicit",
    "album_type",
}


@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))


def prepare_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "The dataset is missing these required columns: "
            + ", ".join(sorted(missing))
        )

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce")
    df["total_tracks"] = pd.to_numeric(df["total_tracks"], errors="coerce")
    df["duration_minutes"] = df["duration_ms"] / 60000.0

    df["artist"] = df["artist"].astype("string").str.strip()
    df["artist"] = df["artist"].str.replace(r"\s+", " ", regex=True)
    df["song"] = df["song"].astype("string").str.strip()

    # Normalize explicit values to numeric 0/1 where possible.
    if df["is_explicit"].dtype == bool:
        df["is_explicit"] = df["is_explicit"].astype(int)
    else:
        explicit_map = {
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0,
        }
        mapped = df["is_explicit"].astype(str).str.lower().map(explicit_map)
        numeric = pd.to_numeric(df["is_explicit"], errors="coerce")
        df["is_explicit"] = mapped.fillna(numeric).fillna(0).astype(int)

    df = df.sort_values(["song", "date"], kind="stable").reset_index(drop=True)

    # Current rank movement. Positive means movement to a larger position number.
    df["Rank_Change"] = df.groupby("song")["position"].diff().fillna(0)
    df["Rank_Change_Current"] = df["Rank_Change"]

    # Days the song has appeared up to the current record.
    df["Days_on_Chart"] = df.groupby("song").cumcount() + 1

    # Seven-record rolling popularity average for each song.
    df["Popularity_Trend_Score"] = (
        df.groupby("song")["popularity"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    )

    # Song-level metrics.
    metrics = (
        df.groupby("song")
        .agg(
            Days_on_Chart_Total=("date", "nunique"),
            Average_Rank=("position", "mean"),
            Best_Rank_Achieved=("position", "min"),
            Rank_Volatility_Index=("position", "std"),
            Average_Popularity=("popularity", "mean"),
        )
        .reset_index()
    )
    metrics["Rank_Volatility_Index"] = metrics["Rank_Volatility_Index"].fillna(0)
    df = df.merge(metrics, on="song", how="left")

    # Target for ML: the next observed playlist position for the same song.
    df["Next_Position"] = df.groupby("song")["position"].shift(-1)

    return df


# Try the original project filename first. If it is not present, allow upload.
def get_data() -> pd.DataFrame | None:
    default_path = Path("Atlantic_United_States.csv")

    if default_path.exists():
        try:
            raw = load_csv_from_path(str(default_path))
            return prepare_data(raw)
        except Exception as exc:
            st.error(f"Could not load Atlantic_United_States.csv: {exc}")
            return None

    st.warning("Atlantic_United_States.csv was not found in the app folder.")
    uploaded = st.file_uploader("Upload the CSV dataset to continue", type=["csv"])
    if uploaded is None:
        st.info("Place Atlantic_United_States.csv in the GitHub repository next to app.py, or upload it above.")
        return None

    try:
        raw = load_uploaded_csv(uploaded.getvalue())
        return prepare_data(raw)
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        return None


df = get_data()

if df is None:
    st.stop()


# =========================================================
# MACHINE-LEARNING MODEL
# =========================================================
MODEL_FEATURES = [
    "position",
    "popularity",
    "duration_minutes",
    "Days_on_Chart",
    "Rank_Change_Current",
    "duration_ms",
    "total_tracks",
    "is_explicit",
    "album_type_compilation",
    "album_type_single",
]


@st.cache_data(show_spinner=False)
def build_ml_data(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    ml = data.dropna(subset=["Next_Position"]).copy()

    x = ml[
        [
            "position",
            "popularity",
            "duration_minutes",
            "Days_on_Chart",
            "Rank_Change_Current",
            "duration_ms",
            "total_tracks",
            "is_explicit",
            "album_type",
        ]
    ].copy()

    x["is_explicit"] = pd.to_numeric(x["is_explicit"], errors="coerce").fillna(0).astype(int)
    x["album_type"] = x["album_type"].fillna("album").astype(str).str.lower()
    x = pd.get_dummies(x, columns=["album_type"], drop_first=True)

    # Force the same feature schema on every dataset.
    for col in ["album_type_compilation", "album_type_single"]:
        if col not in x.columns:
            x[col] = 0

    x = x[MODEL_FEATURES]
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = pd.to_numeric(ml["Next_Position"], errors="coerce")

    valid = y.notna()
    x = x.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    return pd.concat([x, y.rename("Next_Position")], axis=1), MODEL_FEATURES


@st.cache_resource(show_spinner=True)
def train_model(data_for_training: pd.DataFrame):
    ml_data, feature_names = build_ml_data(data_for_training)

    if len(ml_data) < 10:
        return None, feature_names, None, None, None, None

    split_index = max(1, int(len(ml_data) * 0.80))
    if split_index >= len(ml_data):
        split_index = len(ml_data) - 1

    x = ml_data[feature_names]
    y = ml_data["Next_Position"]

    x_train = x.iloc[:split_index]
    x_test = x.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Train only when both training and test sets contain usable rows.
    if len(x_train) < 2 or len(x_test) < 1:
        return None, feature_names, None, None, None, None

    # RandomForestRegressor requires a numeric, finite feature matrix.
    x_train = x_train.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    x_test = x_test.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    y_train = pd.to_numeric(y_train, errors="coerce").fillna(y_train.median())
    y_test = pd.to_numeric(y_test, errors="coerce")

    if y_train.isna().any() or y_test.isna().all():
        return None, feature_names, None, None, None, None

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )

    try:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
    except Exception as exc:
        st.error(f"Model training failed: {exc}")
        return None, feature_names, None, None, None, None

    valid_test = y_test.notna()
    if not valid_test.any():
        return None, feature_names, None, None, None, None

    y_test_valid = y_test.loc[valid_test]
    predictions_valid = predictions[valid_test.to_numpy()]

    mae = mean_absolute_error(y_test_valid, predictions_valid)
    rmse = np.sqrt(mean_squared_error(y_test_valid, predictions_valid))
    r2 = r2_score(y_test_valid, predictions_valid) if len(y_test_valid) > 1 else np.nan

    importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    return model, feature_names, mae, rmse, r2, importance


try:
    model, feature_names, mae, rmse, r2, importance = train_model(df)
except Exception as exc:
    st.error(f"Could not train the prediction model: {exc}")
    model, feature_names, mae, rmse, r2, importance = None, MODEL_FEATURES, None, None, None, None


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Data Analysis", "Rank Prediction", "Model Information"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(df):,} records")


# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":
    st.header("📊 Playlist Performance Dashboard")

    total_records = len(df)
    unique_songs = df["song"].nunique()
    unique_artists = df["artist"].nunique()
    unique_dates = df["date"].nunique()

    top_artist_counts = df["artist"].value_counts(dropna=True)
    top_artist = top_artist_counts.index[0] if not top_artist_counts.empty else "N/A"
    artist_dominance = (
        round(top_artist_counts.iloc[0] / total_records * 100, 2)
        if total_records and not top_artist_counts.empty
        else 0
    )
    explicit_share = round(df["is_explicit"].mean() * 100, 2) if total_records else 0
    average_rank = round(df["position"].mean(), 2)
    average_popularity = round(df["popularity"].mean(), 2)
    volatility = round(df["position"].std(), 2) if total_records > 1 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎵 Records", f"{total_records:,}")
    c2.metric("🎶 Unique Songs", f"{unique_songs:,}")
    c3.metric("🎤 Unique Artists", f"{unique_artists:,}")
    c4.metric("📅 Unique Dates", f"{unique_dates:,}")

    st.markdown("---")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("📈 Average Rank", f"{average_rank:.2f}")
    c6.metric("⭐ Avg Popularity", f"{average_popularity:.2f}")
    c7.metric("👑 Top Artist", str(top_artist))
    c8.metric("🔞 Explicit Content", f"{explicit_share:.2f}%")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("📈 Popularity Trend")
        trend = (
            df.dropna(subset=["date"])
            .groupby("date", as_index=False)["popularity"]
            .mean()
            .sort_values("date")
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(trend["date"], trend["popularity"])
        ax.set_xlabel("Date")
        ax.set_ylabel("Average Popularity")
        ax.set_title("Average Popularity Over Time")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with right:
        st.subheader("🏆 Top 10 Artists by Appearances")
        artist_counts = df["artist"].value_counts().head(10).sort_values()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(artist_counts.index.astype(str), artist_counts.values)
        ax.set_xlabel("Playlist Appearances")
        ax.set_ylabel("Artist")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    st.subheader("📌 Project Summary")
    st.write(
        "This project analyzes playlist rankings to understand song performance, "
        "popularity, ranking movement, artist dominance, explicit content, and "
        "playlist longevity. A Random Forest Regression model is used to predict "
        "the next observed playlist position of a song."
    )


# =========================================================
# DATA ANALYSIS
# =========================================================
elif page == "Data Analysis":
    st.header("🔎 Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Dataset", "Data Quality", "Song Metrics", "Charts"]
    )

    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

        st.subheader("Dataset Information")
        info = pd.DataFrame(
            {
                "Metric": [
                    "Rows",
                    "Columns",
                    "Unique Songs",
                    "Unique Artists",
                    "Unique Dates",
                    "Start Date",
                    "End Date",
                ],
                "Value": [
                    len(df),
                    len(df.columns),
                    df["song"].nunique(),
                    df["artist"].nunique(),
                    df["date"].nunique(),
                    df["date"].min(),
                    df["date"].max(),
                ],
            }
        )
        st.dataframe(info, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Missing Values")
        missing = df.isnull().sum().sort_values(ascending=False)
        st.dataframe(
            missing.to_frame("Missing Values"),
            use_container_width=True,
        )

        st.subheader("Invalid Playlist Positions")
        invalid = df[(df["position"] < 1) | (df["position"] > 50)]
        if invalid.empty:
            st.success("All playlist positions are between 1 and 50.")
        else:
            st.warning(f"Found {len(invalid):,} invalid position records.")
            st.dataframe(invalid.head(20), use_container_width=True, hide_index=True)

        st.subheader("Duplicate Song-Date Records")
        duplicates = df[df.duplicated(subset=["date", "song"], keep=False)]
        if duplicates.empty:
            st.success("No duplicate song-date entries found.")
        else:
            st.warning(f"Found {len(duplicates):,} duplicate song-date records.")
            st.dataframe(duplicates.head(20), use_container_width=True, hide_index=True)

    with tab3:
        song_metrics = (
            df.groupby("song")
            .agg(
                Days_on_Chart=("date", "nunique"),
                Average_Rank=("position", "mean"),
                Best_Rank_Achieved=("position", "min"),
                Rank_Volatility_Index=("position", "std"),
                Average_Popularity=("popularity", "mean"),
            )
            .reset_index()
        )
        song_metrics["Rank_Volatility_Index"] = song_metrics[
            "Rank_Volatility_Index"
        ].fillna(0)

        st.subheader("Song Performance Metrics")
        st.dataframe(song_metrics.head(50), use_container_width=True, hide_index=True)

        st.subheader("Longest-Running Songs")
        st.dataframe(
            song_metrics.sort_values("Days_on_Chart", ascending=False).head(10),
            use_container_width=True,
            hide_index=True,
        )

    with tab4:
        chart1, chart2 = st.columns(2)

        with chart1:
            st.subheader("Top 10 Songs by Average Popularity")
            songs = (
                df.groupby("song")["popularity"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .sort_values()
            )
            fig, ax = plt.subplots()
            ax.barh(songs.index.astype(str), songs.values)
            ax.set_xlabel("Average Popularity")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

        with chart2:
            st.subheader("Playlist Position Distribution")
            fig, ax = plt.subplots()
            ax.hist(df["position"].dropna(), bins=range(1, 52), rwidth=0.8)
            ax.set_xlabel("Playlist Position")
            ax.set_ylabel("Frequency")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)


# =========================================================
# RANK PREDICTION
# =========================================================
elif page == "Rank Prediction":
    st.header("🤖 Next Playlist Position Prediction")
    st.write(
        "Enter the current song information below. The Random Forest model "
        "will predict the next observed playlist position."
    )

    if model is None:
        st.error("There are not enough usable records to train the prediction model.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        position = st.number_input(
            "Current Playlist Position",
            min_value=1,
            max_value=50,
            value=25,
            step=1,
        )
        popularity = st.number_input(
            "Popularity Score",
            min_value=0,
            max_value=100,
            value=85,
            step=1,
        )
        duration_minutes = st.number_input(
            "Song Duration (minutes)",
            min_value=0.1,
            max_value=15.0,
            value=3.0,
            step=0.1,
        )
        days_on_chart = st.number_input(
            "Days on Chart",
            min_value=1,
            max_value=1000,
            value=30,
            step=1,
        )
        rank_change = st.number_input(
            "Current Rank Change",
            min_value=-50.0,
            max_value=50.0,
            value=0.0,
            step=1.0,
        )

    with col2:
        duration_ms = st.number_input(
            "Duration (milliseconds)",
            min_value=1000,
            max_value=900000,
            value=180000,
            step=1000,
        )
        total_tracks = st.number_input(
            "Total Tracks in Album",
            min_value=1,
            max_value=100,
            value=20,
            step=1,
        )
        explicit = st.selectbox("Explicit Content", ["No", "Yes"])
        album_type = st.selectbox(
            "Album Type",
            ["album", "single", "compilation"],
        )

    if st.button("🔮 Predict Next Position", use_container_width=True):
        input_data = pd.DataFrame(
            {
                "position": [position],
                "popularity": [popularity],
                "duration_minutes": [duration_minutes],
                "Days_on_Chart": [days_on_chart],
                "Rank_Change_Current": [rank_change],
                "duration_ms": [duration_ms],
                "total_tracks": [total_tracks],
                "is_explicit": [1 if explicit == "Yes" else 0],
                "album_type_compilation": [1 if album_type == "compilation" else 0],
                "album_type_single": [1 if album_type == "single" else 0],
            }
        )
        input_data = input_data[feature_names]

        prediction = float(model.predict(input_data)[0])
        prediction = max(1.0, min(50.0, prediction))

        st.success(f"🎯 Predicted Next Playlist Position: #{prediction:.0f}")
        st.caption(f"Raw model prediction: {prediction:.2f}")


# =========================================================
# MODEL INFORMATION
# =========================================================
else:
    st.header("🌲 Random Forest Model")
    st.write(
        "The application uses a Random Forest Regression model to predict "
        "the next observed playlist position."
    )

    if model is None:
        st.warning("The model could not be trained because there are too few usable records.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.2f}")
    c2.metric("RMSE", f"{rmse:.2f}")
    c3.metric("R² Score", f"{r2:.4f}" if pd.notna(r2) else "N/A")

    st.markdown("---")
    st.subheader("⭐ Feature Importance")
    st.dataframe(importance, use_container_width=True, hide_index=True)

    top_features = importance.head(10).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top_features["Feature"], top_features["Importance"])
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top Factors Influencing Next Playlist Position")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.subheader("🧮 Model Features")
    st.write(feature_names)

    st.info(
        "The model is trained automatically when the app starts. Streamlit caching "
        "prevents unnecessary retraining during normal page navigation."
    )


# =========================================================
# OPTIONAL MODEL EXPORT
# =========================================================
# The export is intentionally disabled by default. Streamlit Cloud does not need
# model files generated at runtime; keeping the model in memory avoids Colab-only
# downloads and prevents deployment failures caused by missing files.
