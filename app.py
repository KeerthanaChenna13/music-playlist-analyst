import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Project 2",
    page_icon="📊",
    layout="wide"
)

st.title("Project 2")
st.write("This application was created from the original Jupyter Notebook.")

# Original notebook code
# ==========================================
# UNITED STATES TOP 50 PLAYLIST ANALYSIS
# Step 1: Import Required Libraries
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

print("Libraries imported successfully!")

# ==========================================
# Step 2: Upload Dataset
# ==========================================

from google.colab import files

uploaded = files.upload()

# ==========================================
# Step 3: Load Dataset
# ==========================================

file_name = "Atlantic_United_States.csv"

df = pd.read_csv(file_name)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

display(df.head())

# ==========================================
# Step 4: Dataset Information
# ==========================================

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# ==========================================
# Step 5: Date Range
# ==========================================

df['date'] = pd.to_datetime(df['date'], dayfirst=True)

print("Start Date:", df['date'].min())
print("End Date:", df['date'].max())
print("Number of unique dates:", df['date'].nunique())

# ==========================================
# Step 6: Validate Playlist Positions
# ==========================================

print("Minimum position:", df['position'].min())
print("Maximum position:", df['position'].max())

invalid_positions = df[
    (df['position'] < 1) | (df['position'] > 50)
]

print("\nInvalid position records:", len(invalid_positions))

if len(invalid_positions) == 0:
    print("✅ All playlist positions are between 1 and 50.")
else:
    print("⚠️ Invalid positions found.")
    display(invalid_positions.head())

# ==========================================
# Step 7: Missing Value Analysis
# ==========================================

missing_values = df.isnull().sum()

print("Missing values in each column:")
display(missing_values.to_frame("Missing Values"))

# ==========================================
# Step 8: Duplicate Song-Date Check
# ==========================================

duplicate_song_date = df[
    df.duplicated(
        subset=['date', 'song'],
        keep=False
    )
]

print("Duplicate song-date records:", len(duplicate_song_date))

if len(duplicate_song_date) == 0:
    print("✅ No duplicate song-date entries found.")
else:
    print("⚠️ Duplicate song-date entries found.")
    display(duplicate_song_date.head(10))

# ==========================================
# Step 9: Dataset Summary
# ==========================================

print("===== DATASET SUMMARY =====")

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Unique Songs:", df['song'].nunique())
print("Unique Artists:", df['artist'].nunique())
print("Unique Dates:", df['date'].nunique())

print("\nAlbum Types:")
print(df['album_type'].value_counts())

print("\nExplicit Status:")
print(df['is_explicit'].value_counts())

# ==========================================
# Step 10: Standardize Artist Names
# ==========================================

# Remove leading/trailing spaces
df['artist'] = df['artist'].str.strip()

# Replace multiple spaces with a single space
df['artist'] = df['artist'].str.replace(r'\s+', ' ', regex=True)

# Check the number of unique artists after standardization
print("Unique artists after standardization:", df['artist'].nunique())

# Display some artist names
print("\nSample artist names:")
print(df['artist'].drop_duplicates().head(20).tolist())

# ==========================================
# Step 11: Create Duration in Minutes
# ==========================================

df['duration_minutes'] = df['duration_ms'] / 60000

print("Duration conversion completed!")

display(
    df[['song', 'duration_ms', 'duration_minutes']].head(10)
)

# ==========================================
# Step 12: Song Performance Metrics
# ==========================================

song_metrics = df.groupby('song').agg(
    Days_on_Chart=('date', 'nunique'),
    Average_Rank=('position', 'mean'),
    Best_Rank_Achieved=('position', 'min'),
    Rank_Volatility_Index=('position', 'std'),
    Average_Popularity=('popularity', 'mean')
).reset_index()

# Replace missing volatility values
# (occurs when a song appears only once)
song_metrics['Rank_Volatility_Index'] = (
    song_metrics['Rank_Volatility_Index'].fillna(0)
)

print("Song-level feature engineering completed!")

print("Number of unique songs:", len(song_metrics))

display(song_metrics.head(10))

# ==========================================
# Step 13: Popularity Trend Score
# ==========================================

df = df.sort_values(['song', 'date'])

df['Popularity_Trend_Score'] = (
    df.groupby('song')['popularity']
      .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
)

print("Popularity Trend Score created!")

display(
    df[
        ['date', 'song', 'popularity', 'Popularity_Trend_Score']
    ].head(15)
)

# ==========================================
# Step 13: Popularity Trend Score
# ==========================================

df = df.sort_values(['song', 'date'])

df['Popularity_Trend_Score'] = (
    df.groupby('song')['popularity']
      .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
)

print("Popularity Trend Score created!")

display(
    df[
        ['date', 'song', 'popularity', 'Popularity_Trend_Score']
    ].head(15)
)

# ==========================================
# Step 14: Merge Song-Level Metrics
# ==========================================

df = df.merge(
    song_metrics,
    on='song',
    how='left'
)

print("Song metrics merged successfully!")

print("\nUpdated dataset shape:", df.shape)

display(df.head())

# ==========================================
# Step 15: Daily Rank Distribution
# ==========================================

rank_distribution = (
    df['position']
    .value_counts()
    .sort_index()
)

print("Playlist Rank Distribution:")
display(rank_distribution.to_frame("Number of Appearances"))

# ==========================================
# Step 16: Rank Distribution Chart
# ==========================================

plt.figure(figsize=(12, 6))

plt.bar(
    rank_distribution.index,
    rank_distribution.values
)

plt.xlabel("Playlist Position")
plt.ylabel("Number of Appearances")
plt.title("Distribution of Songs Across Top 50 Positions")
plt.xticks(range(1, 51))

plt.show()

# ==========================================
# Step 17: Daily Average Rank
# ==========================================

daily_rank = (
    df.groupby('date')
      .agg(
          Average_Rank=('position', 'mean'),
          Best_Rank=('position', 'min'),
          Worst_Rank=('position', 'max')
      )
      .reset_index()
)

display(daily_rank.head(10))

# ==========================================
# Step 18: Daily Average Rank Trend
# ==========================================

plt.figure(figsize=(14, 6))

plt.plot(
    daily_rank['date'],
    daily_rank['Average_Rank']
)

plt.xlabel("Date")
plt.ylabel("Average Playlist Position")
plt.title("Daily Average Playlist Rank Trend")
plt.gca().invert_yaxis()

plt.show()

# ==========================================
# Step 19: Rank Movement
# ==========================================

df = df.sort_values(['song', 'date'])

df['Previous_Rank'] = (
    df.groupby('song')['position']
      .shift(1)
)

df['Rank_Change'] = (
    df['Previous_Rank'] - df['position']
)

display(
    df[
        ['date', 'song', 'position', 'Previous_Rank', 'Rank_Change']
    ].head(20)
)

# ==========================================
# Step 20: Fast Risers
# ==========================================

fast_risers = (
    df[df['Rank_Change'] > 0]
    .sort_values('Rank_Change', ascending=False)
)

print("Top Fast-Rising Song Movements:")

display(
    fast_risers[
        ['date', 'song', 'artist',
         'Previous_Rank', 'position', 'Rank_Change']
    ].head(20)
)

# ==========================================
# Step 21: Declining Songs
# ==========================================

decliners = (
    df[df['Rank_Change'] < 0]
    .sort_values('Rank_Change')
)

print("Top Declining Song Movements:")

display(
    decliners[
        ['date', 'song', 'artist',
         'Previous_Rank', 'position', 'Rank_Change']
    ].head(20)
)

# ==========================================
# Step 22: Song Entry Detection
# ==========================================

song_first_date = (
    df.groupby('song')['date']
      .transform('min')
)

df['Is_Entry'] = df['date'] == song_first_date

print("Total song entry records:", df['Is_Entry'].sum())

# ==========================================
# Step 23: Entry Analysis
# ==========================================

entries = df[df['Is_Entry']]

entry_summary = (
    entries.groupby('date')
           .size()
           .reset_index(name='New_Songs')
)

display(entry_summary.head(20))

# ==========================================
# Step 24: Song Exit Detection
# ==========================================

song_last_date = (
    df.groupby('song')['date']
      .transform('max')
)

df['Is_Exit'] = df['date'] == song_last_date

print("Total song exit records:", df['Is_Exit'].sum())

# ==========================================
# STEP 15: SONG-LEVEL PERFORMANCE ANALYSIS
# ==========================================

print("========== SONG-LEVEL PERFORMANCE ANALYSIS ==========\n")

# Create song-level summary
song_performance = df.groupby(
    ['song', 'artist'],
    as_index=False
).agg(
    Days_on_Chart=('Days_on_Chart', 'max'),
    Average_Rank=('Average_Rank', 'mean'),
    Best_Rank_Achieved=('Best_Rank_Achieved', 'min'),
    Rank_Volatility_Index=('Rank_Volatility_Index', 'mean'),
    Average_Popularity=('popularity', 'mean'),
    Average_Duration=('duration_minutes', 'mean'),
    Album_Type=('album_type', 'first'),
    Is_Explicit=('is_explicit', 'first')
)

# ------------------------------------------
# 1. Songs with longest playlist presence
# ------------------------------------------

longest_chart_songs = song_performance.sort_values(
    'Days_on_Chart',
    ascending=False
).head(10)

print("Top 10 Songs with Longest Playlist Presence:")
display(longest_chart_songs[
    ['song', 'artist', 'Days_on_Chart', 'Average_Rank',
     'Best_Rank_Achieved', 'Average_Popularity']
])

# ------------------------------------------
# 2. Songs with highest average popularity
# ------------------------------------------

most_popular_songs = song_performance.sort_values(
    'Average_Popularity',
    ascending=False
).head(10)

print("\nTop 10 Songs with Highest Average Popularity:")
display(most_popular_songs[
    ['song', 'artist', 'Average_Popularity',
     'Days_on_Chart', 'Average_Rank', 'Best_Rank_Achieved']
])

# ------------------------------------------
# 3. Songs with best average rank
# ------------------------------------------

best_average_rank = song_performance.sort_values(
    'Average_Rank',
    ascending=True
).head(10)

print("\nTop 10 Songs with Best Average Rank:")
display(best_average_rank[
    ['song', 'artist', 'Average_Rank',
     'Best_Rank_Achieved', 'Days_on_Chart',
     'Average_Popularity']
])

# ------------------------------------------
# 4. Most volatile songs
# ------------------------------------------

most_volatile_songs = song_performance.sort_values(
    'Rank_Volatility_Index',
    ascending=False
).head(10)

print("\nTop 10 Most Volatile Songs:")
display(most_volatile_songs[
    ['song', 'artist', 'Rank_Volatility_Index',
     'Days_on_Chart', 'Average_Rank',
     'Best_Rank_Achieved']
])

print("\nSong-level analysis completed successfully!")

# ==========================================
# STEP 16: PEAK RANK VS LONGEVITY ANALYSIS
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(
    song_performance['Best_Rank_Achieved'],
    song_performance['Days_on_Chart'],
    alpha=0.6
)

plt.xlabel("Best Rank Achieved")
plt.ylabel("Days on Chart")
plt.title("Peak Rank vs Song Longevity")

# Lower rank is better, so reverse x-axis
plt.gca().invert_xaxis()

plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# STEP 17: ARTIST PERFORMANCE ANALYSIS
# ==========================================

print("========== ARTIST PERFORMANCE ANALYSIS ==========\n")

artist_performance = df.groupby(
    'artist',
    as_index=False
).agg(
    Unique_Songs=('song', 'nunique'),
    Total_Days_Appeared=('date', 'nunique'),
    Total_Appearances=('song', 'count'),
    Average_Rank=('position', 'mean'),
    Average_Popularity=('popularity', 'mean')
)

# ------------------------------------------
# Top artists by number of unique songs
# ------------------------------------------

top_artists_songs = artist_performance.sort_values(
    'Unique_Songs',
    ascending=False
).head(10)

print("Top 10 Artists by Number of Unique Songs:")
display(top_artists_songs)

# ------------------------------------------
# Top artists by total appearances
# ------------------------------------------

top_artists_presence = artist_performance.sort_values(
    'Total_Appearances',
    ascending=False
).head(10)

print("\nTop 10 Artists by Total Playlist Appearances:")
display(top_artists_presence)

# ------------------------------------------
# Top artists by average popularity
# ------------------------------------------

top_artists_popularity = artist_performance.sort_values(
    'Average_Popularity',
    ascending=False
).head(10)

print("\nTop 10 Artists by Average Popularity:")
display(top_artists_popularity)

print("\nArtist performance analysis completed successfully!")

# ==========================================
# STEP 18: ARTIST DOMINANCE INDEX
# ==========================================

total_playlist_appearances = len(df)

artist_performance['Artist_Dominance_Index'] = (
    artist_performance['Total_Appearances']
    / total_playlist_appearances
) * 100

artist_dominance = artist_performance.sort_values(
    'Artist_Dominance_Index',
    ascending=False
).head(15)

print("========== ARTIST DOMINANCE LEADERBOARD ==========\n")

display(
    artist_dominance[
        [
            'artist',
            'Unique_Songs',
            'Total_Appearances',
            'Total_Days_Appeared',
            'Average_Rank',
            'Average_Popularity',
            'Artist_Dominance_Index'
        ]
    ]
)

# ==========================================
# SONG-LEVEL PERFORMANCE ANALYSIS
# ==========================================

# Create song-level summary
song_performance = df.groupby(
    ['song', 'artist']
).agg(
    Days_on_Chart=('date', 'nunique'),
    Average_Rank=('position', 'mean'),
    Best_Rank_Achieved=('position', 'min'),
    Rank_Volatility_Index=('position', 'std'),
    Average_Popularity=('popularity', 'mean'),
    Max_Popularity=('popularity', 'max')
).reset_index()

# Replace missing volatility for songs appearing only once
song_performance['Rank_Volatility_Index'] = (
    song_performance['Rank_Volatility_Index'].fillna(0)
)

# Sort by longest chart presence
longest_chart_songs = song_performance.sort_values(
    'Days_on_Chart',
    ascending=False
)

print("===== SONG-LEVEL PERFORMANCE =====")
print("Total unique songs:", len(song_performance))

print("\nTop 10 Songs with Longest Playlist Presence:")
display(longest_chart_songs.head(10))

# ==========================================
# TOP SONGS BY AVERAGE RANK
# ==========================================

top_average_rank = song_performance.sort_values(
    'Average_Rank',
    ascending=True
).head(10)

print("===== TOP SONGS BY AVERAGE RANK =====")
display(top_average_rank)

# ==========================================
# SONGS WITH BEST PEAK RANK
# ==========================================

best_peak_songs = song_performance.sort_values(
    'Best_Rank_Achieved',
    ascending=True
).head(10)

print("===== SONGS WITH BEST PEAK RANK =====")
display(best_peak_songs)

# ==========================================
# MOST POPULAR SONGS
# ==========================================

most_popular_songs = song_performance.sort_values(
    'Average_Popularity',
    ascending=False
).head(10)

print("===== MOST POPULAR SONGS =====")
display(most_popular_songs)

# ==========================================
# PEAK RANK VS LONGEVITY
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(
    song_performance['Days_on_Chart'],
    song_performance['Best_Rank_Achieved'],
    alpha=0.6
)

plt.xlabel("Days on Chart")
plt.ylabel("Best Rank Achieved")
plt.title("Peak Rank vs Chart Longevity")

# Rank 1 should appear at the top
plt.gca().invert_yaxis()

plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# ARTIST PERFORMANCE ANALYSIS
# ==========================================

# Create artist-level performance summary
artist_performance = df.groupby('artist').agg(
    Unique_Songs=('song', 'nunique'),
    Total_Days_Appeared=('date', 'nunique'),
    Total_Chart_Appearances=('song', 'count'),
    Average_Rank=('position', 'mean'),
    Best_Rank_Achieved=('position', 'min'),
    Average_Popularity=('popularity', 'mean')
).reset_index()

# Artist Dominance Index
# Measures the share of total playlist appearances
total_appearances = artist_performance['Total_Chart_Appearances'].sum()

artist_performance['Artist_Dominance_Index'] = (
    artist_performance['Total_Chart_Appearances']
    / total_appearances
) * 100

# Sort by total chart appearances
artist_performance = artist_performance.sort_values(
    'Total_Chart_Appearances',
    ascending=False
)

print("===== ARTIST PERFORMANCE ANALYSIS =====")

print("Unique Artists:", len(artist_performance))

print("\nTop 10 Artists by Chart Appearances:")
display(artist_performance.head(10))

# ==========================================
# ARTISTS WITH MOST UNIQUE SONGS
# ==========================================

top_artists_songs = artist_performance.sort_values(
    'Unique_Songs',
    ascending=False
).head(10)

print("===== TOP ARTISTS BY UNIQUE SONGS =====")
display(top_artists_songs)

# ==========================================
# ARTIST DOMINANCE LEADERBOARD
# ==========================================

top_dominant_artists = artist_performance.sort_values(
    'Artist_Dominance_Index',
    ascending=False
).head(15)

print("===== ARTIST DOMINANCE LEADERBOARD =====")

display(
    top_dominant_artists[
        [
            'artist',
            'Unique_Songs',
            'Total_Days_Appeared',
            'Total_Chart_Appearances',
            'Average_Rank',
            'Average_Popularity',
            'Artist_Dominance_Index'
        ]
    ]
)

# ==========================================
# ARTIST DOMINANCE CHART
# ==========================================

import matplotlib.pyplot as plt

plot_data = top_dominant_artists.sort_values(
    'Artist_Dominance_Index'
)

plt.figure(figsize=(10, 6))

plt.barh(
    plot_data['artist'],
    plot_data['Artist_Dominance_Index']
)

plt.xlabel("Artist Dominance Index (%)")
plt.ylabel("Artist")
plt.title("Top Artists by Playlist Dominance")

plt.tight_layout()
plt.show()

# ==========================================
# ARTIST DOMINANCE OVER TIME
# ==========================================

# Find top 5 artists overall
top_5_artists = (
    artist_performance
    .head(5)['artist']
    .tolist()
)

# Count appearances by date and artist
artist_daily = (
    df[df['artist'].isin(top_5_artists)]
    .groupby(['date', 'artist'])
    .size()
    .reset_index(name='Appearances')
)

plt.figure(figsize=(12, 6))

for artist in top_5_artists:
    artist_data = artist_daily[
        artist_daily['artist'] == artist
    ]

    plt.plot(
        artist_data['date'],
        artist_data['Appearances'],
        label=artist
    )

plt.xlabel("Date")
plt.ylabel("Number of Songs in Top 50")
plt.title("Top Artist Dominance Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# POPULARITY VS RANK CORRELATION
# ==========================================

correlation = df['position'].corr(df['popularity'])

print("===== POPULARITY VS RANK =====")
print("Correlation between Playlist Rank and Popularity:",
      round(correlation, 4))

if correlation < 0:
    print("Interpretation: Higher popularity generally corresponds "
          "to better playlist positions.")
elif correlation > 0:
    print("Interpretation: Higher popularity generally corresponds "
          "to lower playlist positions.")
else:
    print("Interpretation: No clear linear relationship.")

# ==========================================
# POPULARITY DISTRIBUTION BY RANK GROUP
# ==========================================

df['Rank_Group'] = pd.cut(
    df['position'],
    bins=[0, 10, 20, 50],
    labels=['Top 10', 'Top 20', 'Top 50']
)

popularity_distribution = df.groupby(
    'Rank_Group',
    observed=False
)['popularity'].agg(
    ['count', 'mean', 'median', 'min', 'max', 'std']
).reset_index()

print("===== POPULARITY DISTRIBUTION =====")
display(popularity_distribution)

# ==========================================
# AVERAGE POPULARITY BY RANK GROUP
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(
    popularity_distribution['Rank_Group'].astype(str),
    popularity_distribution['mean']
)

plt.xlabel("Playlist Rank Group")
plt.ylabel("Average Popularity")
plt.title("Average Popularity Across Top 10, Top 20 and Top 50")

plt.tight_layout()
plt.show()

# ==========================================
# POPULARITY STABILITY VS CHART VOLATILITY
# ==========================================

song_stability = df.groupby(
    ['song', 'artist']
).agg(
    Average_Popularity=('popularity', 'mean'),
    Popularity_Std=('popularity', 'std'),
    Rank_Volatility=('position', 'std'),
    Days_on_Chart=('date', 'nunique')
).reset_index()

# Replace missing standard deviation for single observations
song_stability['Popularity_Std'] = (
    song_stability['Popularity_Std'].fillna(0)
)

song_stability['Rank_Volatility'] = (
    song_stability['Rank_Volatility'].fillna(0)
)

print("===== POPULARITY STABILITY VS CHART VOLATILITY =====")

display(
    song_stability.sort_values(
        'Rank_Volatility',
        ascending=False
    ).head(10)
)

# ==========================================
# POPULARITY VS RANK VOLATILITY
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    song_stability['Popularity_Std'],
    song_stability['Rank_Volatility'],
    alpha=0.5
)

plt.xlabel("Popularity Volatility")
plt.ylabel("Rank Volatility")
plt.title("Popularity Stability vs Chart Rank Volatility")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# DAILY POPULARITY TREND
# ==========================================

daily_popularity = df.groupby('date').agg(
    Average_Popularity=('popularity', 'mean'),
    Median_Popularity=('popularity', 'median')
).reset_index()

plt.figure(figsize=(12, 6))

plt.plot(
    daily_popularity['date'],
    daily_popularity['Average_Popularity'],
    label='Average Popularity'
)

plt.plot(
    daily_popularity['date'],
    daily_popularity['Median_Popularity'],
    label='Median Popularity'
)

plt.xlabel("Date")
plt.ylabel("Popularity Score")
plt.title("Daily Playlist Popularity Trend")
plt.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# FINAL KPI CALCULATION
# ==========================================

print("==========================================")
print("       PROJECT KEY PERFORMANCE INDICATORS")
print("==========================================")

# ------------------------------------------
# KPI 1: Days on Chart
# ------------------------------------------

average_days_on_chart = song_performance['Days_on_Chart'].mean()
max_days_on_chart = song_performance['Days_on_Chart'].max()

# ------------------------------------------
# KPI 2: Average Rank
# ------------------------------------------

overall_average_rank = df['position'].mean()

# ------------------------------------------
# KPI 3: Rank Volatility Index
# ------------------------------------------

average_rank_volatility = song_performance[
    'Rank_Volatility_Index'
].mean()

# ------------------------------------------
# KPI 4: Popularity Score Trend
# ------------------------------------------

average_popularity = df['popularity'].mean()

# Overall change from first day to last day
daily_popularity = df.groupby('date')['popularity'].mean().reset_index()

first_day_popularity = daily_popularity.iloc[0]['popularity']
last_day_popularity = daily_popularity.iloc[-1]['popularity']

popularity_trend_change = (
    last_day_popularity - first_day_popularity
)

# ------------------------------------------
# KPI 5: Artist Dominance Index
# ------------------------------------------

top_artist = artist_performance.iloc[0]['artist']
top_artist_dominance = artist_performance.iloc[0][
    'Artist_Dominance_Index'
]

# ------------------------------------------
# KPI 6: Explicit Content Share
# ------------------------------------------

explicit_share = (
    df['is_explicit'].mean() * 100
)

# ------------------------------------------
# DISPLAY KPIs
# ------------------------------------------

print("\nKPI 1 — DAYS ON CHART")
print("Average Days on Chart:",
      round(average_days_on_chart, 2))
print("Maximum Days on Chart:",
      round(max_days_on_chart, 2))

print("\nKPI 2 — AVERAGE RANK")
print("Overall Average Rank:",
      round(overall_average_rank, 2))

print("\nKPI 3 — RANK VOLATILITY INDEX")
print("Average Rank Volatility:",
      round(average_rank_volatility, 2))

print("\nKPI 4 — POPULARITY SCORE TREND")
print("Average Popularity:",
      round(average_popularity, 2))
print("First Day Popularity:",
      round(first_day_popularity, 2))
print("Last Day Popularity:",
      round(last_day_popularity, 2))
print("Popularity Change:",
      round(popularity_trend_change, 2))

print("\nKPI 5 — ARTIST DOMINANCE INDEX")
print("Top Artist:", top_artist)
print("Top Artist Dominance:",
      round(top_artist_dominance, 2), "%")

print("\nKPI 6 — EXPLICIT CONTENT SHARE")
print("Explicit Content Share:",
      round(explicit_share, 2), "%")

print("\n==========================================")
print("             KPI CALCULATION DONE")
print("==========================================")

# ==========================================
# KPI SUMMARY TABLE
# ==========================================

kpi_summary = pd.DataFrame({
    'KPI': [
        'Days on Chart',
        'Average Rank',
        'Rank Volatility Index',
        'Popularity Score Trend',
        'Artist Dominance Index',
        'Explicit Content Share'
    ],

    'Value': [
        round(average_days_on_chart, 2),
        round(overall_average_rank, 2),
        round(average_rank_volatility, 2),
        round(average_popularity, 2),
        round(top_artist_dominance, 2),
        round(explicit_share, 2)
    ],

    'Description': [
        'Average number of days songs remain on the playlist',
        'Overall average playlist position',
        'Average variation in song rankings',
        'Average listener popularity score',
        'Share of playlist appearances by the leading artist',
        'Percentage of playlist records that are explicit'
    ]
})

display(kpi_summary)

# ==========================================
# STEP 10: FEATURE ENGINEERING FOR ML
# ==========================================

import pandas as pd
import numpy as np

# Make a copy so the original dataframe is safe
ml_df = df.copy()

# Make sure date is datetime
ml_df['date'] = pd.to_datetime(ml_df['date'])

# Sort by song and date
ml_df = ml_df.sort_values(['song', 'date'])

# ------------------------------------------
# TARGET: NEXT PLAYLIST POSITION
# ------------------------------------------

ml_df['Next_Position'] = (
    ml_df.groupby('song')['position']
    .shift(-1)
)

# ------------------------------------------
# CURRENT PERFORMANCE FEATURES
# ------------------------------------------

ml_df['Rank_Change_Current'] = ml_df['Rank_Change'].fillna(0)

ml_df['Days_on_Chart_Current'] = ml_df['Days_on_Chart']

ml_df['Popularity_Current'] = ml_df['popularity']

ml_df['Duration_Minutes'] = ml_df['duration_minutes']

# ------------------------------------------
# SELECT FEATURES
# ------------------------------------------

features = [
    'position',
    'popularity',
    'duration_minutes',
    'Days_on_Chart',
    'Rank_Change_Current',
    'duration_ms',
    'total_tracks',
    'is_explicit',
    'album_type'
]

target = 'Next_Position'

ml_data = ml_df[features + [target] + ['date', 'song', 'artist']].copy()

# Remove rows where next position is unavailable
ml_data = ml_data.dropna(subset=[target])

print("===== ML DATASET =====")
print("Rows:", len(ml_data))
print("Columns:", len(ml_data.columns))

display(ml_data.head(10))

# ==========================================
# STEP 11: ML DATASET CHECK
# ==========================================

print("Missing values:")
display(ml_data.isnull().sum())

print("\nTarget statistics:")
print(ml_data['Next_Position'].describe())

print("\nTarget range:")
print("Minimum:", ml_data['Next_Position'].min())
print("Maximum:", ml_data['Next_Position'].max())

# ==========================================
# STEP 12: PREPARE FEATURES
# ==========================================

from sklearn.model_selection import train_test_split

X = ml_data[features].copy()
y = ml_data[target].copy()

# Convert True/False to 0/1
X['is_explicit'] = X['is_explicit'].astype(int)

# Convert categorical columns into numerical columns
X = pd.get_dummies(
    X,
    columns=['album_type'],
    drop_first=True
)

print("Feature shape:", X.shape)
print("\nFeatures:")
print(X.columns.tolist())

# ==========================================
# STEP 13: RANDOM FOREST REGRESSION
# ==========================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

# ------------------------------------------
# TIME-BASED TRAIN/TEST SPLIT
# ------------------------------------------

# Sort by date
ml_data = ml_data.sort_values('date')

X = ml_data[features].copy()
y = ml_data[target].copy()

# Convert categorical variables
X['is_explicit'] = X['is_explicit'].astype(int)

X = pd.get_dummies(
    X,
    columns=['album_type'],
    drop_first=True
)

# 80% training, 20% testing
split_index = int(len(X) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# ------------------------------------------
# RANDOM FOREST MODEL
# ------------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Prediction
y_pred = rf_model.predict(X_test)

# ------------------------------------------
# MODEL EVALUATION
# ------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\n==========================================")
print("       RANDOM FOREST MODEL RESULTS")
print("==========================================")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# ==========================================
# STEP 14: FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
})

importance = importance.sort_values(
    'Importance',
    ascending=False
)

print("===== FEATURE IMPORTANCE =====")

display(importance)

import matplotlib.pyplot as plt

top_features = importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features['Feature'],
    top_features['Importance']
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top Factors Influencing Next Playlist Position")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

import joblib

# Save Random Forest model
joblib.dump(rf_model, "playlist_random_forest.pkl")

# Save feature names
joblib.dump(feature_name, "feature_name.pkl")

print("Model saved successfully!")
print("Features saved successfully!")

import joblib

# Save the trained Random Forest model
joblib.dump(rf_model, "playlist_random_forest.pkl")

# Your feature names
feature_names = [
    'position',
    'popularity',
    'duration_minutes',
    'Days_on_Chart',
    'Rank_Change_Current',
    'duration_ms',
    'total_tracks',
    'is_explicit',
    'album_type_compilation',
    'album_type_single'
]

# Save feature names
joblib.dump(feature_names, "feature_names.pkl")

print("Model saved successfully!")
print("Feature names saved successfully!")

from google.colab import files

files.download("playlist_random_forest.pkl")
files.download("feature_names.pkl")

import joblib

# Save Random Forest model
joblib.dump(rf_model, "playlist_rank_model.pkl")

# Save feature names
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("Model and feature names saved successfully!")

from google.colab import files

files.download("playlist_rank_model.pkl")
files.download("feature_names.pkl")

import joblib

model1 = joblib.load("playlist_rank_model.pkl")

print(type(model1))

import os

print(os.path.exists("playlist_rank_model.pkl"))
print(os.path.exists("feature_names.pkl"))

!pip install -q streamlit pyngrok

%%writefile app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Spotify Playlist Analytics",
    page_icon="🎵",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    model = joblib.load("playlist_rank_model.pkl")
    features = joblib.load("feature_names.pkl")
    return model, features

model, feature_names = load_model()

# =========================================================
# TITLE
# =========================================================

st.title("🎵 Spotify Playlist Analytics Dashboard")

st.markdown(
    "### Analyze playlist performance and predict the next playlist position"
)

st.divider()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎧 Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard Overview",
        "Rank Prediction",
        "Model Information"
    ]
)

# =========================================================
# DASHBOARD OVERVIEW
# =========================================================

if page == "Dashboard Overview":

    st.header("📊 Project Key Performance Indicators")

    # KPI values from your analysis
    avg_days = 28.41
    max_days = 536
    avg_rank = 25.50
    volatility = 7.71
    avg_popularity = 87.67
    popularity_change = -4.04
    top_artist = "Taylor Swift"
    artist_dominance = 7.25
    explicit_share = 47.97
    total_songs = 977

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🎵 Total Unique Songs",
            f"{total_songs:,}"
        )

    with col2:
        st.metric(
            "📅 Avg. Days on Chart",
            f"{avg_days}"
        )

    with col3:
        st.metric(
            "🏆 Average Rank",
            f"{avg_rank}"
        )

    with col4:
        st.metric(
            "🔥 Avg. Popularity",
            f"{avg_popularity}"
        )

    st.markdown("---")

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            "📈 Rank Volatility",
            f"{volatility}"
        )

    with col6:
        st.metric(
            "👑 Top Artist",
            top_artist
        )

    with col7:
        st.metric(
            "🎤 Artist Dominance",
            f"{artist_dominance}%"
        )

    with col8:
        st.metric(
            "🔞 Explicit Content",
            f"{explicit_share}%"
        )

    st.markdown("---")

    # Popularity trend
    st.subheader("📈 Popularity Score Trend")

    trend_data = pd.DataFrame({
        "Period": ["First Day", "Average", "Last Day"],
        "Popularity": [90.06, 87.67, 86.02]
    })

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(
        trend_data["Period"],
        trend_data["Popularity"],
        marker="o",
        linewidth=2
    )

    ax.set_ylabel("Popularity Score")
    ax.set_title("Popularity Score Trend")
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    st.info(
        "Popularity decreased by 4.04 points from the first day "
        "to the last day in the analyzed period."
    )

    # Project summary
    st.subheader("📌 Project Summary")

    st.write(
        """
        This project analyzes Spotify playlist rankings to understand
        song performance, popularity, ranking movements, artist dominance,
        and playlist longevity.

        A Random Forest Regression model was developed to predict the
        next playlist position of a song based on its current performance
        characteristics.
        """
    )


# =========================================================
# RANK PREDICTION
# =========================================================

elif page == "Rank Prediction":

    st.header("🤖 Next Playlist Position Prediction")

    st.write(
        "Enter the current song information below. "
        "The Random Forest model will predict the next playlist position."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        position = st.number_input(
            "Current Playlist Position",
            min_value=1,
            max_value=50,
            value=25
        )

        popularity = st.number_input(
            "Popularity Score",
            min_value=0,
            max_value=100,
            value=85
        )

        duration_minutes = st.number_input(
            "Song Duration (minutes)",
            min_value=0.5,
            max_value=15.0,
            value=3.0,
            step=0.1
        )

        days_on_chart = st.number_input(
            "Days on Chart",
            min_value=1.0,
            max_value=1000.0,
            value=30.0
        )

        rank_change = st.number_input(
            "Current Rank Change",
            min_value=-49.0,
            max_value=49.0,
            value=0.0
        )

    with col2:

        duration_ms = st.number_input(
            "Duration (milliseconds)",
            min_value=30000,
            max_value=900000,
            value=180000
        )

        total_tracks = st.number_input(
            "Total Tracks in Album",
            min_value=1,
            max_value=100,
            value=20
        )

        explicit = st.selectbox(
            "Explicit Content",
            ["No", "Yes"]
        )

        album_type = st.selectbox(
            "Album Type",
            ["album", "single", "compilation"]
        )

    st.markdown("---")

    predict_button = st.button(
        "🔮 Predict Next Position",
        use_container_width=True
    )

    if predict_button:

        # Convert explicit value
        is_explicit = 1 if explicit == "Yes" else 0

        # Album encoding
        album_type_single = 1 if album_type == "single" else 0
        album_type_compilation = (
            1 if album_type == "compilation" else 0
        )

        # Create input dataframe
        input_data = pd.DataFrame({
            "position": [position],
            "popularity": [popularity],
            "duration_minutes": [duration_minutes],
            "Days_on_Chart": [days_on_chart],
            "Rank_Change_Current": [rank_change],
            "duration_ms": [duration_ms],
            "total_tracks": [total_tracks],
            "is_explicit": [is_explicit],
            "album_type_compilation": [
                album_type_compilation
            ],
            "album_type_single": [
                album_type_single
            ]
        })

        # Make sure columns are exactly the same
        input_data = input_data[feature_names]

        prediction = model.predict(input_data)[0]

        # Keep prediction within playlist range
        prediction = max(1, min(50, prediction))

        st.success(
            f"🎯 Predicted Next Playlist Position: "
            f"#{prediction:.0f}"
        )

        if prediction <= 10:
            st.info("🔥 Excellent! The song is predicted to remain near the top.")
        elif prediction <= 25:
            st.info("📈 Good performance. The song is predicted to remain in the upper half.")
        elif prediction <= 40:
            st.warning("⚠️ Moderate performance. The song may move toward the lower half.")
        else:
            st.error("📉 The song is predicted to move close to the bottom of the playlist.")


# =========================================================
# MODEL INFORMATION
# =========================================================

elif page == "Model Information":

    st.header("🌲 Random Forest Model")

    st.write(
        "The project uses a Random Forest Regression model "
        "to predict the next playlist position."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE", "2.76")

    with col2:
        st.metric("RMSE", "4.15")

    with col3:
        st.metric("R² Score", "0.9162")

    st.markdown("---")

    st.subheader("📊 Model Performance")

    st.write(
        """
        The Random Forest model achieved an R² score of 0.9162,
        meaning that the model explains a large proportion of the
        variation in the next playlist position.

        The Mean Absolute Error (MAE) is 2.76 positions, meaning
        predictions are approximately 2.76 playlist positions away
        from the actual value on average.
        """
    )

    st.subheader("⭐ Feature Importance")

    importance_data = pd.DataFrame({
        "Feature": [
            "position",
            "Days_on_Chart",
            "popularity",
            "Rank_Change_Current",
            "duration_ms",
            "duration_minutes",
            "total_tracks",
            "is_explicit",
            "album_type_single",
            "album_type_compilation"
        ],
        "Importance": [
            0.912703,
            0.022402,
            0.018899,
            0.018388,
            0.008739,
            0.008403,
            0.007591,
            0.001792,
            0.000792,
            0.000291
        ]
    })

    st.dataframe(
        importance_data,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Current playlist position is the strongest feature used "
        "by the model, with approximately 91.27% feature importance."
    )

    st.subheader("🧮 Model Features")

    st.write(feature_names)

!ls -lh app.py

!streamlit run app.py &>/content/log.txt &

!pip install -q pyngrok

from pyngrok import ngrok

public_url = ngrok.connect(8501)

print("Your Streamlit Dashboard:")
print(public_url)

!pip install streamlit -q

!streamlit run app.py &>/content/streamlit.log &

from google.colab.output import eval_js

url = eval_js("google.colab.kernel.proxyPort(8501)")
print(url)

!cat /content/streamlit.log

!ps aux | grep streamlit

!pkill -f streamlit

from google.colab.output import eval_js

url = eval_js("google.colab.kernel.proxyPort(8501)")
print(url)

!streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.enableXsrfProtection=false --server.enableCORS=false

import streamlit as st

st.success("Application loaded successfully.")
