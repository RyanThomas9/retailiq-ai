"""Public-demo weather exposure engine. All historical store weather is synthetic.

Rules are genericized for this portfolio: precipitation >=3 in/day OR gust >=60 mph;
snowfall >=2 in/day. Counts store-days, not distinct stores. These events may overlap.
"""
import numpy as np
import pandas as pd
import requests
import streamlit as st

REGION_CITIES = {
    'North': ('Chicago', 41.8781, -87.6298),
    'South': ('Atlanta', 33.7490, -84.3880),
    'East': ('New York', 40.7128, -74.0060),
    'West': ('Denver', 39.7392, -104.9903),
}


def exposure_flags(precip_in, gust_mph, snow_in):
    return (precip_in >= 3.0) | (gust_mph >= 60.0), snow_in >= 2.0


@st.cache_data(show_spinner=False)
def synthetic_weather(store_meta, end_date, days=1000, seed=2201):
    """Synthetic independent store-day weather; NOT real historical weather observations."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp(end_date), periods=days, freq='D')
    frames = []
    for store, region in store_meta[['store', 'region']].drop_duplicates().itertuples(index=False, name=None):
        n = len(dates)
        winter = np.isin(dates.month, [12, 1, 2, 3])
        snow_probability = np.where(winter, 0.018 if region in ('North', 'West', 'East') else 0.003, 0.0002)
        snow = np.where(rng.random(n) < snow_probability, rng.uniform(2, 8, n), 0.0)
        rain = np.where(rng.random(n) < 0.012, rng.uniform(3, 6, n), rng.uniform(0, 0.6, n))
        gust = np.where(rng.random(n) < 0.007, rng.uniform(60, 85, n), rng.uniform(5, 40, n))
        severe, heavy_snow = exposure_flags(rain, gust, snow)
        frames.append(pd.DataFrame({'date': dates, 'store': store, 'region': region,
                                    'severe': severe.astype(int), 'snow': heavy_snow.astype(int)}))
    return pd.concat(frames, ignore_index=True)


def weekly_exposure(weather):
    """Sunday-Saturday retail weeks, TY / comparable LY / 2YA by 364-day shift."""
    df = weather.copy()
    df['week_start'] = df['date'] - pd.to_timedelta((df['date'].dt.dayofweek + 1) % 7, unit='D')
    weeks = df.groupby('week_start', as_index=False)[['severe', 'snow']].sum()
    prev = weeks.rename(columns={'week_start': 'week_prior', 'severe': 'severe_ly', 'snow': 'snow_ly'})
    prev2 = weeks.rename(columns={'week_start': 'week_2ya', 'severe': 'severe_2ya', 'snow': 'snow_2ya'})
    weeks['week_prior'] = weeks.week_start - pd.Timedelta(days=364)
    weeks['week_2ya'] = weeks.week_start - pd.Timedelta(days=728)
    return weeks.merge(prev, on='week_prior', how='left').merge(prev2, on='week_2ya', how='left')


@st.cache_data(ttl=3600, show_spinner=False)
def public_forecast(region):
    """Representative city only; public forecast, not synthetic store-level weather."""
    city, lat, lon = REGION_CITIES[region]
    response = requests.get('https://api.open-meteo.com/v1/forecast', params={
        'latitude': lat, 'longitude': lon,
        'daily': 'precipitation_sum,snowfall_sum,wind_gusts_10m_max',
        'forecast_days': 7, 'timezone': 'America/New_York',
    }, timeout=8)
    response.raise_for_status()
    daily = response.json()['daily']
    result = pd.DataFrame({'date': pd.to_datetime(daily['time']),
                           'precip_in': np.array(daily['precipitation_sum'], dtype=float) / 25.4,
                           'snow_in': np.array(daily['snowfall_sum'], dtype=float) / 2.54,
                           'gust_mph': np.array(daily['wind_gusts_10m_max'], dtype=float) / 1.609344})
    result['severe'], result['snow'] = exposure_flags(result.precip_in, result.gust_mph, result.snow_in)
    result['city'] = city
    result['region'] = region
    return result


def forecast_summary():
    frames, errors = [], []
    for region in REGION_CITIES:
        try:
            frames.append(public_forecast(region))
        except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
            errors.append(region)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), errors


def render_weather(scored):
    st.subheader('Weather Intelligence')
    st.caption('Historical store-day exposure is synthetic. Public forecasts use representative regional cities, not real store locations.')
    meta = scored[['store', 'region']].drop_duplicates()
    weather = synthetic_weather(meta, scored.date.max(), days=1000)
    weekly = weekly_exposure(weather)
    latest = weekly.iloc[-1]
    a, b, c = st.columns(3)
    a.metric('Severe-weather store-days · latest week', f'{int(latest.severe):,}',
             None if pd.isna(latest.severe_ly) else f'{latest.severe - latest.severe_ly:+.0f} vs LY')
    b.metric('Heavy-snow store-days · latest week', f'{int(latest.snow):,}',
             None if pd.isna(latest.snow_ly) else f'{latest.snow - latest.snow_ly:+.0f} vs LY')
    b.caption('A store-day may count in both categories.')
    c.metric('Estimated lost sales', 'Not modeled')
    st.line_chart(weekly.tail(52).set_index('week_start')[['severe', 'snow']], height=220)
    st.markdown('#### Next 7 days · public regional forecast')
    forecast, errors = forecast_summary()
    if forecast.empty:
        st.warning('Forecast service unavailable. No weather-risk estimate displayed.')
    else:
        events = forecast.loc[forecast.severe | forecast.snow].copy()
        if events.empty:
            st.success('No threshold events forecast at the four representative city locations.')
        else:
            events['event'] = np.select([events.severe & events.snow, events.snow, events.severe],
                                        ['Severe weather + heavy snow', 'Heavy snow', 'Severe weather'], default='')
            st.dataframe(events[['date', 'city', 'region', 'event', 'precip_in', 'snow_in', 'gust_mph']].round(1),
                         hide_index=True, use_container_width=True)
        if errors:
            st.warning('Forecast data unavailable for: ' + ', '.join(errors))
    st.caption('Thresholds: ≥3 inches daily precipitation or ≥60 mph wind gust; ≥2 inches daily snowfall. '
               'Historical comparisons use 364-day fiscal alignment. Public forecasts cover seven days, not sixteen weeks. '
               'No causal sales impact is asserted.')
