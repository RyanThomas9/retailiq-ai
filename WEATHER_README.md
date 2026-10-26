# Weather Intelligence (public portfolio demo)

The Weather Intelligence page uses synthetic historical weather at store/day grain and real public 7-day forecasts for four representative cities (Chicago, Atlanta, New York, Denver). It does **not** use Tailored Brands weather records, locations, sales-impact coefficients, or proprietary queries.

Thresholds used in this demo: severe weather = daily precipitation >= 3 inches OR wind gust >= 60 mph; heavy snow = daily snowfall >= 2 inches. A store-day can be in both categories. The historical summary uses Sunday-start retail weeks, with 364-day and 728-day fiscal comparisons. Public forecasts are regional proxies and should not be interpreted as forecasts for any actual store.

Forecast data source: Open-Meteo forecast API (https://open-meteo.com/), refreshed at most hourly in Streamlit cache. Historical weather is generated synthetically and deterministically. Forecast failures show an unavailable status; they do not generate fake results.

This release does not claim or quantify weather-caused lost sales. A future validated impact model would require independent historical weather, sales, store geography, and appropriate controls.
