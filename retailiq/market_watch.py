"""Public-news market watch. Deterministic scoring, not LLM-generated analysis."""
import datetime as dt
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import quote, urlparse
import requests
import streamlit as st

SEARCHES = [
    'US retail sales consumer spending apparel',
    'US jobs inflation consumer confidence retail',
    'US apparel stores tariffs clothing retail',
]
POSITIVE = {'growth', 'rebound', 'rise', 'rises', 'increase', 'increases', 'improves', 'stronger', 'surge'}
NEGATIVE = {'decline', 'declines', 'fall', 'falls', 'weakens', 'weak', 'layoffs', 'slump', 'drop', 'drops', 'tariffs'}
RETAIL = {'retail': 4, 'consumer': 3, 'spending': 3, 'apparel': 5, 'clothing': 5,
          'stores': 2, 'sales': 2, 'shopping': 2, 'footwear': 3, 'employment': 2,
          'unemployment': 3, 'inflation': 3, 'confidence': 3, 'tariffs': 2, 'jobs': 2}


def score_headline(title):
    tokens = set(re.findall(r"[a-z]+", title.lower()))
    return sum(weight for word, weight in RETAIL.items() if word in tokens)


def direction(title):
    tokens = set(re.findall(r"[a-z]+", title.lower()))
    if tokens & NEGATIVE and not tokens & POSITIVE: return 'Potential headwind'
    if tokens & POSITIVE and not tokens & NEGATIVE: return 'Potential tailwind'
    return 'Monitor'


def relevance(title):
    lower = title.lower()
    if any(x in lower for x in ['apparel', 'clothing', 'footwear']):
        return 'Apparel demand and category performance'
    if any(x in lower for x in ['jobs', 'employment', 'unemployment', 'layoffs']):
        return 'Household employment and discretionary purchasing power'
    if any(x in lower for x in ['inflation', 'prices', 'tariffs']):
        return 'Consumer purchasing power and merchandise costs'
    return 'Consumer spending and retail traffic'


def parse_feed(xml_bytes, now):
    root = ET.fromstring(xml_bytes)
    out = []
    for item in root.findall('.//item'):
        title = (item.findtext('title') or '').strip()
        link = (item.findtext('link') or '').strip()
        raw = (item.findtext('pubDate') or '').strip()
        if not title or not link or not link.startswith('https://'):
            continue
        try:
            published = parsedate_to_datetime(raw)
            if published.tzinfo is None: published = published.replace(tzinfo=dt.timezone.utc)
            age = (now - published.astimezone(dt.timezone.utc)).total_seconds() / 3600
        except (ValueError, TypeError, OverflowError):
            continue
        if age < -2 or age > 96: continue
        score = score_headline(title)
        if score < 4: continue
        out.append({'title': title, 'url': link, 'published': published.isoformat(),
                    'score': score, 'direction': direction(title), 'relevance': relevance(title)})
    return out


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_market_news():
    now = dt.datetime.now(dt.timezone.utc)
    records, errors = [], 0
    for query in SEARCHES:
        url = 'https://news.google.com/rss/search?q=' + quote(query + ' when:3d') + '&hl=en-US&gl=US&ceid=US:en'
        try:
            response = requests.get(url, timeout=9, headers={'User-Agent':'Mozilla/5.0 RetailIQ-Portfolio/1.0'})
            response.raise_for_status()
            records.extend(parse_feed(response.content, now))
        except (requests.RequestException, ET.ParseError):
            errors += 1
    dedup = {}
    for row in records:
        key = re.sub(r'[^a-z0-9]+', '', row['title'].lower().split(' - ')[0])[:95]
        if key not in dedup or row['score'] > dedup[key]['score']:
            dedup[key] = row
    ranked = sorted(dedup.values(), key=lambda x: (x['score'], x['published']), reverse=True)
    return ranked[:15], errors, now.isoformat()


def render_market_watch():
    st.markdown('<div class="ri-section">◉ &nbsp; AI Market Watch</div>', unsafe_allow_html=True)
    st.markdown('<div class="ri-watch-sub">Current retail news and economic signals · Free, rules-based relevance analysis</div>', unsafe_allow_html=True)
    try:
        items, errors, checked = fetch_market_news()
    except Exception:
        items, errors, checked = [], len(SEARCHES), None
    if not items:
        st.info('No verified recent market headlines available. Market Watch cannot assess news right now; no risks are inferred.')
    else:
        st.markdown('<div class="ri-watch-caption">Potential business relevance — not a verified sales impact or AI causal finding</div>', unsafe_allow_html=True)
        for item in items[:3]:
            from html import escape
            stamp = dt.datetime.fromisoformat(item['published']).strftime('%b %d, %Y')
            # st.link_button uses the original HTTPS source link without injecting it into HTML.
            with st.container(border=True):
                st.markdown(f'<div class="ri-news-direction">{escape(item["direction"])} · {stamp}</div>', unsafe_allow_html=True)
                st.markdown('**' + item['title'].replace('**','') + '**')
                st.caption('Why it may matter: ' + item['relevance'] + '. Check sales and traffic before attributing any change.')
                st.link_button('Read source', item['url'], type='tertiary')
    if errors:
        st.caption(f'{errors} of {len(SEARCHES)} news feeds unavailable; results may be incomplete.')
    st.caption('News via Google News RSS (third-party headlines); refreshed at most every 6 hours. Sources and timestamps should be checked before decisions. No paid LLM is used.')
