"""Conservative relevance rules for the small active stock universe."""
import re

VERSION = 'financial-headlines-v2'
QUERY_CONTEXT = '(stock OR shares OR earnings OR revenue OR nasdaq)'


def financial_headline(text, ticker):
    # AMZN is also a video-source tag. Apple Hospitality is a different issuer.
    if re.search(r'\b(web[- .]?dl|webrip|h[ .]?26[45]|1080p|2160p|720p)\b', text, re.I):
        return False
    if re.search(r'\b(stock\s+pot|in[- ]stock\s+alert|coupon|promo code|shipped from|shopping deals)\b', text, re.I):
        return False
    if ticker == 'AAPL' and re.search(r'\bApple\s+(Hospitality|REIT|Bank|Leisure)\b', text, re.I):
        return False
    return bool(re.search(r'\b(stock|stocks|shares|earnings|revenue|profits?|nasdaq|dividends?|'
                          r'investors?|valuation|market cap|price target|cash flow|guidance|analysts?|'
                          r'quarterly results|buy|sell|bullish|bearish)\b', text, re.I))
