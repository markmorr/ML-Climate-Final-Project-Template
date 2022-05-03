# United States of America Python Dictionary to translate States,
# Districts & Territories to Two-Letter codes and vice versa.
#
# Canonical URL: https://gist.github.com/rogerallen/1583593
#
# Dedicated to the public domain.  To the extent possible under law,
# Roger Allen has waived all copyright and related or neighboring
# rights to this code.  Data originally from Wikipedia at the url:
# https://en.wikipedia.org/wiki/ISO_3166-2:US
#
# Automatically Generated 2021-09-11 18:04:36 via Jupyter Notebook from
# https://gist.github.com/rogerallen/d75440e8e5ea4762374dfd5c1ddf84e0 
# CITATION: NOT MY CODE--THIS CAME FROM ROGERA ALLEN, AT THE FOLLOWING URL: https://gist.github.com/rogerallen/1583593
# 

def get_us_state_to_abbrev():
    us_state_to_abbrev = {
    "alabama": "al",
    "alaska": "ak",
    "arizona": "az",
    "arkansas": "ar",
    "california": "ca",
    "colorado": "co",
    "connecticut": "ct",
    "delaware": "de",
    "florida": "fl",
    "georgia": "ga",
    "hawaii": "hi",
    "idaho": "id",
    "illinois": "il",
    "indiana": "in",
    "iowa": "ia",
    "kansas": "ks",
    "kentucky": "ky",
    "louisiana": "la",
    "maine": "me",
    "maryland": "md",
    "massachusetts": "ma",
    "michigan": "mi",
    "minnesota": "mn",
    "mississippi": "ms",
    "missouri": "mo",
    "montana": "mt",
    "nebraska": "ne",
    "nevada": "nv",
    "new hampshire": "nh",
    "new jersey": "nj",
    "new mexico": "nm",
    "new york": "ny",
    "north carolina": "nc",
    "north dakota": "nd",
    "ohio": "oh",
    "oklahoma": "ok",
    "oregon": "or",
    "pennsylvania": "pa",
    "rhode island": "ri",
    "south carolina": "sc",
    "south dakota": "sd",
    "tennessee": "tn",
    "texas": "tx",
    "utah": "ut",
    "vermont": "vt",
    "virginia": "va",
    "washington": "wa",
    "west virginia": "wv",
    "wisconsin": "wi",
    "wyoming": "wy",
    "district of columbia": "dc",
    "american samoa": "as",
    "guam": "gu",
    "northern mariana islands": "mp",
    "puerto rico": "pr",
    "united states minor outlying islands": "um",
    "u.s. virgin islands": "vi",
}

    return us_state_to_abbrev
    
# invert the dictionary
# abbrev_to_us_state = dict(map(reversed, us_state_to_abbrev.items()))