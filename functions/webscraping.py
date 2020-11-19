# webscraping functions
import requests as rq
from bs4 import BeautifulSoup as bs

# string manipulation
import re

# lyrics retrieval
from lyricsgenius import Genius


# simplify souping process
def soupify(url):
    page = rq.get(url)
    soup = bs(page.content, 'html.parser')
    return soup


# find top 10 song titles and artists for each week in a year
def yearly_top10s(
        url,
        pattern=r'\s{1,2}(?:[1-9]|0[1-9]|10)\s[\d\w][\d\w]?0?\s(.+) –•– ([\w\d\s,\.’&-]+)'):

    # convert webpage to soup object
    soup = soupify(url)

    # find all `p` html tags
    p_tags = soup.find_all('p')

    # find instances with a list of top songs
    weeks = [week.contents for week in p_tags if len(week.contents) > 10]

    # pull important info and wrangle into better format
    weeks_updated = [[' ' + str(line) for line in week
                      if str(line) != '<br/>'
                      if str(line) != '\n——']
                     for week in weeks]

    # process text to grab only the top 10 for each week
    weekly_top10s = [[(re.match(pattern, line).group(1).strip(),
                       re.match(pattern, line).group(2).strip())
                      for line in week if re.match(pattern, line)]
                     for week in weeks_updated]

    # delistify each week, excluding edge case songs that get captured outside of the top 10
    top10s = [song for week in weekly_top10s for song in week if len(week) > 5]

    # only unique songs
    unique_top10s = list(set(top10s))

    return sorted(unique_top10s, key=lambda x: x[1])


# lyrics through genius.com
def lyrics_grabber(access_token, search_term, base_url='https://genius.com'):

    # instantiate genius object
    genius = Genius(access_token)

    # search for song
    song = genius.search(search_term)

    # find url for top hit
    url_addon = song['hits'][0]['result']['path']

    # retrieve lyrics
    lyrics = genius.lyrics(base_url + url_addon)

    return lyrics


# remove brackets and words within
def remove_brackets(lyrics):
    try:
        return re.sub(r'[\[\{].*?[\]\}]', '', lyrics).strip()
    except BaseException:
        return lyrics


# remove certain number of lines from an endline separated string
def remove_n_lines(song, begin_n, end_n=0):

    # convert to list of lines
    lines = song.split('\n')

    # number of lines for indexing purposes
    num_lines = len(lines)

    # slice lines and rejoin
    sliced_str = '\n'.join(lines[begin_n:num_lines-end_n])

    return sliced_str


# an attempt at a catch-all rescraping function (not quite)
def rescrape(url, **kwargs):
    soup = soupify(url)
    lyrics = '\n'.join(
        [line for line in soup.find(**kwargs).contents if isinstance(line, str) if line.strip()]
)
    return lyrics.strip()
