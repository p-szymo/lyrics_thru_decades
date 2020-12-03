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

    # delistify each week, excluding edge case songs that get captured outside
    # of the top 10
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


# remove brackets and asterisks and words within
def remove_brackets(lyrics):
    try:
        return re.sub(r'[\[\{\*].*?[\]\}\*]', '', lyrics).strip()
    except BaseException:
        return lyrics


# remove certain number of lines from an endline separated string
def remove_n_lines(song, begin_n, end_n=0):

    # convert to list of lines
    lines = song.split('\n')

    # number of lines for indexing purposes
    num_lines = len(lines)

    # slice lines and rejoin
    sliced_str = '\n'.join(lines[begin_n:num_lines - end_n])

    return sliced_str


# an attempt at a catch-all rescraping function (not quite)
def rescrape(url, **kwargs):
    soup = soupify(url)
    lyrics = '\n'.join([line for line in soup.find(
        **kwargs).contents if isinstance(line, str) if line.strip()])
    return lyrics.strip()


# rescrape by only using the main (non-featured) artist in the search
def featuring(df, ind, access_token):

    # capture main artist
    main_artist = df.loc[ind, 'artist'].lower().split(' featuring ')[0]

    # update search term
    search_term = f"{df.loc[ind, 'title']} {main_artist}"

    # rescrape
    return lyrics_grabber(access_token, search_term)


# separate double-songs
def split_combos(df, ind):

    # split titles
    first_title = df.loc[ind, 'title'].split(' / ')[0]
    second_title = df.loc[ind, 'title'].split(' / ')[1]

    # capture artist and year
    artist = df.loc[ind, 'artist']
    year = df.loc[ind, 'year']

    # overwrite row with first title
    df.loc[ind, 'title'] = first_title

    # rescrape first title's lyrics (or NaN)
    try:
        df.loc[ind, 'lyrics'] = lyrics_grabber(
            access_token, f'{first_title} {artist}')
    except BaseException:
        df.loc[ind, 'lyrics'] = np.nan

    # rescrape second title's lyrics (or NaN)
    try:
        second_lyrics = lyrics_grabber(
            access_token, f'{second_title} {artist}')
    except BaseException:
        second_lyrics = np.nan

    # use info to make a dictionary
    second_dict = {'year': year,
                   'title': second_title,
                   'artist': artist,
                   'lyrics': second_lyrics}

    return second_dict


# remove words preceding colons
def colon_killer(song, split_on=':', total_kill=False):

    # convert to list
    lines = song.split('\n')

    # iterate over index and list of lines
    for i, line in enumerate(lines):

        # only perform on lines with target
        if split_on in line:

            # split using target
            temp = line.split(split_on)

            # string before target, as a list of words
            temp_words = temp[0].split()

            # if only one or two words
            if len(temp_words) < 3:

                # use only what is after the target (may be empty string)
                lines[i] = temp[1]

            # if 3 or more words precede target
            else:

                # remove 2 words preceding target if `total_kill=True`
                if total_kill:
                    lines[i] = ' '.join(temp_words[:-2]) + temp[1]

                # otherwise skip it
                else:
                    continue

    # convert back to string
    return '\n'.join(lines)


# remove artist names from lyrics (most likely requires everything to be lowercase)
def artist_remover(df, ind, artist_col, lyrics_col, stopwords=None):

    # main artist and featured artist(s), as two strings
    artists_full = df.loc[ind, artist_col].split(' featuring ')

    # each name/word as its own string in a list
    artists_split = [name for artist in artists_full for name in artist.split()]

    if stopwords:
        artists_split = [word for word in artists_split if word not in stopwords]

    # original lyrics
    lyrics = df.loc[ind, lyrics_col]

    # remove instances of those names in lyrics string
    for name in artists_split:
        lyrics = lyrics.replace(name, '')

    # lyrics without leading or trailing whitepace
    return lyrics.strip()
