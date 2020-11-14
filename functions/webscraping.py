# webscraping functions
import requests as rq
from bs4 import BeautifulSoup as bs

# string manipulation
import re


def yearly_top10s(
    url,
    pattern=r'\s{1,2}(?:[1-9]|0[1-9]|10) [1-9][0-9]? ([\w\s\.,’“”\(\)\–\-]+) –•– ([A-Za-z\s,]+)'):
    
    # convert webpage to soup object
    page = rq.get(url)
    soup = bs(page.content, 'html.parser')
    
    # find all `p` html tags
    p_tags = soup.find_all('p')
    
    # find instances with a list of top songs
    weeks = [week.contents for week in p_tags if len(week.contents) > 80]
    
    # pull important info and wrangle into better format
    weeks_updated = [[' ' + str(line) for line in week
                          if str(line) != '<br/>'
                          if str(line) != '\n——']
                     for week in weeks]
    
    # process text to grab just the top 10 for each week
    weekly_top10s = [[(re.match(pattern, line).group(1).strip(),
                       re.match(pattern, line).group(2).strip())
                      for line in week if re.match(pattern, line)]
                     for week in weeks_updated]
    
    # delistify each week
    top10s = [song for week in weekly_top10s for song in week]
    
    # only unique songs
    unique_top10s = list(set(top10s))
    
    return unique_top10s