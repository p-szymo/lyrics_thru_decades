# Music Lyrics Through the Years

#### *Note*: This project is in its early (webscraping) stages. 

## Summary
After scraping top 10 music lyrics from [Top 40 Weekly](https://top40weekly.com/), I'll use natural language processing (NLP) techniques to analyze the form, structure, and word usage of six decades: 1960s, 1970s, 1980s, 1990s, 2000s, and 2010s.

Through text processing, feature engineering, and exploratory data analysis, I hope to discover insights into how popular songwriting has changed (or stayed the same) throughout the years. I'll then create predictive models to provide further insight and confirm my findings during EDA.

## Objectives

TBD

## Findings

TBD

# Conclusions
TBD

## Next steps
TBD

## List of files
- **archives** folder - scrap and backup files
- **data** folder - datasets, corpora, and models
- **functions** folder - python functions files
	- **webscraping.py** - file with functions used in webscraping and data cleaning
- **.gitignore** - list of files and pathways to ignore
- **01_webscraping.ipynb** - notebook of scraping and compiling data
- **README.md** - this very file!
- **__init__.py** - file to direct to the functions folder

## Repo structure
```
.
├── 01_webscraping.ipynb
├── README.md
├── __init__.py
├── archives
│   ├── 01_webscraping-BACKUP_2020-11-11_early_process-Copy1.ipynb
│   ├── 01_webscraping-BACKUP_2020-11-11_early_process.ipynb
│   └── 01_webscraping-BACKUP_2020-11-14_before-abandoning-selenium.ipynb
├── data
│   ├── api_keys.json
│   └── lyrics_df.pkl
└── functions
    ├── __init__.py
    └── webscraping.py
```