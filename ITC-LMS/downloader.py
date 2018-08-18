import re
from pathlib import Path

import bs4
import requests
from settings import *


class NameFormatter:
    def __init__(self, pattern):
        self.pattern = pattern
        self.regex = re.compile(pattern)

    def format(self, name):
        formatted_name = self.regex.match(name).group()
        assert formatted_name
        return formatted_name


account = Account()
login_page = LoginPage()
stat_page = StatIsPage()

login_info = {
    'username': account.username,
    'password': account.password,
    'initialURI': login_page.initialURI,
}

# login
session = requests.session()
res = session.post(login_page.url, data=login_info)
res.raise_for_status()
print('logined')

res = session.get(stat_page.url)
res.raise_for_status()

soup = bs4.BeautifulSoup(res.text, features='lxml')
download_a_lis = soup.find(class_='content').find_all('a')
download_files = {a.text: a.attrs['href'] for a in download_a_lis}

name_formatter = NameFormatter(r'^.+\.(pdf|ipynb)')
download_files = {name_formatter.format(name): url for name, url in download_files.items()}

save_dir = Path(SAVE_PATH)
for f_name, f_url in download_files.items():
    print(f'downloading {f_name}')
    res = session.get(f_url)
    res.raise_for_status()
    with save_dir.joinpath(f_name).open('wb') as f:
        for chunk in res.iter_content(100000):
            f.write(chunk)

print('complete')
