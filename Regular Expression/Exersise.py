# 1. 

import re

data = ['789', '123', '004']
def is_octal(s):
    return bool(re.fullmatch(r'[0-7]+', s))

octal_only = [s for s in data if is_octal(s)]
print(octal_only)







# 2. 

import re

emails = ['zuck@facebook.com', 'sundar@google.com', 'jeff@amazon.com']
def extract_parts(email):
    match = re.match(r'(\w+)@(\w+)\.(\w+)', email)
    return match.groups() if match else ()

results = [extract_parts(email) for email in emails]
print(results)







# 3. 

import re

sentence = "A very   very     irregular sentence"
def normalize_spacing(text):
    return re.sub(r'\s+', ' ', text).strip()

cleaned = normalize_spacing(sentence)
print(cleaned)







# 4. 

import re

tweet = '''Good advice! RT @TheNextWeb: What I would do differently if I was learning to code today http://t.co/lbwej0pX0d cc: @garybernhardt #rstats'''

def clean_tweet(text):
    text = re.sub(r"http\S+", "", text)         
    text = re.sub(r"@\w+", "", text)             
    text = re.sub(r"#\w+", "", text)             
    text = re.sub(r"\bRT\b|\bcc\b:?", "", text)   
    text = re.sub(r"[^\w\s]", "", text)           
    return text.strip()

print(clean_tweet(tweet))






# 5. 

import requests
from bs4 import BeautifulSoup

url = 'https://raw.githubusercontent.com/selva86/datasets/master/sample.html'
r = requests.get(url)

def extract_visible_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    return [tag.get_text(strip=True) for tag in soup.find_all() if tag.get_text(strip=True)]

desired_output = extract_visible_text(r.text)
print(desired_output)






# 6. 

words = ['civic', 'trust', 'words', 'maximum', 'museums', 'aas']

def match_start_end(word_list):
    return [word for word in word_list if word[0].lower() == word[-1].lower()]

print(match_start_end(words))
