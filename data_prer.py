import re
import requests


url_1 = "https://sherlock-holm.es/stories/plain-text/cnus.txt"
response_1 = requests.get(url_1)

with open("data/sherlock/input.txt", "w", encoding="utf-8") as file:
    file.write(response_1.text)

print("Sherlock Holmes dataset downloaded successfully!")

url_2 = "https://www.gutenberg.org/files/11/11-0.txt"
response_2 = requests.get(url_2)

with open("data/alice/input.txt", "w", encoding="utf-8") as file:
    file.write(response_2.text)

print("Alice dataset downloaded successfully!")

# read Sherlock Holmes data
with open("data/sherlock/input.txt", "r", encoding="utf-8") as f:
    sherlock_text = f.read()

# read Alice data
with open("data/alice/input.txt", "r", encoding="utf-8") as f:
    alice_text = f.read()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text

sherlock_text = clean_text(sherlock_text)
alice_text = clean_text(alice_text)


combined_text = sherlock_text + "\n\n" + alice_text


with open("data/combined/input.txt", "w", encoding="utf-8") as f:
    f.write(combined_text)




