import sys, os, csv
from random import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import spacy
nlp = spacy.load('en_core_web_sm')

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn", device_map="auto")
model = model.to('cuda')

summaries = open("lesswrong_summaries.md","w")
summaries.write("# The abbreviated Eliezer Yudkowsky\n")
post_title = ""
self_text = ""
last_text = ""

def summarize(text):
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to('cuda')
    summary_ids = model.generate(input_ids, num_beams=2, do_sample=True, min_length=0, max_length=200)
    output = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output

with open("lesswrong.txt","r") as corpus:
    fails = 0
    posts = 0
    for line in corpus.readlines():
        if line=="\n": # there are a ton of extra newlines for some reason
            pass
        elif line.startswith("Eliezer Yudkowsky, "): # this always follows the title and precedes the post
            post_title = last_text.replace("\n","")
            print(f"Post found: {post_title}")
            self_text = ""
            posts += 1
        elif line.startswith("<|endoftext|>"): # note: I replaced "* * *" with this via GSAR for GPT-2 training
            if post_title:
                # following loop was introduced to deal with BART's tendency to attribute EY statements to made-up journalists
                for attempt in range(10): 
                    summarized_text = summarize(self_text)
                    doc = nlp(summarized_text)
                    hallucination = False # initialize
                    if doc.ents: # check summary for named entities
                        for ent in doc.ents:
                            # make sure the named entity exists in the original post
                            if ent.text not in self_text:
                                hallucination = True
                                print(f'Entity not found ({attempt}): "{ent.text}"')
                                break
                    if not hallucination:
                        summaries.write(f"## {post_title}\n")
                        summaries.write(summarized_text+"\n")
                        post_title = ""
                        break
                    elif attempt==9:
                        print(f"Failed to summarize post due to persistent hallucinations")
                        fails += 1
            self_text = ""
        else:
            self_text += line # accumulate the lines to form the post
            last_text = line # keeping track of this in case it's the title

print(f"Read {posts} posts; {fails} not written due to persistent hallucinations.")
summaries.close()
