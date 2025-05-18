import re

def split_sentences(text: str) -> list[str]:
    lines = text.splitlines()
    sentences = [sentence for line in lines for sentence in re.split(r'[\.\?\!]\ (?=[A-Z])', line)]
    
    """
        The following syntax is the short version of the following:
        
        ```
            for s in sentences:
                for part in split_into_two(s):
                    if len(part) > 0:
                        return part
        ```
        
        It's very confusing at first (I got confused at first as well, asked AI to explain it to me),
        but it's a simple and fast implementation of flatMap() in Python.
    """
    
    sentences = [part for s in sentences for part in split_sentence_into_two(s) if len(part) > 0]
    return sentences

def split_sentence_into_two(sentence: str, min_words=20):
    words = sentence.split(' ')
    
    if len(words) > min_words:
        midpoint = len(words) // 2
        return ' '.join(words[:midpoint]), ' '.join(words[midpoint:])
    
    return sentence, ''