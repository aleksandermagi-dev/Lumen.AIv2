from __future__ import annotations

import re


class TextNormalizer:
    """Applies lightweight normalization for messy conversational input."""

    PHRASE_REPLACEMENTS = {
        "w/": "with",
        "b/c": "because",
        "aintnobodygonnadoitlikethat": "aint nobody gonna do it like that",
        "idontthinkthatsright": "i do not think that is right",
        "imnotsureboutthat": "i am not sure about that",
        "how r u": "how are you",
        "how ru": "how are you",
        "whatcha": "what are you",
        "whatcha got on ur mind": "what do you have on your mind",
        "whatcha got on your mind": "what do you have on your mind",
        "real quick": "quickly",
        "can ya": "can you",
        "could ya": "could you",
        "would ya": "would you",
        "lemme see": "let me see",
        "gotta be honest": "got to be honest",
        "sorta feels like": "sort of feels like",
    }

    TOKEN_REPLACEMENTS = {
        "aint": "is not",
        "whats": "what is",
        "what's": "what is",
        "hows": "how is",
        "how's": "how is",
        "im": "i am",
        "i'm": "i am",
        "ive": "i have",
        "i've": "i have",
        "ill": "i will",
        "i'll": "i will",
        "idk": "i do not know",
        "ik": "i know",
        "imo": "in my opinion",
        "tbh": "to be honest",
        "fwiw": "for what it is worth",
        "dont": "do not",
        "don't": "do not",
        "didnt": "did not",
        "doesnt": "does not",
        "isnt": "is not",
        "arent": "are not",
        "wasnt": "was not",
        "werent": "were not",
        "cant": "cannot",
        "can't": "cannot",
        "wont": "will not",
        "won't": "will not",
        "shouldnt": "should not",
        "wouldnt": "would not",
        "couldnt": "could not",
        "ur": "your",
        "u": "you",
        "r": "are",
        "w": "with",
        "abt": "about",
        "ya'll": "you all",
        "thx": "thanks",
        "thanx": "thanks",
        "ty": "thank you",
        "tysm": "thank you so much",
        "plz": "please",
        "pls": "please",
        "bc": "because",
        "cuz": "because",
        "coz": "because",
        "bcuz": "because",
        "cus": "because",
        "cause": "because",
        "gonna": "going to",
        "wanna": "want to",
        "gotta": "got to",
        "lemme": "let me",
        "gimme": "give me",
        "finna": "going to",
        "tryna": "trying to",
        "ima": "i am going to",
        "gotchu": "got you",
        "wassup": "what is up",
        "sup": "what is up",
        "yo": "yo",
        "uh": "",
        "umm": "",
        "uhh": "",
        "hmm": "",
        "kinda": "kind of",
        "sorta": "sort of",
        "bout": "about",
        "outta": "out of",
        "lotta": "lot of",
        "dunno": "do not know",
        "idc": "i do not care",
        "nvm": "never mind",
        "rn": "right now",
        "asap": "as soon as possible",
        "tho": "though",
        "tho.": "though",
        "tho?": "though",
        "cya": "see you",
        "cmon": "come on",
        "gotcha": "got you",
        "ya": "you",
        "yall": "you all",
        "lumin": "lumen",
        "wuts": "what is",
        "wanna?": "want to",
        "nope": "no",
        "nahh": "no",
        "yep": "yes",
        "yeahh": "yes",
    }

    @classmethod
    def normalize(cls, text: str) -> str:
        normalized = " ".join(text.strip().lower().split())
        for source, target in cls.PHRASE_REPLACEMENTS.items():
            normalized = normalized.replace(source, target)
        normalized = normalized.replace("/", " ")
        normalized = re.sub(r"[ \t]*[,:;.!?]+[ \t]*", " ", normalized)
        tokens = normalized.split()
        expanded: list[str] = []
        for token in tokens:
            replacement = cls.TOKEN_REPLACEMENTS.get(token, token)
            if replacement:
                expanded.extend(replacement.split())
        return " ".join(part for part in expanded if part)
