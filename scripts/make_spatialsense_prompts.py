from datasets import load_dataset
import json
import os

def parse_caption(caption):
    words = caption.lower().strip().split()

    if len(words) < 3:
        return None

    subject = words[0]

    if words[1] in ["on", "under", "behind", "above", "below"]:
        relation = words[1]
        obj = " ".join(words[2:])

    elif words[1] == "in" and len(words) >= 5 and words[2] == "front" and words[3] == "of":
        relation = "in front of"
        obj = " ".join(words[4:])

    elif len(words) >= 6 and words[1] == "to" and words[2] == "the" and words[3] == "left" and words[4] == "of":
        relation = "to the left of"
        obj = " ".join(words[5:])

    elif len(words) >= 6 and words[1] == "to" and words[2] == "the" and words[3] == "right" and words[4] == "of":
        relation = "to the right of"
        obj = " ".join(words[5:])

    elif words[1] == "next" and len(words) >= 4 and words[2] == "to":
        relation = "next to"
        obj = " ".join(words[3:])

    else:
        return None

    if obj.strip() == "":
        return None

    return subject, relation, obj


ds = load_dataset("AsphyXIA/spatial-sense-flattened", split="train")

valid_relations = [
    "on",
    "