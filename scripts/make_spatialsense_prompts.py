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
    "under",
    "behind",
    "in front of",
    "to the left of",
    "to the right of",
    "above",
    "below",
    "next to",
]

output = []
idx = 0

for row in ds:
    parsed = parse_caption(row["caption"])

    if parsed is None:
        continue

    subject, relation, obj = parsed

    if relation not in valid_relations:
        continue

    question = (
        f"<image>\nUSER: Where is the {subject} in relation to the {obj}? "
        "Answer with one of: on, under, behind, in front of, to the left of, "
        "to the right of, above, below, next to.\n"
        "ASSISTANT:"
    )

    output.append({
        "id": idx,
        "question": question,
        "answer": [relation]
    })

    idx += 1

    #if idx == 100:
        #break

os.makedirs("prompts", exist_ok=True)

with open("prompts/SpatialSense_with_answer_nine_options.jsonl", "w") as f:
    for item in output:
        f.write(json.dumps(item) + "\n")

print(f"Saved {len(output)} examples")