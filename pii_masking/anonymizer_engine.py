def mask_entities(text, entities):
    for e in sorted(entities, key=lambda x: -x["position"][0]):
        start, end = e["position"]
        text = text[:start] + f"[{e['label'].upper()}]" + text[end:]
    return text
