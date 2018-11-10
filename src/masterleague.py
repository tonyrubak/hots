from urllib.request import urlopen, Request
import json

def masterleague_get(request_url, options=""):
    apibase = "https://api.masterleague.net/"
    request = Request(apibase + request_url + "?format=json" + options)
    with urlopen(request) as response:
        html = response.read()
    return html

def get_maps():
    response = json.loads(masterleague_get("maps"))
    return response

def get_heroes():
    heroes = []
    total = 0
    page = 1
    num_heroes = 1
    while total < num_heroes:
        response = json.loads(masterleague_get("heroes", options="&page=" + str(page)))
        num_heroes = response["count"]
        for hero in response["results"]:
            heroes.append({hero["name"]: hero})        
        total += len(response["results"])
        page += 1
    return heroes

def get_match(match_id):
    response = json.loads(masterleague_get("matches/" + str(match_id)))
    return response

def get_matches(max_matches):
    matches = []
    total = 0
    page = 1
    num_matches = 1
    while total < max_matches:
        response = json.loads(masterleague_get("matches", options="&page=" + str(page)))
        num_heroes = response["count"]
        for m in response["results"]:
            matches.append(m)        
        total += len(response["results"])
        page += 1
    return matches

def heroes_to_csv(heroes):
    tab = str.maketrans('', '', string.punctuation)
    df = pd.DataFrame(heroes)
    df["name"] = df["name"].str.lower().str.translate(tab).str.replace("the ","").str.replace("lost ", "").str.replace(" ", "")
    df = df.set_index("id")[["name"]]
    df.to_csv("heroes.csv")

def heroes_to_df(heroes):
    tab = str.maketrans('', '', string.punctuation)
    df = pd.DataFrame(heroes)
    df["name"] = df["name"].str.lower().str.translate(tab).str.replace("the ","").str.replace("lost ", "").str.replace(" ", "")
    df = df.set_index("id")[["name"]]
    return df
    
def heroes_to_dict(heroes):
    tab = str.maketrans('', '', string.punctuation)
    df = pd.DataFrame(heroes)
    df["name"] = df["name"].str.lower().str.translate(tab).str.replace("the ","").str.replace("lost ", "").str.replace(" ", "")
    df = df.set_index("id")[["name"]]
    d = dict()
    for (i,h) in df.iterrows():
        d[i] = h["name"]
    return d

def match_to_draft(match, maps, heroes):
    first = match["drafts"][0]
    second =  match["drafts"][1]
    draft = dict()
    # First Ban Phase
    draft["ban1"] = heroes[first["bans"][0]]
    draft["ban2"] = heroes[second["bans"][0]]
    draft["ban3"] = heroes[first["bans"][1]]
    draft["ban4"] = heroes[second["bans"][1]]
    draft["pick1"] = heroes[first["picks"][0]["hero"]]
    draft["pick2"] = heroes[second["picks"][0]["hero"]]
    draft["pick3"] = heroes[second["picks"][1]["hero"]]
    draft["pick4"] = heroes[first["picks"][1]["hero"]]
    draft["pick5"] = heroes[first["picks"][2]["hero"]]
    # Second Ban Phase
    draft["ban5"] = heroes[second["bans"][2]]
    draft["ban6"] = heroes[first["bans"][2]]
    draft["pick6"] = heroes[second["picks"][2]["hero"]]
    draft["pick7"] = heroes[second["picks"][3]["hero"]]
    draft["pick8"] = heroes[first["picks"][3]["hero"]]
    draft["pick9"] = heroes[first["picks"][4]["hero"]]
    draft["pick10"] = heroes[second["picks"][4]["hero"]]
    draft["map"] = maps[match["map"]]
    return draft


# Usage
heroes_dict = heroes_to_dict(heroes)
m = get_match(7772)
maps = pd.read_csv("maps")
maps_dict = dict()
for (idx, m) in maps.iterrows():
    maps_dict[m["id"]] = m["name"]
match_to_draft(m, maps_dict, heroes_dict)

matches = get_matches(100)

for match in matches:
    drafts = drafts.append(match_to_draft(match, maps_dict, heroes_dict), ignore_index=True)

