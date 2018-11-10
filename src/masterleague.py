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
    draft = []
    # First Ban Phase
    draft.append(first["bans"][0])
    draft.append(second["bans"][0])
    draft.append(first["bans"][1])
    draft.append(second["bans"][1])
    draft.append(first["picks"][0]["hero"])
    draft.append(second["picks"][0]["hero"])
    draft.append(second["picks"][1]["hero"])
    draft.append(first["picks"][1]["hero"])
    draft.append(first["picks"][2]["hero"])
    # Second Ban Phase
    draft.append(second["bans"][2])
    draft.append(first["bans"][2])
    draft.append(second["picks"][2]["hero"])
    draft.append(second["picks"][3]["hero"])
    draft.append(first["picks"][3]["hero"])
    draft.append(first["picks"][4]["hero"])
    draft.append(second["picks"][4]["hero"])
    m = maps[match["map"]]
    heroes = [hero for hero in map(lambda x: heroes[x], draft)]
    return [m, *heroes]

# Usage
heroes_dict = heroes_to_dict(heroes)
m = get_match(7772)
maps = pd.read_csv("maps")
maps_dict = dict()
for (idx, m) in maps.iterrows():
    maps_dict[m["id"]] = m["name"]
match_to_draft(m, maps_dict, heroes_dict)

mlist = [7772, 
