
#!/usr/bin/env python3

from __future__ import annotations

import json, random

from pathlib import Path

from collections import defaultdict

ROOT=Path('.'); DATA=ROOT/'data/s1a_confirmatory'; RES=ROOT/'results/s1a'

random.seed(20260710)

def load(p): return [json.loads(l) for l in open(p,encoding='utf-8') if l.strip()]

items={r['item_id']:r for r in load(DATA/'s1a_items.jsonl')}

pairs=load(DATA/'s1a_pairs.jsonl')

# Stratified helper by discipline.

def stratified(rows,n,key=lambda x:x.get('discipline')):

    buckets=defaultdict(list)

    for r in rows: buckets[key(r)].append(r)

    out=[]; total=len(rows)

    for k,b in buckets.items():

        m=max(1,round(n*len(b)/total)); random.shuffle(b); out.extend(b[:m])

    random.shuffle(out)

    return out[:n]

# Faithfulness: authentic nodes only.

auth=[items[p['authentic_item_id']] for p in pairs]

faith=stratified(auth,150)

# Pair choice: tier2 primary, blinded A/B order.

pair_sample=stratified(pairs,100)

forced=[]

for p in pair_sample:

    a=items[p['authentic_item_id']]; s=items[p['tier2_item_id']]

    rows=[('A',a,'authentic'),('B',s,'synthetic')]

    if random.random()<0.5: rows=list(reversed(rows))

    forced.append({'pair_id':p['pair_id'],'discipline':p.get('discipline'),'node_depth':p.get('node_depth'),'A_item_id':rows[0][1]['item_id'],'A_parent_title':rows[0][1].get('parent_title'),'A_child_titles':rows[0][1].get('child_titles'),'B_item_id':rows[1][1]['item_id'],'B_parent_title':rows[1][1].get('parent_title'),'B_child_titles':rows[1][1].get('child_titles'),'answer_key':rows[0][0] if rows[0][2]=='authentic' else rows[1][0]})

(DATA/'s1a_human_faithfulness_150.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in faith)+'\n',encoding='utf-8')

(DATA/'s1a_human_forced_choice_100.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in forced)+'\n',encoding='utf-8')

print({'faithfulness':len(faith),'forced_choice':len(forced),'paths':['data/s1a_confirmatory/s1a_human_faithfulness_150.jsonl','data/s1a_confirmatory/s1a_human_forced_choice_100.jsonl']})

