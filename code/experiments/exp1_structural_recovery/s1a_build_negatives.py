
#!/usr/bin/env python3

from __future__ import annotations

import json, os, random, hashlib

from collections import Counter

from pathlib import Path

ROOT=Path('.')

S2E=Path(os.environ.get('S1A_S2E_DIR', ROOT/'data/s2e_confirmatory'))

OUT=Path(os.environ.get('S1A_OUT_DIR', ROOT/'data/s1a_confirmatory'))

OUT.mkdir(parents=True, exist_ok=True)



def stable_int(*parts): return int(hashlib.sha256('||'.join(map(str,parts)).encode()).hexdigest()[:16],16)

def node_id(n): return f"{n['arxiv_id']}::{n['node_path']}"

def text_of(n): return ' '.join([n.get('node_title') or '', *(n.get('child_titles') or [])]).lower()

def load_nodes():

    rows=[]

    for split in ['train','heldout']:

        p=S2E/f'nodes_{split}.jsonl'

        with open(p,encoding='utf-8') as f:

            for line in f:

                if line.strip():

                    n=json.loads(line); n['node_id']=node_id(n); rows.append(n)

    return rows



def cite_proxy(n): return min(int(n.get('n_children') or 0), int(n.get('n_child_union_cites') or 0))

def choose_tier1(t,pool):

    cand=[n for n in pool if n['arxiv_id']!=t['arxiv_id'] and n.get('discipline')==t.get('discipline') and n.get('n_children')==t.get('n_children')]

    if not cand: cand=[n for n in pool if n['arxiv_id']!=t['arxiv_id'] and n.get('discipline')==t.get('discipline')]

    if not cand: cand=[n for n in pool if n['arxiv_id']!=t['arxiv_id']]

    return cand[stable_int('tier1',t['node_id'])%len(cand)] if cand else None



def choose_tier2(t,pool):

    tset=set(text_of(t).split()); scored=[]

    for n in pool:

        if n['arxiv_id']==t['arxiv_id'] or n.get('discipline')!=t.get('discipline'): continue

        score=0

        score += 4 if n.get('n_children')==t.get('n_children') else -abs((n.get('n_children') or 0)-(t.get('n_children') or 0))

        score += 2 if n.get('node_depth')==t.get('node_depth') else -0.5*abs((n.get('node_depth') or 0)-(t.get('node_depth') or 0))

        score += 1 if n.get('review_length_bucket')==t.get('review_length_bucket') else 0

        score += 1 if cite_proxy(n)==cite_proxy(t) else 0

        nset=set(text_of(n).split()); score += len(tset & nset)/max(1,len(tset | nset))

        scored.append((score,n))

    if not scored: return choose_tier1(t,pool)

    scored.sort(key=lambda x:(-x[0],x[1]['node_id']))

    top=scored[:25]

    return top[stable_int('tier2',t['node_id'])%len(top)][1]



def make_item(pid,tier,condition,t,donor=None):

    child_titles=t.get('child_titles') if condition=='authentic' else donor.get('child_titles')

    return {'item_id':f'{pid}::{condition if tier==0 else tier}','pair_id':pid,'tier':tier,'condition':condition,'target_node_id':t['node_id'],'donor_node_id':donor.get('node_id') if donor else None,'arxiv_id':t['arxiv_id'],'discipline':t.get('discipline'),'node_depth':t.get('node_depth'),'review_length_bucket':t.get('review_length_bucket'),'parent_title':t.get('parent_title') or t.get('node_title'),'child_titles':child_titles,'n_children':len(child_titles or []),'target_n_children':t.get('n_children')}



def main():

    nodes=load_nodes()
    use_all_reviews = os.environ.get("S1A_USE_ALL_REVIEWS", "").lower() in {"1", "true", "yes"}
    eval_nodes = [
        n for n in nodes
        if int(n.get('n_children') or 0) >= 2 and (use_all_reviews or n.get('split') == 'heldout')
    ]
    pool=[n for n in nodes if int(n.get('n_children') or 0)>=2]

    items=[]; pairs=[]; diag=Counter()

    for t in eval_nodes:

        pid=t['node_id']; d1=choose_tier1(t,pool); d2=choose_tier2(t,pool)

        if not d1 or not d2: continue

        auth=make_item(pid,0,'authentic',t); tier1=make_item(pid,'tier1','synthetic',t,d1); tier2=make_item(pid,'tier2','synthetic',t,d2)

        rows=[auth,tier1,tier2]; random.Random(stable_int('shuffle',pid)).shuffle(rows); items.extend(rows)

        pairs.append({'pair_id':pid,'authentic_item_id':auth['item_id'],'tier1_item_id':tier1['item_id'],'tier2_item_id':tier2['item_id'],'target_node_id':t['node_id'],'tier1_donor':d1['node_id'],'tier2_donor':d2['node_id'],'discipline':t.get('discipline'),'node_depth':t.get('node_depth'),'n_children':t.get('n_children')})

        diag['pairs']+=1; diag[f"discipline::{t.get('discipline')}"]+=1

    (OUT/'s1a_items.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in items)+'\n',encoding='utf-8')

    (OUT/'s1a_pairs.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in pairs)+'\n',encoding='utf-8')

    selection = (
        'all parsed social-science roots; no model fitting is performed in this extension'
        if use_all_reviews else 'heldout S2e nodes only'
    )
    summary={'n_eval_nodes':len(eval_nodes),'n_pairs':len(pairs),'n_items':len(items),'diagnostics':dict(diag),'primary':f'tier2 hard matched synthetic; {selection}; old LLM labels not read'}

    (OUT/'s1a_negative_construction_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

    print(json.dumps(summary,ensure_ascii=False,indent=2)[:4000])

if __name__=='__main__': main()

