
#!/usr/bin/env python3

from __future__ import annotations

import json, os, statistics

from collections import Counter, defaultdict

from pathlib import Path

ROOT=Path('.')

PARSED=Path(os.environ.get('S1A_PARSED_DIR', ROOT/'data/e1_parsed'))

S2E=Path(os.environ.get('S1A_S2E_DIR', ROOT/'data/s2e_confirmatory'))

OUT=Path(os.environ.get('S1A_OUT_DIR', ROOT/'data/s1a_confirmatory'))

RES=Path(os.environ.get('S1A_RESULTS_DIR', ROOT/'results/s1a'))

OUT.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)



def walk(nodes):

    rows=[]

    def rec(n,path,depth,parent):

        row={'path':path,'depth':depth,'level':int(n.get('level',0) or 0),'title':n.get('title',''),'parent_title':parent,'n_children':len(n.get('children') or []),'n_own_cites':len(set(n.get('own_cite_keys') or [])),'n_total_cites':len(set(n.get('total_cite_keys') or []))}

        rows.append(row)

        for i,c in enumerate(n.get('children') or []): rec(c,f'{path}.{i}',depth+1,n.get('title',''))

    for i,n in enumerate(nodes or []): rec(n,f'top.{i}',1,'__review_root__')

    return rows



def main():

    ps=PARSED/'_parse_summary.json'

    parse_summary=json.load(open(ps,encoding='utf-8')) if ps.exists() else {}

    records=[]; failures=Counter(); by_disc=defaultdict(Counter); node_rows=[]; total=0

    for p in sorted(PARSED.glob('*/*.json')):

        total+=1; r=json.load(open(p,encoding='utf-8')); disc=r.get('discipline') or p.parent.name

        if r.get('parse_status')!='ok':

            reason=r.get('fail_reason','unknown'); failures[reason]+=1; by_disc[disc][reason]+=1; continue

        tree=r.get('tree') or []

        if len(tree)<2:

            failures['fewer_than_two_top_level_sections']+=1; by_disc[disc]['fewer_than_two_top_level_sections']+=1; continue

        nodes=walk(tree); branch=[n for n in nodes if n['n_children']>=2]

        records.append({'arxiv_id':r['arxiv_id'],'discipline':disc,'n_sections':r.get('n_sections',len(nodes)),'n_unique_cite_keys':r.get('n_unique_cite_keys',0),'n_top_level_sections':len(tree),'n_all_nodes':len(nodes),'n_branching_nodes_without_synthetic_root':len(branch),'max_depth':max([n['depth'] for n in nodes],default=0)})

        by_disc[disc]['ok']+=1

        for n in branch: node_rows.append({'arxiv_id':r['arxiv_id'],'discipline':disc,**n})

    s2e=[]

    for split in ['train','heldout']:

        fp=S2E/f'nodes_{split}.jsonl'

        if fp.exists():

            for line in open(fp,encoding='utf-8'):

                if line.strip(): s2e.append(json.loads(line))

    cov=[n.get('citation_child_coverage') for n in s2e if isinstance(n.get('citation_child_coverage'),(int,float))]

    summary={'input_parse_summary':parse_summary,'total_json_records_seen':total,'recoverable_reviews_layer1':len(records),'failure_counts_recomputed':dict(failures),'by_discipline':{k:dict(v) for k,v in sorted(by_disc.items())},'deterministic_tree_stats':{'n_reviews':len(records),'n_branching_nodes_without_synthetic_root':len(node_rows),'mean_sections':statistics.mean([r['n_sections'] for r in records]) if records else 0,'median_sections':statistics.median([r['n_sections'] for r in records]) if records else 0,'mean_max_depth':statistics.mean([r['max_depth'] for r in records]) if records else 0,'median_max_depth':statistics.median([r['max_depth'] for r in records]) if records else 0},'s2e_root_repaired_corpus':{'n_reviews':len({n['arxiv_id'] for n in s2e}) if s2e else 0,'n_organization_nodes_with_synthetic_root':len(s2e),'n_synthetic_review_root_nodes':sum(1 for n in s2e if n.get('is_review_root')),'citation_child_coverage_mean':statistics.mean(cov) if cov else None,'citation_child_coverage_median':statistics.median(cov) if cov else None,'n_nodes_with_citation_coverage':len(cov)},'layer1_interpretation':'Deterministic LaTeX-to-ordered-tree recovery with citation assignment; no semantic labels used.'}

    (OUT/'layer1_review_manifest.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in records)+'\n',encoding='utf-8')

    (OUT/'layer1_branching_nodes_no_root.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in node_rows)+'\n',encoding='utf-8')

    (RES/'s1a_layer1_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

    print(json.dumps(summary,ensure_ascii=False,indent=2)[:5000])

if __name__=='__main__': main()

