"""Cumulative, one-component ablation. Gold labels are used only AFTER retrieval.
Run: python ablation.py [--offline] [--data-folder public_test]
"""
import argparse
import csv
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hw4 as h
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

VARIANTS = [
 ('B0', 'Naive get_text + recursive split + whitespace BM25'),
 ('V1', 'Replace HTML tables with labelled table text'),
 ('V2', 'Use section-aware chunks with section/caption headers'),
 ('V3', 'Use regex tokens, stop words and lexical aliases'),
 ('V4', 'Parse queries and filter company/year/document metadata'),
 ('V5', 'Remove company/year terms from the ranking query'),
 ('V6', 'Allocate comparison evidence per company'),
 ('V7', 'Reject unknown companies, unsupported years/metrics'),
]

def make_chunks(corpus, level):
    if level >= 2:
        return [c for f,s in corpus.items() for c in h.chunk_document(f,s)]
    chunks=[]
    splitter=h.RecursiveCharacterTextSplitter(chunk_size=h.MAX_CHUNK_CHARS,chunk_overlap=100)
    for f,html in corpus.items():
        soup=BeautifulSoup(html,'html.parser'); meta=h._metadata(f,soup)
        if level == 1:
            for table in soup.find_all('table'):
                table.replace_with(h.table_to_text(str(table)))
        text=soup.get_text()
        chunks.extend(h.Document(page_content=t,metadata=dict(meta)) for t in splitter.split_text(text))
    return chunks

def run(folder, out, offline=False):
    corpus=h.load_corpus(folder/'corpus'); questions=h.load_json(folder/'questions.json')
    gold=h.load_json(folder/'ground_truth.json'); out.mkdir(exist_ok=True)
    summaries=[]
    for level,(label,change) in enumerate(VARIANTS):
        start=time.perf_counter(); chunks=make_chunks(corpus,level)
        ix=h.build_index(chunks)
        if level<3:
            ix['bm25']=BM25Okapi([c.page_content.lower().split() for c in chunks])
        retrieved={}
        for qid,q in questions.items():
            if level<3:
                scores=ix['bm25'].get_scores(q.lower().split())
                retrieved[qid]=[chunks[i] for i in sorted(range(len(chunks)),key=lambda i:-scores[i])[:5]]
            else:
                retrieved[qid]=h._retrieve(ix,q,filtering=level>=4,metric_query=level>=5,diversify=level>=6,abstain=level>=7)
        retrieval_s=time.perf_counter()-start
        if offline:
            answers={qid:'' for qid in questions}
        else:
            # Fail loudly on API errors: no fabricated answers or silently inflated scores.
            with ThreadPoolExecutor(max_workers=8) as pool:
                answers=dict(zip(questions,pool.map(lambda k:h.generate(questions[k],retrieved[k]),questions)))
        rows=[]
        for qid,q in questions.items():
            g=gold[qid]; r=retrieved[qid]; parsed=h.parse_query(q) if level>=4 else None
            rows.append(dict(id=qid,category=g['category'],question=q,parsed=json.dumps(parsed),
                retrieved=';'.join(sorted({c.metadata['source'] for c in r})),
                retrieval_hit=h.retrieval_hit(r,g) if g['evidence'] else '',
                answer=answers[qid],correct=h.answer_correct(answers[qid],g) if not offline else '',
                expected=g['answer'],nonempty=bool(r)))
        with (out/f'{label}_results.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
        yes=[r for r in rows if gold[r['id']]['evidence']]; no=[r for r in rows if not gold[r['id']]['evidence']]
        summary=dict(variant=label,change=change,chunks=len(chunks),
            recall=sum(r['retrieval_hit'] for r in yes)/len(yes),
            kept=sum(r['nonempty'] for r in yes)/len(yes),
            rejected=sum(not r['nonempty'] for r in no)/len(no),
            accuracy=sum(r['correct'] for r in rows)/len(rows) if not offline else None,
            retrieval_seconds=round(retrieval_s,3),total_seconds=round(time.perf_counter()-start,3))
        summaries.append(summary)
        (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
        print(json.dumps(summary),flush=True)
    (out/'usage.json').write_text(json.dumps(h.GENERATOR_USAGE.summary(),indent=2)+'\n')
    lines=['| Variant | Change | Recall@5 | Kept | Rejected | Accuracy | Retrieval s |',
           '|---|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        acc='Not run' if s['accuracy'] is None else f"{s['accuracy']:.1%}"
        lines.append(f"| {s['variant']} | {s['change']} | {s['recall']:.1%} | {s['kept']:.1%} | {s['rejected']:.1%} | {acc} | {s['retrieval_seconds']} |")
    (out/'table.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data-folder',default='public_test'); p.add_argument('--offline',action='store_true'); p.add_argument('--output',default='ablation_results')
    a=p.parse_args(); run(Path(a.data_folder),Path(a.output),a.offline)
