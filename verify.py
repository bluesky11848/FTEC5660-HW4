"""Offline checks; verifies labels only after building the retrieval pipeline."""
import json
from pathlib import Path
import hw4 as h

root=Path('public_test')
corpus=h.load_corpus(root/'corpus'); docs=h.load_json(root/'docs.json')
chunks=[]
for f,html in corpus.items():
    result=h.chunk_document(f,html)
    assert not h.document_problems(result,None,docs[f],f), f
    chunks.extend(result)
for t in h.load_json(root/'tables.json'):
    assert h.table_score(h.table_to_text(t['html']),t)==1
index=h.build_index(chunks)
gold=h.load_json(root/'ground_truth.json')
for qid,q in h.load_json(root/'questions.json').items():
    parsed=h.parse_query(q); result=h.retrieve(index,q)
    assert h.parse_score(parsed,gold[qid])==1,(qid,parsed)
    assert len(result)<=5 and all(any(c is old for old in chunks) for c in result)
    if gold[qid]['evidence']:
        assert h.retrieval_hit(result,gold[qid]),qid
    else:
        assert result==[],qid
# A renamed corpus tests that entity extraction/indexing do not depend on public names or filenames.
name='Aster Vale Instruments'
old=next(iter(docs.values()))['company']
f=next(iter(docs)); html=corpus[f].replace(old,name)
new=h.chunk_document('unseen.html',html)
assert all(h.company_key(c.metadata['company'])==h.company_key(name) for c in new)
assert h.parse_query("What was Aster Vale Instruments' revenue in FY2024?")['companies']==[name]
assert h.parse_query("What was Aster Vale Systems' revenue in FY2024?")['companies']==['Aster Vale Systems']
assert h.retrieve(h.build_index(new),"What was Aster Vale Systems' revenue in FY2024?")==[]
assert h.USAGE.llm_calls==0
print('PASS: tables, all document probes/metadata/limits, query fields, recall, abstention, chunk identity, and unseen entities.')
