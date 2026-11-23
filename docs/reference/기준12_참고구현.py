import json,glob,re,csv,sys,collections
W=sys.argv[1]
cots={}
for f in glob.glob(W+'/raw/cot/**/*.json',recursive=True):
    d=json.load(open(f,encoding='utf-8')); cots[str(d['cot_id'])]=d
J={r['cot_id']:r for r in csv.DictReader(open('/mnt/user-data/uploads/finance-eda/outputs/judgments.csv',encoding='utf-8-sig'))}
NM=r'(공격투자형|적극투자형|위험중립형|안정추구형|안전추구형|안정형)'
norm=lambda s:'안정추구형' if s=='안전추구형' else s
IT={
'목적':r'목적|목표|자산\s*증식|노후|은퇴|주택|결혼|학자금|교육비|여유\s*자금|마련|생활\s*자금',
'재산':r'소득|연봉|자산|보유\s*(?:자금|금액)|투자금|[천백]?\s*만\s*원|억\s*원|부채|대출|저축',
'경험':r'거래해\s*(?:본|온|왔)|거래한|거래\s*(?:이력|경험|해\s*왔)|위주로만?\s*(?:거래|운용|투자|가입)|보유(?:한|하고\s*있는|\s*중인)\s*(?:상품|펀드|주식|채권)|경험|투자해\s*(?:본|온|왔)|운용해\s*(?:본|온|왔)|운용한|처음\s*(?:투자|접)|(?:상품|펀드|주식|채권|ETF|예금|적금)[^.?!]{0,20}(?:보유|운용|투자해)',
'이해도':r'이해|알고\s*있|잘\s*모르|처음\s*접하|구조를',
'위험태도':r'감수|감내|손실|원금\s*보장|안정적|변동성|위험\s*등급|위험을|위험은|위험도',
'연령':r'\d0\s*대|\d{2}\s*세',
}
def prop(d):
    c2=d['cot2']
    par={norm(x) for x in re.findall(NM+r"\s*[(（:]\s*(?:허용[^)）]*)?\d",c2)}
    if len(par)>1: return '복수','cot2 정형'
    m=re.findall(NM+r"\s*[,(（]\s*허용",c2)+re.findall(r"'"+NM+r"\s*,\s*허용",c2)
    s={norm(x) for x in m}
    if len(s)==1: return list(s)[0],'cot2 정형'
    s={norm(x) for x in re.findall(NM,c2)}
    if len(s)==1: return list(s)[0],'cot2'
    if len(s)>1: return '복수','cot2'
    s={norm(x) for x in re.findall(NM,d['cot1'])}
    if len(s)==1: return list(s)[0],'cot1'
    return ('복수' if s else '없음'),'cot1'
rows=[];C=collections.Counter()
for cid,j in J.items():
    d=cots[cid]; t=d['question']+' '+d['cot1']
    it={k:bool(re.search(p,t)) for k,p in IT.items()}
    if not it['연령'] and d.get('age'): it['연령']='메타'
    p,src=prop(d)
    stated={norm(x) for x in re.findall(NM,d['question'])}
    rec=bool(d.get('product_names'))
    legal_miss=[k for k in ('목적','재산','경험') if not it[k]]
    if p in ('없음','복수'): c2='②-b 성향 미확정 추천' if rec else '판정 불가(추천 없음)'
    elif len(stated)==1 and p in stated: c2='결함 아님(고객 진술 성향)'
    elif legal_miss: c2='②-a 근거 부족 확정'
    else: c2='결함 아님'
    n6=sum(1 for v in it.values() if v)
    rows.append(dict(cot_id=cid,분할=j['분할'],성향=p,성향출처=src,고객진술=('|'.join(sorted(stated))),**{k:('메타' if v=='메타' else ('확인' if v else '미확인')) for k,v in it.items()},확인수=n6,법정미확인='|'.join(legal_miss),기준2=c2,사람확인사유=j['사람 확인 사유'],이관분류=j['이관 분류']))
w=csv.DictWriter(open(W+'/c12_result.csv','w',encoding='utf-8-sig',newline=''),fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
tot=len(rows);print('총',tot)
for k in IT: print(k,'확인', sum(1 for r in rows if r[k]!='미확인'), round(100*sum(1 for r in rows if r[k]!='미확인')/tot,1))
print('확인수 분포',sorted(collections.Counter(r['확인수'] for r in rows).items()))
for sp in ('Training','Validation',None):
    sub=[r for r in rows if sp is None or r['분할']==sp]
    print(sp or '전체',collections.Counter(r['기준2'] for r in sub).most_common())
print('성향 출처',collections.Counter(r['성향출처'] for r in rows).most_common())
print('성향',collections.Counter(r['성향'] for r in rows).most_common())
h=[r for r in rows if '성향' in r['사람확인사유'] or r['사람확인사유'].startswith('성향')]
print('사람확인 성향불명 행 수',len(h),collections.Counter(r['기준2'] for r in h).most_common())
print('사람확인사유 값',collections.Counter(r['사람확인사유'] for r in rows if r['사람확인사유']).most_common())
print('법정 미확인 조합',collections.Counter(r['법정미확인'] for r in rows if r['기준2'].startswith('②-a')).most_common())
