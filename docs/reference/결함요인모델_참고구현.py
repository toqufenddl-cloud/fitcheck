import json,glob,re,csv,sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score
S=sys.argv[1]
C={}
for f in glob.glob(S+'/raw/cot/**/*.json',recursive=True):
    d=json.load(open(f,encoding='utf-8')); C[str(d['cot_id'])]=d
P={}
for f in glob.glob(S+'/raw/src/**/*.json',recursive=True):
    d=json.load(open(f,encoding='utf-8')); P[d['product_name'].strip()]=d
J={r['cot_id']:r for r in csv.DictReader(open('/mnt/user-data/uploads/finance-eda/outputs/judgments.csv',encoding='utf-8-sig'))}
K={r['cot_id']:r for r in csv.DictReader(open(S+'/c12_result.csv',encoding='utf-8-sig'))}
AGE={'20대':2,'30대':3,'40대':4,'50대':5,'50대 이상':5,'60대':6,'60대 이상':6,'70대':7,'80대 이상':8}
TYPES=['주식형','채권형','재간접','단기금융','파생형','퇴직연금']
rows=[]
for cid,j in J.items():
    d=C[cid]; q=d['question']; pn=d['product_names'] if isinstance(d['product_names'],list) else []
    types={P[n.strip()]['product_type'] for n in pn if n.strip() in P}
    x=dict(cot_id=cid,split=j['분할'],
      유형_조건기반=int(d['query_type']=='투자성향 및 조건기반형'),유형_상품비교=int(d['query_type']=='상품 비교 추천형'),
      연령=AGE.get(d['age'],np.nan),여성=int(d['gender']=='여'),추천수=len(pn),
      질문_숫자등급=int(bool(re.search(r'[1-6]\s*등급',q))),
      질문_높은위험명칭=int(bool(re.search(r'(매우\s*높은|다소\s*높은|높은)\s*위험',q))),
      질문_고려표현=int(bool(re.search(r'감수|수용|감내|고민|고려|거부감',q))),
      질문_회피표현=int(bool(re.search(r'부담|원하지\s*않|원치\s*않|피하|꺼리|지양|너무\s*높은|지나치게',q))),
      질문_경험언급=int(K[cid]['경험']=='확인'),질문_이해도언급=int(K[cid]['이해도']=='확인'),
      고객진술성향=int(bool(K[cid]['고객진술'])),질문길이=len(q)/100,답변길이=len(d['answer'])/100)
    for t in TYPES: x['상품_'+t]=int(t in types)
    x['y_범위밖제외']=int(j['기준③ 처리']=='제외'); x['y_등급설명없음']=int(j['기준④ 판정']=='기재 없음')
    rows.append(x)
df=pd.DataFrame(rows).dropna()
F=[c for c in df.columns if c not in('cot_id','split') and not c.startswith('y_')]
tr,te=df[df.split=='Training'],df[df.split=='Validation']
out={}
for y in ('y_범위밖제외','y_등급설명없음'):
    res={'base_train':round(tr[y].mean(),3),'base_test':round(te[y].mean(),3)}
    lr=LogisticRegression(max_iter=2000,class_weight='balanced').fit((tr[F]-tr[F].mean())/tr[F].std(),tr[y])
    p=lr.predict_proba((te[F]-tr[F].mean())/tr[F].std())[:,1]
    res['LR_AUC']=round(roc_auc_score(te[y],p),3);res['LR_AP']=round(average_precision_score(te[y],p),3)
    pr=(p>=0.5).astype(int);res['LR_recall']=round(recall_score(te[y],pr),3);res['LR_precision']=round(precision_score(te[y],pr),3)
    coef=sorted(zip(F,lr.coef_[0]),key=lambda t:-abs(t[1]))[:8]
    dt=DecisionTreeClassifier(max_depth=3,class_weight='balanced',min_samples_leaf=50,random_state=0).fit(tr[F],tr[y])
    q=dt.predict_proba(te[F])[:,1];res['DT_AUC']=round(roc_auc_score(te[y],q),3);res['DT_AP']=round(average_precision_score(te[y],q),3)
    print('=====',y,res); print('LR 상위 계수(표준화):',[(f,round(c,2)) for f,c in coef])
    print(export_text(dt,feature_names=F,show_weights=False))
    imp=sorted(zip(F,dt.feature_importances_),key=lambda t:-t[1])[:6];print('DT 중요도',[(f,round(v,3)) for f,v in imp])
    # simple lift table on strongest single feature
print(len(df),'rows', len(tr),len(te))
for f in ('질문_숫자등급','질문_높은위험명칭','질문_고려표현','고객진술성향','상품_퇴직연금','유형_상품비교'):
    g=df.groupby(f)['y_범위밖제외'].agg(['mean','count']);print(f,[(i,round(r['mean']*100,1),int(r['count'])) for i,r in g.iterrows()])
from sklearn.model_selection import cross_val_score, StratifiedKFold
print('---CV')
for y in ('y_범위밖제외','y_등급설명없음'):
    Xs=(tr[F]-tr[F].mean())/tr[F].std()
    s=cross_val_score(LogisticRegression(max_iter=2000,class_weight='balanced'),Xs,tr[y],cv=StratifiedKFold(5,shuffle=True,random_state=0),scoring='roc_auc')
    print(y,'CV AUC',round(s.mean(),3),'±',round(s.std(),3))
print('---요인별 비율')
df['재간접또는퇴직연금']=((df['상품_재간접']+df['상품_퇴직연금'])>0).astype(int)
df['질문유형']=np.where(df['유형_조건기반']==1,'조건기반',np.where(df['유형_상품비교']==1,'상품비교','고객특성'))
for y in ('y_범위밖제외','y_등급설명없음'):
    print('##',y,'전체',round(df[y].mean()*100,1))
    for f in ('질문유형','재간접또는퇴직연금','상품_단기금융','질문_높은위험명칭','질문_숫자등급'):
        g=df.groupby(f)[y].agg(['mean','count']);print(' ',f,[(i,round(r['mean']*100,1),int(r['count'])) for i,r in g.iterrows()])
    g=df.groupby(['질문유형','재간접또는퇴직연금'])[y].agg(['mean','count']);print('  조합',[(i,round(r['mean']*100,1),int(r['count'])) for i,r in g.iterrows()])
