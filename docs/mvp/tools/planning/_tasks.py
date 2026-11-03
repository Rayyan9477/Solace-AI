import re,sys,glob
for f in sys.argv[1:]:
    txt=open(f,encoding='utf-8').read()
    parts=re.split(r'(?m)^### Task ',txt)
    print('==',f)
    for p in parts[1:]:
        head=p.split('\n',1)[0]
        def g(k):
            m=re.search(r'\*\*'+k+r':\*\*\s*(.*)',p)
            return m.group(1).strip()[:150] if m else '-'
        ph=g('Phase')
        print(head[:110],'| W',g('Wave'),'| Ph',ph,'| Cov',g('Covers')[:60],'| Dep',g('Depends on')[:120])
