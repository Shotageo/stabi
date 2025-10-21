from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import ezdxf
except Exception:
    ezdxf = None

_ST_RE = re.compile(r"(?:STA|KP|No\.?|NO\.?|No|NO)?\s*([0-9]+)\s*[+−-]\s*([0-9]+)", re.IGNORECASE)
def normalize_no_key(text: str) -> Optional[str]:
    t = str(text).replace("−", "-"); m = _ST_RE.search(t)
    if not m: return None
    a,b = int(m.group(1)), int(m.group(2))
    return f"No.{a}+{b:02d}"

@dataclass
class LayerInfo:
    name: str
    entity_counts: Dict[str,int]
    length_sum: float

def _safe_len(e): 
    try: return float(e.length())
    except Exception: return 0.0

def list_layers(path: str) -> List[LayerInfo]:
    if ezdxf is None: raise RuntimeError("ezdxf not installed.")
    doc=ezdxf.readfile(path); msp=doc.modelspace()
    d:Dict[str,LayerInfo]={}
    for e in msp:
        nm=e.dxf.layer; info=d.get(nm) or LayerInfo(nm,{},0.0)
        t=e.dxftype(); info.entity_counts[t]=info.entity_counts.get(t,0)+1
        info.length_sum+=_safe_len(e); d[nm]=info
    return sorted(d.values(), key=lambda x:(-x.length_sum,x.name.lower()))

def _sample_entity_2d(ent):
    import numpy as _np
    if ent.dxftype()=="LINE":
        return _np.array([[ent.dxf.start.x, ent.dxf.start.y],[ent.dxf.end.x,ent.dxf.end.y]],float)
    try:
        return _np.array([[p[0],p[1]] for p in ent.flattening(0.25)],float)
    except Exception:
        try: return _np.array([[v[0],v[1]] for v in ent.get_points()],float)
        except Exception: return _np.empty((0,2),float)

def read_centerline(path: str, allow: List[str], unit_scale=1.0) -> np.ndarray:
    if ezdxf is None: raise RuntimeError("ezdxf not installed.")
    doc=ezdxf.readfile(path); msp=doc.modelspace(); allow=set(allow or [])
    cands=[]
    for e in msp.query("*"):
        if allow and e.dxf.layer not in allow: continue
        if e.dxftype() in ("LWPOLYLINE","POLYLINE","LINE","SPLINE"): cands.append((_safe_len(e),e))
    if not cands: raise ValueError("no centerline candidate")
    ent=max(cands,key=lambda t:t[0])[1]
    xy=_sample_entity_2d(ent)*unit_scale
    d=np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    if d[-1]<=0: raise ValueError("centerline length zero")
    return xy

def project_point_to_polyline(poly: np.ndarray, pt: np.ndarray) -> Tuple[float,float]:
    if poly.shape[0]<2: return 0.0,float("inf")
    segs=np.diff(poly,axis=0); lens=np.linalg.norm(segs,axis=1); cum=np.r_[0.0,np.cumsum(lens)]
    best=(0.0,float("inf")); P0=poly[:-1]
    for i,(p,v,L) in enumerate(zip(P0,segs,lens)):
        if L==0: continue
        t=float(np.dot(pt-p,v)/(L*L)); t=max(0.0,min(1.0,t)); proj=p+t*v
        dist=float(np.linalg.norm(pt-proj)); s=float(cum[i]+t*L)
        if dist<best[1]: best=(s,dist)
    return best

def extract_no_labels(path: str, label_layers: List[str], unit_scale=1.0)->List[Dict]:
    if ezdxf is None: raise RuntimeError("ezdxf not installed.")
    doc=ezdxf.readfile(path); msp=doc.modelspace(); allow=set(label_layers or []); out=[]
    for e in msp.query("TEXT MTEXT"):
        if allow and e.dxf.layer not in allow: continue
        try:
            raw=e.dxf.text if e.dxftype()=="TEXT" else e.plain_text()
            key=normalize_no_key(raw); if not key: continue
            ins=e.dxf.insert; pos=(float(ins.x)*unit_scale,float(ins.y)*unit_scale)
            out.append({"key":key,"pos":pos,"raw":raw,"layer":e.dxf.layer})
        except Exception: pass
    return out

def extract_circles(path: str, circle_layers: List[str], unit_scale=1.0)->List[Dict]:
    if ezdxf is None: raise RuntimeError("ezdxf not installed.")
    doc=ezdxf.readfile(path); msp=doc.modelspace(); allow=set(circle_layers or []); out=[]
    for c in msp.query("CIRCLE"):
        if allow and c.dxf.layer not in allow: continue
        try:
            cx,cy=float(c.dxf.center.x)*unit_scale,float(c.dxf.center.y)*unit_scale
            r=float(c.dxf.radius)*unit_scale; out.append({"center":(cx,cy),"r":r,"layer":c.dxf.layer})
        except Exception: pass
    try:
        for ref in msp.query("INSERT"):
            for v in ref.virtual_entities():
                if v.dxftype()=="CIRCLE":
                    if allow and v.dxf.layer not in allow: continue
                    cx,cy=float(v.dxf.center.x)*unit_scale,float(v.dxf.center.y)*unit_scale
                    r=float(v.dxf.radius)*unit_scale; out.append({"center":(cx,cy),"r":r,"layer":v.dxf.layer})
    except Exception: pass
    return out

def _is_closed_poly(pts,tol=1e-6): 
    return pts.shape[0]>=3 and float(np.linalg.norm(pts[0]-pts[-1]))<tol
def _bbox(pts): xmin,ymin=pts.min(0); xmax,ymax=pts.max(0); return xmin,xmax,ymin,ymax
def _poly_length(pts): 
    return 0.0 if pts.shape[0]<2 else float(np.sum(np.linalg.norm(np.diff(pts,axis=0),axis=1)))
def _rect_like_penalty(pts):
    if pts.shape[0]<4: return 0.0
    v=np.diff(pts,axis=0); v=v[np.linalg.norm(v,axis=1)>1e-9]
    if len(v)<2: return 0.0
    u=v[:-1]/np.linalg.norm(v[:-1],axis=1)[:,None]; w=v[1:]/np.linalg.norm(v[1:],axis=1)[:,None]
    cos=np.clip((u*w).sum(1),-1.0,1.0); ang=np.degrees(np.arccos(cos))
    return float(np.mean((np.abs(ang-90.0)<10.0).astype(float)))
def _groundlike_score(pts):
    if pts.shape[0]<2: return 0.0
    L=_poly_length(pts); xmin,xmax,ymin,ymax=_bbox(pts); w=max(1e-9,xmax-xmin); h=max(1e-9,ymax-ymin)
    aspect=h/w; n=pts.shape[0]; closed=_is_closed_poly(pts); rect=_rect_like_penalty(pts)
    s_len=1.0-np.exp(-L/20.0); s_asp=np.clip(1.0-aspect*2.0,0.0,1.0); s_nv=np.clip((n-4)/20.0,0.0,1.0)
    s_open=0.0 if closed else 1.0; s_rect=1.0-np.clip(rect,0.0,1.0)
    return float(np.clip(0.35*s_asp+0.25*s_len+0.15*s_nv+0.15*s_open+0.10*s_rect,0.0,1.0))

def read_single_section_file(path, layer_name=None, unit_scale=1.0,
                             aggregate="median", smooth_k=0, max_slope=0.0, target_step=0.20):
    if path.lower().endswith(".csv"):
        import numpy as np
        try:
            data=np.loadtxt(path,delimiter=",",dtype=float)
            P=np.asarray(data[:,:2],float)*unit_scale
        except Exception: return None
        if P.ndim!=2 or P.shape[1]<2 or len(P)<2: return None
        order=np.argsort(P[:,0]); P=P[order]
        u=P[:,0]; v=P[:,1]
        uu=np.linspace(u.min(),u.max(),max(10,int((u.max()-u.min())/max(target_step,1e-3))))
        vv=np.interp(uu,u,v); return np.column_stack([uu,vv])
    if ezdxf is None: raise RuntimeError("ezdxf not installed.")
    try: doc=ezdxf.readfile(path)
    except Exception: return None
    msp=doc.modelspace()

    scored=[]
    for e in msp.query("LWPOLYLINE"):
        if (layer_name is None) or (e.dxf.layer==layer_name):
            pts=_sample_entity_2d(e)
            if pts.size: scored.append((_groundlike_score(pts.astype(float)*unit_scale),
                                        pts.astype(float)*unit_scale))
    scored.sort(key=lambda t:t[0], reverse=True)
    if scored and scored[0][0]>=0.45:
        pts=scored[0][1]; d=np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
        if d[-1]<=0: return None
        uu=np.arange(0,d[-1]+target_step*0.5, max(target_step,1e-3))
        x=np.interp(uu,d,pts[:,0]); y=np.interp(uu,d,pts[:,1])
        return np.column_stack([x,y])
    pts_list=[]
    for e in msp.query("LWPOLYLINE"):
        if (layer_name is None) or (e.dxf.layer==layer_name):
            Q=_sample_entity_2d(e)
            if Q.size: pts_list.append(Q)
    for ln in msp.query("LINE"):
        if (layer_name is None) or (ln.dxf.layer==layer_name):
            pts_list.append(np.array([[ln.dxf.start.x,ln.dxf.start.y],[ln.dxf.end.x,ln.dxf.end.y]],float))
    if not pts_list: return None
    P=(np.vstack(pts_list)).astype(float)*unit_scale
    mu=P.mean(0); X=P-mu; cov=np.cov(X.T); w,V=np.linalg.eigh(cov); V=V[:,::-1]; U=X@V
    order=np.argsort(U[:,0]); uo=U[order,0]; vo=U[order,1]
    du=np.diff(uo); med=np.median(np.abs(du)) or 1.0; cuts=np.where(du>6.0*med)[0]
    idx=np.r_[0,cuts+1,len(uo)]; segs=[slice(idx[i],idx[i+1]) for i in range(len(idx)-1)]
    if not segs: return None
    best=max(segs, key=lambda s:s.stop-s.start); u=uo[best]; v=vo[best]
    umin,umax=float(u.min()),float(u.max()); nbin=max(60, int((umax-umin)/max(target_step,1e-3)))
    bins=np.linspace(umin,umax,nbin+1); ids=np.digitize(u,bins)-1
    xs,ys=[],[]
    for b in range(nbin):
        mask=(ids==b)
        if np.count_nonzero(mask)>=3:
            xs.append(0.5*(bins[b]+bins[b+1]))
            vb=v[mask]
            if aggregate=="lower": ys.append(float(np.percentile(vb,10)))
            elif aggregate=="upper": ys.append(float(np.percentile(vb,90)))
            else: ys.append(float(np.median(vb)))
    if not xs: return None
    P=np.column_stack([np.asarray(xs),np.asarray(ys)]); order=np.argsort(P[:,0]); P=P[order]
    return P
