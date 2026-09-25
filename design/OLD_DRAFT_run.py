"""Exactly 80 paired gravity clips + 20 color controls; no fitted model or variants."""
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
SEED = 20260924

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def setup():
    rng = np.random.default_rng(SEED)
    split = rng.permutation(40).tolist()
    controls = sorted(np.random.default_rng(SEED+1).choice(40,20,replace=False).tolist())
    scenes = [dict(id=j*8+i, x0=float(x), y0=float(y), vx=2., vy=0.)
              for j,y in enumerate(np.linspace(10,14,5)) for i,x in enumerate(np.linspace(-7,7,8))]
    cfg = dict(seed=SEED, scenes=scenes, train=split[:30], test=split[30:], controls=controls,
        gravity=[9.8,4.9], frames=16, times=np.linspace(0,1,16).tolist(), resolution=384,
        radius_m=0.5, world=[-10,10,0,20], background=[32,32,32], grid=[44,44,44],
        baseline_color=[235,235,235], control_color=[60,140,235],
        model='V-JEPA 2.1 ViT-B/16 EMA encoder', dtype='float32', pooling='mean of final normalized spatial-temporal tokens',
        source_commit='204698b45b3712590f06245fbfba32d3be539812',
        mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225],
        transport='d=mean(z4.9-z9.8) on 30 train scenes; predict z9.8+d on 10 test scenes; no fitted scale')
    p=ROOT/'setup.json'
    if p.exists(): assert json.loads(p.read_text())==cfg, 'Frozen setup changed'
    else: save_json(p,cfg)
    return cfg

def render(scene,g,color,cfg):
    S=cfg['resolution']; scale=S/20
    bg=Image.new('RGB',(S,S),tuple(cfg['background'])); draw=ImageDraw.Draw(bg)
    for p in range(0,S,round(2*scale)):
        draw.line((p,0,p,S-1),fill=tuple(cfg['grid']))
        draw.line((0,p,S-1,p),fill=tuple(cfg['grid']))
    frames=[]
    for t in cfg['times']:
        x=scene['x0']+scene['vx']*t; y=scene['y0']+scene['vy']*t-0.5*g*t*t
        r=cfg['radius_m']; assert -10+r<x<10-r and r<y<20-r
        cx=(x+10)*scale; cy=(20-y)*scale
        im=bg.copy(); ImageDraw.Draw(im).ellipse((cx-r*scale,cy-r*scale,cx+r*scale,cy+r*scale),fill=tuple(color))
        frames.append(np.asarray(im))
    return np.stack(frames)

def generate(cfg):
    import imageio.v2 as iio
    out=ROOT/'videos'; out.mkdir(exist_ok=True)
    rows=[]
    for s in cfg['scenes']:
        conditions=[('g98',9.8,cfg['baseline_color']),('g49',4.9,cfg['baseline_color'])]
        if s['id'] in cfg['controls']: conditions.append(('color',9.8,cfg['control_color']))
        for condition,g,color in conditions:
            name=f"scene_{s['id']:02d}_{condition}"
            frames=render(s,g,color,cfg)
            path=out/(name+'.mkv')
            if not path.exists():
                iio.mimwrite(path,frames,format='FFMPEG',fps=15,codec='ffv1',pixelformat='bgr0',macro_block_size=1)
            decoded=np.stack(iio.mimread(path,format='FFMPEG'))
            assert np.array_equal(decoded,frames), 'Video round-trip was not lossless'
            rows.append(dict(name=name,scene=s['id'],condition=condition,gravity=g,path=str(path.relative_to(ROOT)),sha256=sha(path)))
    assert len(rows)==100
    save_json(ROOT/'manifest.json',rows)
    # Contact sheet depicts actual frames, not additional experimental clips.
    sheet=Image.new('RGB',(384*3,384*2))
    for j,(g,color) in enumerate([(9.8,cfg['baseline_color']),(4.9,cfg['baseline_color']),(9.8,cfg['control_color'])]):
        frames=render(cfg['scenes'][cfg['controls'][0]],g,color,cfg)
        for k,index in enumerate([0,15]): sheet.paste(Image.fromarray(frames[index]),(j*384,k*384))
    sheet.save(ROOT/'stimulus_preview.png')
    print('Generated and losslessly verified 100 videos',flush=True)

def extract(cfg,args):
    import torch, subprocess
    sys.path.insert(0,str(Path(args.repo).resolve()))
    from app.vjepa_2_1.models.vision_transformer import vit_base
    from src.hub.backbones import _clean_backbone_key
    assert subprocess.check_output(['git','-C',args.repo,'rev-parse','HEAD'],text=True).strip()==cfg['source_commit']
    torch.set_num_threads(4); torch.manual_seed(SEED); torch.use_deterministic_algorithms(True)
    model=vit_base(patch_size=16,img_size=(384,384),num_frames=16,tubelet_size=2,
        use_sdpa=True,use_SiLU=False,wide_SiLU=True,uniform_power=False,use_rope=True,
        img_temporal_dim_size=1,interpolate_rope=True)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(_clean_backbone_key(checkpoint['ema_encoder']),strict=True)
    del checkpoint
    model.eval().requires_grad_(False)
    assert not model.return_hierarchical
    mean=torch.tensor(cfg['mean']).view(1,3,1,1,1); std=torch.tensor(cfg['std']).view(1,3,1,1,1)
    out=ROOT/'embeddings'; out.mkdir(exist_ok=True)
    rows=json.loads((ROOT/'manifest.json').read_text())
    import imageio.v2 as iio
    started=time.time()
    for i,row in enumerate(rows):
        target=out/(row['name']+'.npy')
        if target.exists(): continue
        frames=np.stack(iio.mimread(ROOT/row['path'],format='FFMPEG'))
        x=torch.from_numpy(frames).permute(3,0,1,2).unsqueeze(0).float()/255
        with torch.inference_mode():
            tokens=model((x-mean)/std)
            assert tokens.shape==(1,4608,768),tokens.shape
            z=tokens.mean(dim=1)[0].numpy()
        assert np.isfinite(z).all()
        np.save(target,z)
        print(f"{i+1}/100 {row['name']} elapsed={time.time()-started:.1f}s",flush=True)
    save_json(ROOT/'provenance.json',dict(torch=torch.__version__,numpy=np.__version__,checkpoint_sha256=sha(args.checkpoint),
        checkpoint_url='https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt',source_commit=cfg['source_commit'],device='cpu',tokens=[1,4608,768]))

def unit(x):
    n=np.linalg.norm(x,axis=-1,keepdims=True)
    if np.any(n<1e-12): raise ValueError('Zero intervention vector: cosine undefined')
    return x/n

def stats(x):
    return dict(n=int(x.size),mean=float(x.mean()),median=float(np.median(x)),std=float(x.std()),min=float(x.min()),max=float(x.max()))

def analyze(cfg):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def read(i,c): return np.load(ROOT/'embeddings'/f'scene_{i:02d}_{c}.npy').astype(np.float64)
    base=np.stack([read(i,'g98') for i in range(40)]); low=np.stack([read(i,'g49') for i in range(40)])
    dg=low-base; dc=np.stack([read(i,'color')-base[i] for i in cfg['controls']])
    gg=unit(dg)@unit(dg).T; gc=unit(dg)@unit(dc).T
    pair=gg[np.triu_indices(40,1)]; matched=gc[cfg['controls'],np.arange(20)]
    train=cfg['train']; test=cfg['test']; direction=dg[train].mean(axis=0)
    pred=base[test]+direction; e0=np.linalg.norm(dg[test],axis=1); e1=np.linalg.norm(pred-low[test],axis=1)
    align=unit(dg[test])@unit(direction)
    result=dict(gravity_pairwise=stats(pair),gravity_color_all=stats(gc),gravity_color_matched=stats(matched),
        transport=dict(train_scenes=train,test_scenes=test,direction_norm=float(np.linalg.norm(direction)),
        mean_direction_cosine=float(align.mean()),mean_identity_error=float(e0.mean()),mean_transport_error=float(e1.mean()),
        relative_rmse=float(np.sqrt(np.sum(e1**2)/np.sum(e0**2))),fraction_improved=float(np.mean(e1<e0))))
    save_json(ROOT/'results.json',result)
    np.savez(ROOT/'analysis_arrays.npz',baseline=base,low_gravity=low,gravity_vectors=dg,color_vectors=dc,
        gravity_cosines=gg,gravity_color_cosines=gc,train_direction=direction,test_predictions=pred)
    np.savetxt(ROOT/'gravity_cosines.csv',gg,delimiter=','); np.savetxt(ROOT/'gravity_color_cosines.csv',gc,delimiter=',')
    with open(ROOT/'transport.csv','w') as f:
        w=csv.writer(f);w.writerow(['scene','identity_error','transport_error','relative_error','direction_cosine'])
        w.writerows(zip(test,e0,e1,e1/e0,align))
    fig,ax=plt.subplots(1,3,figsize=(16,4.5))
    im=ax[0].imshow(gg,vmin=-1,vmax=1,cmap='coolwarm');ax[0].set(title='Gravity intervention cosines',xlabel='Scene',ylabel='Scene');fig.colorbar(im,ax=ax[0])
    ax[1].hist(pair,bins=np.linspace(-1,1,31),alpha=.55,density=True,label='Gravity–gravity (780 pairs)')
    ax[1].hist(matched,bins=np.linspace(-1,1,31),alpha=.55,density=True,label='Gravity–color (20 matched)')
    ax[1].set(xlabel='Cosine',ylabel='Density',title='Intervention alignment');ax[1].legend(fontsize=8)
    ax[2].bar(np.arange(10)-.2,e0,.4,label='No transport');ax[2].bar(np.arange(10)+.2,e1,.4,label='Train mean transport')
    ax[2].set(xticks=np.arange(10),xticklabels=test,xlabel='Held-out scene',ylabel='Euclidean error',title='30-train / 10-test transport');ax[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'results.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['generate','extract','analyze','all']);p.add_argument('--repo',default='/tmp/gravity-vjepa2');p.add_argument('--checkpoint',default='/mnt/d/gravity-latent-experiment-runtime/vjepa2_1_vitb_dist_vitG_384.pt');args=p.parse_args()
    cfg=setup()
    if args.stage in ('generate','all'): generate(cfg)
    if args.stage in ('extract','all'): extract(cfg,args)
    if args.stage in ('analyze','all'): analyze(cfg)
