"""Draw a 2-D LBM solid mask in a local browser and save solid[x, y].

Run from the repository root:
    uv run --locked python src/geometry/draw_grid2d.py
Then open http://127.0.0.1:8765 in a browser. Ctrl+C stops the server.
"""

import argparse
import base64
import binascii
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shlex

import numpy as np


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "manual_2d" / "solid.npy"
MAX_SIDE = 1024
MAX_BODY = 2 * 1024 * 1024


def validate_solid(solid):
    if (not isinstance(solid, np.ndarray) or solid.ndim != 2 or
            any(not 3 <= side <= MAX_SIDE for side in solid.shape)):
        raise ValueError(f"solid.npy must have shape (nx, ny), each side 3–{MAX_SIDE}")
    if not np.isin(solid, (0, 1)).all():
        raise ValueError("solid.npy must contain only 0 (pore) and 1 (solid)")
    return np.asarray(solid, dtype=np.uint8)


def encode_grid(solid):
    """Browser bytes are rows from y=0 upward; NumPy uses solid[x, y]."""
    return base64.b64encode(np.ascontiguousarray(solid.T).tobytes()).decode("ascii")


def decode_grid(nx, ny, encoded):
    if not isinstance(nx, int) or isinstance(nx, bool) or not 3 <= nx <= MAX_SIDE:
        raise ValueError(f"width must be 3–{MAX_SIDE}")
    if not isinstance(ny, int) or isinstance(ny, bool) or not 3 <= ny <= MAX_SIDE:
        raise ValueError(f"height must be 3–{MAX_SIDE}")
    if not isinstance(encoded, str):
        raise ValueError("grid data is missing")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != nx * ny:
        raise ValueError("grid data length does not match width × height")
    return validate_solid(np.frombuffer(raw, dtype=np.uint8).reshape(ny, nx).T)


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>二维 LBM 几何绘制</title>
<style>
:root{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#1c2630;background:#f3f6f8}
*{box-sizing:border-box} body{margin:0;padding:24px} main{max-width:1240px;margin:auto}
h1{font-size:1.55rem;margin:0 0 8px} p{line-height:1.55;margin:8px 0 18px;color:#51606c}
.layout{display:grid;grid-template-columns:minmax(0,800px) minmax(260px,340px);gap:22px;align-items:start}
.card{background:#fff;border:1px solid #dbe2e7;border-radius:12px;padding:16px;box-shadow:0 2px 12px #21313d0b}
.stage{position:relative;max-width:768px;border:1px solid #62798a;touch-action:none;line-height:0;background:#fff}
canvas{width:100%;height:auto;image-rendering:pixelated;display:block;touch-action:none}
#overlay{position:absolute;inset:0;pointer-events:none}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0}
button,select,input{font:inherit}button{background:#edf4f7;color:#173b50;border:1px solid #b5c7d0;border-radius:7px;padding:7px 10px;cursor:pointer}
button:hover{background:#dceef5}button.primary{background:#147da4;border-color:#147da4;color:white}
button.active{background:#173b50;color:white}button:disabled{opacity:.5;cursor:default}
input[type=number]{width:78px;padding:6px;border:1px solid #b5c7d0;border-radius:6px}
input[type=range]{width:120px}label{display:inline-flex;align-items:center;gap:5px}
.label{font-size:.9rem;color:#52616d;margin-top:15px}#status{min-height:2.5em;color:#1e5c75;line-height:1.4}
code{display:block;white-space:pre-wrap;word-break:break-all;background:#f1f5f7;padding:10px;border-radius:6px;font-size:.83rem}
.small{font-size:.88rem;color:#536572}.legend{display:flex;gap:12px;margin:12px 0}.swatch{display:inline-block;width:14px;height:14px;border:1px solid #98a8b0;vertical-align:-2px}
@media(max-width:980px){.layout{grid-template-columns:1fr}body{padding:12px}}
</style></head><body><main>
<h1>二维 LBM 几何绘制</h1>
<p>黑色是固体（1），白色是孔隙（0）。坐标原点在左下角；保存文件为 <b>solid[x, y]</b>，可直接交给本项目的二维 LBM 求解器。</p>
<div class="layout"><section class="card">
<div class="stage"><canvas id="image"></canvas><canvas id="overlay"></canvas></div>
<div class="legend"><span><i class="swatch" style="background:#223a4b"></i> 固体</span><span><i class="swatch" style="background:white"></i> 孔隙</span></div>
<div id="stats" class="small"></div><div id="coords" class="small">坐标：(x, y)</div>
</section><aside class="card">
<div class="label">绘制工具</div>
<div class="row" id="tools"><button data-tool="solid" class="active">画固体</button><button data-tool="pore">擦成孔隙</button><button data-tool="circle">实心圆</button><button data-tool="rect">实心矩形</button></div>
<div class="row"><label>画笔半径 <input id="radius" type="range" min="0" max="40" value="3"><span id="radiusValue">3</span> 格</label></div>
<div class="row"><label><input id="periodic" type="checkbox" checked> 画笔 / 圆形跨边界回绕</label></div>
<div class="row"><label><input id="gridlines" type="checkbox"> 显示网格线</label></div>
<div class="row"><button id="undo">撤销</button><button id="redo">重做</button><button id="clear">全部清空</button></div>
<div class="label">新建网格</div>
<div class="row"><label>宽 nx <input id="width" type="number" min="3" max="1024"></label><label>高 ny <input id="height" type="number" min="3" max="1024"></label><button id="new">新建</button></div>
<div class="label">导入已有 solid.npy（不会改动原文件）</div>
<div class="row"><input id="file" type="file" accept=".npy,application/octet-stream"></div>
<div class="label">保存并用于 LBM</div>
<div class="row"><button class="primary" id="save">保存 solid.npy</button><a id="download" href="/api/download" hidden>下载已保存文件</a></div>
<div class="small">输出到：<code id="path"></code></div>
<div class="small">在项目根目录运行：</div><code id="command"></code>
<div id="status" role="status"></div>
<p class="small">求解器的外边界是周期性的；若希望固体结构本身也周期接续，请勾选跨边界回绕来绘制边缘。画笔、圆形和矩形均会填满内部。</p>
</aside></div></main>
<script>
const image=document.getElementById('image'),overlay=document.getElementById('overlay');
const ctx=image.getContext('2d'),octx=overlay.getContext('2d');
let nx=0,ny=0,cells=new Uint8Array(),tool='solid',drawing=false,start=null,last=null,snapshot=null;
let undoStack=[],redoStack=[],saved=false;
const $=id=>document.getElementById(id);
const status=(message,error=false)=>{const el=$('status');el.textContent=message;el.style.color=error?'#a2352b':'#1e5c75'};
const idx=(x,y)=>y*nx+x;
const wrap=(v,n)=>(v%n+n)%n;
function setGrid(width,height,data){
  nx=width;ny=height;cells=new Uint8Array(data);undoStack=[];redoStack=[];saved=false;
  image.width=nx;image.height=ny;overlay.width=nx;overlay.height=ny;
  $('width').value=nx;$('height').value=ny;$('download').hidden=true;render();
}
function render(){
  const img=ctx.createImageData(nx,ny);let count=0;
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
    const solid=cells[idx(x,y)];count+=solid;
    const p=4*((ny-1-y)*nx+x);
    img.data[p]=solid?34:255;img.data[p+1]=solid?58:255;img.data[p+2]=solid?75:255;img.data[p+3]=255;
  }
  ctx.putImageData(img,0,0);
  $('stats').textContent=`${nx} × ${ny} 格｜固体 ${count} 格｜孔隙率 ${((1-count/cells.length)*100).toFixed(2)}%${saved?'｜已保存':'｜未保存'}`;
  drawOverlay();
}
function drawOverlay(){
  octx.clearRect(0,0,nx,ny);
  if($('gridlines').checked){
    octx.lineWidth=Math.max(.1,nx/768*.45);octx.strokeStyle='rgba(37,90,110,.35)';octx.beginPath();
    for(let x=0;x<=nx;x++){octx.moveTo(x,0);octx.lineTo(x,ny)}
    for(let y=0;y<=ny;y++){octx.moveTo(0,y);octx.lineTo(nx,y)}octx.stroke();
  }
  if(!drawing||!start||!last||!['circle','rect'].includes(tool))return;
  octx.fillStyle='rgba(20,125,164,.35)';octx.strokeStyle='#147da4';octx.lineWidth=Math.max(.5,nx/768);
  const x1=start.x,y1=ny-1-start.y,x2=last.x,y2=ny-1-last.y;
  if(tool==='rect'){
    const left=Math.min(x1,x2),top=Math.min(y1,y2),w=Math.abs(x2-x1)+1,h=Math.abs(y2-y1)+1;
    octx.fillRect(left,top,w,h);octx.strokeRect(left,top,w,h);
  }else{
    const r=Math.hypot(last.x-start.x,last.y-start.y);
    octx.beginPath();octx.arc(x1+.5,y1+.5,r,0,2*Math.PI);octx.fill();octx.stroke();
  }
}
function point(e){
  const rect=image.getBoundingClientRect();
  return {x:Math.max(0,Math.min(nx-1,Math.floor((e.clientX-rect.left)/rect.width*nx))),
          y:Math.max(0,Math.min(ny-1,ny-1-Math.floor((e.clientY-rect.top)/rect.height*ny)))};
}
function setCell(x,y,value,periodic){
  if(periodic){x=wrap(x,nx);y=wrap(y,ny)}
  if(x>=0&&x<nx&&y>=0&&y<ny)cells[idx(x,y)]=value;
}
function stamp(x,y,value){
  const r=Number($('radius').value),periodic=$('periodic').checked;
  for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++)if(dx*dx+dy*dy<=r*r)setCell(x+dx,y+dy,value,periodic);
}
function segment(a,b,value){
  const n=Math.max(Math.abs(b.x-a.x),Math.abs(b.y-a.y));
  for(let i=0;i<=n;i++){const t=n?i/n:0;stamp(Math.round(a.x+(b.x-a.x)*t),Math.round(a.y+(b.y-a.y)*t),value)}
}
function commitShape(a,b){
  if(tool==='rect'){
    for(let y=Math.min(a.y,b.y);y<=Math.max(a.y,b.y);y++)
      for(let x=Math.min(a.x,b.x);x<=Math.max(a.x,b.x);x++)setCell(x,y,1,false);
  }else{
    const radius=Math.hypot(b.x-a.x,b.y-a.y),periodic=$('periodic').checked;
    const reach=Math.ceil(radius);
    for(let dy=-reach;dy<=reach;dy++)for(let dx=-reach;dx<=reach;dx++)
      if(dx*dx+dy*dy<=radius*radius)setCell(a.x+dx,a.y+dy,1,periodic);
  }
}
function historyPush(before){undoStack.push(before);if(undoStack.length>20)undoStack.shift();redoStack=[];saved=false;$('download').hidden=true}
function finish(e){
  if(!drawing)return;
  const end=e?point(e):last;
  if(['circle','rect'].includes(tool))commitShape(start,end);
  drawing=false;historyPush(snapshot);snapshot=null;start=null;last=null;render();
}
image.addEventListener('pointerdown',e=>{
  if(e.button!==0)return;e.preventDefault();image.setPointerCapture(e.pointerId);
  drawing=true;start=point(e);last=start;snapshot=cells.slice();
  if(tool==='solid'||tool==='pore'){stamp(start.x,start.y,tool==='solid'?1:0);render()}else drawOverlay();
});
image.addEventListener('pointermove',e=>{
  const p=point(e);$('coords').textContent=`坐标：(${p.x}, ${p.y})`;
  if(!drawing)return;
  if(tool==='solid'||tool==='pore'){segment(last,p,tool==='solid'?1:0);last=p;render()}
  else{last=p;drawOverlay()}
});
image.addEventListener('pointerup',finish);
image.addEventListener('pointercancel',()=>{if(drawing){cells=snapshot;drawing=false;start=null;last=null;snapshot=null;render()}});
image.addEventListener('lostpointercapture',()=>{if(drawing)finish(null)});
$('tools').addEventListener('click',e=>{
  const b=e.target.closest('button[data-tool]');if(!b)return;
  tool=b.dataset.tool;document.querySelectorAll('#tools button').forEach(v=>v.classList.toggle('active',v===b));
});
$('radius').addEventListener('input',()=>{$('radiusValue').textContent=$('radius').value});
$('gridlines').addEventListener('change',drawOverlay);
$('undo').onclick=()=>{if(!undoStack.length)return;redoStack.push(cells.slice());cells=undoStack.pop();saved=false;$('download').hidden=true;render()};
$('redo').onclick=()=>{if(!redoStack.length)return;undoStack.push(cells.slice());cells=redoStack.pop();saved=false;$('download').hidden=true;render()};
$('clear').onclick=()=>{if(!confirm('清空所有固体格点？'))return;historyPush(cells.slice());cells.fill(0);render()};
$('new').onclick=()=>{
  const width=Number($('width').value),height=Number($('height').value);
  if(!Number.isInteger(width)||!Number.isInteger(height)||width<3||height<3||width>1024||height>1024){status('宽和高必须是 3–1024 的整数。',true);return}
  if(!confirm('新建网格会清除当前未保存的绘制内容，继续吗？'))return;
  setGrid(width,height,new Uint8Array(width*height));status('已新建空白网格。');
};
function toBase64(data){
  let s='';const chunk=16384;
  for(let i=0;i<data.length;i+=chunk)s+=String.fromCharCode(...data.subarray(i,i+chunk));
  return btoa(s);
}
$('save').onclick=async()=>{
  let solids=0;for(const value of cells)solids+=value;
  if(solids===0||solids===cells.length){status('求解器需要同时有孔隙和固体；请先绘制有效几何。',true);return}
  try{
    const response=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({nx,ny,data:toBase64(cells)})});
    const result=await response.json();if(!response.ok)throw Error(result.error);
    saved=true;$('download').hidden=false;render();status(`已保存 ${result.path}，孔隙率 ${(result.porosity*100).toFixed(2)}%。`);
  }catch(err){status(`保存失败：${err.message}`,true)}
};
$('file').addEventListener('change',async e=>{
  const file=e.target.files[0];if(!file)return;
  try{
    const response=await fetch('/api/import',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:await file.arrayBuffer()});
    const result=await response.json();if(!response.ok)throw Error(result.error);
    const raw=atob(result.data),data=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)data[i]=raw.charCodeAt(i);
    setGrid(result.nx,result.ny,data);status(`已导入 ${file.name}；修改后点击保存。`);
  }catch(err){status(`导入失败：${err.message}`,true)}finally{e.target.value=''}
});
fetch('/api/state').then(r=>r.json()).then(result=>{
  const raw=atob(result.data),data=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)data[i]=raw.charCodeAt(i);
  setGrid(result.nx,result.ny,data);saved=result.saved;if(saved){$('download').hidden=false;render()}
  $('path').textContent=result.output;$('command').textContent=result.command;
  status('可以开始绘制。保存后在终端运行上面的 LBM 命令。');
}).catch(err=>status(`加载失败：${err.message}`,true));
</script></body></html>"""


def make_handler(initial_solid, output, initial_saved=False):
    initial_solid = validate_solid(initial_solid)
    output = output.resolve()
    solver = Path(__file__).resolve().parents[1] / "solver" / "2D_DARCY" / "main.py"
    command = f"uv run --locked python {shlex.quote(str(solver))} --solid {shlex.quote(str(output))} --force 4e-6"

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, data, content_type, status=200, extra_headers=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, data, status=200):
            self.send_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"),
                            "application/json; charset=utf-8", status)

        def do_GET(self):
            if self.path == "/":
                self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self.send_json({"nx": initial_solid.shape[0], "ny": initial_solid.shape[1],
                                "data": encode_grid(initial_solid), "output": str(output),
                                "command": command, "saved": initial_saved})
            elif self.path == "/api/download" and output.is_file():
                self.send_bytes(output.read_bytes(), "application/octet-stream", extra_headers={
                    "Content-Disposition": 'attachment; filename="solid.npy"'})
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path not in ("/api/save", "/api/import"):
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY:
                    raise ValueError("request body is empty or too large")
                body = self.rfile.read(size)
                if self.path == "/api/import":
                    solid = validate_solid(np.load(io.BytesIO(body), allow_pickle=False))
                    self.send_json({"nx": solid.shape[0], "ny": solid.shape[1],
                                    "data": encode_grid(solid)})
                    return
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("request must be a JSON object")
                solid = decode_grid(payload.get("nx"), payload.get("ny"),
                                    payload.get("data"))
                if not solid.any() or solid.all():
                    raise ValueError("LBM requires both pore and solid cells")
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(output.name + ".tmp")
                with temporary.open("wb") as handle:
                    np.save(handle, solid)
                temporary.replace(output)
                self.send_json({"path": str(output), "porosity": float(1-solid.mean())})
            except (ValueError, TypeError, KeyError, binascii.Error) as exc:
                self.send_json({"error": str(exc)}, 400)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=256, help="new blank grid width")
    parser.add_argument("--ny", type=int, default=256, help="new blank grid height")
    parser.add_argument("--input", type=Path, help="load an existing solid.npy at startup")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="file written by the Save button")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    initial_saved = not args.input and args.output.is_file()
    if args.input:
        solid = validate_solid(np.load(args.input, allow_pickle=False))
    elif args.output.is_file():
        solid = validate_solid(np.load(args.output, allow_pickle=False))
    else:
        if not 3 <= args.nx <= MAX_SIDE or not 3 <= args.ny <= MAX_SIDE:
            parser.error(f"nx and ny must be 3–{MAX_SIDE}")
        solid = np.zeros((args.nx, args.ny), dtype=np.uint8)
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1–65535")
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 make_handler(solid, args.output, initial_saved))
    print(f"Open http://127.0.0.1:{args.port} in your browser", flush=True)
    print(f"Save target: {args.output.resolve()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
