import os
import io
import base64
import zipfile
import numpy as np
import pandas as pd
import trimesh
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from math import pi
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc


def load_stl_and_split(file_bytes):
    mesh = trimesh.load(io.BytesIO(file_bytes), file_type='stl')
    if isinstance(mesh, trimesh.Scene):
# 若為 Scene，合併
        mesh = trimesh.util.concatenate(tuple(m for m in mesh.geometry.values()))
# 分割連通組件
    components = mesh.split(only_watertight=False)
    return components

def compute_particle_metrics(mesh, use_center_of_mass=True):

#回傳一個 dict，包含各種幾何參數與指標。

# 基本幾何量
    volume = float(mesh.volume) if mesh.is_volume else np.nan
    area = float(mesh.area)

複製
# 凸殼
try:
    ch = mesh.convex_hull
    ch_vol = float(ch.volume)
except Exception:
    ch_vol = np.nan

# 質心
centroid = mesh.center_mass if (use_center_of_mass and mesh.is_volume) else mesh.vertices.mean(axis=0)

# PCA 主軸
verts = mesh.vertices - centroid
cov = np.cov(verts.T)
eigvals, eigvecs = np.linalg.eigh(cov)
# 由小到大，反轉成大到小
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

# 投影以求範圍 (直徑)
proj = verts @ eigvecs
ranges = proj.max(axis=0) - proj.min(axis=0)   # 長、中、短
a, b, c = ranges[0], ranges[1], ranges[2]

# EI, FI
EI = b / a if a > 0 else np.nan
FI = c / b if b > 0 else np.nan

# Roundness Index
# IR = S / (4π ((a+b+c)/6)^2 )
mean_radius = (a + b + c) / 6.0
IR = area / (4 * pi * mean_radius**2) if mean_radius > 0 else np.nan

# Sphericity
# psi = ((36 π V^2)^(1/3)) / S
if volume > 0:
    sph = ((36 * pi * (volume**2))**(1/3)) / area
else:
    sph = np.nan

# Convexity
convexity = volume / ch_vol if (ch_vol and ch_vol > 0 and volume > 0) else np.nan

# 內外球（近似）
dists = np.linalg.norm(verts, axis=1)
r_in = dists.min()
r_out = dists.max()

return {
    "Volume": volume,
    "SurfaceArea": area,
    "ConvexHullVolume": ch_vol,
    "CentroidX": centroid[0],
    "CentroidY": centroid[1],
    "CentroidZ": centroid[2],
    "Major_d_a": a,
    "Intermediate_d_b": b,
    "Minor_d_c": c,
    "EI_b_over_a": EI,
    "FI_c_over_b": FI,
    "RoundnessIndex_IR": IR,
    "Sphericity": sph,
    "Convexity": convexity,
    "InscribedSphere_r": r_in,
    "CircumscribedSphere_r": r_out,
    "EigenVec1": eigvecs[:,0],
    "EigenVec2": eigvecs[:,1],
    "EigenVec3": eigvecs[:,2]
}
def fit_kde_density(centers, bandwidth=None, method='scott'):
    n = centers.shape[0]
if n < 2:
    return np.zeros(n), None
if bandwidth is None:
# 自動估 bandwidth
if method == 'scott':
# Scott's rule: h = n^(-1/(d+4)) * σ
sigma = np.mean(centers.std(axis=0))
h = sigma * n ** (-1.0 / (centers.shape[1] + 4))
elif method == 'silverman':
sigma = np.mean(centers.std(axis=0))
h = ( (4/(centers.shape[1]+2))(1.0/(centers.shape[1]+4)) ) * sigma * n(-1.0/(centers.shape[1]+4))
else:
h = 0.2
else:
h = bandwidth
kde = KernelDensity(bandwidth=h, kernel='gaussian')
kde.fit(centers)
log_dens = kde.score_samples(centers)
dens = np.exp(log_dens)
# normalize
dens_norm = (dens - dens.min())/(dens.max()-dens.min()+1e-12)
return dens_norm, kde

def pair_correlation(centers, r_max=None, bins=30):
"""
簡易 g(r) 計算（未做邊界修正）。
"""
n = centers.shape[0]
if n < 3:
return None
dvec = pdist(centers)
if r_max is None:
r_max = dvec.max() * 0.6
hist, edges = np.histogram(dvec, bins=bins, range=(0, r_max))
r = 0.5*(edges[1:]+edges[:-1])
dr = edges[1]-edges[0]
# 密度
vol_total = np.prod( (centers.max(axis=0) - centers.min(axis=0)) )
rho = n / vol_total if vol_total > 0 else 0
shell_vol = 4 * np.pi * r ** 2 * dr
exp_count = shell_vol * rho * n / 2.0  # i<j 組合
with np.errstate(divide='ignore', invalid='ignore'):
g = hist / (exp_count + 1e-12)
return r, g

def ripley_K_L(centers, r_values):
"""
簡易 Ripley K 與 L（未邊界修正）。
"""
n = len(centers)
if n < 3:
return None
vol_total = np.prod( (centers.max(axis=0) - centers.min(axis=0)) )
rho = n / vol_total
tree = cKDTree(centers)
K = []
for r in r_values:
counts = tree.query_ball_point(centers, r)
# each counts[i] 包含自身
sum_counts = sum(len(c)-1 for c in counts)
K_r = sum_counts / (n * rho)
K.append(K_r)
K = np.array(K)
L = (K / ((4/3)*pi))**(1/3)
return K, L

============== 視覺化輔助 ==============
def mesh_to_plotly(mesh, color='lightgray', opacity=1.0, name='mesh', show_edges=False):
tri = mesh.faces
verts = mesh.vertices
x,y,z = verts[:,0], verts[:,1], verts[:,2]
i,j,k = tri[:,0], tri[:,1], tri[:,2]
return go.Mesh3d(
x=x, y=y, z=z,
i=i, j=j, k=k,
color=color,
opacity=opacity,
name=name,
showscale=False,
flatshading=True
)

def generate_all_particles_fig(particle_meshes, df, color_mode='uniform'):
fig = go.Figure()
if len(particle_meshes)==0:
fig.update_layout(title="尚未載入資料")
return fig
# 決定顏色
if color_mode == 'density' and 'DensityNorm' in df.columns:
cmap = plt_colormap_turbo()
dens = df['DensityNorm'].values
colors = [cmap(float(d)) for d in dens]
elif color_mode == 'sphericity':
sph = df['Sphericity'].fillna(0).values
sph_norm = (sph - sph.min())/(sph.max()-sph.min()+1e-9)
cmap = plt_colormap_viridis()
colors = [cmap(float(s)) for s in sph_norm]
else:
colors = ['#888888']*len(particle_meshes)

複製
# 以透明度較低方式顯示全部顆粒
for idx, m in enumerate(particle_meshes):
    fig.add_trace(mesh_to_plotly(m, color=colors[idx], opacity=0.55, name=f"ID {idx}"))
# 顯示中心點
fig.add_trace(go.Scatter3d(
    x=df.CentroidX, y=df.CentroidY, z=df.CentroidZ,
    mode='markers',
    marker=dict(size=4, color=df.index, colorscale='Turbo', showscale=False),
    name='Centers'
))
fig.update_layout(
    scene=dict(aspectmode='data'),
    margin=dict(l=0,r=0,b=0,t=30),
    title="所有顆粒 (點擊顆粒中心附近可選擇)"
)
return fig
def generate_single_particle_fig(mesh, row, show_ellipsoid=True, show_inscribed=True, show_circumscribed=True, show_axes=True):
fig = go.Figure()
fig.add_trace(mesh_to_plotly(mesh, color='#1f77b4', opacity=0.85, name='Particle'))
centroid = np.array([row.CentroidX, row.CentroidY, row.CentroidZ])

複製
# 主軸
eig1 = row.EigenVec1
eig2 = row.EigenVec2
eig3 = row.EigenVec3
if isinstance(eig1, str):
    # 從字串轉回
    eig1 = np.fromstring(eig1.strip("[]"), sep=' ')
    eig2 = np.fromstring(row.EigenVec2.strip("[]"), sep=' ')
    eig3 = np.fromstring(row.EigenVec3.strip("[]"), sep=' ')

proj = (mesh.vertices - centroid) @ np.vstack([eig1,eig2,eig3]).T
ranges = proj.max(axis=0) - proj.min(axis=0)
a,b,c = ranges
# 半軸
ra, rb, rc = a/2, b/2, c/2

# 擬合橢圓
if show_ellipsoid:
    u = np.linspace(0, 2*np.pi, 32)
    v = np.linspace(0, np.pi, 16)
    uu, vv = np.meshgrid(u, v)
    x = ra * np.cos(uu) * np.sin(vv)
    y = rb * np.sin(uu) * np.sin(vv)
    z = rc * np.cos(vv)
    # 旋轉回原座標
    R = np.vstack([eig1,eig2,eig3]).T
    ellip_pts = np.stack([x,y,z], axis=-1) @ R.T + centroid
    fig.add_trace(go.Surface(
        x=ellip_pts[:,:,0],
        y=ellip_pts[:,:,1],
        z=ellip_pts[:,:,2],
        opacity=0.25,
        showscale=False,
        name='Fitted Ellipsoid',
        colorscale='Blues'
    ))

# 內切球
if show_inscribed:
    r_in = row.InscribedSphere_r
    sphere = sphere_param(center=centroid, r=r_in, n_u=20, n_v=12)
    fig.add_trace(go.Surface(
        x=sphere[0], y=sphere[1], z=sphere[2],
        opacity=0.25, showscale=False, name='Inscribed Sphere', colorscale='Greens'
    ))
# 外接球
if show_circumscribed:
    r_out = row.CircumscribedSphere_r
    sphere2 = sphere_param(center=centroid, r=r_out, n_u=24, n_v=16)
    fig.add_trace(go.Surface(
        x=sphere2[0], y=sphere2[1], z=sphere2[2],
        opacity=0.15, showscale=False, name='Circumscribed Sphere', colorscale='Reds'
    ))

# 主軸線
if show_axes:
    scale = 0.6
    axes_data = []
    for vec, length, color, name in [
        (eig1, a, 'red', 'Major'),
        (eig2, b, 'green', 'Intermediate'),
        (eig3, c, 'purple', 'Minor')
    ]:
        p1 = centroid - vec*(length/2)
        p2 = centroid + vec*(length/2)
        axes_data.append(go.Scatter3d(
            x=[p1[0],p2[0]], y=[p1[1],p2[1]], z=[p1[2],p2[2]],
            mode='lines',
            line=dict(color=color, width=6),
            name=f"{name} axis"
        ))
    for tr in axes_data:
        fig.add_trace(tr)

fig.update_layout(
    scene=dict(aspectmode='data'),
    margin=dict(l=0,r=0,b=0,t=30),
    title="選擇顆粒詳細視圖"
)
return fig
def sphere_param(center, r, n_u=24, n_v=16):
u = np.linspace(0, 2*np.pi, n_u)
v = np.linspace(0, np.pi, n_v)
uu, vv = np.meshgrid(u,v)
x = center[0] + r * np.cos(uu) * np.sin(vv)
y = center[1] + r * np.sin(uu) * np.sin(vv)
z = center[2] + r * np.cos(vv)
return x,y,z

簡單 colormap (避免依賴 matplotlib)
def plt_colormap_turbo():
# 取樣一個簡單插值（縮短程式；實務可用 matplotlib）
# 回傳函數 f(t) -> '#RRGGBB'
import colorsys
# 用一組預定顏色點 (Turbo 大致風格近似)
stops = [
(0.0, (48,18,59)),
(0.2, (0,115,188)),
(0.4, (0,182,188)),
(0.6, (220,223,65)),
(0.8, (238,117,0)),
(1.0, (180,4,38))
]
def interp(t):
t = max(0,min(1,t))
for i in range(len(stops)-1):
t0,c0 = stops[i]
t1,c1 = stops[i+1]
if t0 <= t <= t1:
f = (t - t0)/(t1-t0+1e-12)
c = tuple(int(c0[k] + f*(c1[k]-c0[k])) for k in range(3))
return 'rgb({},{},{})'.format(*c)
return 'rgb(0,0,0)'
return interp

def plt_colormap_viridis():
stops = [
(0.0,(68,1,84)),
(0.25,(58,82,139)),
(0.5,(32,144,140)),
(0.75,(94,201,98)),
(1.0,(253,231,36))
]
def interp(t):
t = max(0,min(1,t))
for i in range(len(stops)-1):
t0,c0 = stops[i]
t1,c1 = stops[i+1]
if t0 <= t <= t1:
f = (t - t0)/(t1-t0+1e-12)
c = tuple(int(c0[k] + f*(c1[k]-c0[k])) for k in range(3))
return 'rgb({},{},{})'.format(*c)
return 'rgb(0,0,0)'
return interp

============== 建立 Dash App ==============
app = Dash(name, external_stylesheets=[dbc.themes.FLATLY])
app.title = "顆粒 3D 形狀與分布分析"

全域暫存
global_store = {
"meshes": [],
"df": None
}

app.layout = dbc.Container(fluid=True, children=[
html.H3("顆粒 3D 形狀指標與分布分析工具"),
dbc.Row([
dbc.Col([
dcc.Upload(
id='upload-stl',
children=html.Div(['拖曳或點擊上傳 STL 檔']),
style={
'width':'100%','height':'80px','lineHeight':'80px',
'borderWidth':'2px','borderStyle':'dashed','borderRadius':'5px',
'textAlign':'center','margin':'10px'
},
multiple=False
),
dbc.Row([
dbc.Col([
html.Label("顆粒著色模式"),
dcc.Dropdown(
id='color-mode',
options=[
{'label':'單一顏色','value':'uniform'},
{'label':'密度 (KDE)','value':'density'},
{'label':'Sphericity','value':'sphericity'}
],
value='uniform'
)
], width=6),
dbc.Col([
html.Label("KDE Bandwidth (空白=自動)"),
dcc.Input(id='kde-bandwidth', type='number', min=0, step=0.0001, style={'width':'100%'}),
html.Small("自動使用 Scott 規則")
], width=6),
]),
html.Br(),
dbc.Button("重新計算密度 / g(r)", id='recompute-density', color='primary', size='sm'),
html.Hr(),
dcc.Graph(id='all-particles-fig', style={'height':'600px'}),
], width=7),

複製
    dbc.Col([        html.Div(id='selected-particle-info', style={'whiteSpace':'pre-line','fontSize':'14px'}),        dbc.Checklist(            id='single-display-options',            options=[                {'label':'擬合橢圓','value':'ellipsoid'},                {'label':'內切球','value':'inscribed'},                {'label':'外接球','value':'circumscribed'},                {'label':'主軸','value':'axes'}            ],
            value=['ellipsoid','inscribed','circumscribed','axes'],
            inline=True
        ),
        dcc.Graph(id='single-particle-fig', style={'height':'600px'}),
        html.Hr(),
        html.H5("g(r) / Ripley L 圖"),
        dcc.Graph(id='gr-fig', style={'height':'300px'}),
    ], width=5),
]),
html.Hr(),
html.H4("顆粒資料表"),
html.Div([    dbc.Button("匯出 CSV", id='export-csv', color='secondary', size='sm', style={'marginRight':'10px'}),    dbc.Button("匯出 Excel", id='export-excel', color='secondary', size='sm'),    dcc.Download(id='download-file')], style={'marginBottom':'10px'}),
dash_table.DataTable(
    id='particles-table',
    columns=[],
    data=[],
    page_size=10,
    filter_action='native',
    sort_action='native',
    row_selectable='single',
    style_table={'overflowX':'auto','maxHeight':'400px'},
    style_cell={'fontSize':'12px','padding':'4px'}
),
html.Br(),
html.Div(id='footer', children="狀態：等待上傳 STL...")
])

============== Callback 與互動邏輯 ==============
def parse_contents(contents, filename):
content_type, content_string = contents.split(',')
decoded = base64.b64decode(content_string)
if filename.lower().endswith('.stl'):
return decoded
else:
raise ValueError("只支援 STL 檔")

@app.callback(
Output('particles-table','data'),
Output('particles-table','columns'),
Output('all-particles-fig','figure'),
Output('footer','children'),
Input('upload-stl','contents'),
State('upload-stl','filename'),
State('color-mode','value'),
prevent_initial_call=True
)
def handle_upload(contents, filename, color_mode):
if contents is None:
return [], [], go.Figure(), "尚未上傳"
try:
file_bytes = parse_contents(contents, filename)
meshes = load_stl_and_split(file_bytes)
records = []
for idx, m in enumerate(meshes):
metrics = compute_particle_metrics(m)
metrics['ID'] = idx
# eigen向量存成字串方便 DataTable 顯示
for k in ['EigenVec1','EigenVec2','EigenVec3']:
metrics[k] = np.array2string(metrics[k], precision=5, separator=' ')
records.append(metrics)
df = pd.DataFrame(records).set_index('ID')

複製
    # 初次密度計算
    centers = df[['CentroidX','CentroidY','CentroidZ']].values
    dens_norm, _ = fit_kde_density(centers)
    df['DensityNorm'] = dens_norm

    global_store['meshes'] = meshes
    global_store['df'] = df

    columns = [{"name": c, "id": c} for c in df.columns]

    fig_all = generate_all_particles_fig(meshes, df, color_mode=color_mode)
    return df.reset_index().to_dict('records'), columns, fig_all, f"載入完成：{filename}，顆粒數={len(meshes)}"

except Exception as e:
    return [], [], go.Figure(), f"錯誤：{e}"
@app.callback(
Output('all-particles-fig','figure'),
Input('color-mode','value'),
Input('recompute-density','n_clicks'),
State('kde-bandwidth','value'),
prevent_initial_call=True
)
def update_all_fig(color_mode, n_clicks, bw):
df = global_store.get('df')
meshes = global_store.get('meshes')
if df is None or len(meshes)==0:
return go.Figure()
# 若點擊重新計算
triggered = [t['prop_id'] for t in callback_context.triggered]
if 'recompute-density.n_clicks' in triggered:
centers = df[['CentroidX','CentroidY','CentroidZ']].values
if bw is not None and bw>0:
dens_norm, _ = fit_kde_density(centers, bandwidth=bw)
else:
dens_norm, _ = fit_kde_density(centers)
df['DensityNorm'] = dens_norm
global_store['df'] = df
fig = generate_all_particles_fig(meshes, df, color_mode=color_mode)
return fig

@app.callback(
Output('single-particle-fig','figure'),
Output('selected-particle-info','children'),
Input('particles-table','selected_rows'),
Input('single-display-options','value'),
prevent_initial_call=True
)
def display_single(selected_rows, options):
df = global_store.get('df')
meshes = global_store.get('meshes')
if df is None or selected_rows is None or len(selected_rows)==0:
return go.Figure(), "尚未選擇顆粒"
row_index = selected_rows[0]
row = df.reset_index().iloc[row_index]
mesh = meshes[row.ID]
fig = generate_single_particle_fig(
mesh, row,
show_ellipsoid=('ellipsoid' in options),
show_inscribed=('inscribed' in options),
show_circumscribed=('circumscribed' in options),
show_axes=('axes' in options)
)
info = []
for k in [
'ID','Volume','SurfaceArea','ConvexHullVolume',
'Major_d_a','Intermediate_d_b','Minor_d_c',
'EI_b_over_a','FI_c_over_b',
'RoundnessIndex_IR','Sphericity','Convexity',
'InscribedSphere_r','CircumscribedSphere_r','DensityNorm'
]:
if k in row:
info.append(f"{k}: {row[k]}")
return fig, "\n".join(info)

@app.callback(
Output('gr-fig','figure'),
Input('recompute-density','n_clicks'),
prevent_initial_call=True
)
def update_gr(n):
df = global_store.get('df')
if df is None or len(df)==0:
return go.Figure()
centers = df[['CentroidX','CentroidY','CentroidZ']].values
pg = pair_correlation(centers, bins=25)
fig = go.Figure()
if pg is not None:
r, g = pg
fig.add_trace(go.Scatter(x=r, y=g, mode='lines+markers', name='g(r)'))
# Ripley L
r_values = np.linspace(0, (centers.max(axis=0)-centers.min(axis=0)).max()*0.4, 25)
RL = ripley_K_L(centers, r_values)
if RL is not None:
K,L = RL
fig.add_trace(go.Scatter(x=r_values, y=L - r_values, mode='lines', name='L(r)-r'))
fig.update_layout(title="Pair Correlation g(r) / Ripley L(r)-r", xaxis_title="r", yaxis_title="g(r) 或 L(r)-r")
return fig

@app.callback(
Output('download-file','data'),
Input('export-csv','n_clicks'),
Input('export-excel','n_clicks'),
prevent_initial_call=True
)
def export_data(n_csv, n_excel):
df = global_store.get('df')
if df is None:
return None
triggered = callback_context.triggered[0]['prop_id']
export_cols = [c for c in df.columns if not c.startswith('EigenVec')]
out_df = df[export_cols].reset_index()
if 'export-csv' in triggered:
return dcc.send_bytes(out_df.to_csv(index=False).encode('utf-8-sig'), "particles_metrics.csv")
else:
# Excel
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
out_df.to_excel(writer, sheet_name='Particles', index=False)
buffer.seek(0)
return dcc.send_bytes(buffer.read(), "particles_metrics.xlsx")

============== 主程式入口 ==============
if name == 'main':
app.run_server(debug=True)