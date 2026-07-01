import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import poisson
from scipy.optimize import minimize
import io
import base64
import unicodedata
import difflib
import xgboost as xgb
from Pronosticador.models import Partido

print("¡Ecosistema de predicción cargado!")

EQUIPOS_MUNDIAL_2026 = {
    "Canada", "Qatar", "Switzerland", "United States", "Australia", "Turkey",
    "Mexico", "South Korea", "Czechia", "Germany", "Ivory Coast", "Ecuador",
    "Saudi Arabia", "Spain", "Cape Verde", "Algeria", "Argentina", "Jordan",
    "Austria", "Belgium", "Iran", "New Zealand", "Bosnia and Herzegovina", "Brazil",
    "Scotland", "South Africa", "Colombia", "Uzbekistan", "DR Congo", "Croatia",
    "England", "Panama", "Curacao", "Egypt", "Morocco", "France", "Iraq",
    "Norway", "Senegal", "Ghana", "Haiti", "Japan", "Tunisia", "Netherlands",
    "Sweden", "Paraguay", "Portugal", "Uruguay"
}

IMPORTANCIA_TORNEO = {
    'FIFA World Cup': 3.0,
    'FIFA World Cup qualification': 2.0,
    'UEFA Euro': 2.5,
    'UEFA Euro qualification': 1.8,
    'Copa América': 2.5,
    'UEFA Nations League': 1.5,
    'African Cup of Nations': 2.0,
    'African Cup of Nations qualification': 1.5,
    'CONCACAF Nations League': 1.3,
    'Friendly': 0.5,
}
IMPORTANCIA_DEFAULT = 1.0

def normalize_name(name):
    return ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn').lower().strip()

def get_best_match(input_name, available_names):
    norm_input = normalize_name(input_name)
    for name in available_names:
        if normalize_name(name) == norm_input:
            return name
    matches = difflib.get_close_matches(input_name, available_names, n=1, cutoff=0.6)
    return matches[0] if matches else None

def _cargar_df_partidos():
    partidos_qs = Partido.objects.filter(equipo_local__nombre__in=EQUIPOS_MUNDIAL_2026, equipo_visitante__nombre__in=EQUIPOS_MUNDIAL_2026).values('fecha', 'equipo_local__nombre', 'equipo_visitante__nombre', 'goles_local', 'goles_visitante', 'neutral', 'torneo')
    df = pd.DataFrame(list(partidos_qs))
    df.rename(columns={'equipo_local__nombre': 'local', 'equipo_visitante__nombre': 'visitante', 'goles_local': 'goles_local', 'goles_visitante': 'goles_visita'}, inplace=True)
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df

def _calcular_pesos_temporales(df, xi=0.004):
    hoy = pd.Timestamp.today()
    peso_tiempo = np.exp(-xi * (hoy - df['fecha']).dt.days.astype(float))
    peso_torneo = df['torneo'].map(lambda t: IMPORTANCIA_TORNEO.get(t, IMPORTANCIA_DEFAULT))
    return peso_tiempo * peso_torneo

def _preparar_arrays(df, equipos):
    idx = {equipo: i for i, equipo in enumerate(equipos)}
    idx_local = np.array([idx[e] for e in df['local']], dtype=np.int32)
    idx_visitante = np.array([idx[e] for e in df['visitante']], dtype=np.int32)
    return idx_local, idx_visitante, df['goles_local'].to_numpy(dtype=np.int32), df['goles_visita'].to_numpy(dtype=np.int32), df['neutral'].to_numpy(dtype=np.int32)

def _log_verosimilitud_dixon_coles(params, n, idx_l, idx_v, gl, gv, neutral, pesos):
    from scipy.special import gammaln
    alphas, betas = params[:n], params[n:2*n]
    gamma, rho = params[2*n], params[2*n + 1]
    log_mu = alphas[idx_l] - betas[idx_v] + (gamma * (1 - neutral))
    log_la = alphas[idx_v] - betas[idx_l]
    mu, la = np.exp(log_mu), np.exp(log_la)
    log_pois_l = gl * log_mu - mu - gammaln(gl + 1)
    log_pois_v = gv * log_la - la - gammaln(gv + 1)
    tau = np.ones(len(gl))
    tau = np.where((gl==0)&(gv==0), 1-mu*la*rho, np.where((gl==1)&(gv==0), 1+la*rho, np.where((gl==0)&(gv==1), 1+mu*rho, np.where((gl==1)&(gv==1), 1-rho, 1.0))))
    return -np.sum(pesos[tau>0] * (np.log(tau[tau>0]) + log_pois_l[tau>0] + log_pois_v[tau>0]))

def _tau_dc(gl, gv, mu, la, rho):
    if gl==0 and gv==0: return 1-mu*la*rho
    if gl==1 and gv==0: return 1+la*rho
    if gl==0 and gv==1: return 1+mu*rho
    if gl==1 and gv==1: return 1-rho
    return 1.0

def calcularFuerzasEquipos(min_partidos=15, xi=0.004):
    df = _cargar_df_partidos()
    conteo = pd.concat([df['local'], df['visitante']]).value_counts()
    equipos_validos = sorted(conteo[conteo >= min_partidos].index.tolist())
    df = df[df['local'].isin(equipos_validos) & df['visitante'].isin(equipos_validos)]
    n = len(equipos_validos)
    idx_l, idx_v, gl, gv, neutral = _preparar_arrays(df, equipos_validos)
    res = minimize(fun=_log_verosimilitud_dixon_coles, x0=np.zeros(2*n+2), args=(n, idx_l, idx_v, gl, gv, neutral, _calcular_pesos_temporales(df, xi=xi).to_numpy()), method='L-BFGS-B')
    params_dict = {'alphas': dict(zip(equipos_validos, res.x[:n])), 'betas': dict(zip(equipos_validos, res.x[n:2*n])), 'rho': res.x[2*n+1]}
    X, y = [], []
    for _, row in df.iterrows():
        loc, vis = row['local'], row['visitante']
        X.append([params_dict['alphas'][loc], params_dict['betas'][loc], params_dict['alphas'][vis], params_dict['betas'][vis], (1 if row['neutral']==0 else 0)])
        y.append(row['goles_local'])
        X.append([params_dict['alphas'][vis], params_dict['betas'][vis], params_dict['alphas'][loc], params_dict['betas'][loc], 0])
        y.append(row['goles_visita'])
    model = xgb.XGBRegressor(
        objective='count:poisson',
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ).fit(np.array(X), np.array(y))
    params_dict['xgb_model'] = model
    return None, params_dict

def predecir_partido(equipo_local, equipo_visitante, params_dict, max_goles=5, es_neutral=False):
    loc_db = get_best_match(equipo_local, params_dict['alphas'].keys())
    vis_db = get_best_match(equipo_visitante, params_dict['alphas'].keys())
    if not loc_db or not vis_db: raise ValueError(f"No pude encontrar '{equipo_local}' o '{equipo_visitante}'")

    ventaja_local = 0 if es_neutral else 1
    xgb_model = params_dict['xgb_model']
    alphas, betas = params_dict['alphas'], params_dict['betas']
    xg_local = float(xgb_model.predict(np.array([[alphas[loc_db], betas[loc_db], alphas[vis_db], betas[vis_db], ventaja_local]]))[0])
    xg_visita = float(xgb_model.predict(np.array([[alphas[vis_db], betas[vis_db], alphas[loc_db], betas[loc_db], 0]]))[0])

    matriz = np.zeros((max_goles+1, max_goles+1))
    for gl in range(max_goles+1):
        for gv in range(max_goles+1):
            tau = _tau_dc(gl, gv, xg_local, xg_visita, params_dict['rho'])
            matriz[gl][gv] = max(tau * poisson.pmf(gl, xg_local) * poisson.pmf(gv, xg_visita), 0)
    matriz = (matriz / matriz.sum()) * 100

    return {
        'matriz': matriz, 'xg_local': round(xg_local, 2), 'xg_visita': round(xg_visita, 2),
        'prob_local': round(float(np.sum(np.tril(matriz, k=-1))), 1), 'prob_empate': round(float(np.sum(np.diag(matriz))), 1),
        'prob_visita': round(float(np.sum(np.triu(matriz, k=1))), 1), 'marcador_probable': (int(np.unravel_index(np.argmax(matriz), matriz.shape)[0]), int(np.unravel_index(np.argmax(matriz), matriz.shape)[1])),
        'equipo_local': loc_db, 'equipo_visitante': vis_db
    }

def generar_tres_paneles_base64(resultado):
    local, visitante = resultado['equipo_local'], resultado['equipo_visitante']
    xg_l, xg_v, matriz = resultado['xg_local'], resultado['xg_visita'], resultado['matriz']
    p_l, p_e, p_v = resultado['prob_local'], resultado['prob_empate'], resultado['prob_visita']

    # ── Colores del tema oscuro ──────────────────────────────────────────────
    BG_BASE    = '#080C12'
    BG_PANEL   = '#0E1420'
    BG_AXES    = '#0E1420'
    COL_TEXT   = '#F0F4FF'
    COL_MUTED  = '#6B7A99'
    COL_EDGE   = '#1E2A40'
    COL_LOCAL  = '#34D399'   # verde  → equipo local
    COL_DRAW   = '#94A3B8'   # gris   → empate
    COL_VISIT  = '#60A5FA'   # azul   → visitante
    COL_TITLE  = '#38BDF8'   # cian   → títulos de panel

    # ── rcParams ANTES de crear la figura ───────────────────────────────────
    plt.rcParams.update({
        'figure.facecolor':  BG_BASE,
        'axes.facecolor':    BG_AXES,
        'axes.edgecolor':    COL_EDGE,
        'axes.labelcolor':   COL_TEXT,
        'axes.titlecolor':   COL_TITLE,
        'axes.titlesize':    11,
        'axes.titleweight':  'bold',
        'text.color':        COL_TEXT,
        'xtick.color':       COL_MUTED,
        'ytick.color':       COL_MUTED,
        'xtick.labelsize':   8,
        'ytick.labelsize':   8,
        'grid.color':        COL_EDGE,
        'grid.linewidth':    0.5,
    })

    fig = plt.figure(figsize=(16, 5), facecolor=BG_BASE)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # ── Panel 1: Probabilidades ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG_AXES)

    etiquetas = [local, 'Empate', visitante]
    valores   = [p_l, p_e, p_v]
    colores   = [COL_LOCAL, COL_DRAW, COL_VISIT]

    bars = ax1.bar(etiquetas, valores, color=colores, width=0.5,
                   edgecolor=BG_BASE, linewidth=1.2)

    for bar in bars:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f'{bar.get_height():.1f}%',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=COL_TEXT
        )

    ax1.set_title('Probabilidad', pad=10)
    ax1.set_ylim(0, max(valores) + 12)
    ax1.tick_params(axis='x', labelsize=8, colors=COL_MUTED)
    ax1.tick_params(axis='y', labelsize=8, colors=COL_MUTED)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.spines[['left', 'bottom']].set_color(COL_EDGE)

    # ── Panel 2: Top 10 marcadores ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG_AXES)

    idx_flat = np.argsort(matriz.flatten())[::-1][:10]
    scores, probs, bar_colors = [], [], []
    for i in idx_flat:
        gl, gv = np.unravel_index(i, matriz.shape)
        scores.append(f"{local[:3].upper()} {gl}–{gv} {visitante[:3].upper()}")
        probs.append(matriz[gl, gv])
        bar_colors.append(COL_LOCAL if gl > gv else (COL_VISIT if gl < gv else COL_DRAW))

    ax2.barh(scores[::-1], probs[::-1], color=bar_colors[::-1],
             edgecolor=BG_BASE, linewidth=0.8, height=0.65)

    ax2.set_title('Top 10 marcadores', pad=10)
    ax2.tick_params(axis='y', labelsize=7.5)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.spines[['left', 'bottom']].set_color(COL_EDGE)

    # ── Panel 3: Heatmap de marcadores ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(BG_AXES)

    sns.heatmap(
        matriz, ax=ax3,
        annot=True, fmt='.1f',
        cmap='Blues', cbar=False,
        linewidths=0.4, linecolor=BG_BASE,
        annot_kws={'size': 8, 'fontweight': 'bold', 'color': '#000000'},
    )

    ax3.set_title('Matriz de marcadores', pad=10)
    ax3.set_xlabel(f'Goles {visitante}', labelpad=6, fontsize=8)
    ax3.set_ylabel(f'Goles {local}',     labelpad=6, fontsize=8)
    ax3.tick_params(colors=COL_MUTED)

    # ── Título general ────────────────────────────────────────────────────────
    fig.suptitle(
        f'Predicción: {local} vs {visitante}',
        fontsize=15, fontweight='bold', color=COL_TEXT, y=1.04
    )

    # ── Exportar a base64 ─────────────────────────────────────────────────────
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=130,
                facecolor=BG_BASE, edgecolor='none')
    img = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close('all')
    return img

def pronosticar(equipo_local, equipo_visitante, es_neutral=False):
    _, params = calcularFuerzasEquipos()
    res = predecir_partido(equipo_local, equipo_visitante, params, es_neutral=es_neutral)
    res['imagen_b64'] = generar_tres_paneles_base64(res)
    return res