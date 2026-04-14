import pandas as pd
import json
import mplsoccer
from mplsoccer import Sblocal
import os
import numpy as np
import warnings

# Ignorar warnings de pandas/pyarrow si es necesario
warnings.filterwarnings('ignore')

def process_data_and_create_shots_df(data_folder="data", output_folder="data_processed"):
    """
    Procesa los archivos de datos de StatsBomb, guarda los DataFrames individuales en formato parquet
    y genera un DataFrame consolidado de tiros con información 360.
    """
    
    # Crear estructura de carpetas de salida
    subfolders = ['events', 'related', 'freeze', 'tactics', 'matches']
    for sub in subfolders:
        os.makedirs(os.path.join(output_folder, sub), exist_ok=True)
        
    parser = Sblocal(dataframe=True)
    
    # 1. Cargar y guardar todos los partidos (matches)
    # Asumimos la ruta estándar de la competición (La Liga 2023/2024 es id 11, season 317 según contexto anterior)
    # Si no existe, intentamos buscar en la carpeta matches
    matches_path = os.path.join(data_folder, "matches/9/281.json")
    if not os.path.exists(matches_path):
        # Fallback: buscar cualquier json en matches/
        matches_dir = os.path.join(data_folder, "matches")
        # Lógica simple para encontrar el archivo de matches si la ruta exacta varía
        for root, dirs, files in os.walk(matches_dir):
            for file in files:
                if file.endswith(".json"):
                    matches_path = os.path.join(root, file)
                    break
    
    print(f"Cargando partidos desde: {matches_path}")
    df_matches = parser.match(matches_path)
    df_matches.to_parquet(os.path.join(output_folder, "matches", "all_matches.parquet"), index=False)
    
    all_shots_360_list = []
    
    # Iterar sobre cada partido
    # Usamos los IDs de los partidos encontrados en df_matches
    match_ids = df_matches['match_id'].unique()
    
    print(f"Procesando {len(match_ids)} partidos...")
    
    for match_id in match_ids:
        try:
            # Rutas de archivos
            event_file = os.path.join(data_folder, "events", f"{match_id}.json")
            three_sixty_file = os.path.join(data_folder, "three-sixty", f"{match_id}.json")
            
            # Verificar existencia de archivos esenciales
            if not os.path.exists(event_file):
                print(f"Falta archivo de eventos para match_id: {match_id}")
                continue
                
            # Cargar datos
            df_events, df_related, df_freeze, df_tactics = parser.event(event_file)
            
            # Función auxiliar para guardar parquet de forma segura
            def save_parquet_safe(df, path):
                try:
                    df.to_parquet(path, index=False)
                except Exception:
                    # Si falla, intentar convertir columnas object a string
                    df_str = df.copy()
                    for col in df_str.columns:
                        if df_str[col].dtype == 'object':
                            df_str[col] = df_str[col].astype(str)
                    df_str.to_parquet(path, index=False)

            # Guardar en parquet individualmente
            save_parquet_safe(df_events, os.path.join(output_folder, "events", f"{match_id}.parquet"))
            save_parquet_safe(df_related, os.path.join(output_folder, "related", f"{match_id}.parquet"))
            save_parquet_safe(df_freeze, os.path.join(output_folder, "freeze", f"{match_id}.parquet"))
            save_parquet_safe(df_tactics, os.path.join(output_folder, "tactics", f"{match_id}.parquet"))
            
            # === PROCESAMIENTO DE TIROS (SHOTS) ===
            
            # Filtrar solo tiros
            shots = df_events[df_events['type_name'] == 'Shot'].copy()
            
            if shots.empty:
                continue

            # Merge con freeze frame (datos 360)
            # df_freeze ya viene del parser.event, pero a veces es mejor usar el parser.frame para el 360 completo si es necesario.
            # Sin embargo, parser.event ya devuelve df_freeze vinculado a eventos.
            # Nota: Sblocal.event devuelve df_freeze que tiene 'id' del evento.
            
            # Asegurarnos de que df_freeze tiene las columnas necesarias
            # A veces df_freeze puede estar vacío si no hay datos 360 para los eventos
            if df_freeze.empty:
                # Si está vacío, creamos las columnas para que no falle el merge, o saltamos
                # Intentamos cargar desde three-sixty folder si el parser.event no trajo suficiente info (aunque debería)
                if os.path.exists(three_sixty_file):
                     df_360_full = parser.frame(three_sixty_file)
                     # Este df_360_full suele tener 'id' (event_uuid) o similar. En mplsoccer suele ser 'id' o 'event_uuid'.
                     # Ajustar si es necesario. Asumimos que df_freeze de parser.event es suficiente o usamos este.
                     # Para consistencia con el código del usuario que usa df_freeze_p1 (del parser.event), usaremos ese.
                     pass

            shots_freeze = shots.merge(
                df_freeze, 
                on='id', 
                how='left',
                suffixes=('', '_freeze')
            )
            
            # Si no hay datos de freeze para los tiros, saltamos la parte de agregación compleja o rellenamos con vacíos
            if 'teammate' not in shots_freeze.columns:
                # Crear columnas dummy si no existen para evitar error
                shots_freeze['teammate'] = False
                shots_freeze['x_freeze'] = np.nan
                shots_freeze['y_freeze'] = np.nan
                shots_freeze['position_name_freeze'] = ""
                shots_freeze['player_id'] = np.nan # Asumiendo que existe player_id en freeze
            
            # Asegurar que player_id existe en freeze (a veces se llama player_id, a veces actor_id, etc. en raw, pero mplsoccer lo estandariza)
            # En mplsoccer df_freeze suele tener: 'teammate', 'actor_player_id' o 'player_id', 'x', 'y' (renombrados a x_freeze, y_freeze tras merge si colisionan, pero aquí vienen de df_freeze directo)
            # Al hacer merge, las columnas de df_freeze se unen. df_freeze tiene 'x', 'y' normalmente. Al hacer merge con suffixes=('', '_freeze'), 'x' del evento se queda 'x', 'x' del freeze se vuelve 'x_freeze'.
            
            # Verificamos nombre de columna de ID de jugador en freeze
            # Normalmente es 'player_id' en df_freeze de mplsoccer.
            # Si colisiona con player_id del tirador (en shots), se convertirá en player_id_freeze.
            player_id_col = 'player_id_freeze' if 'player_id_freeze' in shots_freeze.columns else 'player_id'
            # Si no existe (raro), usaremos None
            
            # Función auxiliar para crear diccionarios
            def create_pos_dict(x_vals, y_vals, ids):
                return [{'x': x, 'y': y, 'player_id': pid} for x, y, pid in zip(x_vals, y_vals, ids)]

            # Agregación
            conteos = (
                shots_freeze
                .groupby("id")
                .agg(
                    # === Conteos ===
                    total_jugadores=('teammate', 'count'),
                    compañeros=('teammate', lambda x: (x == True).sum()),
                    rivales=('teammate', lambda x: (x == False).sum()),
                    porteros=('position_name_freeze', lambda x: (x == "Goalkeeper").sum()), # position_name en freeze suele ser position_name

                    # === POSICIONES (Lista de diccionarios) ===
                    posiciones_compañeros=('x_freeze', lambda x: [
                        {'x': x_val, 'y': y_val, 'player_id': pid_val} 
                        for x_val, y_val, team, pid_val in zip(
                            shots_freeze.loc[x.index, 'x_freeze'],
                            shots_freeze.loc[x.index, 'y_freeze'],
                            shots_freeze.loc[x.index, 'teammate'],
                            shots_freeze.loc[x.index, player_id_col] if player_id_col in shots_freeze.columns else [None]*len(x)
                        ) if team
                    ]),
                    posiciones_rivales=('x_freeze', lambda x: [
                        {'x': x_val, 'y': y_val, 'player_id': pid_val} 
                        for x_val, y_val, team, pid_val in zip(
                            shots_freeze.loc[x.index, 'x_freeze'],
                            shots_freeze.loc[x.index, 'y_freeze'],
                            shots_freeze.loc[x.index, 'teammate'],
                            shots_freeze.loc[x.index, player_id_col] if player_id_col in shots_freeze.columns else [None]*len(x)
                        ) if not team
                    ]),
                    posiciones_porteros=('x_freeze', lambda x: [
                        {'x': x_val, 'y': y_val, 'player_id': pid_val} 
                        for x_val, y_val, pos, pid_val in zip(
                            shots_freeze.loc[x.index, 'x_freeze'],
                            shots_freeze.loc[x.index, 'y_freeze'],
                            shots_freeze.loc[x.index, 'position_name_freeze'],
                            shots_freeze.loc[x.index, player_id_col] if player_id_col in shots_freeze.columns else [None]*len(x)
                        ) if pos == "Goalkeeper"
                    ])
                )
                .reset_index()
            )
            
            # Merge final para este partido
            shots_360 = shots.merge(conteos, on="id", how="left")
            
            # Añadir match_id
            shots_360['match_id'] = match_id

            # Columna gol
            shots_360['is_goal'] = (shots_360['outcome_name'] == "Goal").astype(int)
            
            # Selección de columnas (asegurando que existan)
            cols_to_keep = [
                'match_id', 'id', 'period', 'timestamp', 'obv_for_net', 'possession_team_name', 'play_pattern_name', 
                'team_name', 'player_name', 'position_name', 'sub_type_name', 
                'x', 'y', 'z', 
                'end_x', 'end_y', 'end_z',
                'outcome_name', 'is_goal', 'shot_statsbomb_xg', 'shot_shot_execution_xg', 
                'shot_shot_execution_xg_uplift','shot_gk_save_difficulty_xg', 
                'shot_gk_positioning_xg_suppression', 'shot_gk_shot_stopping_xg_suppression',
                'body_part_name', 'under_pressure', 'total_jugadores', 'compañeros', 'rivales', 
                'porteros', 'posiciones_compañeros', 'posiciones_rivales', 'posiciones_porteros'
            ]
            
            # Filtrar solo columnas que existen en el dataframe
            existing_cols = [c for c in cols_to_keep if c in shots_360.columns]
            shots_360_limpio = shots_360[existing_cols].copy()
            
            # Rellenar nulos (cuidado con las columnas de listas, fillna podría afectarlas si son NaN, pero agg devuelve listas vacías o listas)
            # Las columnas numéricas o de texto pueden rellenarse.
            # shots_360_limpio = shots_360_limpio.fillna("") # Esto convierte NaN a string vacío, cuidado con columnas numéricas.
            
            all_shots_360_list.append(shots_360_limpio)
            
        except Exception as e:
            print(f"Error procesando partido {match_id}: {e}")
            continue

    # Concatenar todo
    if all_shots_360_list:
        all_shots_df_frames = pd.concat(all_shots_360_list, ignore_index=True)
        
        # Guardar resultado final
        output_path = os.path.join(output_folder, "all_shots_360_processed.parquet")
        
        # Convertir columnas complejas a JSON string para evitar problemas con Parquet
        cols_complex = ['posiciones_compañeros', 'posiciones_rivales', 'posiciones_porteros']
        for col in cols_complex:
            if col in all_shots_df_frames.columns:
                all_shots_df_frames[col] = all_shots_df_frames[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)

        try:
            all_shots_df_frames.to_parquet(output_path, index=False)
            print(f"DataFrame consolidado guardado en: {output_path}")
        except Exception as e:
            print(f"Error guardando parquet: {e}")
            # Intentar guardar convirtiendo todo a string como fallback
            try:
                df_str = all_shots_df_frames.copy()
                for col in df_str.columns:
                    if df_str[col].dtype == 'object':
                        df_str[col] = df_str[col].astype(str)
                df_str.to_parquet(output_path, index=False)
                print(f"DataFrame consolidado guardado (convertido a string) en: {output_path}")
            except Exception as e2:
                print(f"Error final guardando parquet: {e2}")
    else:
        print("No se encontraron tiros para procesar.")

if __name__ == "__main__":
    process_data_and_create_shots_df()
