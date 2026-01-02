"""
Script com melhorias críticas para o projeto D4Maia
Funções auxiliares para implementar as correções identificadas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy import stats


def plot_k_distance(X, k=4, title="K-Distance Plot para DBSCAN"):
    """
    Cria k-distance plot para escolha rigorosa de eps no DBSCAN.
    
    Args:
        X: Dados normalizados
        k: Número de vizinhos mais próximos (geralmente min_samples-1)
        title: Título do gráfico
    
    Returns:
        distances: Distâncias ordenadas
    """
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Distância ao k-ésimo vizinho mais próximo (excluir o próprio ponto)
    k_distances = distances[:, k]
    k_distances_sorted = np.sort(k_distances)[::-1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(k_distances_sorted)), k_distances_sorted, 'b-', linewidth=2)
    ax.set_xlabel('Pontos ordenados por distância', fontsize=12)
    ax.set_ylabel(f'Distância ao {k}-ésimo vizinho mais próximo', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Marcar "cotovelo" sugerido
    # Estratégia: encontrar ponto onde a curva muda de inclinação
    if len(k_distances_sorted) > 10:
        # Calcular derivada (diferença entre pontos consecutivos)
        diff = np.diff(k_distances_sorted)
        # Encontrar ponto de maior mudança (máximo da segunda derivada)
        diff2 = np.diff(diff)
        if len(diff2) > 0:
            elbow_idx = np.argmax(diff2) if len(diff2) > 0 else len(k_distances_sorted) // 4
            suggested_eps = k_distances_sorted[elbow_idx]
            ax.axhline(y=suggested_eps, color='r', linestyle='--', 
                      label=f'eps sugerido ≈ {suggested_eps:.3f}')
            ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return k_distances_sorted


def kmeans_with_balancing(X, k_range, scaler=None, min_cluster_size_pct=0.05, 
                         n_init=20, random_state=42):
    """
    K-Means com garantia de balanceamento mínimo dos clusters.
    
    Args:
        X: Dados (normalizados ou não)
        k_range: Range de k a testar
        scaler: Scaler usado (para logging)
        min_cluster_size_pct: Percentagem mínima de pontos por cluster
        n_init: Número de inicializações
        random_state: Seed
    
    Returns:
        results: DataFrame com resultados para cada k
        best_k: Melhor k escolhido
        best_model: Melhor modelo KMeans
        best_labels: Labels do melhor modelo
    """
    results = []
    
    for k in k_range:
        if k > len(X):
            continue
            
        kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=random_state, max_iter=500)
        labels = kmeans.fit_predict(X)
        
        # Verificar balanceamento
        cluster_sizes = pd.Series(labels).value_counts()
        min_size = cluster_sizes.min()
        min_size_pct = min_size / len(X)
        
        # Calcular métricas apenas se clusters estão balanceados
        if min_size_pct >= min_cluster_size_pct and len(set(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)
            davies = davies_bouldin_score(X, labels)
            inertia = kmeans.inertia_
            
            # Score combinado (priorizar silhouette)
            combined_score = 0.5 * silhouette + 0.3 * (calinski / 1000) - 0.2 * (davies / 10)
        else:
            silhouette = np.nan
            calinski = np.nan
            davies = np.nan
            inertia = kmeans.inertia_
            combined_score = -np.inf
        
        results.append({
            'k': k,
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies,
            'inertia': inertia,
            'min_cluster_size': min_size,
            'min_cluster_size_pct': min_size_pct,
            'combined_score': combined_score,
            'balanced': min_size_pct >= min_cluster_size_pct
        })
    
    results_df = pd.DataFrame(results)
    
    # Escolher melhor k (apenas entre os balanceados)
    valid_results = results_df[results_df['balanced'] == True]
    
    if len(valid_results) > 0:
        best_idx = valid_results['combined_score'].idxmax()
        best_k = int(valid_results.loc[best_idx, 'k'])
        
        # Treinar modelo final
        best_model = KMeans(n_clusters=best_k, n_init=n_init, random_state=random_state, max_iter=500)
        best_labels = best_model.fit_predict(X)
    else:
        # Fallback: usar k médio do range
        best_k = int(np.median(list(k_range)))
        best_model = KMeans(n_clusters=best_k, n_init=n_init, random_state=random_state, max_iter=500)
        best_labels = best_model.fit_predict(X)
        print(f"⚠️  AVISO: Nenhum k balanceado encontrado. Usando k={best_k} como fallback.")
    
    return results_df, best_k, best_model, best_labels


def detect_outliers_multiple_methods(df, feature_cols, methods=['iqr', 'zscore', 'isolation']):
    """
    Detecta outliers usando múltiplos métodos.
    
    Args:
        df: DataFrame com dados
        feature_cols: Lista de colunas a analisar
        methods: Lista de métodos a usar ('iqr', 'zscore', 'isolation')
    
    Returns:
        results: DataFrame com resultados por método
        combined_mask: Máscara combinada (True = outlier em pelo menos um método)
    """
    results = []
    masks = {}
    
    # IQR Method
    if 'iqr' in methods:
        iqr_mask = pd.Series([False] * len(df), index=df.index)
        for col in feature_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                col_mask = (df[col] < lower) | (df[col] > upper)
                iqr_mask = iqr_mask | col_mask
        
        n_outliers_iqr = iqr_mask.sum()
        masks['iqr'] = iqr_mask
        results.append({
            'method': 'IQR',
            'n_outliers': n_outliers_iqr,
            'pct_outliers': n_outliers_iqr / len(df) * 100
        })
    
    # Z-Score Method
    if 'zscore' in methods:
        zscore_mask = pd.Series([False] * len(df), index=df.index)
        for col in feature_cols:
            if col in df.columns:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                col_mask = pd.Series([False] * len(df), index=df.index)
                col_mask.loc[df[col].dropna().index] = z_scores > 3
                zscore_mask = zscore_mask | col_mask
        
        n_outliers_zscore = zscore_mask.sum()
        masks['zscore'] = zscore_mask
        results.append({
            'method': 'Z-Score (|z|>3)',
            'n_outliers': n_outliers_zscore,
            'pct_outliers': n_outliers_zscore / len(df) * 100
        })
    
    # Isolation Forest (se disponível)
    if 'isolation' in methods:
        try:
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_labels = iso_forest.fit_predict(df[feature_cols].fillna(0))
            isolation_mask = pd.Series(iso_labels == -1, index=df.index)
            
            n_outliers_iso = isolation_mask.sum()
            masks['isolation'] = isolation_mask
            results.append({
                'method': 'Isolation Forest',
                'n_outliers': n_outliers_iso,
                'pct_outliers': n_outliers_iso / len(df) * 100
            })
        except ImportError:
            print("⚠️  Isolation Forest não disponível. Pulando método.")
    
    # Máscara combinada (outlier em pelo menos um método)
    if masks:
        combined_mask = pd.Series([False] * len(df), index=df.index)
        for mask in masks.values():
            combined_mask = combined_mask | mask
    else:
        combined_mask = pd.Series([False] * len(df), index=df.index)
    
    results_df = pd.DataFrame(results)
    
    return results_df, combined_mask, masks


def create_simple_vs_advanced_features(df_series, cpe_col='CPE', target_col='PotActiva'):
    """
    Cria versões simples e avançadas de features para comparação.
    
    Args:
        df_series: DataFrame com séries temporais
        cpe_col: Nome da coluna CPE
        target_col: Nome da coluna alvo
    
    Returns:
        features_simple: DataFrame com features simples (10-15 features básicas)
        features_advanced: DataFrame com features avançadas (50+ features)
    """
    features_simple_list = []
    features_advanced_list = []
    
    for cpe in df_series[cpe_col].unique():
        df_cpe = df_series[df_series[cpe_col] == cpe].copy()
        
        if len(df_cpe) == 0:
            continue
        
        # FEATURES SIMPLES (10-15 básicas)
        simple_features = {
            'CPE': cpe,
            'consumo_mean': df_cpe[target_col].mean(),
            'consumo_std': df_cpe[target_col].std(),
            'consumo_min': df_cpe[target_col].min(),
            'consumo_max': df_cpe[target_col].max(),
            'consumo_median': df_cpe[target_col].median(),
            'consumo_q25': df_cpe[target_col].quantile(0.25),
            'consumo_q75': df_cpe[target_col].quantile(0.75),
            'consumo_cv': df_cpe[target_col].std() / df_cpe[target_col].mean() if df_cpe[target_col].mean() > 0 else 0,
        }
        
        # Adicionar features temporais básicas
        if 'tstamp' in df_cpe.columns:
            df_cpe['hour'] = pd.to_datetime(df_cpe['tstamp']).dt.hour
            df_cpe['day_of_week'] = pd.to_datetime(df_cpe['tstamp']).dt.dayofweek
            df_cpe['is_weekend'] = df_cpe['day_of_week'].isin([5, 6])
            
            simple_features['hora_pico'] = df_cpe.groupby('hour')[target_col].mean().idxmax()
            simple_features['consumo_weekend'] = df_cpe[df_cpe['is_weekend']][target_col].mean()
            simple_features['consumo_weekday'] = df_cpe[~df_cpe['is_weekend']][target_col].mean()
            simple_features['racio_weekend_util'] = simple_features['consumo_weekend'] / simple_features['consumo_weekday'] if simple_features['consumo_weekday'] > 0 else 0
        
        features_simple_list.append(simple_features)
        
        # FEATURES AVANÇADAS (50+ features com autocorrelação, etc.)
        # Copiar features simples
        advanced_features = simple_features.copy()
        
        # Adicionar features avançadas
        if 'tstamp' in df_cpe.columns:
            # Features por hora (24 valores)
            for h in range(24):
                advanced_features[f'consumo_h{h:02d}'] = df_cpe[df_cpe['hour'] == h][target_col].mean()
            
            # Features por dia da semana (7 valores)
            for d in range(7):
                advanced_features[f'consumo_dia_{d}'] = df_cpe[df_cpe['day_of_week'] == d][target_col].mean()
            
            # Autocorrelação (lag 96 = 1 dia, lag 672 = 1 semana)
            if len(df_cpe) > 672:
                series = df_cpe[target_col].values
                lag_96 = np.corrcoef(series[:-96], series[96:])[0, 1] if len(series) > 96 else 0
                lag_672 = np.corrcoef(series[:-672], series[672:])[0, 1] if len(series) > 672 else 0
                advanced_features['autocorr_lag_96'] = lag_96
                advanced_features['autocorr_lag_672'] = lag_672
            
            # Load factor (média / máximo)
            advanced_features['load_factor'] = df_cpe[target_col].mean() / df_cpe[target_col].max() if df_cpe[target_col].max() > 0 else 0
            
            # Variabilidade intra-dia
            daily_means = df_cpe.groupby(df_cpe['tstamp'].dt.date)[target_col].mean()
            advanced_features['variabilidade_inter_dia'] = daily_means.std() if len(daily_means) > 1 else 0
        
        features_advanced_list.append(advanced_features)
    
    features_simple = pd.DataFrame(features_simple_list)
    features_advanced = pd.DataFrame(features_advanced_list)
    
    return features_simple, features_advanced


def add_cluster_features_to_supervised(df_supervised, df_clusters, cluster_cols=['cluster_kmeans', 'cluster_dbscan']):
    """
    Adiciona clusters como features ao dataset supervisionado.
    
    Args:
        df_supervised: DataFrame supervisionado
        df_clusters: DataFrame com clusters por CPE
        cluster_cols: Colunas de clusters a adicionar
    
    Returns:
        df_supervised_enriched: DataFrame com clusters como features
    """
    df_enriched = df_supervised.copy()
    
    # Merge com clusters
    for col in cluster_cols:
        if col in df_clusters.columns:
            df_enriched = df_enriched.merge(
                df_clusters[['CPE', col]], 
                on='CPE', 
                how='left'
            )
            # Preencher nulos com -1 (categoria "sem cluster")
            df_enriched[col] = df_enriched[col].fillna(-1).astype(int)
    
    return df_enriched

