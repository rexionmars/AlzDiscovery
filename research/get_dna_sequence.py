# Importação de bibliotecas necessárias
import os
import pandas as pd
import numpy as np
import requests
import urllib.request
import gzip
import shutil
from Bio import Entrez, SeqIO
import GEOparse

# Configuração para acesso às bases de dados NCBI
Entrez.email = "leoanrdimelo43@gmail.com"  # Substitua pelo seu email

# Diretório para salvar os dados
base_dir = "alzheimer_mouse_data"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Função para buscar estudos relacionados a modelos murinos de Alzheimer no GEO
def search_geo_datasets(query, max_results=10):
    """
    Busca datasets no GEO relacionados à query
    """
    print(f"Buscando datasets relacionados a: {query}")
    handle = Entrez.esearch(db="gds", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    
    return record["IdList"]

# Função para obter informações detalhadas sobre um dataset GEO
def get_geo_dataset_info(gds_id):
    """
    Obtém informações detalhadas sobre um dataset GEO
    """
    handle = Entrez.esummary(db="gds", id=gds_id)
    record = Entrez.read(handle)
    handle.close()
    
    return record[0]

# Função para baixar e processar um dataset GEO
def download_geo_dataset(gse_id, save_dir):
    """
    Baixa e processa um dataset GEO
    """
    try:
        print(f"Baixando dataset GSE{gse_id}...")
        gse = GEOparse.get_GEO(geo=gse_id, destdir=save_dir)
        
        # Salvando metadados
        metadata_file = os.path.join(save_dir, f"{gse_id}_metadata.csv")
        gse.phenotype_data.to_csv(metadata_file)
        
        print(f"Metadados salvos em: {metadata_file}")
        return gse
    except Exception as e:
        print(f"Erro ao baixar GSE{gse_id}: {e}")
        return None

# Função para baixar dados de sequência do SRA
def download_sra_data(sra_id, save_dir):
    """
    Baixa dados de sequenciamento do SRA
    """
    try:
        print(f"Baixando dados SRA: {sra_id}...")
        # Esta é uma versão simplificada; para projetos reais, considere usar ferramentas como SRA Toolkit
        url = f"https://trace.ncbi.nlm.nih.gov/Traces/sra-reads-be/fastq?acc={sra_id}"
        save_path = os.path.join(save_dir, f"{sra_id}.fastq.gz")
        
        # Baixa o arquivo
        urllib.request.urlretrieve(url, save_path)
        
        # Descompacta o arquivo
        with gzip.open(save_path, 'rb') as f_in:
            with open(save_path[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"Dados SRA salvos em: {save_path[:-3]}")
        return save_path[:-3]
    except Exception as e:
        print(f"Erro ao baixar dados SRA {sra_id}: {e}")
        return None

# Função para baixar dados de modelos específicos de Alzheimer
def download_alzheimer_mouse_model_data(model_name, save_dir):
    """
    Busca e baixa dados específicos de um modelo murino de Alzheimer
    """
    query = f"{model_name}[Title] AND mouse[Organism] AND Alzheimer[Title] AND RNA-seq[Method]"
    gds_list = search_geo_datasets(query)
    
    model_dir = os.path.join(save_dir, model_name.replace("/", "_"))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    datasets = []
    
    for gds_id in gds_list:
        info = get_geo_dataset_info(gds_id)
        gse_id = info.get("GSE", "").replace("GSE", "")
        
        if gse_id:
            print(f"Processando {info['title']} (GSE{gse_id})")
            gse = download_geo_dataset(gse_id, model_dir)
            if gse:
                datasets.append(gse)
    
    return datasets

# Função para processar e consolidar os dados baixados
def process_alzheimer_data(datasets, save_dir):
    """
    Processa e consolida os dados de diferentes datasets
    """
    all_data = []
    metadata = []
    
    for gse in datasets:
        # Processamento básico dos dados de expressão
        for gpl in gse.gpls:
            expression_data = gse.gse_gpl_data[gpl].pivot_samples("VALUE")
            all_data.append(expression_data)
            
            # Adiciona metadados
            meta = gse.phenotype_data.copy()
            meta["dataset_id"] = gse.name
            meta["platform"] = gpl
            metadata.append(meta)
    
    # Consolida os dados (isso é simplificado e precisaria ser adaptado ao seu caso específico)
    if all_data:
        # Tenta combinar dados de diferentes plataformas (simplificado)
        # Na prática, você precisa de métodos mais sofisticados para normalização entre plataformas
        combined_data = pd.concat(all_data, axis=1)
        combined_meta = pd.concat(metadata)
        
        # Salva os dados consolidados
        combined_data.to_csv(os.path.join(save_dir, "combined_expression_data.csv"))
        combined_meta.to_csv(os.path.join(save_dir, "combined_metadata.csv"))
        
        print(f"Dados consolidados salvos em {save_dir}")
        return combined_data, combined_meta
    else:
        print("Nenhum dado encontrado para processar")
        return None, None

# Função principal
def main():
    print("Iniciando coleta de dados genômicos para modelos murinos de Alzheimer...")
    
    # Modelos murinos de Alzheimer a serem buscados
    models = ["APP/PS1", "5xFAD", "3xTg-AD", "Tg2576", "APPNL-G-F"]
    
    all_datasets = []
    
    # Baixa dados para cada modelo
    for model in models:
        print(f"\n{'='*50}")
        print(f"Buscando dados para o modelo {model}")
        print(f"{'='*50}")
        
        datasets = download_alzheimer_mouse_model_data(model, base_dir)
        all_datasets.extend(datasets)
    
    # Processa todos os dados
    if all_datasets:
        expression_data, metadata = process_alzheimer_data(all_datasets, base_dir)
        
        print("\nResumo dos dados coletados:")
        print(f"Total de datasets: {len(all_datasets)}")
        if expression_data is not None:
            print(f"Número de genes: {len(expression_data)}")
            print(f"Número de amostras: {expression_data.shape[1]}")
    else:
        print("Nenhum dataset encontrado para os modelos especificados.")
    
    print("\nColeta de dados concluída!")

if __name__ == "__main__":
    main()