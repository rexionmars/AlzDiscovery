import os
import pandas as pd
import numpy as np
import requests
import gzip
import shutil
import tarfile
import urllib.request
import re
import time
from ftplib import FTP

# Diretório base para armazenar os dados
BASE_DIR = "alzheimer_mouse_data"

# Função para criar diretórios
def create_directories(model_list):
    """Cria os diretórios para cada modelo"""
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    
    for model in model_list:
        model_dir = os.path.join(BASE_DIR, model.replace("/", "_"))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    return True

# Função para download direto do GEO/FTP
def download_geo_direct(gse_id, save_dir):
    """
    Faz o download direto dos arquivos do GEO via FTP
    """
    print(f"Tentando baixar diretamente arquivos do GEO para {gse_id}...")
    
    # Constantes para o FTP do NCBI
    FTP_HOST = "ftp.ncbi.nlm.nih.gov"
    
    # Determinar o diretório FTP correto baseado no formato do ID
    gse_prefix = gse_id[:5]
    gse_suffix = gse_id[5:]
    if len(gse_suffix) <= 3:
        ftp_dir = f"/geo/series/{gse_prefix}nnn/{gse_id}"
    else:
        ftp_dir = f"/geo/series/{gse_prefix}nnn/{gse_id}"

    # Conectar ao servidor FTP
    try:
        ftp = FTP(FTP_HOST)
        ftp.login()
        
        # Verificar se o diretório existe
        try:
            ftp.cwd(ftp_dir)
            print(f"Conectado ao diretório FTP: {ftp_dir}")
            
            # Listar conteúdo do diretório
            directories = []
            ftp.retrlines('LIST', lambda x: directories.append(x))
            
            # Verificar se o diretório 'suppl' existe
            if any('suppl' in dir_info for dir_info in directories):
                # Tentar navegar para o diretório suppl
                ftp.cwd(f"{ftp_dir}/suppl")
                print("Acessando diretório de arquivos suplementares...")
                
                # Listar arquivos no diretório suppl
                files = []
                ftp.retrlines('LIST', lambda x: files.append(x))
                
                # Verificar se há arquivos
                if not files:
                    print("Nenhum arquivo encontrado no diretório 'suppl'")
                    
                    # Tentar baixar o arquivo RAW.tar como alternativa
                    try:
                        raw_url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={gse_id}&format=file"
                        raw_file = os.path.join(save_dir, f"{gse_id}_RAW.tar")
                        print(f"Tentando baixar o arquivo RAW via HTTP: {raw_url}")
                        
                        urllib.request.urlretrieve(raw_url, raw_file)
                        print(f"Download do arquivo RAW concluído: {raw_file}")
                        
                        # Extrair o arquivo tar
                        with tarfile.open(raw_file, 'r') as tar:
                            tar.extractall(path=save_dir)
                        print(f"Arquivos extraídos em: {save_dir}")
                        
                        return save_dir
                    except Exception as e:
                        print(f"Erro ao baixar arquivo RAW: {str(e)}")
                        return None
                
                # Baixar cada arquivo do diretório suppl
                for file_info in files:
                    # Extrair nome do arquivo
                    file_parts = file_info.split()
                    if len(file_parts) < 9:
                        continue
                    
                    file_name = file_parts[-1]
                    file_size = int(file_parts[4])
                    
                    # Ignorar diretórios
                    if file_info.startswith('d'):
                        continue
                    
                    # Caminho para salvar o arquivo
                    file_path = os.path.join(save_dir, file_name)
                    
                    # Baixar o arquivo
                    print(f"Baixando {file_name} ({file_size} bytes)...")
                    with open(file_path, 'wb') as f:
                        ftp.retrbinary(f'RETR {file_name}', f.write)
                    
                    # Extrair arquivos compactados
                    if file_name.endswith('.tar.gz') or file_name.endswith('.tgz'):
                        print(f"Extraindo {file_name}...")
                        with tarfile.open(file_path, 'r:gz') as tar:
                            tar.extractall(path=save_dir)
                    elif file_name.endswith('.gz') and not file_name.endswith('.tar.gz'):
                        print(f"Descomprimindo {file_name}...")
                        with gzip.open(file_path, 'rb') as f_in:
                            with open(file_path[:-3], 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                
                print(f"Download concluído para {gse_id}")
                return save_dir
            
            # Se não há diretório 'suppl', tentar baixar matriz
            print("Diretório 'suppl' não encontrado, tentando matriz...")
            
            # Tentar navegar para o diretório matrix
            try:
                ftp.cwd(f"{ftp_dir}/matrix")
                print("Acessando diretório de matrizes...")
                
                # Listar arquivos no diretório matrix
                files = []
                ftp.retrlines('LIST', lambda x: files.append(x))
                
                # Baixar arquivos de matriz
                for file_info in files:
                    # Extrair nome do arquivo
                    file_parts = file_info.split()
                    if len(file_parts) < 9:
                        continue
                    
                    file_name = file_parts[-1]
                    
                    # Ignorar diretórios
                    if file_info.startswith('d'):
                        continue
                    
                    # Caminho para salvar o arquivo
                    file_path = os.path.join(save_dir, file_name)
                    
                    # Baixar o arquivo
                    print(f"Baixando {file_name}...")
                    with open(file_path, 'wb') as f:
                        ftp.retrbinary(f'RETR {file_name}', f.write)
                    
                    # Extrair arquivos compactados
                    if file_name.endswith('.gz'):
                        print(f"Descomprimindo {file_name}...")
                        with gzip.open(file_path, 'rb') as f_in:
                            with open(file_path[:-3], 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                
                print(f"Download de matriz concluído para {gse_id}")
                return save_dir
                
            except Exception as e:
                print(f"Erro ao acessar diretório de matrizes: {str(e)}")
            
            # Se chegou aqui, tentar baixar via HTTP
            try:
                raw_url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={gse_id}&format=file"
                raw_file = os.path.join(save_dir, f"{gse_id}_RAW.tar")
                print(f"Tentando baixar o arquivo RAW via HTTP: {raw_url}")
                
                urllib.request.urlretrieve(raw_url, raw_file)
                print(f"Download do arquivo RAW concluído: {raw_file}")
                
                # Extrair o arquivo tar
                with tarfile.open(raw_file, 'r') as tar:
                    tar.extractall(path=save_dir)
                print(f"Arquivos extraídos em: {save_dir}")
                
                return save_dir
            except Exception as e:
                print(f"Erro ao baixar arquivo RAW: {str(e)}")
        
        except Exception as e:
            print(f"Erro ao navegar pelo diretório FTP: {str(e)}")
            
            # Tentar alternativa via HTTP
            try:
                print("Tentando método alternativo via HTTP...")
                raw_url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={gse_id}&format=file"
                raw_file = os.path.join(save_dir, f"{gse_id}_RAW.tar")
                
                urllib.request.urlretrieve(raw_url, raw_file)
                print(f"Download concluído: {raw_file}")
                
                # Extrair o arquivo tar
                with tarfile.open(raw_file, 'r') as tar:
                    tar.extractall(path=save_dir)
                print(f"Arquivos extraídos em: {save_dir}")
                
                return save_dir
            except Exception as e:
                print(f"Erro no método alternativo: {str(e)}")
                
        finally:
            ftp.quit()
    
    except Exception as e:
        print(f"Erro na conexão FTP: {str(e)}")
        
        # Tentar baixar diretamente via HTTP como último recurso
        try:
            print("Tentando baixar via HTTP como último recurso...")
            raw_url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={gse_id}&format=file"
            raw_file = os.path.join(save_dir, f"{gse_id}_RAW.tar")
            
            urllib.request.urlretrieve(raw_url, raw_file)
            print(f"Download concluído: {raw_file}")
            
            # Extrair o arquivo tar
            with tarfile.open(raw_file, 'r') as tar:
                tar.extractall(path=save_dir)
            print(f"Arquivos extraídos em: {save_dir}")
            
            return save_dir
        except Exception as e:
            print(f"Falha no último método de download: {str(e)}")
    
    # Se todas as tentativas falharem
    print("Todos os métodos de download falharam.")
    print(f"Recomendação: Baixe manualmente o dataset {gse_id} através da interface web do GEO:")
    print(f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}")
    
    return None

# Função para baixar dados específicos do dataset 5xFAD
def download_5xfad_data():
    """
    Baixa dados do modelo 5xFAD do dataset GSE168137
    """
    model_dir = os.path.join(BASE_DIR, "5xFAD")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Dataset específico para o 5xFAD
    gse_id = "GSE168137"
    
    # Tentar baixar diretamente
    dataset_dir = download_geo_direct(gse_id, model_dir)
    
    if not dataset_dir:
        # Método alternativo - tentar baixar através da URL do Synapse
        try:
            print("\nTentando método alternativo via Synapse...")
            print(f"Os dados do 5xFAD podem ser baixados manualmente em:")
            print(f"https://www.synapse.org/#!Synapse:syn23628482")
            print("Note que o download através do portal Synapse pode requerer cadastro.")
            
            # Alternativa AMP-AD Knowledge Portal
            print("\nAlternativa AMP-AD Knowledge Portal:")
            print("https://adknowledgeportal.org")
            print("Busque por '5xFAD' ou 'GSE168137' para encontrar os dados.")
            
            return None
        except Exception as e:
            print(f"Erro no método alternativo: {str(e)}")
            return None
    
    return dataset_dir

# Função para processar dados de RNA-seq
def process_rnaseq_data(data_dir):
    """
    Processa os dados de RNA-seq após o download
    
    Args:
        data_dir: Diretório contendo os dados baixados
    """
    print("Processando dados de RNA-seq...")
    
    try:
        # Buscar arquivos de expressão gênica
        count_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".txt") or file.endswith(".csv") or file.endswith(".tsv"):
                    if "count" in file.lower() or "expression" in file.lower() or "matrix" in file.lower():
                        count_files.append(os.path.join(root, file))
        
        # Também buscar por arquivos de expressão em diretórios específicos
        for root, dirs, files in os.walk(data_dir):
            for dir_name in dirs:
                if "expression" in dir_name.lower() or "counts" in dir_name.lower():
                    for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir_name)):
                        for file in subfiles:
                            if file.endswith(".txt") or file.endswith(".csv") or file.endswith(".tsv"):
                                count_files.append(os.path.join(subroot, file))
        
        if not count_files:
            print("Não foram encontrados arquivos de expressão gênica.")
            
            # Verificar se existem arquivos .CEL ou arquivos FASTQ
            raw_files = []
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(".CEL") or file.endswith(".fastq") or file.endswith(".fastq.gz"):
                        raw_files.append(os.path.join(root, file))
            
            if raw_files:
                print(f"Encontrados {len(raw_files)} arquivos brutos (.CEL ou FASTQ).")
                print("Estes arquivos precisam ser processados com ferramentas específicas.")
                print("Exemplos de arquivos encontrados:")
                for i, file in enumerate(raw_files[:5]):
                    print(f"  - {os.path.basename(file)}")
                if len(raw_files) > 5:
                    print(f"  - ... e mais {len(raw_files) - 5} arquivo(s)")
            
            return None
        
        print(f"Encontrados {len(count_files)} arquivos de expressão gênica.")
        print("Primeiros arquivos encontrados:")
        for i, file in enumerate(count_files[:5]):
            print(f"  - {os.path.basename(file)}")
        if len(count_files) > 5:
            print(f"  - ... e mais {len(count_files) - 5} arquivo(s)")
        
        # Processar o primeiro arquivo encontrado
        print(f"\nProcessando arquivo: {os.path.basename(count_files[0])}")
        
        # Determinar o separador com base na extensão do arquivo
        if count_files[0].endswith(".csv"):
            sep = ","
        else:
            sep = "\t"
        
        # Tentar carregar o arquivo
        try:
            # Primeiro tentar ler as primeiras linhas para verificar o formato
            with open(count_files[0], 'r') as f:
                first_lines = [next(f) for _ in range(5) if f]
            
            print("Primeiras linhas do arquivo:")
            for line in first_lines:
                print(line.strip())
            
            # Verificar se tem cabeçalho
            has_header = True
            
            # Carregar os dados
            expression_data = pd.read_csv(count_files[0], sep=sep, index_col=0, header=0 if has_header else None)
            print(f"Dados carregados com sucesso. Formato: {expression_data.shape}")
            
            # Salvar uma versão processada
            processed_file = os.path.join(data_dir, "processed_expression_data.csv")
            expression_data.to_csv(processed_file)
            print(f"Dados processados salvos em: {processed_file}")
            
            return expression_data
        
        except Exception as e:
            print(f"Erro ao carregar o arquivo de expressão: {str(e)}")
            print("Para processar este arquivo, pode ser necessário ajustar o código")
            print("de acordo com o formato específico do arquivo.")
            return None
    
    except Exception as e:
        print(f"Erro ao processar dados de RNA-seq: {str(e)}")
        return None

# Função principal
def main():
    """Função principal para controlar o fluxo de execução"""
    print("Iniciando download de dados genômicos para modelos murinos de Alzheimer...")
    
    # Lista de modelos a serem baixados
    models = ["5xFAD", "APP_PS1", "3xTg-AD", "APPNL-G-F", "Tg2576"]
    
    # Criar estrutura de diretórios
    create_directories(models)
    
    # Baixar dados do 5xFAD (foco principal)
    print("\n" + "="*50)
    print("Baixando dados do modelo 5xFAD")
    print("="*50)
    
    data_dir = download_5xfad_data()
    
    if data_dir:
        # Processar dados de RNA-seq
        expression_data = process_rnaseq_data(data_dir)
        
        if expression_data is not None:
            print("\nResumo dos dados:")
            print(f"Número de genes: {expression_data.shape[0]}")
            print(f"Número de amostras: {expression_data.shape[1]}")
            
            # Exibir primeiras linhas dos dados
            print("\nPrimeiras linhas dos dados de expressão:")
            print(expression_data.head())
        else:
            print("\nNão foi possível processar automaticamente os dados de expressão.")
            print("Recomendação: Verifique manualmente os arquivos baixados e ajuste")
            print("o código de processamento conforme necessário.")
    else:
        print("\nNão foi possível baixar os dados automaticamente.")
        print("Recomendação: Tente baixar manualmente os dados através da interface web do GEO:")
        print("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE168137")
        print("Ou através do portal Synapse: https://www.synapse.org/#!Synapse:syn23628482")

    print("\nColeta de dados concluída!")

if __name__ == "__main__":
    main()