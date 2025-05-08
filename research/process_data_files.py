import os
import pandas as pd
import numpy as np
import gzip
import shutil
import xml.etree.ElementTree as ET
import re

# Diretório atual onde estão os arquivos
DATA_DIR = "/home/rexionmars/estudos/dns-sequence/data"
# Diretório para arquivos processados
PROCESSED_DIR = "./processed"

# Criar diretório para arquivos processados
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

def extract_gz_file(gz_file_path, output_file_path=None):
    """Extrai um arquivo gzip"""
    if output_file_path is None:
        output_file_path = gz_file_path[:-3]  # Remove .gz do final
    
    print(f"Extraindo {gz_file_path} para {output_file_path}...")
    with gzip.open(gz_file_path, 'rb') as f_in: 
        with open(output_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    return output_file_path

def process_count_list(count_file):
    """Processa o arquivo de contagem de genes"""
    print(f"Processando arquivo de contagem: {count_file}")
    
    # Extrair o arquivo se estiver compactado
    if count_file.endswith('.gz'):
        count_file = extract_gz_file(count_file)
    
    # Ler o arquivo de contagem
    try:
        # Primeiro ler as primeiras linhas para entender o formato
        with open(count_file, 'r') as f:
            first_lines = [next(f) for _ in range(5)]
        
        print("Primeiras linhas do arquivo de contagem:")
        for line in first_lines:
            print(line.strip())
        
        # Tentar carregar o arquivo como TSV
        counts_df = pd.read_csv(count_file, sep='\t', index_col=0)
        
        # Salvar versão processada
        processed_file = os.path.join(PROCESSED_DIR, "processed_counts.csv")
        counts_df.to_csv(processed_file)
        
        print(f"Dados de contagem processados. Formato: {counts_df.shape}")
        return counts_df
    
    except Exception as e:
        print(f"Erro ao processar arquivo de contagem: {str(e)}")
        
        # Tentar método alternativo de leitura
        try:
            print("Tentando método alternativo...")
            
            # Pode ser necessário ajustar os parâmetros dependendo do formato real
            counts_df = pd.read_csv(count_file, sep='\t', comment='#', header=0)
            
            processed_file = os.path.join(PROCESSED_DIR, "processed_counts.csv")
            counts_df.to_csv(processed_file)
            
            print(f"Dados processados com método alternativo. Formato: {counts_df.shape}")
            return counts_df
        
        except Exception as e2:
            print(f"Falha no método alternativo: {str(e2)}")
            return None

def process_expression_list(expression_file):
    """Processa o arquivo de expressão gênica"""
    print(f"Processando arquivo de expressão: {expression_file}")
    
    # Extrair o arquivo se estiver compactado
    if expression_file.endswith('.gz'):
        expression_file = extract_gz_file(expression_file)
    
    # Ler o arquivo de expressão
    try:
        # Primeiro ler as primeiras linhas para entender o formato
        with open(expression_file, 'r') as f:
            first_lines = [next(f) for _ in range(5)]
        
        print("Primeiras linhas do arquivo de expressão:")
        for line in first_lines:
            print(line.strip())
        
        # Tentar carregar o arquivo como TSV
        expression_df = pd.read_csv(expression_file, sep='\t', index_col=0)
        
        # Salvar versão processada
        processed_file = os.path.join(PROCESSED_DIR, "processed_expression.csv")
        expression_df.to_csv(processed_file)
        
        print(f"Dados de expressão processados. Formato: {expression_df.shape}")
        return expression_df
    
    except Exception as e:
        print(f"Erro ao processar arquivo de expressão: {str(e)}")
        
        # Tentar método alternativo de leitura
        try:
            print("Tentando método alternativo...")
            
            # Pode ser necessário ajustar os parâmetros dependendo do formato real
            expression_df = pd.read_csv(expression_file, sep='\t', comment='#', header=0)
            
            processed_file = os.path.join(PROCESSED_DIR, "processed_expression.csv")
            expression_df.to_csv(processed_file)
            
            print(f"Dados processados com método alternativo. Formato: {expression_df.shape}")
            return expression_df
        
        except Exception as e2:
            print(f"Falha no método alternativo: {str(e2)}")
            return None

def process_series_matrix(matrix_file):
    """Processa o arquivo de matriz de série para extrair metadados"""
    print(f"Processando arquivo de matriz: {matrix_file}")
    
    # Extrair o arquivo se estiver compactado
    if matrix_file.endswith('.gz'):
        matrix_file = extract_gz_file(matrix_file)
    
    try:
        # Leitura dos metadados
        metadata = {}
        sample_ids = []
        sample_characteristics = {}
        
        with open(matrix_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Extrair IDs de amostras
                if line.startswith('!Sample_geo_accession'):
                    sample_ids = line.split('\t')[1:]
                    for sample_id in sample_ids:
                        sample_characteristics[sample_id] = {}
                
                # Extrair características das amostras
                elif line.startswith('!Sample_characteristics_ch'):
                    parts = line.split('\t')
                    header = parts[0]
                    values = parts[1:]
                    
                    for i, sample_id in enumerate(sample_ids):
                        if i < len(values):
                            value = values[i]
                            # Extrair tipo de característica e valor
                            if ':' in value:
                                char_type, char_value = value.split(':', 1)
                                sample_characteristics[sample_id][char_type.strip()] = char_value.strip()
                            else:
                                # Se não tem separador, usar o cabeçalho como chave
                                char_type = f"characteristic_{header.split('_')[-1]}"
                                sample_characteristics[sample_id][char_type] = value.strip()
                
                # Extrair informações globais do estudo
                elif line.startswith('!Series_'):
                    parts = line.split('=', 1)
                    if len(parts) > 1:
                        key = parts[0].replace('!Series_', '').strip()
                        value = parts[1].strip()
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]  # Remover aspas
                        metadata[key] = value
        
        # Criar DataFrame de metadados
        metadata_df = pd.DataFrame.from_dict(sample_characteristics, orient='index')
        
        # Salvar metadados processados
        processed_metadata = os.path.join(PROCESSED_DIR, "metadata.csv")
        metadata_df.to_csv(processed_metadata)
        
        # Salvar informações globais do estudo
        study_info = os.path.join(PROCESSED_DIR, "study_info.txt")
        with open(study_info, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Metadados processados. Número de amostras: {len(sample_ids)}")
        return metadata_df, metadata
    
    except Exception as e:
        print(f"Erro ao processar arquivo de matriz: {str(e)}")
        return None, None

def analyze_5xfad_data(counts_df, expression_df, metadata_df):
    """Realiza análise inicial dos dados do modelo 5xFAD"""
    print("\nIniciando análise dos dados do modelo 5xFAD...")
    
    if counts_df is not None:
        print("\nAnálise dos dados de contagem:")
        print(f"Total de genes: {counts_df.shape[0]}")
        print(f"Total de amostras: {counts_df.shape[1]}")
        
        # Estatísticas básicas
        print("\nEstatísticas básicas das contagens:")
        counts_stats = counts_df.describe()
        print(counts_stats)
        
        # Salvar estatísticas
        stats_file = os.path.join(PROCESSED_DIR, "counts_statistics.csv")
        counts_stats.to_csv(stats_file)
    
    if expression_df is not None:
        print("\nAnálise dos dados de expressão:")
        print(f"Total de genes: {expression_df.shape[0]}")
        print(f"Total de amostras: {expression_df.shape[1]}")
        
        # Estatísticas básicas
        print("\nEstatísticas básicas da expressão:")
        expression_stats = expression_df.describe()
        print(expression_stats)
        
        # Salvar estatísticas
        stats_file = os.path.join(PROCESSED_DIR, "expression_statistics.csv")
        expression_stats.to_csv(stats_file)
    
    if metadata_df is not None:
        print("\nAnálise dos metadados:")
        print(f"Total de amostras: {metadata_df.shape[0]}")
        print(f"Variáveis disponíveis: {', '.join(metadata_df.columns)}")
        
        # Verificar quais colunas podem conter informações sobre genótipo ou condição
        for col in metadata_df.columns:
            unique_values = metadata_df[col].unique()
            if len(unique_values) <= 10:  # Mostrar apenas se houver poucos valores únicos
                print(f"\nValores únicos em '{col}':")
                for val in unique_values:
                    count = metadata_df[col].value_counts()[val]
                    print(f"  - {val}: {count} amostras")
        
        # Tentar identificar grupos experimentais
        genotype_cols = [col for col in metadata_df.columns if 'genotype' in col.lower() or 'strain' in col.lower()]
        condition_cols = [col for col in metadata_df.columns if 'condition' in col.lower() or 'treatment' in col.lower()]
        age_cols = [col for col in metadata_df.columns if 'age' in col.lower() or 'month' in col.lower()]
        
        if genotype_cols:
            print("\nPossíveis colunas de genótipo:", genotype_cols)
        if condition_cols:
            print("Possíveis colunas de condição:", condition_cols)
        if age_cols:
            print("Possíveis colunas de idade:", age_cols)

def main():
    """Função principal para processar os arquivos do dataset 5xFAD"""
    print("Iniciando processamento de dados do modelo 5xFAD (GSE168137)...")
    
    # Paths para os arquivos
    count_file = os.path.join(DATA_DIR, "GSE168137_countList.txt.gz")
    expression_file = os.path.join(DATA_DIR, "GSE168137_expressionList.txt.gz")
    matrix_file = os.path.join(DATA_DIR, "GSE168137_series_matrix.txt.gz")
    
    # Verificar existência dos arquivos
    if not os.path.exists(count_file):
        print(f"Arquivo não encontrado: {count_file}")
    if not os.path.exists(expression_file):
        print(f"Arquivo não encontrado: {expression_file}")
    if not os.path.exists(matrix_file):
        print(f"Arquivo não encontrado: {matrix_file}")
    
    # Processar arquivos
    counts_df = None
    expression_df = None
    metadata_df = None
    
    if os.path.exists(count_file):
        counts_df = process_count_list(count_file)
    
    if os.path.exists(expression_file):
        expression_df = process_expression_list(expression_file)
    
    if os.path.exists(matrix_file):
        metadata_df, study_metadata = process_series_matrix(matrix_file)
    
    # Realizar análise inicial dos dados
    analyze_5xfad_data(counts_df, expression_df, metadata_df)
    
    print("\nProcessamento concluído! Os dados processados estão no diretório:", PROCESSED_DIR)

if __name__ == "__main__":
    main()