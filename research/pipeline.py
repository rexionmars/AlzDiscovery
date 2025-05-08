#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline para Descoberta de Medicamentos para Alzheimer
Combinando análise transcriptômica e modelos de linguagem médicos
"""

import pandas as pd
import numpy as np
import os
import json
import requests
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import re
import time
import logging
from datetime import datetime

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"alzheimer_drug_discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AlzheimerDrugDiscovery")

class AlzheimerDrugDiscovery:
    """Pipeline para descoberta de medicamentos para Alzheimer usando LLMs e transcriptômica"""
    
    def __init__(self, 
                 count_file="GSE168137_countList.txt", 
                 expression_file="GSE168137_expressionList.txt",
                 meditron_model="meditron",
                 biomistral_model="adrienbrault/biomistral-7b:Q2_K",
                 llm_api_url="http://localhost:11434/api/generate"):
        """
        Inicializa o pipeline de descoberta de medicamentos
        
        Args:
            count_file: Caminho para o arquivo de contagens brutas do RNA-seq
            expression_file: Caminho para o arquivo de expressão gênica normalizada
            meditron_model: Nome do modelo médico especializado
            biomistral_model: Nome do modelo biomédico
            llm_api_url: URL da API local para os modelos LLM
        """
        self.count_file = count_file
        self.expression_file = expression_file
        self.meditron_model = meditron_model
        self.biomistral_model = biomistral_model
        self.llm_api_url = llm_api_url
        
        # Variáveis para armazenar resultados intermediários
        self.expression_data = None
        self.metadata = None
        self.deg_results = None
        self.upregulated_genes = []
        self.downregulated_genes = []
        self.pathway_results = None
        self.top_pathways = []
        self.compounds = []
        self.compound_interactions = {}
        self.compound_ranking = None
        self.combination_therapies = {}
        self.literature_validation = {}
        self.final_report = None
        
        logger.info("Pipeline de Descoberta de Medicamentos para Alzheimer inicializado")
    
    def load_data(self):
        """Carrega e prepara os dados transcriptômicos"""
        try:
            logger.info(f"Carregando dados de expressão de {self.expression_file}")
            
            # Verifica se o arquivo está comprimido
            if self.expression_file.endswith('.gz'):
                self.expression_data = pd.read_csv(self.expression_file, sep='\t', compression='gzip')
            else:
                self.expression_data = pd.read_csv(self.expression_file, sep='\t')
            
            # Assumindo que a primeira coluna contém os IDs dos genes
            gene_ids = self.expression_data.iloc[:, 0]
            expression_matrix = self.expression_data.iloc[:, 1:]
            
            # Extrai metadados dos nomes das colunas
            sample_names = expression_matrix.columns
            self.metadata = pd.DataFrame({
                'sample_id': sample_names,
                'genotype': ['5xFAD' if '5xFAD' in name else 'BL6' for name in sample_names],
                'region': ['cortex' if 'CX' in name else 'hippocampus' if 'HP' in name else 'unknown' for name in sample_names],
                'age': [re.search(r'(\d+)m', name).group(1) if re.search(r'(\d+)m', name) else 'unknown' for name in sample_names],
                'sex': ['male' if '_M_' in name else 'female' if '_F_' in name else 'unknown' for name in sample_names]
            })
            
            logger.info(f"Dados carregados com sucesso. {self.expression_data.shape[0]} genes, {self.expression_data.shape[1]-1} amostras.")
            logger.info(f"Distribuição de genótipos: {self.metadata['genotype'].value_counts().to_dict()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            return False
    
    def run_differential_expression(self, save_results=True):
        """
        Realiza análise de expressão diferencial
        
        Como os pacotes DESeq2 e edgeR são em R, aqui simularemos resultados
        para fins de demonstração. Em um cenário real, você usaria rpy2 para 
        chamar estas bibliotecas diretamente ou executaria scripts R separados.
        """
        logger.info("Iniciando análise de expressão diferencial")
        
        try:
            # Simulação de resultados de expressão diferencial
            # Em um cenário real, você usaria DESeq2 ou edgeR via rpy2
            
            # Cria um DataFrame para armazenar os resultados simulados
            n_genes = 1000  # Simulando 1000 genes para demonstração
            np.random.seed(42)  # Para reprodutibilidade
            
            self.deg_results = pd.DataFrame({
                'gene_id': [f"GENE_{i}" for i in range(n_genes)],
                'baseMean': np.random.exponential(500, n_genes),
                'log2FoldChange': np.random.normal(0, 2, n_genes),
                'lfcSE': np.random.uniform(0.1, 0.5, n_genes),
                'stat': np.random.normal(0, 1, n_genes),
                'pvalue': np.random.beta(1, 10, n_genes),
                'padj': np.random.beta(1, 20, n_genes)  # Valores ajustados são geralmente mais conservadores
            })
            
            # Identificar genes diferencialmente expressos
            significant_genes = self.deg_results[self.deg_results['padj'] < 0.05].copy()
            
            # Classificar por fold change e extrair top genes
            self.upregulated_genes = significant_genes[significant_genes['log2FoldChange'] > 1].sort_values(
                'log2FoldChange', ascending=False)['gene_id'].tolist()[:50]
            
            self.downregulated_genes = significant_genes[significant_genes['log2FoldChange'] < -1].sort_values(
                'log2FoldChange', ascending=True)['gene_id'].tolist()[:50]
            
            if save_results:
                self.deg_results.to_csv("differential_expression_results.csv", index=False)
                
            logger.info(f"Análise de expressão diferencial concluída. {len(self.upregulated_genes)} genes up-regulados, {len(self.downregulated_genes)} genes down-regulados.")
            return True
            
        except Exception as e:
            logger.error(f"Erro na análise de expressão diferencial: {str(e)}")
            return False
    
    def run_pathway_analysis(self, save_results=True):
        """
        Realiza análise de enriquecimento de vias
        
        Assim como a análise diferencial, aqui simulamos resultados
        que normalmente viriam de ferramentas como clusterProfiler ou gProfiler
        """
        logger.info("Iniciando análise de vias biológicas")
        
        try:
            # Simulando resultados de análise de vias
            # Em um cenário real, você usaria bibliotecas como gseapy, goatools, etc.
            
            # Lista de vias biológicas relevantes para Alzheimer
            alzheimer_pathways = [
                "Amyloid-beta metabolism",
                "Tau protein binding",
                "Neuroinflammatory response",
                "Microglial activation",
                "Synaptic dysfunction",
                "Oxidative stress response",
                "Mitochondrial dysfunction",
                "Autophagy-lysosomal system",
                "Endosomal-lysosomal system",
                "Lipid metabolism",
                "Calcium signaling",
                "Insulin signaling pathway",
                "MAPK signaling cascade",
                "Apoptotic process",
                "Proteolysis",
                "Cellular response to stress",
                "Cytokine-mediated signaling",
                "Complement activation",
                "Neurotrophic signaling",
                "Blood-brain barrier permeability"
            ]
            
            np.random.seed(43)  # Para reprodutibilidade
            n_pathways = len(alzheimer_pathways)
            
            self.pathway_results = pd.DataFrame({
                'pathway_id': [f"PATH_{i}" for i in range(n_pathways)],
                'pathway_name': alzheimer_pathways,
                'gene_count': np.random.randint(10, 100, n_pathways),
                'p_value': np.random.beta(1, 10, n_pathways),
                'adjusted_p_value': np.random.beta(1, 15, n_pathways),
                'enrichment_score': np.random.uniform(1.5, 4.0, n_pathways)
            })
            
            # Ordenar por significância
            self.pathway_results = self.pathway_results.sort_values('adjusted_p_value')
            
            # Extrair top vias
            self.top_pathways = self.pathway_results['pathway_name'].tolist()[:10]
            
            if save_results:
                self.pathway_results.to_csv("enrichment_results.csv", index=False)
            
            logger.info(f"Análise de vias concluída. Top vias: {', '.join(self.top_pathways[:3])}")
            return True
            
        except Exception as e:
            logger.error(f"Erro na análise de vias: {str(e)}")
            return False
    
    def query_llm(self, prompt, model, max_retries=3, retry_delay=2):
        """
        Envia uma consulta para o modelo LLM e retorna a resposta
        
        Args:
            prompt: O prompt a ser enviado para o modelo
            model: O nome do modelo a ser consultado
            max_retries: Número máximo de tentativas em caso de falha
            retry_delay: Tempo de espera entre tentativas (segundos)
        
        Returns:
            O texto da resposta do modelo ou None em caso de falha
        """
        payload = {
            "model": model,
            "prompt": prompt
        }
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Enviando prompt para o modelo {model} (tentativa {attempt+1}/{max_retries})")
                response = requests.post(self.llm_api_url, data=json.dumps(payload))
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '')
                else:
                    logger.warning(f"Resposta não-200 da API: {response.status_code}, {response.text}")
            
            except Exception as e:
                logger.warning(f"Erro ao consultar LLM: {str(e)}")
            
            # Espera antes de tentar novamente
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        
        logger.error(f"Falha após {max_retries} tentativas de consultar o modelo {model}")
        return None
    
    def identify_potential_compounds(self):
        """
        Consulta os modelos LLM para identificar compostos potenciais
        com base nos genes e vias alterados
        """
        logger.info("Consultando LLMs para identificação de compostos potenciais")
        
        # Limitar o número de genes para não sobrecarregar o prompt
        up_genes_subset = self.upregulated_genes[:15] if len(self.upregulated_genes) > 15 else self.upregulated_genes
        down_genes_subset = self.downregulated_genes[:15] if len(self.downregulated_genes) > 15 else self.downregulated_genes
        pathways_subset = self.top_pathways[:10] if len(self.top_pathways) > 10 else self.top_pathways
        
        # Construir o prompt
        prompt_template = """
        Based on a transcriptomic analysis of 5xFAD mouse model of Alzheimer's disease, 
        we identified the following key dysregulated genes:

        Upregulated genes: {upregulated}

        Downregulated genes: {downregulated}

        The most significantly affected biological pathways are:
        {pathways}

        Given this gene expression profile:
        1. What existing compounds or drugs might target these pathways to reverse the Alzheimer's pathology?
        2. What novel therapeutic approaches could be developed to normalize these gene expression patterns?
        3. What combinations of compounds might work synergistically to address multiple dysregulated pathways?
        4. What potential side effects should be monitored based on these targets?

        Please provide specific molecular mechanisms for how these compounds would affect the disease pathology.
        Format your response as a numbered list of compounds, with each compound followed by its mechanism.
        """
        
        prompt = prompt_template.format(
            upregulated=", ".join(up_genes_subset),
            downregulated=", ".join(down_genes_subset),
            pathways="\n".join([f"- {p}" for p in pathways_subset])
        )
        
        # Consultar ambos os modelos
        logger.info(f"Consultando modelo {self.biomistral_model}")
        biomistral_response = self.query_llm(prompt, self.biomistral_model)
        
        logger.info(f"Consultando modelo {self.meditron_model}")
        meditron_response = self.query_llm(prompt, self.meditron_model)
        
        # Extrair compostos das respostas
        all_compounds = []
        
        if biomistral_response:
            biomistral_compounds = self.extract_compounds(biomistral_response)
            logger.info(f"Extraídos {len(biomistral_compounds)} compostos do {self.biomistral_model}")
            all_compounds.extend(biomistral_compounds)
        
        if meditron_response:
            meditron_compounds = self.extract_compounds(meditron_response)
            logger.info(f"Extraídos {len(meditron_compounds)} compostos do {self.meditron_model}")
            all_compounds.extend(meditron_compounds)
        
        # Remover duplicatas
        unique_compounds = {}
        for comp in all_compounds:
            if comp['name'] not in unique_compounds:
                unique_compounds[comp['name']] = comp
        
        self.compounds = list(unique_compounds.values())
        logger.info(f"Identificados {len(self.compounds)} compostos únicos potenciais para tratamento")
        
        # Salvar os resultados
        with open("potential_compounds.json", "w") as f:
            json.dump(self.compounds, f, indent=2)
        
        return len(self.compounds) > 0
    
    def extract_compounds(self, llm_response):
        """
        Extrai nomes de compostos e mecanismos da resposta do LLM
        
        Args:
            llm_response: Texto da resposta do LLM
            
        Returns:
            Lista de dicionários contendo nome e mecanismo de cada composto
        """
        compounds = []
        lines = llm_response.split('\n')
        
        # Padrões para identificar compostos
        compound_patterns = [
            r'^[0-9]+\.[\s]*([A-Z][a-z]+[a-zA-Z\s\-]+):', # 1. Compound:
            r'^[0-9]+\.[\s]*([A-Z][a-z]+[a-zA-Z\s\-]+) -', # 1. Compound -
            r'^[*\-•][\s]*([A-Z][a-z]+[a-zA-Z\s\-]+):', # • Compound:
            r'^([A-Z][a-z]+[a-zA-Z\s\-]+)\s*\((?:drug|compound|inhibitor|therapy)\):', # Compound (drug):
        ]
        
        current_compound = None
        current_mechanism = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Verificar se esta linha inicia um novo composto
            compound_match = None
            for pattern in compound_patterns:
                match = re.search(pattern, line)
                if match:
                    compound_match = match
                    break
            
            if compound_match:
                # Se encontramos um novo composto, salvamos o anterior
                if current_compound:
                    compounds.append({
                        'name': current_compound,
                        'mechanism': ' '.join(current_mechanism)
                    })
                
                # Iniciamos um novo composto
                current_compound = compound_match.group(1).strip()
                current_mechanism = []
                
                # Capturar qualquer texto após o nome do composto
                remainder = line[compound_match.end():].strip()
                if remainder and remainder not in [':', '-']:
                    current_mechanism.append(remainder)
            elif current_compound and line:
                current_mechanism.append(line)
        
        # Adicionar o último composto
        if current_compound:
            compounds.append({
                'name': current_compound,
                'mechanism': ' '.join(current_mechanism)
            })
        
        return compounds
    
    def simulate_drug_gene_interactions(self, top_n=5):
        """
        Simula interações entre os compostos identificados e os genes alvo
        
        Args:
            top_n: Número de compostos mais promissores a considerar
        """
        logger.info(f"Simulando interações medicamento-gene para os {top_n} compostos mais promissores")
        
        if not self.compounds:
            logger.error("Nenhum composto identificado para simular interações")
            return False
        
        # Selecionar os compostos mais promissores
        # Em uma implementação real, você poderia ranqueá-los primeiro
        top_compounds = self.compounds[:top_n]
        
        # Selecionar genes alvo para consulta
        target_genes = []
        if self.upregulated_genes:
            target_genes.extend(self.upregulated_genes[:10])
        if self.downregulated_genes:
            target_genes.extend(self.downregulated_genes[:10])
        
        if not target_genes:
            logger.error("Nenhum gene alvo encontrado para simulação")
            return False
        
        self.compound_interactions = {}
        for compound in top_compounds:
            logger.info(f"Analisando interações para o composto {compound['name']}")
            
            interaction_prompt = f"""
            For the compound {compound['name']}, which is proposed to work via the following mechanism:
            "{compound['mechanism']}"
            
            Please predict the specific molecular interactions with the following Alzheimer's-associated genes:
            {', '.join(target_genes[:20])}  # Limitar para 20 genes
            
            For each interaction:
            1. What is the likely effect (activation, inhibition, modulation)?
            2. Through what molecular pathway or mechanism does this interaction occur?
            3. How might this interaction contribute to ameliorating Alzheimer's pathology?
            4. Is there supporting evidence from literature or similar compounds?
            
            Be specific about molecular mechanisms and provide quantitative predictions when possible.
            """
            
            # Usar o modelo meditron para maior especificidade médica
            interaction_result = self.query_llm(interaction_prompt, self.meditron_model)
            if interaction_result:
                self.compound_interactions[compound['name']] = interaction_result
            else:
                logger.warning(f"Não foi possível obter interações para {compound['name']}")
        
        # Salvar os resultados
        with open("compound_interactions.json", "w") as f:
            json.dump(self.compound_interactions, f, indent=2)
        
        return len(self.compound_interactions) > 0
    
    def rank_compounds(self):
        """
        Consulta o LLM para ranquear os compostos com base em suas interações previstas
        """
        logger.info("Ranqueando compostos baseado em suas interações previstas")
        
        if not self.compound_interactions:
            logger.error("Nenhuma interação composto-gene para ranquear")
            return False
        
        # Construir prompt para ranqueamento
        ranking_prompt = f"""
        Given the following candidate compounds for treating Alzheimer's disease in the 5xFAD mouse model,
        along with their predicted gene interactions:

        {json.dumps(self.compound_interactions, indent=2)}

        Please rank these compounds based on:
        1. Potential efficacy for treating Alzheimer's pathology
        2. Alignment with known disease mechanisms
        3. Safety profile and potential side effects
        4. Novelty compared to existing treatments
        5. Feasibility for development and testing

        For each compound, provide a score (1-10) and detailed rationale, focusing on specific molecular mechanisms.
        Format as: "Rank 1: [Compound Name] - Score: [X/10]" followed by rationale on separate lines.
        """
        
        # Consultar o modelo Biomistral
        self.compound_ranking = self.query_llm(ranking_prompt, self.biomistral_model)
        
        if self.compound_ranking:
            # Salvar o ranqueamento
            with open("compound_ranking.txt", "w") as f:
                f.write(self.compound_ranking)
            logger.info("Ranqueamento de compostos concluído com sucesso")
            return True
        else:
            logger.error("Falha ao ranquear compostos")
            return False
    
    def simulate_combination_therapy(self, top_n=3):
        """
        Simula combinações de terapias para os compostos mais promissores
        
        Args:
            top_n: Número de compostos a considerar para combinações
        """
        logger.info(f"Simulando terapias combinadas com os {top_n} compostos mais promissores")
        
        if not self.compounds or len(self.compounds) < 2:
            logger.error("Número insuficiente de compostos para simular combinações")
            return False
        
        # Selecionar os compostos mais promissores
        top_compounds = self.compounds[:top_n]
        compound_names = [c['name'] for c in top_compounds]
        
        # Gerar todas as combinações possíveis de pares
        compound_pairs = list(combinations(compound_names, 2))
        
        self.combination_therapies = {}
        for pair in compound_pairs:
            logger.info(f"Analisando combinação: {pair[0]} + {pair[1]}")
            
            combo_prompt = f"""
            For the combination therapy of {pair[0]} and {pair[1]} for Alzheimer's disease:
            
            1. Predict potential synergistic effects that would exceed individual treatments
            2. Identify possible molecular pathways where these compounds might complement each other
            3. Assess potential risks of combining these compounds
            4. Suggest optimal dosing strategy for this combination
            5. Predict how this combination might affect different cell types in the brain (neurons, microglia, astrocytes)
            
            Base your assessment on the known mechanisms of these compounds and their interaction with Alzheimer's pathology.
            """
            
            # Alternando entre os dois modelos para diversificar as análises
            model = self.meditron_model if len(self.combination_therapies) % 2 == 0 else self.biomistral_model
            
            combo_result = self.query_llm(combo_prompt, model)
            if combo_result:
                self.combination_therapies[f"{pair[0]} + {pair[1]}"] = combo_result
            else:
                logger.warning(f"Não foi possível avaliar a combinação {pair[0]} + {pair[1]}")
        
        # Salvar os resultados
        with open("combination_therapies.json", "w") as f:
            json.dump(self.combination_therapies, f, indent=2)
        
        return len(self.combination_therapies) > 0
    
    def validate_with_literature(self, top_n=3):
        """
        Valida os compostos mais promissores com literatura científica
        
        Args:
            top_n: Número de compostos a validar
        """
        logger.info(f"Validando os {top_n} compostos mais promissores com literatura")
        
        if not self.compounds:
            logger.error("Nenhum composto para validar")
            return False
        
        # Selecionar os compostos mais promissores
        top_compounds = self.compounds[:top_n]
        
        self.literature_validation = {}
        for compound in top_compounds:
            logger.info(f"Pesquisando literatura para o composto {compound['name']}")
            
            literature_prompt = f"""
            For the compound {compound['name']} proposed for Alzheimer's disease treatment:
            
            1. What existing scientific literature supports its use in Alzheimer's or similar neurodegenerative conditions?
            2. Have there been any clinical trials or animal studies using this compound for Alzheimer's?
            3. What is the evidence strength (strong, moderate, preliminary, theoretical)?
            4. Are there any contradictory findings in the literature?
            
            Please cite specific studies when possible and evaluate the quality of evidence.
            Include years of publication when mentioning studies.
            """
            
            literature_review = self.query_llm(literature_prompt, self.biomistral_model)
            if literature_review:
                self.literature_validation[compound['name']] = literature_review
            else:
                logger.warning(f"Não foi possível obter validação da literatura para {compound['name']}")
        
        # Salvar os resultados
        with open("literature_validation.json", "w") as f:
            json.dump(self.literature_validation, f, indent=2)
        
        return len(self.literature_validation) > 0
    
    def generate_final_report(self):
        """
        Gera um relatório final integrando todos os resultados
        """
        logger.info("Gerando relatório final integrado")
        
        # Verificar se temos dados suficientes para gerar o relatório
        if not (self.compounds and self.compound_interactions):
            logger.error("Dados insuficientes para gerar relatório final")
            return False
        
        # Construir prompt para o relatório final
        final_report_prompt = f"""
        Based on a comprehensive analysis combining transcriptomic data from 5xFAD mouse models 
        and advanced LLM-based drug discovery methods, please generate a structured report on 
        potential Alzheimer's treatments with the following sections:

        1. EXECUTIVE SUMMARY
           Key findings and top recommendations

        2. METHODOLOGY OVERVIEW
           How transcriptomic data was integrated with LLM analysis

        3. TOP CANDIDATE COMPOUNDS
           {json.dumps([c['name'] for c in self.compounds[:5]], indent=2)}
           
        4. MECHANISM OF ACTION
           How these compounds address the dysregulated pathways in Alzheimer's

        5. PREDICTED EFFICACY
           Based on gene interaction models
           
        6. PROMISING COMBINATION THERAPIES
           {json.dumps(list(self.combination_therapies.keys()), indent=2) if self.combination_therapies else "No combination data available"}
           
        7. LITERATURE VALIDATION
           Summary of existing evidence

        8. EXPERIMENTAL DESIGN RECOMMENDATIONS
           Suggested in vivo validation experiments

        9. TRANSLATIONAL POTENTIAL
           Pathway to human applications

        Please format this as a comprehensive scientific report suitable for drug discovery researchers.
        """
        
        # Consultar o modelo meditron para o relatório final
        self.final_report = self.query_llm(final_report_prompt, self.meditron_model)
        
        if self.final_report:
            # Salvar o relatório final
            with open("final_report.md", "w") as f:
                f.write(self.final_report)
            logger.info("Relatório final gerado com sucesso")
            return True
        else:
            logger.error("Falha ao gerar relatório final")
            return False
    
    def run_pipeline(self):
        """
        Executa o pipeline completo de descoberta de medicamentos
        """
        logger.info("Iniciando pipeline completo de descoberta de medicamentos para Alzheimer")
        
        # Etapa 1: Carregar e preparar dados
        if not self.load_data():
            logger.error("Falha na etapa de carregamento de dados. Abortando pipeline.")
            return False
        
        # Etapa 2: Análise de expressão diferencial
        if not self.run_differential_expression():
            logger.warning("Falha na análise de expressão diferencial. Continuando com dados simulados.")
            
            # Criar dados simulados para continuar
            self.upregulated_genes = [f"UPR_GENE_{i}" for i in range(50)]
            self.downregulated_genes = [f"DOWN_GENE_{i}" for i in range(50)]
        
        # Etapa 3: Análise de vias biológicas
        if not self.run_pathway_analysis():
            logger.warning("Falha na análise de vias. Continuando com dados simulados.")
            
            # Criar dados simulados para continuar
            self.top_pathways = [
                "Amyloid-beta metabolism",
                "Tau protein binding",
                "Neuroinflammatory response",
                "Microglial activation",
                "Synaptic dysfunction"
            ]
        
        # Etapa 4: Identificação de compostos potenciais
        if not self.identify_potential_compounds():
            logger.error("Falha na identificação de compostos. Abortando pipeline.")
            return False
        
        # Etapa 5: Simulação de interações medicamento-gene
        if not self.simulate_drug_gene_interactions(top_n=5):
            logger.error("Falha na simulação de interações. Abortando pipeline.")
            return False
        
        # Etapa 6: Ranqueamento de compostos
        if not self.rank_compounds():
            logger.warning("Falha no ranqueamento de compostos. Continuando com dados não ranqueados.")
        
        # Etapa 7: Simulação de terapias combinadas
        if not self.simulate_combination_therapy(top_n=3):
            logger.warning("Falha na simulação de terapias combinadas. Continuando sem dados de combinação.")
        
        # Etapa 8: Validação com literatura
        if not self.validate_with_literature(top_n=3):
            logger.warning("Falha na validação com literatura. Continuando sem validação bibliográfica.")
        
        # Etapa 9: Geração do relatório final
        if not self.generate_final_report():
            logger.error("Falha na geração do relatório final.")
            return False
        
        logger.info("Pipeline de descoberta de medicamentos concluído com sucesso!")
        return True


def main():
    """Função principal para executar o pipeline"""
    
    # Verificar argumentos da linha de comando
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline de Descoberta de Medicamentos para Alzheimer")
    parser.add_argument("--count-file", default="GSE168137_countList.txt", help="Arquivo de contagens brutas")
    parser.add_argument("--expression-file", default="GSE168137_expressionList.txt", help="Arquivo de expressão normalizada")
    parser.add_argument("--meditron-model", default="meditron", help="Nome do modelo médico")
    parser.add_argument("--biomistral-model", default="adrienbrault/biomistral-7b:Q2_K", help="Nome do modelo biomédico")
    parser.add_argument("--llm-api-url", default="http://localhost:11434/api/generate", help="URL da API dos modelos LLM")
    parser.add_argument("--simulate-data", action="store_true", help="Usar dados simulados em vez de arquivos reais")
    args = parser.parse_args()
    
    # Configurar o pipeline
    pipeline = AlzheimerDrugDiscovery(
        count_file=args.count_file,
        expression_file=args.expression_file,
        meditron_model=args.meditron_model,
        biomistral_model=args.biomistral_model,
        llm_api_url=args.llm_api_url
    )
    
    # Executar o pipeline
    success = pipeline.run_pipeline()
    
    if success:
        logger.info("Pipeline concluído com sucesso. Os seguintes arquivos foram gerados:")
        logger.info("- differential_expression_results.csv: Genes diferencialmente expressos")
        logger.info("- enrichment_results.csv: Resultados da análise de vias")
        logger.info("- potential_compounds.json: Compostos potenciais identificados")
        logger.info("- compound_interactions.json: Interações previstas entre compostos e genes")
        logger.info("- compound_ranking.txt: Ranqueamento dos compostos")
        logger.info("- combination_therapies.json: Análise de terapias combinadas")
        logger.info("- literature_validation.json: Validação com literatura científica")
        logger.info("- final_report.md: Relatório final integrado")
        
        print("\nRelatório final gerado com sucesso: final_report.md")
    else:
        logger.error("Pipeline falhou. Verifique o log para mais detalhes.")
        print("\nFalha na execução do pipeline. Verifique o log para mais detalhes.")


if __name__ == "__main__":
    main()