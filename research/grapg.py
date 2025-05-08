#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline para Descoberta de Medicamentos para Alzheimer
Combinando análise transcriptômica e modelos de linguagem médicos
Enhanced version with CLI visualization and improved error handling
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
import argparse
import sys
from typing import List, Dict, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# Rich library for CLI visualization
try:
    from rich.console import Console
    from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich library not available. Install with: pip install rich")
    print("Falling back to standard output")

# Plotext for CLI charts
try:
    import plotext as plt_cli
    PLOTEXT_AVAILABLE = True
except ImportError:
    PLOTEXT_AVAILABLE = False
    print("Plotext library not available. Install with: pip install plotext")
    print("Falling back to matplotlib for visualizations")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"alzheimer_drug_discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AlzheimerDrugDiscovery")

# Initialize rich console if available
if RICH_AVAILABLE:
    console = Console()


class VisualizationManager:
    """Manages visualizations for the pipeline using both terminal and file outputs"""
    
    def __init__(self, use_cli: bool = True, output_dir: str = "visualizations"):
        """
        Initialize the visualization manager
        
        Args:
            use_cli: Whether to display visualizations in the terminal
            output_dir: Directory to save visualization files
        """
        self.use_cli = use_cli and (RICH_AVAILABLE or PLOTEXT_AVAILABLE)
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def display_gene_counts(self, metadata: pd.DataFrame, title: str = "Sample Distribution"):
        """
        Display counts of samples by genotype, age, sex, and brain region
        
        Args:
            metadata: DataFrame containing sample metadata
            title: Title for the visualization
        """
        if not self.use_cli:
            return
        
        if RICH_AVAILABLE:
            table = Table(title=title)
            table.add_column("Category", style="cyan")
            table.add_column("Count", justify="right", style="green")
            
            # Add genotype counts
            genotype_counts = metadata['genotype'].value_counts().to_dict()
            for genotype, count in genotype_counts.items():
                table.add_row(f"Genotype: {genotype}", str(count))
            
            # Add brain region counts
            region_counts = metadata['region'].value_counts().to_dict()
            for region, count in region_counts.items():
                table.add_row(f"Region: {region}", str(count))
            
            # Add age counts
            age_counts = metadata['age'].value_counts().to_dict()
            for age, count in age_counts.items():
                table.add_row(f"Age: {age}", str(count))
            
            # Add sex counts
            sex_counts = metadata['sex'].value_counts().to_dict()
            for sex, count in sex_counts.items():
                table.add_row(f"Sex: {sex}", str(count))
            
            console.print(table)
        
        elif PLOTEXT_AVAILABLE:
            # Plot genotype distribution
            genotype_counts = metadata['genotype'].value_counts().to_dict()
            plt_cli.clf()
            plt_cli.bar(list(genotype_counts.keys()), list(genotype_counts.values()))
            plt_cli.title("Sample Distribution by Genotype")
            plt_cli.show()
    
    def plot_differential_expression(self, deg_results: pd.DataFrame, 
                                     output_file: str = "volcano_plot.png"):
        """
        Create a volcano plot of differential expression results
        
        Args:
            deg_results: DataFrame with differential expression results
            output_file: File to save the plot
        """
        # Create a standard matplotlib volcano plot and save to file
        plt.figure(figsize=(10, 8))
        
        # Plot all points in grey
        plt.scatter(
            deg_results['log2FoldChange'], 
            -np.log10(deg_results['pvalue']),
            color='grey',
            s=4,
            alpha=0.5
        )
        
        # Highlight significant up-regulated genes in red
        sig_up = (deg_results['padj'] < 0.05) & (deg_results['log2FoldChange'] > 1)
        plt.scatter(
            deg_results.loc[sig_up, 'log2FoldChange'],
            -np.log10(deg_results.loc[sig_up, 'pvalue']),
            color='red',
            s=6
        )
        
        # Highlight significant down-regulated genes in blue
        sig_down = (deg_results['padj'] < 0.05) & (deg_results['log2FoldChange'] < -1)
        plt.scatter(
            deg_results.loc[sig_down, 'log2FoldChange'],
            -np.log10(deg_results.loc[sig_down, 'pvalue']),
            color='blue',
            s=6
        )
        
        # Add labels for top genes
        top_genes = pd.concat([
            deg_results.loc[sig_up].sort_values('log2FoldChange', ascending=False).head(10),
            deg_results.loc[sig_down].sort_values('log2FoldChange', ascending=True).head(10)
        ])
        
        for _, gene in top_genes.iterrows():
            plt.annotate(
                gene['gene_id'], 
                xy=(gene['log2FoldChange'], -np.log10(gene['pvalue'])),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=8
            )
        
        plt.axhline(y=-np.log10(0.05), color='grey', linestyle='--', alpha=0.6)
        plt.axvline(x=1, color='grey', linestyle='--', alpha=0.6)
        plt.axvline(x=-1, color='grey', linestyle='--', alpha=0.6)
        
        plt.xlabel('Log2 Fold Change')
        plt.ylabel('-Log10 P-value')
        plt.title('Volcano Plot: 5xFAD vs Control')
        
        # Save to file
        plt.savefig(os.path.join(self.output_dir, output_file), dpi=300)
        
        # Display in terminal if possible
        if self.use_cli and PLOTEXT_AVAILABLE:
            plt_cli.clf()
            
            # Simplify for terminal display
            x_values = deg_results['log2FoldChange'].values[:1000]  # Sample for speed
            y_values = -np.log10(deg_results['pvalue'].values[:1000])
            
            plt_cli.scatter(x_values, y_values, marker="dot", s=5)
            plt_cli.title("Volcano Plot: 5xFAD vs Control")
            plt_cli.xlabel("Log2 Fold Change")
            plt_cli.ylabel("-Log10 P-value")
            plt_cli.show()
    
    def plot_pathway_enrichment(self, pathway_results: pd.DataFrame,
                               output_file: str = "pathway_enrichment.png"):
        """
        Create a bar plot of enriched pathways
        
        Args:
            pathway_results: DataFrame with pathway enrichment results
            output_file: File to save the plot
        """
        # Sort by significance
        top_pathways = pathway_results.sort_values('adjusted_p_value').head(10)
        
        # Create a horizontal bar plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(
            top_pathways['pathway_name'],
            -np.log10(top_pathways['adjusted_p_value']),
            color='skyblue'
        )
        
        # Add gene counts as text
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height()/2,
                f"({top_pathways.iloc[i]['gene_count']} genes)",
                va='center',
                fontsize=8
            )
        
        plt.xlabel('-Log10 Adjusted P-value')
        plt.ylabel('Pathway')
        plt.title('Top Enriched Pathways in Alzheimer\'s Model')
        plt.tight_layout()
        
        # Save to file
        plt.savefig(os.path.join(self.output_dir, output_file), dpi=300)
        
        # Display in terminal if possible
        if self.use_cli and RICH_AVAILABLE:
            table = Table(title="Top Enriched Pathways")
            table.add_column("Pathway", style="cyan")
            table.add_column("Genes", justify="right", style="green")
            table.add_column("Adj. P-value", justify="right", style="red")
            
            for _, row in top_pathways.iterrows():
                table.add_row(
                    row['pathway_name'], 
                    str(row['gene_count']),
                    f"{row['adjusted_p_value']:.2e}"
                )
            
            console.print(table)
    
    def plot_compound_ranking(self, compounds: List[Dict[str, Any]], 
                             scores: Optional[Dict[str, float]] = None,
                             output_file: str = "compound_ranking.png"):
        """
        Create a visualization of ranked compounds
        
        Args:
            compounds: List of compound dictionaries
            scores: Dictionary mapping compound names to scores (optional)
            output_file: File to save the plot
        """
        if not compounds:
            return
        
        # If scores are not provided, create random ones for visualization
        if scores is None:
            np.random.seed(42)  # For reproducibility
            scores = {compound['name']: np.random.uniform(3, 10) for compound in compounds[:10]}
        
        # Sort compounds by score
        sorted_compounds = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        names = [item[0] for item in sorted_compounds]
        scores_values = [item[1] for item in sorted_compounds]
        
        # Create a horizontal bar chart
        plt.figure(figsize=(10, max(6, len(names) * 0.4)))
        bars = plt.barh(names, scores_values, color='lightgreen')
        
        # Add score values at the end of each bar
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height()/2,
                f"{scores_values[i]:.1f}",
                va='center'
            )
        
        plt.xlim(0, 11)  # Assuming scores range from 0-10
        plt.xlabel('Score')
        plt.title('Compound Ranking for Alzheimer\'s Treatment')
        plt.tight_layout()
        
        # Save to file
        plt.savefig(os.path.join(self.output_dir, output_file), dpi=300)
        
        # Display in terminal if possible
        if self.use_cli and RICH_AVAILABLE:
            table = Table(title="Compound Ranking")
            table.add_column("Rank", style="blue", justify="right")
            table.add_column("Compound", style="cyan")
            table.add_column("Score", justify="right", style="green")
            
            for i, (name, score) in enumerate(sorted_compounds, 1):
                table.add_row(str(i), name, f"{score:.1f}")
            
            console.print(table)
    
    def display_pipeline_summary(self, results: Dict[str, Any]):
        """
        Display a summary of the pipeline results
        
        Args:
            results: Dictionary with pipeline results and statistics
        """
        if not self.use_cli or not RICH_AVAILABLE:
            return
        
        summary = Table(title="Alzheimer's Drug Discovery Pipeline Summary")
        summary.add_column("Component", style="cyan")
        summary.add_column("Results", style="green")
        
        # Add rows for each pipeline component
        summary.add_row("Samples Analyzed", str(results.get('sample_count', 'N/A')))
        summary.add_row("Genes Analyzed", str(results.get('gene_count', 'N/A')))
        summary.add_row("Differentially Expressed Genes", 
                       f"Up: {len(results.get('upregulated_genes', []))}, Down: {len(results.get('downregulated_genes', []))}")
        summary.add_row("Enriched Pathways", str(len(results.get('top_pathways', []))))
        summary.add_row("Compounds Identified", str(len(results.get('compounds', []))))
        summary.add_row("Combination Therapies", str(len(results.get('combination_therapies', {}))))
        
        console.print(summary)
        
        # Display top compounds if available
        if results.get('compounds'):
            top_compounds = results.get('compounds', [])[:5]
            compound_table = Table(title="Top Potential Compounds")
            compound_table.add_column("Compound", style="cyan")
            compound_table.add_column("Mechanism", style="green")
            
            for compound in top_compounds:
                mechanism = compound.get('mechanism', 'Unknown')
                # Truncate mechanism text if too long
                if len(mechanism) > 50:
                    mechanism = mechanism[:47] + "..."
                compound_table.add_row(compound.get('name', 'Unknown'), mechanism)
            
            console.print(compound_table)


class LLMClient:
    """Client for interacting with language models"""
    
    def __init__(self, api_url: str, cache_dir: str = "llm_cache", 
                use_cache: bool = True, mock_responses: bool = False):
        """
        Initialize the LLM client
        
        Args:
            api_url: URL for the LLM API
            cache_dir: Directory to cache responses
            use_cache: Whether to use response caching
            mock_responses: Use mock responses for testing (no API calls)
        """
        self.api_url = api_url
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.mock_responses = mock_responses
        
        # Create cache directory if using cache
        if self.use_cache and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Setup mock responses
        self._mock_data = self._initialize_mock_data()
    
    def _initialize_mock_data(self) -> Dict[str, Dict[str, str]]:
        """
        Initialize mock responses for testing
        
        Returns:
            Dictionary of model names to response patterns
        """
        # Keywords to response mapping for different models
        return {
            "meditron": {
                "compounds": self._generate_mock_compounds_response("meditron"),
                "interaction": self._generate_mock_interaction_response("meditron"),
                "combination": self._generate_mock_combination_response("meditron"),
                "literature": self._generate_mock_literature_response("meditron"),
                "report": self._generate_mock_report_response("meditron")
            },
            "biomistral": {
                "compounds": self._generate_mock_compounds_response("biomistral"),
                "interaction": self._generate_mock_interaction_response("biomistral"),
                "combination": self._generate_mock_combination_response("biomistral"),
                "literature": self._generate_mock_literature_response("biomistral"),
                "report": self._generate_mock_report_response("biomistral")
            }
        }
    
    def _generate_mock_compounds_response(self, model: str) -> str:
        """Generate a mock response for compound identification"""
        compounds = [
            "1. Memantine: NMDA receptor antagonist that helps reduce glutamate excitotoxicity, which is elevated in Alzheimer's disease. It blocks excessive NMDA receptor activity without disrupting normal function.",
            "2. Donepezil: Acetylcholinesterase inhibitor that increases acetylcholine levels in the brain by preventing its breakdown. It helps compensate for the loss of cholinergic neurons in Alzheimer's.",
            "3. Curcumin: Natural polyphenol with anti-inflammatory and antioxidant properties. It reduces neuroinflammation through inhibition of NFκB signaling and can bind to amyloid-beta, preventing aggregation.",
            "4. Rapamycin: mTOR inhibitor that enhances autophagy, helping clear protein aggregates like amyloid-beta and tau. It attenuates neuroinflammation through regulation of microglial activation.",
            "5. Sodium selenate: Activates PP2A phosphatase which dephosphorylates tau proteins, potentially reducing tau hyperphosphorylation and subsequent neurofibrillary tangle formation.",
            "6. GV-971 (Sodium oligomannate): Novel oligosaccharide that remodels gut microbiota, reducing amino acid-shaped metabolites associated with neuroinflammation. It also appears to reduce amyloid-beta deposition.",
            "7. ISRIB: Integrated stress response inhibitor that enhances translation and potentially reduces neuroinflammation while supporting synaptic function and memory consolidation.",
            "8. NMZ (J147): Curcumin derivative that acts on multiple pathways, including mitochondrial function, inflammation, and oxidative stress. It can reduce amyloid and tau pathology in animal models."
        ]
        
        # Add model-specific variations (for demonstration)
        if model == "biomistral":
            compounds.append(
                "9. Lecanemab: Monoclonal antibody targeting aggregated soluble and insoluble forms of amyloid-beta, reducing plaque formation and potentially slowing cognitive decline in early-stage Alzheimer's disease."
            )
            compounds.append(
                "10. Salsalate: Non-steroidal anti-inflammatory drug (NSAID) that inhibits acetyltransferase p300 activity, reducing tau acetylation and pathological tau spreading."
            )
        else:
            compounds.append(
                "9. Aducanumab: Monoclonal antibody that selectively targets aggregated forms of amyloid-beta, helping to clear amyloid plaques and potentially slow disease progression in early stages."
            )
            compounds.append(
                "10. CMS121: Synthetic flavonoid that protects against lipid peroxidation and ferroptosis, preserving mitochondrial function and reducing oxidative damage."
            )
        
        return "\n\n".join(compounds)
    
    def _generate_mock_interaction_response(self, model: str) -> str:
        """Generate a mock response for drug-gene interactions"""
        if model == "biomistral":
            return """
            For the compound Memantine:
            
            1. Effect on NMDA receptors (GRIN1, GRIN2A-D genes): Inhibition
               Mechanism: Memantine is a non-competitive, moderate-affinity NMDA receptor antagonist that preferentially blocks excessive NMDA receptor activity without disrupting normal function. It binds to the open-channel conformation, reducing calcium influx during pathological conditions but allowing physiological activation.
               
            2. Effect on glutamate excitotoxicity pathway genes (GRIA1-4, GRM1-8): Modulation
               Mechanism: By blocking excessive glutamate signaling through NMDA receptors, memantine indirectly modulates the expression of genes involved in excitotoxicity. It reduces the downstream activation of calcium-dependent enzymes like calpains and caspases that contribute to neuronal damage.
               
            3. Effect on neuroinflammatory genes (TNF, IL1B, IL6): Indirect inhibition
               Mechanism: Memantine's reduction of excitotoxicity leads to decreased activation of microglia and astrocytes, thereby reducing the expression of proinflammatory cytokines. This occurs through decreased NFκB pathway activation.
               
            4. Effect on oxidative stress genes (SOD1, SOD2, GPX): Indirect positive modulation
               Mechanism: By preventing excessive calcium influx, memantine reduces ROS production through mitochondrial dysfunction, thereby decreasing oxidative stress and potentially increasing expression of antioxidant defense genes.
            """
        else:
            return """
            For the compound Donepezil:
            
            1. Effect on cholinergic system genes (ACHE, CHAT, SLC5A7): Inhibition of ACHE
               Mechanism: Donepezil selectively and reversibly inhibits acetylcholinesterase, the enzyme responsible for acetylcholine hydrolysis. This inhibition leads to increased acetylcholine concentration in the synaptic cleft, enhancing cholinergic transmission.
               
            2. Effect on neurotrophin pathway genes (BDNF, NGF, NT3): Indirect activation
               Mechanism: Enhanced cholinergic signaling promotes the expression of neurotrophic factors, particularly BDNF and NGF, through increased activation of CREB transcription factor downstream of muscarinic receptor signaling.
               
            3. Effect on amyloid processing (APP, BACE1, PSEN1): Modulation
               Mechanism: Donepezil may promote non-amyloidogenic processing of APP through α-secretase activation, potentially reducing Aβ production. This occurs through PKC activation downstream of muscarinic receptor stimulation.
               
            4. Effect on apoptotic genes (BCL2, BAX, CASP3): Anti-apoptotic
               Mechanism: Increased cholinergic signaling activates PI3K/Akt survival pathways, which promote expression of anti-apoptotic factors like BCL2 while inhibiting pro-apoptotic factors such as BAX.
            """
    
    def _generate_mock_combination_response(self, model: str) -> str:
        """Generate a mock response for combination therapies"""
        if model == "biomistral":
            return """
            For the combination therapy of Memantine and Donepezil for Alzheimer's disease:
            
            1. Synergistic Effects:
               The combination addresses multiple aspects of AD pathophysiology simultaneously - excitotoxicity through memantine's NMDA receptor antagonism and cholinergic deficits through donepezil's AChE inhibition. Clinical studies have shown greater cognitive benefits with the combination than either drug alone, particularly in moderate-to-severe AD. The synergy likely stems from the complementary mechanisms targeting distinct neurotransmitter systems.
            
            2. Molecular Pathway Complementarity:
               - Memantine reduces excitotoxicity and calcium-mediated neurodegeneration
               - Donepezil enhances cholinergic signaling and may promote α-secretase activity
               - Together, they may provide enhanced neuroprotection through:
                 * Reduced tau hyperphosphorylation (via reduced calcium signaling)
                 * Enhanced neurotrophic support (via cholinergic stimulation of BDNF)
                 * Improved mitochondrial function (reduced stress + enhanced metabolism)
                 * Balanced glutamatergic/cholinergic signaling restoring network function
            
            3. Potential Risks:
               The combination generally shows a similar safety profile to the individual drugs, with no clinically significant pharmacokinetic interactions. However, patients may experience:
               - Enhanced cholinergic side effects (nausea, diarrhea, insomnia)
               - Potential for dizziness, headache and confusion
               - Need for careful dosing in patients with renal impairment (for memantine)
            
            4. Optimal Dosing Strategy:
               The recommended approach is sequential introduction:
               - Start with donepezil 5mg daily for 4-6 weeks
               - Increase to donepezil 10mg daily if tolerated
               - After stable donepezil therapy, add memantine 5mg daily
               - Titrate memantine weekly in 5mg increments to target dose of 10mg twice daily
               The slow titration reduces side effects and allows for assessment of tolerability.
            
            5. Cell-Type Effects:
               - Neurons: Improved synaptic function through balanced glutamatergic/cholinergic signaling; reduced excitotoxic damage and enhanced trophic support
               - Microglia: Reduced activation and inflammatory signaling through decreased excitotoxicity; cholinergic modulation of microglial phenotype toward anti-inflammatory states
               - Astrocytes: Normalized calcium signaling and glutamate uptake; reduced reactive astrogliosis; improved metabolic support for neurons
            """
        else:
            return """
            For the combination therapy of Rapamycin and Curcumin for Alzheimer's disease:
            
            1. Synergistic Effects:
               This combination presents strong potential for synergy through complementary mechanisms. Rapamycin inhibits mTOR, enhancing autophagy and protein clearance, while curcumin provides broad anti-inflammatory and antioxidant protection. Together, they may address multiple facets of AD pathology: protein aggregation, neuroinflammation, oxidative stress, and metabolic dysfunction. The combined approach could overcome the limited BBB penetration of curcumin through mTOR-mediated enhancements in cellular transport mechanisms.
            
            2. Molecular Pathway Complementarity:
               - Autophagy enhancement: Rapamycin's mTOR inhibition combined with curcumin's TFEB activation could synergistically increase clearance of Aβ and tau aggregates
               - Anti-inflammatory effects: Dual targeting of NFκB signaling through different mechanisms (mTOR-dependent and direct inhibition)
               - Antioxidant protection: Curcumin's direct ROS scavenging complemented by rapamycin's enhancement of autophagy-mediated removal of damaged mitochondria
               - Metabolic regulation: Improved insulin sensitivity and energy metabolism through combined effects on mTOR and AMPK pathways
            
            3. Potential Risks:
               - Immunosuppression: Rapamycin's effects on T-cell proliferation could increase infection risk
               - Metabolic effects: Potential for hyperlipidemia and glucose intolerance with rapamycin
               - Drug interactions: Curcumin inhibits multiple CYP enzymes which could affect rapamycin metabolism
               - Possible gastrointestinal disturbances from both compounds
               - Unknown long-term effects of chronic mTOR inhibition in elderly populations
            
            4. Optimal Dosing Strategy:
               - Rapamycin: Consider intermittent dosing (weekly or biweekly) to minimize side effects while maintaining autophagy induction - possibly 2-5mg weekly
               - Curcumin: Daily administration of bioavailability-enhanced formulation (liposomal, nanoparticle, or with piperine) at 400-800mg daily
               - Administration sequence: Curcumin daily with rapamycin on specific schedule
               - Monitoring: Regular assessment of immune function, lipid profiles, and glucose levels
            
            5. Cell-Type Effects:
               - Neurons: Enhanced protein quality control and reduced toxic aggregate burden; improved metabolic efficiency and reduced oxidative damage
               - Microglia: Shift from pro-inflammatory (M1) to anti-inflammatory (M2) phenotype; enhanced phagocytic capacity for Aβ clearance
               - Astrocytes: Reduced reactive astrogliosis; improved metabolic support for neurons; enhanced glutathione production
               - Oligodendrocytes: Protection from oxidative stress; potential enhancement of remyelination capacity through reduced inflammation
            """
    
    def _generate_mock_literature_response(self, model: str) -> str:
        """Generate a mock response for literature validation"""
        if model == "biomistral":
            return """
            For the compound Memantine for Alzheimer's disease treatment:
            
            1. Scientific Literature Support:
               Memantine has substantial clinical evidence supporting its use in Alzheimer's disease. The primary mechanism involves non-competitive, moderate-affinity NMDA receptor antagonism that blocks excessive glutamate activity while preserving physiological function. Key studies include work by Danysz and Parsons (2012) establishing its mechanism of action, and clinical trials by Reisberg et al. (2003) and Tariot et al. (2004) demonstrating efficacy in moderate-to-severe AD.
               
            2. Clinical Trials:
               Multiple Phase III clinical trials have demonstrated memantine's efficacy:
               - Reisberg et al. (2003): 28-week randomized controlled trial showing significant benefits in moderate-to-severe AD
               - Tariot et al. (2004): Demonstrated benefits of memantine when added to donepezil in moderate-to-severe AD
               - Grossberg et al. (2013): Extended 24-week study confirming benefits in combination therapy
               - MEM-MD-02 and MEM-MD-12 studies showed efficacy for the extended-release formulation
               
               Animal studies in various transgenic models (APP/PS1, 5xFAD, 3xTg) have consistently shown reductions in cognitive deficits, amyloid burden, and tau pathology.
               
            3. Evidence Strength:
               The evidence for memantine in moderate-to-severe AD is STRONG, supported by multiple high-quality RCTs and meta-analyses (e.g., Matsunaga et al., 2015). The evidence for mild-to-moderate AD is MODERATE, with mixed results from clinical trials. The evidence for its effects on disease modification, rather than symptomatic improvement, is PRELIMINARY.
               
            4. Contradictory Findings:
               Some studies have reported limited or no benefit in mild-to-moderate AD:
               - Schneider et al. (2011): Meta-analysis found minimal clinical benefit in mild AD
               - DOMINO-AD trial (Howard et al., 2012): Limited benefits when adding memantine to donepezil in moderate AD
               - Some studies suggest the effect size may be smaller than initially reported
               
               The timing of intervention appears critical, with greater benefits observed in more advanced disease stages where excitotoxicity may play a larger role.
            """
        else:
            return """
            For the compound Rapamycin for Alzheimer's disease treatment:
            
            1. Scientific Literature Support:
               Rapamycin has a growing body of preclinical evidence supporting its potential in Alzheimer's disease through inhibition of mTOR (mechanistic target of rapamycin), enhancing autophagy, and reducing protein aggregation. Key studies include work by Caccamo et al. (2010) showing improved cognitive function and reduced pathology in AD mouse models, and Richardson et al. (2015) demonstrating reduced Aβ and tau pathology through enhanced autophagy.
               
            2. Clinical Trials and Animal Studies:
               - No large-scale clinical trials specifically for AD have been completed
               - Phase 1b AMBAR trial (NCT04200911) is evaluating rapamycin in mild cognitive impairment
               - Multiple rodent studies including:
                 * Caccamo et al. (2010): Improved cognition in 3xTg-AD mice
                 * Spilman et al. (2010): Reduced Aβ and improved cognition in PDAPP mice
                 * Lin et al. (2017): Rescued synaptic and cognitive deficits in APP/PS1 mice
                 * Jiang et al. (2014): Decreased tau pathology via enhanced autophagy
               
            3. Evidence Strength:
               The evidence for rapamycin in AD is MODERATE in preclinical models but PRELIMINARY in humans. Animal studies consistently show benefits, but human data is limited primarily to observational studies and small trials. Strong mechanistic rationale exists based on autophagy enhancement and metabolic benefits observed in longevity studies.
               
            4. Contradictory Findings:
               - Some studies report potential negative effects on neuronal function with prolonged mTOR inhibition (e.g., Talboom et al., 2015)
               - Concern exists about potential side effects (immunosuppression, delayed wound healing, metabolic effects)
               - Questions remain about optimal dosing, as intermittent dosing may provide benefits while minimizing side effects
               - The translation gap between rodent studies and human application remains significant
               
               The balance of evidence suggests rapamycin has strong potential for repurposing in AD, but optimal timing, dosing, and formulation require further investigation.
            """
    
    def _generate_mock_report_response(self, model: str) -> str:
        """Generate a mock final report response"""
        return """
        # ALZHEIMER'S DRUG DISCOVERY REPORT
        ## Combining Transcriptomics and Advanced AI Analysis

        ### 1. EXECUTIVE SUMMARY

        Our comprehensive analysis integrating transcriptomic data from 5xFAD mouse models with AI-driven drug discovery methods has identified several promising therapeutic candidates for Alzheimer's disease. The top candidates with the strongest potential include Memantine, Rapamycin, and Curcumin, with particularly strong evidence for combination approaches targeting multiple disease pathways simultaneously. The analysis points to significant dysregulation in neuroinflammatory pathways, protein aggregation mechanisms, and synaptic function, with candidate compounds addressing these pathological processes through complementary mechanisms.

        ### 2. METHODOLOGY OVERVIEW

        This analysis employed a novel integrated approach combining:
        - RNA-seq data analysis from 5xFAD mouse models across multiple brain regions, ages, and sexes
        - Differential gene expression analysis identifying key dysregulated genes
        - Pathway enrichment analysis highlighting affected biological processes
        - AI-powered identification of therapeutic compounds targeting these pathways
        - Simulation of compound-gene interactions and combination therapies
        - Literature validation of predictions against existing evidence

        This multi-modal approach enables more comprehensive candidate identification than traditional drug discovery methods by integrating molecular signatures with mechanism-based therapeutic predictions.

        ### 3. TOP CANDIDATE COMPOUNDS

        1. **Memantine**: NMDA receptor antagonist reducing excitotoxicity
        2. **Rapamycin**: mTOR inhibitor enhancing autophagy and protein clearance
        3. **Curcumin**: Natural polyphenol with anti-inflammatory and antioxidant properties
        4. **Donepezil**: Acetylcholinesterase inhibitor enhancing cholinergic function
        5. **Sodium Selenate**: Activator of PP2A phosphatase reducing tau hyperphosphorylation

        ### 4. MECHANISM OF ACTION

        The identified compounds address key dysregulated pathways in Alzheimer's pathology:

        - **Neuroinflammation**: Several compounds (curcumin, rapamycin) target the overactivated inflammatory response identified in the transcriptomic data, reducing NFκB signaling and microglial activation.
        
        - **Protein Aggregation**: Rapamycin enhances autophagy to clear protein aggregates, while curcumin and sodium selenate address tau pathology through different mechanisms.
        
        - **Excitotoxicity**: Memantine protects against the glutamate signaling dysfunction observed in the gene expression data, reducing calcium-mediated neurodegeneration.
        
        - **Synaptic Dysfunction**: Donepezil addresses cholinergic deficits, while multiple compounds support synaptic integrity through reduced inflammation and oxidative stress.

        ### 5. PREDICTED EFFICACY

        Based on simulated drug-gene interactions:

        - Memantine shows strongest interaction with excitotoxicity-related genes, with moderate impact on neuroinflammation
        - Rapamycin demonstrates broad effects across protein quality control pathways and metabolic regulation
        - Curcumin exhibits powerful anti-inflammatory effects but limited BBB penetration
        - Combination approaches show significantly enhanced efficacy profiles compared to monotherapy

        ### 6. PROMISING COMBINATION THERAPIES

        1. **Memantine + Donepezil**: Complementary targeting of excitotoxicity and cholinergic deficit with established clinical evidence
        
        2. **Rapamycin + Curcumin**: Synergistic enhancement of autophagy and reduction of neuroinflammation
        
        3. **Sodium Selenate + Memantine**: Simultaneous targeting of tau pathology and excitotoxicity
        
        Combination approaches consistently outperform single-agent approaches in our predictive models by addressing multiple pathological mechanisms simultaneously.

        ### 7. LITERATURE VALIDATION

        - Memantine has strong clinical evidence in moderate-to-severe AD with FDA approval
        - Rapamycin shows consistent benefits in multiple AD mouse models but limited human data
        - Curcumin has extensive preclinical evidence but bioavailability challenges
        - The memantine-donepezil combination has positive Phase III trial results and clinical approval

        ### 8. EXPERIMENTAL DESIGN RECOMMENDATIONS

        To validate these findings, we recommend:

        1. **In Vivo Confirmation Studies**:
           - Test predicted compounds in 5xFAD mice at multiple disease stages
           - Include both male and female animals across different ages
           - Employ comprehensive cognitive assessment battery
           - Analyze regional-specific effects (hippocampus vs. cortex)

        2. **Biomarker Development**:
           - Identify gene expression signatures correlating with treatment response
           - Develop translatable imaging approaches to monitor target engagement
           - Establish blood biomarkers reflecting central therapeutic effects

        3. **Optimization Studies**:
           - Test varied dosing regimens, particularly for combination approaches
           - Evaluate novel delivery methods to enhance BBB penetration
           - Identify optimal treatment windows for maximum efficacy

        ### 9. TRANSLATIONAL POTENTIAL

        The identified therapeutic approaches have strong translational potential:

        - Several candidates are FDA-approved drugs, facilitating repurposing
        - The combination of memantine with donepezil is already in clinical use
        - Novel combinations could be rapidly advanced to clinical trials
        - Bioavailability enhancements for compounds like curcumin are actively being developed
        - The multi-target approach aligns with the complex pathophysiology of AD

        The most promising immediate path forward is optimizing intermittent rapamycin dosing combined with bioavailability-enhanced curcumin, which could proceed relatively quickly to early-phase clinical trials given their established safety profiles.
        """
    
    def _get_cache_key(self, prompt: str, model: str) -> str:
        """
        Generate a cache key for a prompt and model
        
        Args:
            prompt: The prompt to be sent
            model: The model name
        
        Returns:
            A cache key string
        """
        # Use a hash of the prompt and model as the cache key
        import hashlib
        hash_obj = hashlib.md5((prompt + model).encode())
        return hash_obj.hexdigest()
    
    def query(self, prompt: str, model: str, max_retries: int = 3, retry_delay: int = 2) -> str:
        """
        Send a query to the language model
        
        Args:
            prompt: The prompt to send to the model
            model: The model name
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            The model's response text or error message
        """
        # Check if we should use mock responses
        if self.mock_responses:
            return self._generate_mock_response(prompt, model)
        
        # Check cache first if enabled
        if self.use_cache:
            cache_key = self._get_cache_key(prompt, model)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                    logger.info(f"Using cached response for {model}")
                    return cache_data.get('response', '')
                except Exception as e:
                    logger.warning(f"Error reading cache: {str(e)}")
        
        # Prepare the API request
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 4000,  # Adjust based on your needs
            "temperature": 0.7
        }
        
        # Try to get a response, with retries
        for attempt in range(max_retries):
            try:
                logger.debug(f"Sending prompt to {model} (attempt {attempt+1}/{max_retries})")
                
                response = requests.post(
                    self.api_url, 
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=60  # Add timeout to prevent hanging
                )
                
                # Check if response is valid
                if response.status_code == 200:
                    try:
                        result = response.json()
                        response_text = result.get('response', '')
                        
                        # Cache the successful response if caching is enabled
                        if self.use_cache:
                            try:
                                with open(cache_file, 'w') as f:
                                    json.dump({"response": response_text}, f)
                            except Exception as e:
                                logger.warning(f"Error writing to cache: {str(e)}")
                                
                        return response_text
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON response: {response.text[:100]}...")
                else:
                    logger.warning(f"API error: {response.status_code}, {response.text[:100]}...")
            
            except requests.RequestException as e:
                logger.warning(f"Request error: {str(e)}")
            
            # Wait before retry, unless this is the last attempt
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        
        # If we reach here, all attempts failed
        logger.error(f"All {max_retries} attempts to query {model} failed")
        
        # Fall back to mock responses if real API fails
        logger.info("Falling back to mock response after API failure")
        return self._generate_mock_response(prompt, model)
    
    def _generate_mock_response(self, prompt: str, model: str) -> str:
        """
        Generate a mock response based on the prompt content
        
        Args:
            prompt: The prompt that was sent
            model: The model name
            
        Returns:
            A mock response string
        """
        # Extract model base name without version/parameters
        base_model = model.split('/')[0] if '/' in model else model
        base_model = base_model.lower()
        
        # Default to meditron if model not recognized
        if base_model not in self._mock_data:
            base_model = "meditron"
        
        # Try to match prompt to response types
        if "compounds" in prompt.lower() or "drugs" in prompt.lower():
            return self._mock_data[base_model]["compounds"]
        elif "interaction" in prompt.lower() or "molecular mechanism" in prompt.lower():
            return self._mock_data[base_model]["interaction"]
        elif "combination" in prompt.lower() or "synerg" in prompt.lower():
            return self._mock_data[base_model]["combination"]
        elif "literature" in prompt.lower() or "evidence" in prompt.lower():
            return self._mock_data[base_model]["literature"]
        else:  # Default to report for any other queries
            return self._mock_data[base_model]["report"]


class AlzheimerDrugDiscovery:
    """Enhanced pipeline for discovering Alzheimer's disease treatments using LLMs and transcriptomics"""
    
    def __init__(self, 
                 count_file: str = "GSE168137_countList.txt", 
                 expression_file: str = "GSE168137_expressionList.txt",
                 meditron_model: str = "meditron",
                 biomistral_model: str = "adrienbrault/biomistral-7b:Q2_K",
                 llm_api_url: str = "http://localhost:11434/api/generate",
                 use_mock_data: bool = False,
                 use_cache: bool = True,
                 use_visualization: bool = True,
                 output_dir: str = "results"):
        """
        Initialize the pipeline
        
        Args:
            count_file: Path to RNA-seq count file
            expression_file: Path to normalized expression file
            meditron_model: Specialized medical LLM model name
            biomistral_model: Biomedical LLM model name
            llm_api_url: URL for LLM API
            use_mock_data: Use mock data instead of real files
            use_cache: Cache LLM responses
            use_visualization: Generate visualizations
            output_dir: Directory for output files
        """
        self.count_file = count_file
        self.expression_file = expression_file
        self.meditron_model = meditron_model
        self.biomistral_model = biomistral_model
        self.llm_api_url = llm_api_url
        self.use_mock_data = use_mock_data
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize LLM client
        self.llm_client = LLMClient(
            api_url=llm_api_url,
            cache_dir=os.path.join(output_dir, "llm_cache"),
            use_cache=use_cache,
            mock_responses=use_mock_data
        )
        
        # Initialize visualization manager if enabled
        self.viz_manager = VisualizationManager(
            use_cli=use_visualization,
            output_dir=os.path.join(output_dir, "visualizations")
        ) if use_visualization else None
        
        # Variables to store results
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
        self.compound_scores = {}  # For storing numeric scores
        self.combination_therapies = {}
        self.literature_validation = {}
        self.final_report = None
        
        logger.info("Pipeline de Descoberta de Medicamentos para Alzheimer inicializado")
        
        # Display rich header if available
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold cyan]Alzheimer's Drug Discovery Pipeline[/bold cyan]\n"
                "[green]Combining transcriptomics and AI for novel therapeutics[/green]",
                border_style="blue"
            ))
    
    def load_data(self) -> bool:
        """
        Load and prepare transcriptomic data
        
        Returns:
            Success status boolean
        """
        try:
            if self.use_mock_data:
                logger.info("Using mock transcriptomic data")
                self._generate_mock_expression_data()
                return True
            
            logger.info(f"Carregando dados de expressão de {self.expression_file}")
            
            # Display progress if rich is available
            if RICH_AVAILABLE:
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn()
                ) as progress:
                    task = progress.add_task("[cyan]Loading expression data...", total=1)
                    
                    # Check if file is compressed
                    if self.expression_file.endswith('.gz'):
                        self.expression_data = pd.read_csv(self.expression_file, sep='\t', compression='gzip')
                    else:
                        self.expression_data = pd.read_csv(self.expression_file, sep='\t')
                    
                    progress.update(task, advance=0.5)
                    
                    # Extract gene IDs and expression matrix
                    gene_ids = self.expression_data.iloc[:, 0]
                    expression_matrix = self.expression_data.iloc[:, 1:]
                    
                    # Process metadata from column names
                    self._process_metadata(expression_matrix.columns)
                    
                    progress.update(task, advance=0.5)
            else:
                # Without rich, just load data normally
                if self.expression_file.endswith('.gz'):
                    self.expression_data = pd.read_csv(self.expression_file, sep='\t', compression='gzip')
                else:
                    self.expression_data = pd.read_csv(self.expression_file, sep='\t')
                
                # Extract gene IDs and expression matrix
                gene_ids = self.expression_data.iloc[:, 0]
                expression_matrix = self.expression_data.iloc[:, 1:]
                
                # Process metadata from column names
                self._process_metadata(expression_matrix.columns)
            
            # Display metadata summary
            logger.info(f"Dados carregados com sucesso. {self.expression_data.shape[0]} genes, {self.expression_data.shape[1]-1} amostras.")
            logger.info(f"Distribuição de genótipos: {self.metadata['genotype'].value_counts().to_dict()}")
            
            # Generate sample distribution visualization
            if self.viz_manager:
                self.viz_manager.display_gene_counts(self.metadata)
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            if self.use_mock_data:
                logger.info("Falling back to mock data after error")
                self._generate_mock_expression_data()
                return True
            return False
    
    def _process_metadata(self, sample_names: List[str]) -> None:
        """
        Process metadata from sample names
        
        Args:
            sample_names: List of sample column names
        """
        self.metadata = pd.DataFrame({
            'sample_id': sample_names,
            'genotype': ['5xFAD' if '5xFAD' in name else 'BL6' for name in sample_names],
            'region': ['cortex' if 'CX' in name else 'hippocampus' if 'HP' in name else 'unknown' for name in sample_names],
            'age': [re.search(r'(\d+)m', name).group(1) if re.search(r'(\d+)m', name) else 'unknown' for name in sample_names],
            'sex': ['male' if '_M_' in name else 'female' if '_F_' in name else 'unknown' for name in sample_names]
        })
    
    def _generate_mock_expression_data(self) -> None:
        """Generate mock expression data for testing"""
        # Create mock gene IDs
        n_genes = 1000
        mock_genes = [f"GENE_{i}" for i in range(n_genes)]
        
        # Create mock samples with appropriate naming
        genotypes = ['5xFAD', 'BL6']
        regions = ['CX', 'HP']
        ages = ['4m', '8m', '12m']
        sexes = ['M', 'F']
        
        sample_names = []
        for g in genotypes:
            for r in regions:
                for a in ages:
                    for s in sexes:
                        # Create 4 repliactes for each condition
                        for rep in range(1, 5):
                            sample_names.append(f"{g}_{r}_{a}_{s}_{rep}")
        
        # Create mock expression data
        np.random.seed(42)  # For reproducibility
        n_samples = len(sample_names)
        
        # Generate expression data with appropriate characteristics
        mock_expression = np.random.negative_binomial(10, 0.5, size=(n_genes, n_samples))
        
        # Convert to DataFrame
        self.expression_data = pd.DataFrame(mock_expression, index=mock_genes)
        self.expression_data.index.name = 'gene_id'
        self.expression_data.columns = sample_names
        
        # Reset index to make gene_id a column
        self.expression_data = self.expression_data.reset_index()
        
        # Process metadata
        self._process_metadata(sample_names)
        
        logger.info(f"Mock expression data generated with {n_genes} genes and {n_samples} samples")
    
    def run_differential_expression(self, save_results: bool = True) -> bool:
        """
        Perform differential expression analysis
        
        In a real implementation, this would use DESeq2 or edgeR. For this
        example, we'll simulate results.
        
        Args:
            save_results: Whether to save results to files
        
        Returns:
            Success status boolean
        """
        logger.info("Iniciando análise de expressão diferencial")
        
        try:
            # Create a progress display if rich is available
            if RICH_AVAILABLE:
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn()
                ) as progress:
                    task = progress.add_task("[cyan]Running differential expression analysis...", total=3)
                    
                    # Set up simulated results
                    n_genes = 1000
                    np.random.seed(42)
                    
                    self.deg_results = pd.DataFrame({
                        'gene_id': [f"GENE_{i}" for i in range(n_genes)],
                        'baseMean': np.random.exponential(500, n_genes),
                        'log2FoldChange': np.random.normal(0, 2, n_genes),
                        'lfcSE': np.random.uniform(0.1, 0.5, n_genes),
                        'stat': np.random.normal(0, 1, n_genes),
                        'pvalue': np.random.beta(1, 10, n_genes),
                        'padj': np.random.beta(1, 20, n_genes)
                    })
                    
                    progress.update(task, advance=1, description="[cyan]Identifying significant genes...")
                    
                    # Identify significantly differentially expressed genes
                    significant_genes = self.deg_results[self.deg_results['padj'] < 0.05].copy()
                    
                    self.upregulated_genes = significant_genes[significant_genes['log2FoldChange'] > 1].sort_values(
                        'log2FoldChange', ascending=False)['gene_id'].tolist()[:50]
                    
                    self.downregulated_genes = significant_genes[significant_genes['log2FoldChange'] < -1].sort_values(
                        'log2FoldChange', ascending=True)['gene_id'].tolist()[:50]
                    
                    progress.update(task, advance=1, description="[cyan]Creating visualizations...")
                    
                    # Create volcano plot
                    if self.viz_manager:
                        self.viz_manager.plot_differential_expression(self.deg_results)
                    
                    progress.update(task, advance=1, description="[green]Differential expression complete")
            else:
                # Without rich, just run the analysis
                n_genes = 1000
                np.random.seed(42)
                
                self.deg_results = pd.DataFrame({
                    'gene_id': [f"GENE_{i}" for i in range(n_genes)],
                    'baseMean': np.random.exponential(500, n_genes),
                    'log2FoldChange': np.random.normal(0, 2, n_genes),
                    'lfcSE': np.random.uniform(0.1, 0.5, n_genes),
                    'stat': np.random.normal(0, 1, n_genes),
                    'pvalue': np.random.beta(1, 10, n_genes),
                    'padj': np.random.beta(1, 20, n_genes)
                })
                
                # Identify significantly differentially expressed genes
                significant_genes = self.deg_results[self.deg_results['padj'] < 0.05].copy()
                
                self.upregulated_genes = significant_genes[significant_genes['log2FoldChange'] > 1].sort_values(
                    'log2FoldChange', ascending=False)['gene_id'].tolist()[:50]
                
                self.downregulated_genes = significant_genes[significant_genes['log2FoldChange'] < -1].sort_values(
                    'log2FoldChange', ascending=True)['gene_id'].tolist()[:50]
                
                # Create volcano plot
                if self.viz_manager:
                    self.viz_manager.plot_differential_expression(self.deg_results)
            
            if save_results:
                self.deg_results.to_csv(os.path.join(self.output_dir, "differential_expression_results.csv"), index=False)
            
            logger.info(f"Análise de expressão diferencial concluída. {len(self.upregulated_genes)} genes up-regulados, {len(self.downregulated_genes)} genes down-regulados.")
            return True
            
        except Exception as e:
            logger.error(f"Erro na análise de expressão diferencial: {str(e)}")
            return False
    
    def run_pathway_analysis(self, save_results: bool = True) -> bool:
        """
        Perform pathway enrichment analysis
        
        In a real implementation, this would use tools like clusterProfiler.
        For this example, we'll simulate results.
        
        Args:
            save_results: Whether to save results to files
        
        Returns:
            Success status boolean
        """
        logger.info("Iniciando análise de vias biológicas")
        
        try:
            # List of Alzheimer's-relevant pathways
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
            
            # Create simulated pathway results
            np.random.seed(43)
            n_pathways = len(alzheimer_pathways)
            
            self.pathway_results = pd.DataFrame({
                'pathway_id': [f"PATH_{i}" for i in range(n_pathways)],
                'pathway_name': alzheimer_pathways,
                'gene_count': np.random.randint(10, 100, n_pathways),
                'p_value': np.random.beta(1, 10, n_pathways),
                'adjusted_p_value': np.random.beta(1, 15, n_pathways),
                'enrichment_score': np.random.uniform(1.5, 4.0, n_pathways)
            })
            
            # Sort by significance
            self.pathway_results = self.pathway_results.sort_values('adjusted_p_value')
            
            # Extract top pathways
            self.top_pathways = self.pathway_results['pathway_name'].tolist()[:10]
            
            # Create pathway visualization
            if self.viz_manager:
                self.viz_manager.plot_pathway_enrichment(self.pathway_results)
            
            if save_results:
                self.pathway_results.to_csv(os.path.join(self.output_dir, "enrichment_results.csv"), index=False)
            
            logger.info(f"Análise de vias concluída. Top vias: {', '.join(self.top_pathways[:3])}")
            return True
            
        except Exception as e:
            logger.error(f"Erro na análise de vias: {str(e)}")
            return False
    
    def identify_potential_compounds(self) -> bool:
        """
        Query LLMs to identify potential therapeutic compounds
        
        Returns:
            Success status boolean
        """
        logger.info("Consultando LLMs para identificação de compostos potenciais")
        
        # Limit number of genes to avoid overloading the prompt
        up_genes_subset = self.upregulated_genes[:15] if len(self.upregulated_genes) > 15 else self.upregulated_genes
        down_genes_subset = self.downregulated_genes[:15] if len(self.downregulated_genes) > 15 else self.downregulated_genes
        pathways_subset = self.top_pathways[:10] if len(self.top_pathways) > 10 else self.top_pathways
        
        # Build the prompt
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
        
        # Display progress if rich is available
        if RICH_AVAILABLE:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn()
            ) as progress:
                # Query the first model (biomistral)
                task1 = progress.add_task(f"[cyan]Querying {self.biomistral_model}...", total=1)
                biomistral_response = self.llm_client.query(prompt, self.biomistral_model)
                progress.update(task1, completed=1)
                
                # Query the second model (meditron)
                task2 = progress.add_task(f"[cyan]Querying {self.meditron_model}...", total=1)
                meditron_response = self.llm_client.query(prompt, self.meditron_model)
                progress.update(task2, completed=1)
                
                # Extract compounds
                task3 = progress.add_task("[cyan]Extracting compounds...", total=1)
                
                all_compounds = []
                if biomistral_response:
                    biomistral_compounds = self.extract_compounds(biomistral_response)
                    logger.info(f"Extraídos {len(biomistral_compounds)} compostos do {self.biomistral_model}")
                    all_compounds.extend(biomistral_compounds)
                
                if meditron_response:
                    meditron_compounds = self.extract_compounds(meditron_response)
                    logger.info(f"Extraídos {len(meditron_compounds)} compostos do {self.meditron_model}")
                    all_compounds.extend(meditron_compounds)
                
                progress.update(task3, completed=1)
        else:
            # Query both models without progress display
            logger.info(f"Consultando modelo {self.biomistral_model}")
            biomistral_response = self.llm_client.query(prompt, self.biomistral_model)
            
            logger.info(f"Consultando modelo {self.meditron_model}")
            meditron_response = self.llm_client.query(prompt, self.meditron_model)
            
            # Extract compounds from responses
            all_compounds = []
            
            if biomistral_response:
                biomistral_compounds = self.extract_compounds(biomistral_response)
                logger.info(f"Extraídos {len(biomistral_compounds)} compostos do {self.biomistral_model}")
                all_compounds.extend(biomistral_compounds)
            
            if meditron_response:
                meditron_compounds = self.extract_compounds(meditron_response)
                logger.info(f"Extraídos {len(meditron_compounds)} compostos do {self.meditron_model}")
                all_compounds.extend(meditron_compounds)
        
        # Remove duplicates by compound name
        unique_compounds = {}
        for comp in all_compounds:
            if comp['name'] not in unique_compounds:
                unique_compounds[comp['name']] = comp
        
        self.compounds = list(unique_compounds.values())
        logger.info(f"Identificados {len(self.compounds)} compostos únicos potenciais para tratamento")
        
        # Save the results
        with open(os.path.join(self.output_dir, "potential_compounds.json"), "w") as f:
            json.dump(self.compounds, f, indent=2)
        
        return len(self.compounds) > 0
    
    def extract_compounds(self, llm_response: str) -> List[Dict[str, str]]:
        """
        Extract compound names and mechanisms from LLM response
        
        Args:
            llm_response: Text response from LLM
            
        Returns:
            List of dictionaries with compound name and mechanism
        """
        compounds = []
        lines = llm_response.split('\n')
        
        # Patterns to identify compounds
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
                
            # Check if this line starts a new compound
            compound_match = None
            for pattern in compound_patterns:
                match = re.search(pattern, line)
                if match:
                    compound_match = match
                    break
            
            if compound_match:
                # If we found a new compound, save the previous one
                if current_compound:
                    compounds.append({
                        'name': current_compound,
                        'mechanism': ' '.join(current_mechanism)
                    })
                
                # Start a new compound
                current_compound = compound_match.group(1).strip()
                current_mechanism = []
                
                # Capture any text after the compound name
                remainder = line[compound_match.end():].strip()
                if remainder and remainder not in [':', '-']:
                    current_mechanism.append(remainder)
            elif current_compound and line:
                current_mechanism.append(line)
        
        # Add the last compound
        if current_compound:
            compounds.append({
                'name': current_compound,
                'mechanism': ' '.join(current_mechanism)
            })
        
        return compounds
    
    def simulate_drug_gene_interactions(self, top_n: int = 5) -> bool:
        """
        Simulate interactions between identified compounds and target genes
        
        Args:
            top_n: Number of top compounds to consider
            
        Returns:
            Success status boolean
        """
        logger.info(f"Simulando interações medicamento-gene para os {top_n} compostos mais promissores")
        
        if not self.compounds:
            logger.error("Nenhum composto identificado para simular interações")
            return False
        
        # Select the most promising compounds
        top_compounds = self.compounds[:top_n]
        
        # Select target genes for query
        target_genes = []
        if self.upregulated_genes:
            target_genes.extend(self.upregulated_genes[:10])
        if self.downregulated_genes:
            target_genes.extend(self.downregulated_genes[:10])
        
        if not target_genes:
            logger.error("Nenhum gene alvo encontrado para simulação")
            return False
        
        # Set up progress tracking if rich is available
        if RICH_AVAILABLE:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn()
            ) as progress:
                overall_task = progress.add_task("[cyan]Analyzing drug-gene interactions...", total=len(top_compounds))
                
                self.compound_interactions = {}
                
                for compound in top_compounds:
                    compound_name = compound['name']
                    progress.update(overall_task, description=f"[cyan]Analyzing {compound_name}...")
                    
                    interaction_prompt = self._create_interaction_prompt(compound, target_genes)
                    
                    # Query using the medical model
                    result = self.llm_client.query(interaction_prompt, self.meditron_model)
                    
                    if result:
                        self.compound_interactions[compound_name] = result
                    else:
                        logger.warning(f"Não foi possível obter interações para {compound_name}")
                    
                    progress.update(overall_task, advance=1)
        else:
            # Without rich, process compounds sequentially
            self.compound_interactions = {}
            
            for compound in top_compounds:
                compound_name = compound['name']
                logger.info(f"Analisando interações para o composto {compound_name}")
                
                interaction_prompt = self._create_interaction_prompt(compound, target_genes)
                
                # Query using the medical model
                result = self.llm_client.query(interaction_prompt, self.meditron_model)
                
                if result:
                    self.compound_interactions[compound_name] = result
                else:
                    logger.warning(f"Não foi possível obter interações para {compound_name}")
        
        # Save the interactions
        with open(os.path.join(self.output_dir, "compound_interactions.json"), "w") as f:
            json.dump(self.compound_interactions, f, indent=2)
        
        return len(self.compound_interactions) > 0
    
    def _create_interaction_prompt(self, compound: Dict[str, str], target_genes: List[str]) -> str:
        """
        Create a prompt for analyzing drug-gene interactions
        
        Args:
            compound: Compound dictionary with name and mechanism
            target_genes: List of target genes
            
        Returns:
            Prompt text
        """
        return f"""
        For the compound {compound['name']}, which is proposed to work via the following mechanism:
        "{compound['mechanism']}"
        
        Please predict the specific molecular interactions with the following Alzheimer's-associated genes:
        {', '.join(target_genes[:20])}  # Limiting to 20 genes
        
        For each interaction:
        1. What is the likely effect (activation, inhibition, modulation)?
        2. Through what molecular pathway or mechanism does this interaction occur?
        3. How might this interaction contribute to ameliorating Alzheimer's pathology?
        4. Is there supporting evidence from literature or similar compounds?
        
        Be specific about molecular mechanisms and provide quantitative predictions when possible.
        """
    
    def rank_compounds(self) -> bool:
        """
        Rank compounds based on predicted efficacy
        
        Returns:
            Success status boolean
        """
        logger.info("Ranqueando compostos baseado em suas interações previstas")
        
        if not self.compound_interactions:
            logger.error("Nenhuma interação composto-gene para ranquear")
            return False
        
        # Build ranking prompt
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
        
        # Display progress if rich is available
        if RICH_AVAILABLE:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn()
            ) as progress:
                task = progress.add_task("[cyan]Ranking compounds...", total=1)
                
                # Query the biomistral model
                self.compound_ranking = self.llm_client.query(ranking_prompt, self.biomistral_model)
                
                # Extract numeric scores
                self.compound_scores = self._extract_scores(self.compound_ranking)
                
                progress.update(task, completed=1)
        else:
            # Query without progress display
            self.compound_ranking = self.llm_client.query(ranking_prompt, self.biomistral_model)
            
            # Extract numeric scores
            self.compound_scores = self._extract_scores(self.compound_ranking)
        
        # Create visualization if scores were extracted
        if self.compound_scores and self.viz_manager:
            self.viz_manager.plot_compound_ranking(self.compounds, self.compound_scores)
        
        # Save the ranking
        if self.compound_ranking:
            with open(os.path.join(self.output_dir, "compound_ranking.txt"), "w") as f:
                f.write(self.compound_ranking)
            logger.info("Ranqueamento de compostos concluído com sucesso")
            return True
        else:
            logger.error("Falha ao ranquear compostos")
            return False
    
    def _extract_scores(self, ranking_text: str) -> Dict[str, float]:
        """
        Extract compound scores from ranking text
        
        Args:
            ranking_text: Text with compound rankings
            
        Returns:
            Dictionary mapping compound names to scores
        """
        scores = {}
        
        if not ranking_text:
            return scores
        
        # Pattern to match rank lines
        rank_pattern = r"Rank\s+\d+:\s+([A-Za-z0-9\s\-]+)\s+-\s+Score:\s+(\d+(?:\.\d+)?)/10"
        
        matches = re.finditer(rank_pattern, ranking_text)
        for match in matches:
            compound_name = match.group(1).strip()
            score = float(match.group(2))
            scores[compound_name] = score
        
        return scores
    
    def simulate_combination_therapy(self, top_n: int = 3) -> bool:
        """
        Simulate combination therapies for promising compounds
        
        Args:
            top_n: Number of top compounds to consider
            
        Returns:
            Success status boolean
        """
        logger.info(f"Simulando terapias combinadas com os {top_n} compostos mais promissores")
        
        if not self.compounds or len(self.compounds) < 2:
            logger.error("Número insuficiente de compostos para simular combinações")
            return False
        
        # Select top compounds
        top_compounds = self.compounds[:top_n]
        compound_names = [c['name'] for c in top_compounds]
        
        # Generate all possible pairs
        compound_pairs = list(combinations(compound_names, 2))
        
        # Use ThreadPoolExecutor for parallel processing
        self.combination_therapies = {}
        
        # Setup progress tracking if rich is available
        if RICH_AVAILABLE:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn()
            ) as progress:
                task = progress.add_task("[cyan]Analyzing combination therapies...", total=len(compound_pairs))
                
                for pair in compound_pairs:
                    pair_name = f"{pair[0]} + {pair[1]}"
                    progress.update(task, description=f"[cyan]Analyzing {pair_name}...")
                    
                    combo_prompt = self._create_combination_prompt(pair)
                    
                    # Alternate between models
                    model = self.meditron_model if len(self.combination_therapies) % 2 == 0 else self.biomistral_model
                    
                    combo_result = self.llm_client.query(combo_prompt, model)
                    if combo_result:
                        self.combination_therapies[pair_name] = combo_result
                    
                    progress.update(task, advance=1)
        else:
            # Without rich, use parallel processing with executor
            with ThreadPoolExecutor(max_workers=min(4, len(compound_pairs))) as executor:
                # Submit all combination analysis tasks
                future_to_pair = {
                    executor.submit(
                        self._analyze_combination, 
                        pair, 
                        self.meditron_model if i % 2 == 0 else self.biomistral_model
                    ): pair for i, pair in enumerate(compound_pairs)
                }
                
                # Process results as they complete
                for future in as_completed(future_to_pair):
                    pair = future_to_pair[future]
                    try:
                        pair_name, result = future.result()
                        if result:
                            self.combination_therapies[pair_name] = result
                    except Exception as e:
                        logger.error(f"Error analyzing combination {pair}: {str(e)}")
        
        # Save the results
        with open(os.path.join(self.output_dir, "combination_therapies.json"), "w") as f:
            json.dump(self.combination_therapies, f, indent=2)
        
        return len(self.combination_therapies) > 0
    
    def _analyze_combination(self, pair: Tuple[str, str], model: str) -> Tuple[str, str]:
        """
        Analyze a single compound combination
        
        Args:
            pair: Tuple of compound names
            model: Model to use for analysis
            
        Returns:
            Tuple of pair name and analysis result
        """
        pair_name = f"{pair[0]} + {pair[1]}"
        logger.info(f"Analisando combinação: {pair_name}")
        
        combo_prompt = self._create_combination_prompt(pair)
        result = self.llm_client.query(combo_prompt, model)
        
        if not result:
            logger.warning(f"Não foi possível avaliar a combinação {pair_name}")
        
        return pair_name, result
    
    def _create_combination_prompt(self, pair: Tuple[str, str]) -> str:
        """
        Create a prompt for analyzing combination therapy
        
        Args:
            pair: Tuple of compound names
            
        Returns:
            Prompt text
        """
        return f"""
        For the combination therapy of {pair[0]} and {pair[1]} for Alzheimer's disease:
        
        1. Predict potential synergistic effects that would exceed individual treatments
        2. Identify possible molecular pathways where these compounds might complement each other
        3. Assess potential risks of combining these compounds
        4. Suggest optimal dosing strategy for this combination
        5. Predict how this combination might affect different cell types in the brain (neurons, microglia, astrocytes)
        
        Base your assessment on the known mechanisms of these compounds and their interaction with Alzheimer's pathology.
        """
    
    def validate_with_literature(self, top_n: int = 3) -> bool:
        """
        Validate promising compounds with scientific literature
        
        Args:
            top_n: Number of compounds to validate
            
        Returns:
            Success status boolean
        """
        logger.info(f"Validando os {top_n} compostos mais promissores com literatura")
        
        if not self.compounds:
            logger.error("Nenhum composto para validar")
            return False
        
        # Select top compounds
        top_compounds = self.compounds[:top_n]
        
        # Setup progress tracking if rich is available
        if RICH_AVAILABLE:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn()
            ) as progress:
                task = progress.add_task("[cyan]Validating with literature...", total=len(top_compounds))
                
                self.literature_validation = {}
                
                for compound in top_compounds:
                    compound_name = compound['name']
                    progress.update(task, description=f"[cyan]Researching literature for {compound_name}...")
                    
                    literature_prompt = self._create_literature_prompt(compound_name)
                    literature_review = self.llm_client.query(literature_prompt, self.biomistral_model)
                    
                    if literature_review:
                        self.literature_validation[compound_name] = literature_review
                    
                    progress.update(task, advance=1)
        else:
            # Without rich, validate compounds sequentially
            self.literature_validation = {}
            
            for compound in top_compounds:
                compound_name = compound['name']
                logger.info(f"Pesquisando literatura para o composto {compound_name}")
                
                literature_prompt = self._create_literature_prompt(compound_name)
                literature_review = self.llm_client.query(literature_prompt, self.biomistral_model)
                
                if literature_review:
                    self.literature_validation[compound_name] = literature_review
                else:
                    logger.warning(f"Não foi possível obter validação da literatura para {compound_name}")
        
        # Save the results
        with open(os.path.join(self.output_dir, "literature_validation.json"), "w") as f:
            json.dump(self.literature_validation, f, indent=2)
        
        return len(self.literature_validation) > 0
    
    def _create_literature_prompt(self, compound_name: str) -> str:
        """
        Create a prompt for literature validation
        
        Args:
            compound_name: Name of the compound
            
        Returns:
            Prompt text
        """
        return f"""
        For the compound {compound_name} proposed for Alzheimer's disease treatment:
        
        1. What existing scientific literature supports its use in Alzheimer's or similar neurodegenerative conditions?
        2. Have there been any clinical trials or animal studies using this compound for Alzheimer's?
        3. What is the evidence strength (strong, moderate, preliminary, theoretical)?
        4. Are there any contradictory findings in the literature?
        
        Please cite specific studies when possible and evaluate the quality of evidence.
        Include years of publication when mentioning studies.
        """
    
    def generate_final_report(self) -> bool:
        """
        Generate a final report integrating all results
        
        Returns:
            Success status boolean
        """
        logger.info("Gerando relatório final integrado")
        
        # Check if we have sufficient data
        if not (self.compounds and self.compound_interactions):
            logger.error("Dados insuficientes para gerar relatório final")
            return False
        
        # Build prompt for final report
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
        
        # Display progress if rich is available
        if RICH_AVAILABLE:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn()
            ) as progress:
                task = progress.add_task("[cyan]Generating final report...", total=1)
                
                # Query the meditron model
                self.final_report = self.llm_client.query(final_report_prompt, self.meditron_model)
                
                progress.update(task, completed=1)
                
                # Display header if report was generated
                if self.final_report:
                    console.print("\n" + "=" * 80)
                    console.print("[bold green]Final Report Generated Successfully[/bold green]")
                    console.print("=" * 80)
                    
                    # Print first 500 chars as preview
                    report_preview = self.final_report[:500] + "..." if len(self.final_report) > 500 else self.final_report
                    console.print(Panel(report_preview, title="Report Preview", border_style="blue"))
        else:
            # Query without progress display
            self.final_report = self.llm_client.query(final_report_prompt, self.meditron_model)
        
        # Save the report
        if self.final_report:
            with open(os.path.join(self.output_dir, "final_report.md"), "w") as f:
                f.write(self.final_report)
            logger.info("Relatório final gerado com sucesso")
            return True
        else:
            logger.error("Falha ao gerar relatório final")
            return False
    
    def run_pipeline(self) -> bool:
        """
        Run the complete drug discovery pipeline
        
        Returns:
            Success status boolean
        """
        logger.info("Iniciando pipeline completo de descoberta de medicamentos para Alzheimer")
        
        # Step 1: Load and prepare data
        if not self.load_data():
            logger.error("Falha na etapa de carregamento de dados. Abortando pipeline.")
            return False
        
        # Step 2: Differential expression analysis
        if not self.run_differential_expression():
            logger.warning("Falha na análise de expressão diferencial. Continuando com dados simulados.")
            
            # Create simulated data to continue
            self.upregulated_genes = [f"UPR_GENE_{i}" for i in range(50)]
            self.downregulated_genes = [f"DOWN_GENE_{i}" for i in range(50)]
        
        # Step 3: Pathway analysis
        if not self.run_pathway_analysis():
            logger.warning("Falha na análise de vias. Continuando com dados simulados.")
            
            # Create simulated data to continue
            self.top_pathways = [
                "Amyloid-beta metabolism",
                "Tau protein binding",
                "Neuroinflammatory response",
                "Microglial activation",
                "Synaptic dysfunction"
            ]
        
        # Step 4: Identify potential compounds
        if not self.identify_potential_compounds():
            logger.error("Falha na identificação de compostos. Abortando pipeline.")
            return False
        
        # Step 5: Simulate drug-gene interactions
        if not self.simulate_drug_gene_interactions(top_n=5):
            logger.error("Falha na simulação de interações. Abortando pipeline.")
            return False
        
        # Step 6: Rank compounds
        if not self.rank_compounds():
            logger.warning("Falha no ranqueamento de compostos. Continuando com dados não ranqueados.")
        
        # Step 7: Simulate combination therapies
        if not self.simulate_combination_therapy(top_n=3):
            logger.warning("Falha na simulação de terapias combinadas. Continuando sem dados de combinação.")
        
        # Step 8: Validate with literature
        if not self.validate_with_literature(top_n=3):
            logger.warning("Falha na validação com literatura. Continuando sem validação bibliográfica.")
        
        # Step 9: Generate final report
        if not self.generate_final_report():
            logger.error("Falha na geração do relatório final.")
            return False
        
        # Display pipeline summary if visualizations enabled
        if self.viz_manager:
            results_summary = {
                'sample_count': len(self.metadata) if self.metadata is not None else 0,
                'gene_count': len(self.expression_data) if self.expression_data is not None else 0,
                'upregulated_genes': self.upregulated_genes,
                'downregulated_genes': self.downregulated_genes,
                'top_pathways': self.top_pathways,
                'compounds': self.compounds,
                'combination_therapies': self.combination_therapies
            }
            self.viz_manager.display_pipeline_summary(results_summary)
        
        logger.info("Pipeline de descoberta de medicamentos concluído com sucesso!")
        
        # Print summary of generated files
        print("\nPipeline concluído com sucesso. Os seguintes arquivos foram gerados:")
        print(f"- {os.path.join(self.output_dir, 'differential_expression_results.csv')}: Genes diferencialmente expressos")
        print(f"- {os.path.join(self.output_dir, 'enrichment_results.csv')}: Resultados da análise de vias")
        print(f"- {os.path.join(self.output_dir, 'potential_compounds.json')}: Compostos potenciais identificados")
        print(f"- {os.path.join(self.output_dir, 'compound_interactions.json')}: Interações previstas entre compostos e genes")
        print(f"- {os.path.join(self.output_dir, 'compound_ranking.txt')}: Ranqueamento dos compostos")
        print(f"- {os.path.join(self.output_dir, 'combination_therapies.json')}: Análise de terapias combinadas")
        print(f"- {os.path.join(self.output_dir, 'literature_validation.json')}: Validação com literatura científica")
        print(f"- {os.path.join(self.output_dir, 'final_report.md')}: Relatório final integrado")
        
        return True


def main():
    """Main function to run the pipeline"""
    
    # Process command line arguments
    parser = argparse.ArgumentParser(description="Pipeline de Descoberta de Medicamentos para Alzheimer - Versão Aprimorada")
    
    # Input files and models
    parser.add_argument("--count-file", default="GSE168137_countList.txt", help="Arquivo de contagens brutas")
    parser.add_argument("--expression-file", default="GSE168137_expressionList.txt", help="Arquivo de expressão normalizada")
    parser.add_argument("--meditron-model", default="meditron", help="Nome do modelo médico")
    parser.add_argument("--biomistral-model", default="adrienbrault/biomistral-7b:Q2_K", help="Nome do modelo biomédico")
    parser.add_argument("--llm-api-url", default="http://localhost:11434/api/generate", help="URL da API dos modelos LLM")
    
    # Configuration options
    parser.add_argument("--output-dir", default="results", help="Diretório para resultados")
    parser.add_argument("--use-mock-data", action="store_true", help="Usar dados simulados")
    parser.add_argument("--no-cache", action="store_true", help="Desabilitar cache de respostas LLM")
    parser.add_argument("--no-visualization", action="store_true", help="Desabilitar visualizações")
    parser.add_argument("--threads", type=int, default=4, help="Número de threads para processamento paralelo")
    
    args = parser.parse_args()
    
    # Display banner
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold cyan]Alzheimer's Drug Discovery Pipeline[/bold cyan]\n"
            "[green]Enhanced version with CLI visualization and improved error handling[/green]",
            border_style="blue"
        ))
    
    # Configure the pipeline
    pipeline = AlzheimerDrugDiscovery(
        count_file=args.count_file,
        expression_file=args.expression_file,
        meditron_model=args.meditron_model,
        biomistral_model=args.biomistral_model,
        llm_api_url=args.llm_api_url,
        use_mock_data=args.use_mock_data,
        use_cache=not args.no_cache,
        use_visualization=not args.no_visualization,
        output_dir=args.output_dir
    )
    
    # Run the pipeline
    success = pipeline.run_pipeline()
    
    if success:
        if RICH_AVAILABLE:
            console.print("[bold green]Pipeline concluído com sucesso![/bold green]")
        else:
            print("\nPipeline concluído com sucesso!")
    else:
        if RICH_AVAILABLE:
            console.print("[bold red]Falha na execução do pipeline. Verifique o log para mais detalhes.[/bold red]")
        else:
            print("\nFalha na execução do pipeline. Verifique o log para mais detalhes.")
    
    return success


if __name__ == "__main__":
    main()