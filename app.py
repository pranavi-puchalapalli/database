import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

# Configure page settings
st.set_page_config(
    page_title="GutBiomeIndex & Analysis Tool",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load data from GitHub
@st.cache_data
def load_reference_data():
    """
    Load the gut microbiome reference database from GitHub.
    Uses the EBI Metagenomics Sequential Genomes TSV file.
    """
    try:
        # GitHub URL for the EBI Metagenomics Sequential Genomes TSV file
        url = "ebi_metagenomics_sequential_genomes.tsv"
        response = requests.get(url)
        
        if response.status_code == 200:
            # Load TSV data
            data = pd.read_csv(StringIO(response.text), sep='\t')
            
            # Make sure we have all required columns, even if empty
            required_columns = ["Genome", "Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species", "GC Content", "Genome Length"]
            for col in required_columns:
                if col not in data.columns:
                    data[col] = None
            
            # Ensure numeric columns are properly formatted
            if "GC Content" in data.columns:
                data["GC Content"] = pd.to_numeric(data["GC Content"], errors='coerce')
            if "Genome Length" in data.columns:
                data["Genome Length"] = pd.to_numeric(data["Genome Length"], errors='coerce')
                
            return data
        else:
            st.error(f"Failed to fetch data: HTTP {response.status_code}")
            # Return sample data as fallback
            return generate_sample_reference_data()
    except Exception as e:
        st.error(f"Error loading reference data: {e}")
        # Return sample data as fallback
        return generate_sample_reference_data()

# Function to generate sample reference data if GitHub data can't be loaded
def generate_sample_reference_data():
    """Generate sample reference data for demonstration purposes"""
    # Sample taxonomic data
    phyla = ["Firmicutes", "Bacteroidetes", "Proteobacteria", "Actinobacteria", "Verrucomicrobia"]
    classes = ["Clostridia", "Bacilli", "Bacteroidia", "Gammaproteobacteria", "Actinobacteria"]
    orders = ["Clostridiales", "Lactobacillales", "Bacteroidales", "Enterobacterales", "Bifidobacteriales"]
    families = ["Lachnospiraceae", "Ruminococcaceae", "Bacteroidaceae", "Enterobacteriaceae", "Bifidobacteriaceae"]
    genera = ["Blautia", "Faecalibacterium", "Bacteroides", "Escherichia", "Bifidobacterium"]
    species = ["coccoides", "prausnitzii", "fragilis", "coli", "longum"]
    
    # Generate 30 sample entries
    n_samples = 30
    data = {
        "Genome": [f"Genome_{i+1}" for i in range(n_samples)],
        "Domain": ["Bacteria"] * n_samples,
        "Phylum": np.random.choice(phyla, n_samples),
        "Class": np.random.choice(classes, n_samples),
        "Order": np.random.choice(orders, n_samples),
        "Family": np.random.choice(families, n_samples),
        "Genus": np.random.choice(genera, n_samples),
        "Species": np.random.choice(species, n_samples),
        "GC Content": np.random.uniform(30, 70, n_samples).round(2),
        "Genome Length": np.random.randint(1500000, 8000000, n_samples)
    }
    
    # Ensure species names match genera
    for i in range(n_samples):
        if data["Genus"][i] == "Blautia":
            data["Species"][i] = "coccoides"
        elif data["Genus"][i] == "Faecalibacterium":
            data["Species"][i] = "prausnitzii"
        elif data["Genus"][i] == "Bacteroides":
            data["Species"][i] = "fragilis"
        elif data["Genus"][i] == "Escherichia":
            data["Species"][i] = "coli"
        elif data["Genus"][i] == "Bifidobacterium":
            data["Species"][i] = "longum"
    
    return pd.DataFrame(data)

# Function for the database reference page
def show_database_page():
    st.title("GutBiomeIndex Database")
    st.write("""
    This database contains reference information for common gut microbiome species.
    Use the filters below to explore specific taxonomic groups or search for particular microbes.
    """)
    
    # Load the reference data
    reference_data = load_reference_data()
    
    # Add sidebar filters
    st.sidebar.header("Database Filters")
    
    # Filter by taxonomy
    taxonomy_filters = {}

    # Create filters for each taxonomic level - adapt to available columns in the data
    available_tax_levels = [level for level in ["Domain", "Phylum", "Class", "Order", "Family", "Genus"] 
                           if level in reference_data.columns and reference_data[level].notna().any()]
    
    for tax_level in available_tax_levels:
        unique_values = sorted(reference_data[tax_level].dropna().unique())
        selected = st.sidebar.multiselect(f"Filter by {tax_level}", unique_values)
        if selected:
            taxonomy_filters[tax_level] = selected
    
    # Apply taxonomy filters
    filtered_data = reference_data.copy()
    for tax_level, values in taxonomy_filters.items():
        filtered_data = filtered_data[filtered_data[tax_level].isin(values)]
    
    # Additional numeric filters - only if these columns exist and contain data
    if "GC Content" in reference_data.columns and reference_data["GC Content"].notna().any():
        gc_min = float(reference_data["GC Content"].min())
        gc_max = float(reference_data["GC Content"].max())
        gc_range = st.sidebar.slider(
            "GC Content Range (%)",
            gc_min,
            gc_max,
            (gc_min, gc_max)
        )
        filtered_data = filtered_data[
            (filtered_data["GC Content"] >= gc_range[0]) & 
            (filtered_data["GC Content"] <= gc_range[1])
        ]
    
    if "Genome Length" in reference_data.columns and reference_data["Genome Length"].notna().any():
        length_min = int(reference_data["Genome Length"].min())
        length_max = int(reference_data["Genome Length"].max())
        genome_length_range = st.sidebar.slider(
            "Genome Length Range (bp)",
            length_min,
            length_max,
            (length_min, length_max)
        )
        filtered_data = filtered_data[
            (filtered_data["Genome Length"] >= genome_length_range[0]) & 
            (filtered_data["Genome Length"] <= genome_length_range[1])
        ]
    
    # Search functionality
    search_term = st.text_input("Search by genome name or taxonomy:")
    if search_term:
        search_mask = filtered_data.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
        filtered_data = filtered_data[search_mask]
    
    # Show results
    st.subheader(f"Database Results ({len(filtered_data)} entries)")
    st.dataframe(filtered_data)
    
    # Download filtered data
    if not filtered_data.empty:
        st.download_button(
            label="Download Filtered Data",
            data=filtered_data.to_csv(index=False),
            file_name="microbiome_filtered_data.csv",
            mime="text/csv"
        )
    
    # Visualizations - only if we have data to visualize
    if not filtered_data.empty:
        st.subheader("Database Visualizations")
        
        # Determine which visualizations are possible based on available data
        has_taxonomy = any(tax in filtered_data.columns and filtered_data[tax].notna().any() 
                         for tax in ["Domain", "Phylum", "Class", "Order", "Family", "Genus"])
        has_gc = "GC Content" in filtered_data.columns and filtered_data["GC Content"].notna().any()
        has_genome_length = "Genome Length" in filtered_data.columns and filtered_data["Genome Length"].notna().any()
        
        if has_taxonomy or has_gc:
            col1, col2 = st.columns(2)
            
            with col1:
                if has_taxonomy:
                    # Taxonomic distribution
                    st.write("Taxonomic Distribution")
                    available_tax_levels = [level for level in ["Domain", "Phylum", "Class", "Order", "Family", "Genus"] 
                                          if level in filtered_data.columns and filtered_data[level].notna().any()]
                    
                    if available_tax_levels:
                        tax_level = st.selectbox(
                            "Select taxonomic level for distribution chart:", 
                            available_tax_levels
                        )
                        
                        # Count the occurrences
                        tax_counts = filtered_data[tax_level].value_counts()
                        
                        # Create pie chart
                        fig = px.pie(
                            values=tax_counts.values,
                            names=tax_counts.index,
                            title=f"Distribution by {tax_level}"
                        )
                        st.plotly_chart(fig)
            
            with col2:
                if has_gc:
                    # GC Content distribution
                    st.write("GC Content Distribution")
                    fig = px.histogram(
                        filtered_data,
                        x="GC Content",
                        nbins=20,
                        title="GC Content Distribution"
                    )
                    st.plotly_chart(fig)
        
        # Genome Length vs GC Content - only if both are available
        if has_gc and has_genome_length:
            st.write("Genome Length vs GC Content")
            
            # Determine color column - prefer Phylum but fall back to another taxonomic level if needed
            color_options = [col for col in ["Phylum", "Domain", "Class", "Order", "Family", "Genus"] 
                            if col in filtered_data.columns and filtered_data[col].notna().any()]
            color_col = color_options[0] if color_options else None
            
            fig = px.scatter(
                filtered_data,
                x="Genome Length",
                y="GC Content",
                color=color_col,
                hover_name=filtered_data.index if "Genome" not in filtered_data.columns else "Genome",
                hover_data=[col for col in ["Genus", "Species"] if col in filtered_data.columns],
                title="Genome Length vs GC Content"
            )
            st.plotly_chart(fig)

# Function for the analysis tool page
def show_analysis_tool():
    st.title("GutBiomeIndex Analysis Tool")
    st.write("Upload your microbiome data to analyze alpha and beta diversity")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "tsv", "txt"])

    if uploaded_file is not None:
        # Try to infer data format
        file_type = None
        try:
            # Try to automatically detect file format
            header = uploaded_file.readline().decode('utf-8').strip()
            uploaded_file.seek(0)  # Reset file pointer
            
            if ',' in header:
                file_type = 'csv'
                df = pd.read_csv(uploaded_file, index_col=0)
            else:
                file_type = 'tsv'
                df = pd.read_csv(uploaded_file, sep='\t', index_col=0)
                
            # Force all columns to be numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
            # Reset index to start from 1
            df.index = range(1, len(df) + 1)
        except Exception as e:
            st.error(f"Error automatically detecting file format: {e}")
            
            # If auto-detection fails, try both formats
            try:
                # Try CSV format
                uploaded_file.seek(0)  # Reset file pointer
                df = pd.read_csv(uploaded_file, index_col=0)
                file_type = 'csv'
            except:
                try:
                    # Try TSV format
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(uploaded_file, sep='\t', index_col=0)
                    file_type = 'tsv'
                except Exception as e2:
                    st.error(f"Failed to read file in any common format: {e2}")
                    st.stop()
                    
            # Force all columns to be numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
            # Reset index to start from 1
            df.index = range(1, len(df) + 1)
        
        # Display file info
        st.write(f"File detected as: {file_type}")
        st.write(f"Dimensions: {df.shape[0]} samples × {df.shape[1]} features")
        
        # Display the first few rows of the dataframe
        st.subheader("Preview of your data")
        st.dataframe(df.head())
        
        # Sidebar for analysis options
        st.sidebar.title("Analysis Options")
        
        # Alpha diversity analysis
        st.sidebar.subheader("Alpha Diversity")
        alpha_metrics = st.sidebar.multiselect(
            "Select Alpha Diversity Metrics",
            ["Shannon Index", "Simpson Index", "Observed Species", "Chao1"],
            default=["Shannon Index"]
        )
        
        if alpha_metrics:
            st.subheader("Alpha Diversity Metrics")
            
            # Create a dataframe to store alpha diversity metrics
            alpha_results = pd.DataFrame(index=df.index)
            
            if "Shannon Index" in alpha_metrics:
                # Calculate Shannon Index
                def shannon_index(row):
                    row_sum = row.sum()
                    if row_sum == 0:
                        return 0
                    proportions = row / row_sum
                    # Remove zeros to avoid log(0)
                    proportions = proportions[proportions > 0]
                    return -np.sum(proportions * np.log(proportions))
                
                alpha_results["Shannon Index"] = df.apply(shannon_index, axis=1)
            
            if "Simpson Index" in alpha_metrics:
                # Calculate Simpson Index
                def simpson_index(row):
                    row_sum = row.sum()
                    if row_sum == 0:
                        return 0
                    proportions = row / row_sum
                    return 1 - np.sum(proportions ** 2)
                
                alpha_results["Simpson Index"] = df.apply(simpson_index, axis=1)
            
            if "Observed Species" in alpha_metrics:
                # Calculate Observed Species (richness)
                alpha_results["Observed Species"] = df.apply(lambda row: np.sum(row > 0), axis=1)
            
            if "Chao1" in alpha_metrics:
                # Calculate Chao1 estimator
                def chao1(row):
                    # Count singletons and doubletons
                    counts = pd.Series(row[row > 0]).value_counts()
                    singletons = counts.get(1, 0)
                    doubletons = counts.get(2, 0)
                    observed = np.sum(row > 0)
                    
                    # Avoid division by zero
                    if doubletons == 0:
                        doubletons = 1
                    
                    return observed + (singletons * (singletons - 1)) / (2 * doubletons)
                
                alpha_results["Chao1"] = df.apply(chao1, axis=1)
            
            # Display alpha diversity results
            st.dataframe(alpha_results)
            
            # Plot alpha diversity metrics
            for metric in alpha_metrics:
                fig = px.box(alpha_results, y=metric, title=f"{metric} Distribution")
                st.plotly_chart(fig)
        
        # Beta diversity analysis
        st.sidebar.subheader("Beta Diversity")
        beta_metrics = st.sidebar.multiselect(
            "Select Beta Diversity Metrics",
            ["Bray-Curtis Dissimilarity", "Jaccard Distance", "Euclidean Distance"],
            default=["Bray-Curtis Dissimilarity"]
        )
        
        if beta_metrics:
            st.subheader("Beta Diversity Analysis")
            
            # Create a function to calculate beta diversity metrics
            def calculate_beta_diversity(data, metric):
                if metric == "Bray-Curtis Dissimilarity":
                    # Calculate Bray-Curtis dissimilarity
                    def bray_curtis(u, v):
                        return np.sum(np.abs(u - v)) / (np.sum(u) + np.sum(v))
                    
                    distance_matrix = squareform(pdist(data, bray_curtis))
                
                elif metric == "Jaccard Distance":
                    # Calculate Jaccard distance
                    def jaccard(u, v):
                        u_binary = u > 0
                        v_binary = v > 0
                        intersection = np.sum(u_binary & v_binary)
                        union = np.sum(u_binary | v_binary)
                        return 1 - (intersection / union if union > 0 else 0)
                    
                    distance_matrix = squareform(pdist(data, jaccard))
                
                elif metric == "Euclidean Distance":
                    # Calculate Euclidean distance
                    distance_matrix = squareform(pdist(data, 'euclidean'))
                
                return distance_matrix
            
            # Perform PCA on the data
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df)
            
            # Create a dataframe with PCA results
            pca_df = pd.DataFrame({
                'PC1': pca_result[:, 0],
                'PC2': pca_result[:, 1],
                'Sample': df.index
            })
            
            # Plot PCA results
            st.subheader("Principal Component Analysis")
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                text='Sample',
                title=f"PCA Plot (Explained Variance: PC1 {pca.explained_variance_ratio_[0]:.2%}, PC2 {pca.explained_variance_ratio_[1]:.2%})"
            )
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig)
            
            # Calculate and visualize beta diversity metrics
            for metric in beta_metrics:
                distance_matrix = calculate_beta_diversity(df.values, metric)
                
                # Create a heatmap of the distance matrix
                fig = px.imshow(
                    distance_matrix,
                    labels=dict(x="Sample", y="Sample", color=metric),
                    x=df.index,
                    y=df.index,
                    title=f"{metric} Heatmap"
                )
                st.plotly_chart(fig)
        
        # Additional analysis options
        st.sidebar.subheader("Additional Analysis")
        additional_analyses = st.sidebar.multiselect(
            "Select Additional Analyses",
            ["Feature Abundance", "Correlation Analysis"],
            default=[]
        )
        
        if "Feature Abundance" in additional_analyses:
            st.subheader("Feature Abundance Analysis")
            
            # Get the top N most abundant features
            top_n = st.slider("Select top N features", 5, 30, 10)
            
            # Calculate feature abundance
            feature_abundance = df.sum(axis=0).sort_values(ascending=False).head(top_n)
            
            # Plot feature abundance
            fig = px.bar(
                x=feature_abundance.index,
                y=feature_abundance.values,
                title=f"Top {top_n} Most Abundant Features",
                labels={'x': 'Feature', 'y': 'Abundance'}
            )
            st.plotly_chart(fig)
        
        if "Correlation Analysis" in additional_analyses:
            st.subheader("Correlation Analysis")
            
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            # Plot correlation heatmap
            fig = px.imshow(
                corr_matrix,
                labels=dict(x="Feature", y="Feature", color="Correlation"),
                title="Feature Correlation Heatmap"
            )
            st.plotly_chart(fig)
        
        # Export options
        st.sidebar.subheader("Export Options")
        if st.sidebar.button("Export Results"):
            # Create a download button for each type of result
            
            # Export raw data
            csv_data = df.to_csv(index=True)
            st.download_button(
                label="Download Raw Data (CSV)",
                data=csv_data,
                file_name="microbiome_raw_data.csv",
                mime="text/csv"
            )
            
            # Export alpha diversity results if available
            if 'alpha_results' in locals() and not alpha_results.empty:
                alpha_csv = alpha_results.to_csv(index=True)
                st.download_button(
                    label="Download Alpha Diversity Results (CSV)",
                    data=alpha_csv,
                    file_name="alpha_diversity_results.csv",
                    mime="text/csv"
                )
            
            # Export all results as JSON
            results = {
                "data": df.to_dict(),
            }
            
            if 'alpha_results' in locals():
                results["alpha_diversity"] = alpha_results.to_dict()
            
            # Convert to JSON for download
            import json
            results_json = json.dumps(results)
            
            st.download_button(
                label="Download All Results (JSON)",
                data=results_json,
                file_name="microbiome_analysis_results.json",
                mime="application/json"
            )
    else:
        st.info("Please upload a CSV or TSV file to begin analysis.")
        
        # Display sample dataset option
        if st.button("Use Sample Dataset"):
            # Generate a sample dataset
            sample_data = pd.DataFrame(
                np.random.randint(0, 100, size=(10, 20)),
                columns=[f"Feature_{i}" for i in range(1, 21)]
            )
            
            # Display the sample dataset
            st.subheader("Sample Dataset")
            st.dataframe(sample_data)
            
            # Provide download option for the sample dataset
            csv = sample_data.to_csv()
            st.download_button(
                label="Download Sample Dataset",
                data=csv,
                file_name="sample_microbiome_data.csv",
                mime="text/csv"
            )
            
            st.write("Download the sample dataset and then upload it to test the analysis features.")

# Main app
def main():
    # Application title
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Klebsiella_Pneumoniae_on_Mueller_Hinton_agar.JPG/1024px-Klebsiella_Pneumoniae_on_Mueller_Hinton_agar.JPG", width=100)
    st.sidebar.title("GutBiomeIndex Explorer")
    
    # Create a sidebar for navigation
    st.sidebar.subheader("Navigation")
    page = st.sidebar.radio("Go to", ["Database Reference", "Analysis Tool"])
    
    # Based on the page selected, show the appropriate content
    if page == "Database Reference":
        show_database_page()
    else:
        show_analysis_tool()
    
    # Add About section to sidebar
    st.sidebar.subheader("About")
    st.sidebar.info(
        "The GutBiomeIndex is a simple and powerful platform to store, analyze, "
        "and compare gut bacteria data. It includes built-in tools to calculate alpha and beta diversity, "
        "helping users understand microbial richness and differences between samples. Designed for researchers, "
        "doctors, and students, it supports common sequencing data and offers clear visual reports for "
        "better gut health analysis."
    )
    
    # Add Disclaimer section to sidebar
    st.sidebar.subheader("Disclaimer")
    st.sidebar.warning(
        "The data in this database is collected from public sources like MGnify. "
        "Microbiome composition can vary based on age, location, diet, and sex. "
        "Results may differ depending on these factors and should be interpreted accordingly."
    )
    
    # Add footer with information
    st.markdown("---")
    st.markdown("""
    **GutBiomeIndex & Analysis Tool**
    - Data source: EBI Metagenomics Sequential Genomes
    - For analysis, upload your data as a CSV or TSV file with samples as rows and features (taxa) as columns.
    - Version 1.0 (May 2025)
    """)
    
    # Add information about the application
    with st.sidebar.expander("About this app"):
        st.write("""
        This application provides access to the EBI Metagenomics Sequential Genomes database
        and tools for analyzing microbiome samples.
        
        The database contains taxonomy and genomic information for gut microbiome species.
        
        The analysis tool allows you to calculate alpha and beta diversity metrics
        for your own microbiome data.
        """)
        
    # Add citation information
    with st.sidebar.expander("How to cite"):
        st.write("""
        If you use this tool in your research, please cite:
        
        Puchalapalli et al. (2025). GutBiomeIndex: A tool for reference database browsing and diversity analysis.
        
        Data source: Mitchell et al. (2024). EBI Metagenomics Sequential Genomes Database.
        """)


# Run the app
if __name__ == "__main__":
    main()
