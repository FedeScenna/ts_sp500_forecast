from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
import json
import os
from datetime import datetime
import requests
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

class CSVAnalyzer:
    def __init__(self):
        self.df = None
        self.df_info = None
        self.chat_history = []
        
    def load_csv(self, file_path):
        """Load and analyze CSV file with chunking for large files"""
        try:
            # First, try to read the entire file
            try:
                self.df = pd.read_csv(file_path)
            except MemoryError:
                # If too large, read in chunks and sample
                print("Large file detected, sampling data...")
                chunk_list = []
                chunk_size = 10000
                sample_size = 50000  # Maximum rows to keep
                
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    chunk_list.append(chunk)
                    if len(chunk_list) * chunk_size >= sample_size:
                        break
                
                self.df = pd.concat(chunk_list, ignore_index=True)
                print(f"Sampled {len(self.df)} rows from large dataset")
            
            # Basic data validation and cleaning
            self.df = self.df.dropna(how='all')  # Remove completely empty rows
            self.df = self.df.loc[:, self.df.notna().any()]  # Remove completely empty columns
            
            self.df_info = self._get_dataframe_info()
            return True, f"Successfully loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns"
        except Exception as e:
            return False, f"Error loading CSV: {str(e)}"
    
    def _get_dataframe_info(self):
        """Extract key information about the dataframe"""
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'sample_data': self.df.head().to_dict('records')
        }
        return info
    
    def perform_basic_analysis(self):
        """Perform basic statistical analysis"""
        analysis = {}
        
        # Basic statistics for numeric columns
        if self.df_info['numeric_columns']:
            analysis['numeric_stats'] = self.df[self.df_info['numeric_columns']].describe().to_dict()
        
        # Categorical analysis
        if self.df_info['categorical_columns']:
            analysis['categorical_stats'] = {}
            for col in self.df_info['categorical_columns']:
                analysis['categorical_stats'][col] = {
                    'unique_values': self.df[col].nunique(),
                    'top_values': self.df[col].value_counts().head().to_dict()
                }
        
        return analysis
    
    def detect_anomalies(self, columns=None):
        """Detect anomalies using Isolation Forest"""
        if not self.df_info['numeric_columns']:
            return None, "No numeric columns available for anomaly detection"
        
        try:
            # Use specified columns or all numeric columns
            cols_to_analyze = columns if columns else self.df_info['numeric_columns']
            cols_to_analyze = [col for col in cols_to_analyze if col in self.df_info['numeric_columns']]
            
            if not cols_to_analyze:
                return None, "No valid numeric columns for analysis"
            
            # Prepare data
            data = self.df[cols_to_analyze].dropna()
            if len(data) < 10:
                return None, "Not enough data points for anomaly detection"
            
            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # Detect anomalies
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(data_scaled)
            
            # Get anomaly indices
            anomaly_indices = data.index[anomaly_labels == -1].tolist()
            anomalies = self.df.loc[anomaly_indices, cols_to_analyze + ['index'] if 'index' in self.df.columns else cols_to_analyze]
            
            return {
                'total_anomalies': len(anomaly_indices),
                'anomaly_percentage': round((len(anomaly_indices) / len(data)) * 100, 2),
                'columns_analyzed': cols_to_analyze,
                'sample_anomalies': anomalies.head(10).to_dict('records')
            }, None
            
        except Exception as e:
            return None, f"Error in anomaly detection: {str(e)}"
    
    def create_visualization(self, viz_type, columns=None, title=""):
        """Create visualizations and return as base64 encoded string"""
        try:
            plt.figure(figsize=(10, 6))
            plt.style.use('seaborn-v0_8')
            
            if viz_type == 'correlation' and len(self.df_info['numeric_columns']) > 1:
                corr_matrix = self.df[self.df_info['numeric_columns']].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title(title or 'Correlation Matrix')
                
            elif viz_type == 'distribution' and columns:
                col = columns[0] if isinstance(columns, list) else columns
                if col in self.df_info['numeric_columns']:
                    sns.histplot(data=self.df, x=col, kde=True)
                    plt.title(title or f'Distribution of {col}')
                elif col in self.df_info['categorical_columns']:
                    sns.countplot(data=self.df, x=col)
                    plt.xticks(rotation=45)
                    plt.title(title or f'Count of {col}')
                    
            elif viz_type == 'boxplot' and columns:
                col = columns[0] if isinstance(columns, list) else columns
                if col in self.df_info['numeric_columns']:
                    sns.boxplot(data=self.df, y=col)
                    plt.title(title or f'Box Plot of {col}')
                    
            elif viz_type == 'scatter' and columns and len(columns) >= 2:
                if all(col in self.df_info['numeric_columns'] for col in columns[:2]):
                    sns.scatterplot(data=self.df, x=columns[0], y=columns[1])
                    plt.title(title or f'{columns[0]} vs {columns[1]}')
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            return img_base64, None
            
        except Exception as e:
            plt.close()
            return None, f"Error creating visualization: {str(e)}"

# Global analyzer instance
analyzer = CSVAnalyzer()

def load_default_csv():
    """Load CSV file from data folder on startup"""
    data_folder = 'data'
    if not os.path.exists(data_folder):
        print("No 'data' folder found. Upload functionality will be available.")
        return False
    
    # Look for CSV files in data folder
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in 'data' folder. Upload functionality will be available.")
        return False
    
    # Use the first CSV file found
    csv_file = csv_files[0]
    filepath = os.path.join(data_folder, csv_file)
    
    print(f"Loading default CSV: {csv_file}")
    success, message = analyzer.load_csv(filepath)
    
    if success:
        print(f"✅ {message}")
        if len(csv_files) > 1:
            print(f"Note: Found {len(csv_files)} CSV files. Using '{csv_file}'. Others: {csv_files[1:]}")
        return True
    else:
        print(f"❌ Error loading {csv_file}: {message}")
        return False

def query_ollama(prompt, context=""):
    """Query Ollama API with context about the CSV data"""
    try:
        system_prompt = f"""You are a helpful data analyst. You have access to a CSV dataset with the following information:
{context}

Provide conversational, insightful responses about the data. Focus on practical insights rather than technical jargon. 
When suggesting analyses, be specific about which columns or relationships to explore.
Keep responses concise but informative."""

        response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   'model': 'llama3.1:8b',
                                   'prompt': f"System: {system_prompt}\n\nUser: {prompt}",
                                   'stream': False
                               })
        
        if response.status_code == 200:
            response_data = response.json()
            # Ollama API returns the response in the 'response' field
            return response_data.get('response', 'No response from model')
        else:
            return f"Sorry, I couldn't connect to the AI model. HTTP Status: {response.status_code}. Please make sure Ollama is running."
            
    except requests.exceptions.ConnectionError:
        return "❌ Connection Error: Cannot connect to Ollama. Please make sure:\n1. Ollama is running in a separate terminal\n2. The service is available at http://localhost:11434\n3. You have the 'llama3' model installed (run: ollama pull llama3)"
    except requests.exceptions.Timeout:
        return "⏰ Timeout Error: Ollama is taking too long to respond. Please try again."
    except Exception as e:
        return f"❌ Error communicating with AI: {str(e)}"

@app.route('/')
def index():
    # Check if data is already loaded
    data_info = None
    basic_analysis = None
    
    if analyzer.df_info:
        data_info = analyzer.df_info
        basic_analysis = analyzer.perform_basic_analysis()
    
    return render_template('index.html', 
                         data_loaded=bool(analyzer.df_info),
                         data_info=data_info,
                         basic_analysis=basic_analysis)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    
    if not analyzer.df_info:
        return jsonify({'error': 'No CSV data loaded. Please add a CSV file to the data folder and restart the application.'})
    
    # Create context about the dataset
    context = f"""
Dataset Information:
- Shape: {analyzer.df_info['shape'][0]} rows, {analyzer.df_info['shape'][1]} columns
- Columns: {', '.join(analyzer.df_info['columns'])}
- Numeric columns: {', '.join(analyzer.df_info['numeric_columns'])}
- Categorical columns: {', '.join(analyzer.df_info['categorical_columns'])}
- Missing values: {sum(analyzer.df_info['missing_values'].values())} total
"""
    
    # Get AI response
    ai_response = query_ollama(user_message, context)
    
    # Check if user is asking for specific analysis
    response_data = {
        'message': ai_response,
        'visualizations': [],
        'analysis_results': None
    }
    
    # Simple keyword detection for triggering analyses
    user_lower = user_message.lower()
    
    if 'anomaly' in user_lower or 'outlier' in user_lower:
        anomaly_result, error = analyzer.detect_anomalies()
        if anomaly_result:
            response_data['analysis_results'] = {
                'type': 'anomaly_detection',
                'data': anomaly_result
            }
    
    if 'correlation' in user_lower and len(analyzer.df_info['numeric_columns']) > 1:
        viz_data, error = analyzer.create_visualization('correlation')
        if viz_data:
            response_data['visualizations'].append({
                'type': 'correlation',
                'data': viz_data,
                'title': 'Correlation Matrix'
            })
    
    # Add to chat history
    analyzer.chat_history.append({
        'user': user_message,
        'assistant': ai_response,
        'timestamp': datetime.now().isoformat()
    })
    
    return jsonify(response_data)

@app.route('/analyze/<analysis_type>')
def analyze(analysis_type):
    if not analyzer.df_info:
        return jsonify({'error': 'Please upload a CSV file first'})
    
    if analysis_type == 'basic':
        analysis = analyzer.perform_basic_analysis()
        return jsonify({'analysis': analysis})
    
    elif analysis_type == 'anomalies':
        result, error = analyzer.detect_anomalies()
        if error:
            return jsonify({'error': error})
        return jsonify({'analysis': result})
    
    return jsonify({'error': 'Unknown analysis type'})

@app.route('/visualize', methods=['POST'])
def visualize():
    if not analyzer.df_info:
        return jsonify({'error': 'Please upload a CSV file first'})
    
    viz_type = request.json.get('type')
    columns = request.json.get('columns')
    title = request.json.get('title', '')
    
    viz_data, error = analyzer.create_visualization(viz_type, columns, title)
    
    if error:
        return jsonify({'error': error})
    
    return jsonify({'visualization': viz_data})

@app.route('/export')
def export_analysis():
    if not analyzer.chat_history:
        return jsonify({'error': 'No analysis to export'})
    
    # Create export data
    export_data = {
        'dataset_info': analyzer.df_info,
        'chat_history': analyzer.chat_history,
        'export_timestamp': datetime.now().isoformat()
    }
    
    # Save to file
    filename = f"analysis_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join('exports', filename)
    
    os.makedirs('exports', exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    return send_file(filepath, as_attachment=True, download_name=filename)

@app.route('/data-status')
def data_status():
    """API endpoint to check if data is loaded"""
    if analyzer.df_info:
        basic_analysis = analyzer.perform_basic_analysis()
        return jsonify({
            'loaded': True,
            'data_info': analyzer.df_info,
            'basic_analysis': basic_analysis
        })
    else:
        return jsonify({'loaded': False})

if __name__ == '__main__':
    print("🚀 Starting CSV Chat Analyzer...")
    
    # Try to load default CSV from data folder
    data_loaded = load_default_csv()
    
    if data_loaded:
        print("💬 Ready to chat! Go to http://localhost:5000")
    else:
        print("📤 Ready for file upload! Go to http://localhost:5000")
    
    app.run(debug=True)