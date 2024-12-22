from flask import Flask, render_template, request, jsonify, session, make_response
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from faker import Faker
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from scipy import stats
from scipy.stats import norm
from io import StringIO
import logging

# Configure logging to only show WARNING and above for faker
logging.getLogger('faker').setLevel(logging.WARNING)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# Create a formatter
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# Create and configure stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session management
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-size
fake = Faker()

class ModelTrainer:
    def __init__(self, test_size=0.2, val_size=0.25, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        # Split models into regression and classification
        self.regression_models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
            'SVR': SVR(kernel='rbf')
        }
        self.classification_models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state)
        }
        self.scaler = StandardScaler()
        
    def train_evaluate(self, X, y):
        try:
            # Determine if this is a classification or regression problem
            is_classification = isinstance(y.dtype, pd.CategoricalDtype) or len(np.unique(y)) < 10
            
            # Select appropriate models
            self.models = self.classification_models if is_classification else self.regression_models
            logger.debug(f"Using {'classification' if is_classification else 'regression'} models")
            
            # Split into train, validation, and test sets
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, 
                test_size=self.test_size, 
                random_state=self.random_state)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, 
                test_size=self.val_size, 
                random_state=self.random_state)
            
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            results = {}
            for name, model in self.models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    
                    # Get predictions for all sets
                    train_pred = model.predict(X_train_scaled)
                    val_pred = model.predict(X_val_scaled)
                    test_pred = model.predict(X_test_scaled)
                    
                    if is_classification:
                        results[name] = {
                            'train_accuracy': float(accuracy_score(y_train, train_pred)),
                            'val_accuracy': float(accuracy_score(y_val, val_pred)),
                            'test_accuracy': float(accuracy_score(y_test, test_pred)),
                            'cv_scores': float(cross_val_score(model, X_train_scaled, y_train, cv=5).mean())
                        }
                    else:
                        results[name] = {
                            'train_mse': float(mean_squared_error(y_train, train_pred)),
                            'val_mse': float(mean_squared_error(y_val, val_pred)),
                            'test_mse': float(mean_squared_error(y_test, test_pred)),
                            'train_r2': float(r2_score(y_train, train_pred)),
                            'val_r2': float(r2_score(y_val, val_pred)),
                            'test_r2': float(r2_score(y_test, test_pred)),
                            'cv_scores': float(cross_val_score(model, X_train_scaled, y_train, cv=5).mean())
                        }
                except Exception as e:
                    logger.error(f"Error in model {name}: {str(e)}")
                    results[name] = {'error': str(e)}
            
            return results, (X_train, X_val, X_test), (y_train, y_val, y_test)
        except Exception as e:
            logger.error(f"Error in train_evaluate: {str(e)}")
            raise

class SyntheticDataGenerator:
    def __init__(self):
        self.faker = Faker()
        self.label_encoders = {}
        
    def generate_feature_data(self, num_samples, feature_config):
        try:
            data = {}
            
            for class_name, class_config in feature_config.items():
                class_data = {}
                for feature, settings in class_config.items():
                    mean = float(settings['mean'])
                    std = float(settings['std'])
                    values = norm.rvs(loc=mean, scale=std, size=num_samples)
                    class_data[feature] = values
                
                data[class_name] = pd.DataFrame(class_data)
            
            return data
        except Exception as e:
            logger.error(f"Error in generate_feature_data: {str(e)}")
            raise

    def generate_data(self, num_samples, feature_config):
        try:
            class_data = self.generate_feature_data(num_samples, feature_config)
            
            combined_data = pd.concat([df.assign(class_name=class_name) 
                                     for class_name, df in class_data.items()],
                                    ignore_index=True)
            
            for column in combined_data.select_dtypes(include=['object']):
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                combined_data[column] = self.label_encoders[column].fit_transform(combined_data[column])
            
            return combined_data
        except Exception as e:
            logger.error(f"Error in generate_data: {str(e)}")
            raise

def create_visualizations(data, model_results, feature_config):
    try:
        visualizations = {}
        
        # Common layout settings
        base_layout = {
            'autosize': True,
            'margin': {'l': 50, 'r': 30, 't': 50, 'b': 50, 'pad': 4},
            'paper_bgcolor': '#1e1e1e',
            'plot_bgcolor': '#1e1e1e',
            'font': {
                'color': '#e0e0e0',
                'size': 12
            },
            'showlegend': True,
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'right',
                'x': 1,
                'bgcolor': 'rgba(30,30,30,0.8)'
            },
            'xaxis': {
                'gridcolor': '#333',
                'linecolor': '#999',
                'tickfont': {'size': 10}
            },
            'yaxis': {
                'gridcolor': '#333',
                'linecolor': '#999',
                'tickfont': {'size': 10}
            }
        }
        
        # Distribution plots for each feature by class
        for feature in data.columns:
            if feature != 'class_name' and data[feature].dtype in ['int64', 'float64']:
                fig = px.histogram(data, x=feature, color='class_name',
                                 title=f'Distribution of {feature} by Class',
                                 marginal='box',
                                 template='plotly_dark')
                
                fig.update_layout(
                    **base_layout,
                    height=400,  # Fixed height for better containment
                    bargap=0.1
                )
                visualizations[f'dist_{feature}'] = fig.to_json()
        
        # Correlation heatmap
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) > 1:
            corr_matrix = data[numerical_cols].corr()
            fig = px.imshow(corr_matrix, 
                           title='Feature Correlation Matrix',
                           color_continuous_scale='RdBu',
                           template='plotly_dark')
            
            fig.update_layout(
                **base_layout,
                height=450,  # Slightly taller for readability
                width=450   # Square aspect ratio for correlation matrix
            )
            visualizations['correlation'] = fig.to_json()
        
        # Model performance comparison
        if model_results:
            model_names = []
            metrics = []
            
            # Determine if we're dealing with classification or regression
            is_classification = 'train_accuracy' in next(iter(model_results.values()))
            
            for name, results in model_results.items():
                if 'error' not in results:
                    model_names.append(name)
                    if is_classification:
                        metrics.extend([
                            results['train_accuracy'],
                            results['val_accuracy'],
                            results['test_accuracy']
                        ])
                    else:
                        metrics.extend([
                            results['train_mse'],
                            results['val_mse'],
                            results['test_mse']
                        ])
            
            # Performance comparison plots
            if is_classification:
                fig = go.Figure(data=[
                    go.Bar(name='Training Accuracy', x=model_names, 
                          y=[results['train_accuracy'] for results in model_results.values() if 'error' not in results]),
                    go.Bar(name='Validation Accuracy', x=model_names, 
                          y=[results['val_accuracy'] for results in model_results.values() if 'error' not in results]),
                    go.Bar(name='Test Accuracy', x=model_names, 
                          y=[results['test_accuracy'] for results in model_results.values() if 'error' not in results])
                ])
                fig.update_layout(
                    **base_layout,
                    title='Model Performance Comparison (Accuracy)',
                    barmode='group',
                    height=400
                )
                visualizations['model_accuracy'] = fig.to_json()
            else:
                # MSE comparison
                fig = go.Figure(data=[
                    go.Bar(name='Training MSE', x=model_names, 
                          y=[results['train_mse'] for results in model_results.values() if 'error' not in results]),
                    go.Bar(name='Validation MSE', x=model_names, 
                          y=[results['val_mse'] for results in model_results.values() if 'error' not in results]),
                    go.Bar(name='Test MSE', x=model_names, 
                          y=[results['test_mse'] for results in model_results.values() if 'error' not in results])
                ])
                fig.update_layout(
                    **base_layout,
                    title='Model Performance Comparison (MSE)',
                    barmode='group',
                    height=400
                )
                visualizations['model_mse'] = fig.to_json()
                
                # R2 comparison
                fig = go.Figure(data=[
                    go.Bar(name='Training R²', x=model_names, 
                          y=[results['train_r2'] for results in model_results.values() if 'error' not in results]),
                    go.Bar(name='Validation R²', x=model_names, 
                          y=[results['val_r2'] for results in model_results.values() if 'error' not in results]),
                    go.Bar(name='Test R²', x=model_names, 
                          y=[results['test_r2'] for results in model_results.values() if 'error' not in results])
                ])
                fig.update_layout(
                    **base_layout,
                    title='Model Performance Comparison (R²)',
                    barmode='group',
                    height=400
                )
                visualizations['model_r2'] = fig.to_json()
            
            # Cross-validation scores
            fig = go.Figure(data=[
                go.Bar(name='CV Score', x=model_names, 
                      y=[results['cv_scores'] for results in model_results.values() if 'error' not in results])
            ])
            fig.update_layout(
                **base_layout,
                title='Cross-validation Scores',
                height=400
            )
            visualizations['cv_scores'] = fig.to_json()
        
        return visualizations
    except Exception as e:
        logger.error(f"Error in create_visualizations: {str(e)}")
        raise

generator = SyntheticDataGenerator()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_data', methods=['POST'])
def generate_data():
    try:
        # Add request validation
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        params = request.json
        logger.debug(f"Received params: {params}")
        
        if not params:
            return jsonify({'error': 'Empty request body'}), 400

        # Validate required parameters
        required_params = ['num_samples', 'feature_config', 'target_column']
        for param in required_params:
            if param not in params:
                return jsonify({'error': f'Missing required parameter: {param}'}), 400
        
        num_samples = int(params.get('num_samples', 1000))
        feature_config = params.get('feature_config', {})
        target_column = params.get('target_column')
        
        # Get split parameters with defaults
        test_size = float(params.get('test_size', 0.2))
        val_size = float(params.get('val_size', 0.25))
        random_state = int(params.get('random_state', 42))
        
        # Store parameters for download endpoint
        session['last_params'] = params
        
        # Generate synthetic data
        synthetic_data = generator.generate_data(num_samples, feature_config)
        
        # Prepare data for modeling
        X = synthetic_data.drop(columns=[target_column, 'class_name'])
        y = synthetic_data[target_column]
        
        # Create trainer with custom parameters and evaluate models
        trainer = ModelTrainer(test_size=test_size, val_size=val_size, random_state=random_state)
        model_results, (X_train, X_val, X_test), (y_train, y_val, y_test) = trainer.train_evaluate(X, y)
        
        # Create complete datasets with target
        train_data = pd.concat([pd.DataFrame(X_train, columns=X.columns), 
                              pd.Series(y_train, name=target_column)], axis=1)
        val_data = pd.concat([pd.DataFrame(X_val, columns=X.columns), 
                            pd.Series(y_val, name=target_column)], axis=1)
        test_data = pd.concat([pd.DataFrame(X_test, columns=X.columns), 
                             pd.Series(y_test, name=target_column)], axis=1)
        
        # Calculate actual split proportions
        total_samples = len(synthetic_data)
        split_info = {
            'train_size': int(len(train_data)),
            'train_ratio': float(len(train_data) / total_samples),
            'val_size': int(len(val_data)),
            'val_ratio': float(len(val_data) / total_samples),
            'test_size': int(len(test_data)),
            'test_ratio': float(len(test_data) / total_samples)
        }
        
        # Generate visualizations
        visualizations = create_visualizations(synthetic_data, model_results, feature_config)
        
        # Prepare response with limited data for display
        response_data = {
            'synthetic_data': synthetic_data.head(1000).to_dict('records'),
            'train_data': train_data.head(1000).to_dict('records'),
            'validation_data': val_data.head(1000).to_dict('records'),
            'test_data': test_data.head(1000).to_dict('records'),
            'model_results': model_results,
            'visualizations': visualizations,
            'split_info': split_info
        }
        
        logger.debug("Response data prepared successfully")
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error in generate_data: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/download/<dataset_type>')
def download_csv(dataset_type):
    if dataset_type not in ['train', 'validation', 'test']:
        return jsonify({'error': 'Invalid dataset type'}), 400
        
    try:
        # Get the parameters from session
        params = session.get('last_params')
        if not params:
            return jsonify({'error': 'No parameters available'}), 404
            
        # Regenerate the data
        synthetic_data = generator.generate_data(params['num_samples'], params['feature_config'])
        X = synthetic_data.drop(columns=[params['target_column'], 'class_name'])
        y = synthetic_data[params['target_column']]
        
        trainer = ModelTrainer(
            test_size=params.get('test_size', 0.2),
            val_size=params.get('val_size', 0.25),
            random_state=params.get('random_state', 42)
        )
        
        _, (X_train, X_val, X_test), (y_train, y_val, y_test) = trainer.train_evaluate(X, y)
        
        # Select the appropriate dataset
        if dataset_type == 'train':
            data = pd.concat([pd.DataFrame(X_train, columns=X.columns), 
                            pd.Series(y_train, name=params['target_column'])], axis=1)
        elif dataset_type == 'validation':
            data = pd.concat([pd.DataFrame(X_val, columns=X.columns), 
                            pd.Series(y_val, name=params['target_column'])], axis=1)
        else:
            data = pd.concat([pd.DataFrame(X_test, columns=X.columns), 
                            pd.Series(y_test, name=params['target_column'])], axis=1)
        
        # Create CSV in memory
        output = StringIO()
        data.to_csv(output, index=False)
        
        # Create the response
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = f"attachment; filename={dataset_type}_data.csv"
        response.headers["Content-type"] = "text/csv"
        
        return response
        
    except Exception as e:
        logger.error(f"Error in download_csv: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)